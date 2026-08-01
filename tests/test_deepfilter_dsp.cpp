// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "models/deepfilter/deepfilter_dsp.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

namespace dsp = speech_core::deepfilter_dsp;

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

bool approx(float actual, double expected, double tolerance) {
    return std::fabs(static_cast<double>(actual) - expected) <= tolerance;
}

void test_upstream_constants() {
    const dsp::Config cfg;
    const std::vector<int> expected_widths = {
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7,
        7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67,
    };
    const auto widths = dsp::make_erb_widths(cfg);
    assert(widths == expected_widths);
    assert(std::accumulate(widths.begin(), widths.end(), 0) == cfg.freq_bins);
    assert(approx(dsp::normalization_alpha(cfg), 0.99, 1e-7));

    const auto mean = dsp::make_mean_norm_state(cfg.erb_bands);
    const auto unit = dsp::make_unit_norm_state(cfg.df_bins);
    assert(approx(mean.front(), -60.0, 1e-7));
    assert(approx(mean.back(), -90.0, 1e-6));
    assert(approx(unit.front(), 0.001, 1e-9));
    assert(approx(unit.back(), 0.0001, 1e-9));

    const auto window = dsp::make_vorbis_window(cfg.fft_size);
    for (int i = 0; i < cfg.hop_size; ++i) {
        const double cola = static_cast<double>(window[static_cast<size_t>(i)]) * window[static_cast<size_t>(i)] +
                            static_cast<double>(window[static_cast<size_t>(i + cfg.hop_size)]) *
                                window[static_cast<size_t>(i + cfg.hop_size)];
        assert(std::fabs(cola - 1.0) < 2e-6);
    }
    std::printf("  PASS: upstream_constants\n");
}

void test_native_960_analysis() {
    const dsp::Config cfg;
    const auto window = dsp::make_vorbis_window(cfg.fft_size);
    const float impulse = 1.0f;
    std::vector<float> real;
    std::vector<float> imag;
    dsp::analyze(&impulse, 1, cfg, window, real, imag);

    assert(dsp::frame_count(1, cfg) == 2);
    assert(real.size() == static_cast<size_t>(2 * cfg.freq_bins));
    const double magnitude = static_cast<double>(window[480]) / 960.0;
    for (int bin : {0, 1, 17, 95, 96, 479, 480}) {
        const double expected = (bin % 2 == 0) ? magnitude : -magnitude;
        assert(approx(real[static_cast<size_t>(bin)], expected, 2e-7));
        assert(std::fabs(imag[static_cast<size_t>(bin)]) < 2e-7f);
    }
    std::printf("  PASS: native_960_analysis\n");
}

void test_streaming_roundtrip_non_aligned() {
    const dsp::Config cfg;
    const auto window = dsp::make_vorbis_window(cfg.fft_size);
    std::vector<float> input(1001);
    for (size_t i = 0; i < input.size(); ++i) {
        const double t = static_cast<double>(i) / cfg.sample_rate;
        input[i] = static_cast<float>(0.31 * std::sin(2.0 * kPi * 437.0 * t) +
                                      0.08 * std::cos(2.0 * kPi * 1903.0 * t));
    }

    std::vector<float> real;
    std::vector<float> imag;
    dsp::analyze(input.data(), input.size(), cfg, window, real, imag);
    std::vector<float> output(input.size());
    dsp::synthesize(real, imag, dsp::frame_count(input.size(), cfg),
                    cfg, window, output.data(), output.size());

    float max_error = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        max_error = std::max(max_error, std::fabs(input[i] - output[i]));
    }
    assert(max_error < 2e-5f);
    std::printf("  PASS: streaming_roundtrip_non_aligned (max_error=%.2g)\n", max_error);
}

void test_feature_normalization_and_layout() {
    const dsp::Config cfg;
    const auto widths = dsp::make_erb_widths(cfg);
    constexpr int frames = 2;
    std::vector<float> real(static_cast<size_t>(frames) * cfg.freq_bins);
    std::vector<float> imag(real.size());
    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < cfg.freq_bins; ++f) {
            const size_t index = static_cast<size_t>(t) * cfg.freq_bins + f;
            real[index] = static_cast<float>(1.0 + t + f * 0.001);
            imag[index] = static_cast<float>(-0.25 + t * 0.75 - f * 0.0002);
        }
    }

    std::vector<float> erb;
    std::vector<float> spec;
    dsp::compute_features(real, imag, frames, cfg, widths, erb, spec);
    assert(erb.size() == static_cast<size_t>(frames * cfg.erb_bands));
    assert(spec.size() == static_cast<size_t>(2 * frames * cfg.df_bins));

    const float alpha = 0.99f;
    float unit_state = 0.001f;
    const size_t channel_stride = static_cast<size_t>(frames) * cfg.df_bins;
    for (int t = 0; t < frames; ++t) {
        const float re = real[static_cast<size_t>(t) * cfg.freq_bins];
        const float im = imag[static_cast<size_t>(t) * cfg.freq_bins];
        unit_state = std::sqrt(re * re + im * im) * (1.0f - alpha) + unit_state * alpha;
        const float divisor = std::sqrt(unit_state);
        const size_t index = static_cast<size_t>(t) * cfg.df_bins;
        assert(approx(spec[index], re / divisor, 2e-6));
        assert(approx(spec[channel_stride + index], im / divisor, 2e-6));
    }

    float first_power = 0.0f;
    for (int f = 0; f < widths.front(); ++f) {
        first_power += real[static_cast<size_t>(f)] * real[static_cast<size_t>(f)] +
                       imag[static_cast<size_t>(f)] * imag[static_cast<size_t>(f)];
    }
    first_power /= static_cast<float>(widths.front());
    const float first_db = 10.0f * std::log10(first_power + 1e-10f);
    const float first_state = first_db * (1.0f - alpha) - 60.0f * alpha;
    assert(approx(erb[0], (first_db - first_state) / 40.0f, 2e-6));
    std::printf("  PASS: feature_normalization_and_layout\n");
}

void test_immutable_deep_filter_and_fusion() {
    const dsp::Config cfg;
    const auto widths = dsp::make_erb_widths(cfg);
    constexpr int frames = 4;
    std::vector<float> real(static_cast<size_t>(frames) * cfg.freq_bins);
    std::vector<float> imag(real.size());
    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < cfg.freq_bins; ++f) {
            const size_t index = static_cast<size_t>(t) * cfg.freq_bins + f;
            real[index] = static_cast<float>(100 * t + f + 1);
            imag[index] = static_cast<float>(-10 * t - f * 0.1f);
        }
    }

    std::vector<float> mask(static_cast<size_t>(frames) * cfg.erb_bands, 0.25f);
    std::vector<float> coefs(
        static_cast<size_t>(cfg.df_order) * frames * cfg.df_bins * 2, 0.0f);
    // tap=1 maps output t to original t-1 (pad_before=2). If filtering is
    // performed in place, later frames incorrectly see already-filtered data.
    for (int t = 0; t < frames; ++t) {
        for (int f = 0; f < cfg.df_bins; ++f) {
            const size_t index =
                ((static_cast<size_t>(1) * frames + t) * cfg.df_bins + f) * 2;
            coefs[index] = 1.0f;
        }
    }

    std::vector<float> enhanced_real;
    std::vector<float> enhanced_imag;
    dsp::apply_network_output(real, imag, mask.data(), coefs.data(), frames,
                              cfg, widths, enhanced_real, enhanced_imag);

    for (int f : {0, 17, 95}) {
        assert(enhanced_real[static_cast<size_t>(f)] == 0.0f);
        for (int t = 1; t < frames; ++t) {
            const size_t output_index = static_cast<size_t>(t) * cfg.freq_bins + f;
            const size_t source_index = static_cast<size_t>(t - 1) * cfg.freq_bins + f;
            assert(enhanced_real[output_index] == real[source_index]);
            assert(enhanced_imag[output_index] == imag[source_index]);
        }
    }
    const int high_bin = cfg.df_bins;
    for (int t = 0; t < frames; ++t) {
        const size_t index = static_cast<size_t>(t) * cfg.freq_bins + high_bin;
        assert(enhanced_real[index] == real[index] * 0.25f);
        assert(enhanced_imag[index] == imag[index] * 0.25f);
    }
    std::printf("  PASS: immutable_deep_filter_and_fusion\n");
}

}  // namespace

int main() {
    std::printf("test_deepfilter_dsp:\n");
    test_upstream_constants();
    test_native_960_analysis();
    test_streaming_roundtrip_non_aligned();
    test_feature_normalization_and_layout();
    test_immutable_deep_filter_and_fusion();
    std::printf("All DeepFilterNet3 DSP tests passed.\n");
    return 0;
}
