#include "deepfilter_dsp.h"

#include "kissfft/kiss_fftr.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace speech_core::deepfilter_dsp {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

void validate_config(const Config& cfg) {
    if (cfg.fft_size <= 0 || cfg.fft_size % 2 != 0 ||
        cfg.hop_size <= 0 || cfg.hop_size * 2 > cfg.fft_size ||
        cfg.erb_bands <= 0 || cfg.df_bins <= 0 ||
        cfg.df_bins > cfg.freq_bins || cfg.df_order <= 0 ||
        cfg.df_lookahead < 0 || cfg.df_lookahead >= cfg.df_order ||
        cfg.freq_bins != cfg.fft_size / 2 + 1 ||
        cfg.sample_rate <= 0 || cfg.min_erb_freqs <= 0 ||
        cfg.norm_tau <= 0.0f) {
        throw std::invalid_argument("DeepFilterNet3: invalid DSP configuration");
    }
}

void validate_widths(const std::vector<int>& widths, const Config& cfg) {
    if (widths.size() != static_cast<size_t>(cfg.erb_bands) ||
        std::any_of(widths.begin(), widths.end(), [](int width) { return width <= 0; }) ||
        std::accumulate(widths.begin(), widths.end(), 0) != cfg.freq_bins) {
        throw std::invalid_argument("DeepFilterNet3: invalid ERB widths");
    }
}

class KissRealPlan {
public:
    KissRealPlan(int fft_size, bool inverse)
        : cfg_(kiss_fftr_alloc(fft_size, inverse ? 1 : 0, nullptr, nullptr)) {
        if (!cfg_) {
            throw std::runtime_error("DeepFilterNet3: kiss_fftr_alloc failed");
        }
    }

    ~KissRealPlan() { kiss_fftr_free(cfg_); }

    KissRealPlan(const KissRealPlan&) = delete;
    KissRealPlan& operator=(const KissRealPlan&) = delete;

    kiss_fftr_cfg get() const { return cfg_; }

private:
    kiss_fftr_cfg cfg_;
};

std::vector<float> linspace(float first, float last, int count) {
    if (count <= 0) return {};
    if (count == 1) return {first};
    std::vector<float> values(static_cast<size_t>(count));
    const float step = (last - first) / static_cast<float>(count - 1);
    for (int i = 0; i < count; ++i) {
        values[static_cast<size_t>(i)] = first + static_cast<float>(i) * step;
    }
    return values;
}

}  // namespace

std::vector<int> make_erb_widths(const Config& cfg) {
    validate_config(cfg);

    auto freq_to_erb = [](float frequency) {
        return 9.265f * std::log1p(frequency / (24.7f * 9.265f));
    };
    auto erb_to_freq = [](float erb) {
        return 24.7f * 9.265f * (std::exp(erb / 9.265f) - 1.0f);
    };

    const float nyquist = static_cast<float>(cfg.sample_rate / 2);
    const float bin_width = static_cast<float>(cfg.sample_rate) /
                            static_cast<float>(cfg.fft_size);
    const float erb_low = freq_to_erb(0.0f);
    const float erb_high = freq_to_erb(nyquist);
    const float step = (erb_high - erb_low) / static_cast<float>(cfg.erb_bands);

    std::vector<int> widths(static_cast<size_t>(cfg.erb_bands), 0);
    int previous_bin = 0;
    int bins_carried = 0;
    for (int band = 1; band <= cfg.erb_bands; ++band) {
        const float frequency = erb_to_freq(erb_low + static_cast<float>(band) * step);
        const int boundary_bin = static_cast<int>(std::round(frequency / bin_width));
        int width = boundary_bin - previous_bin - bins_carried;
        if (width < cfg.min_erb_freqs) {
            bins_carried = cfg.min_erb_freqs - width;
            width = cfg.min_erb_freqs;
        } else {
            bins_carried = 0;
        }
        widths[static_cast<size_t>(band - 1)] = width;
        previous_bin = boundary_bin;
    }

    // Include Nyquist, then remove any enforced-bin overflow from the final
    // band exactly as libdf::erb_fb does.
    ++widths.back();
    const int excess = std::accumulate(widths.begin(), widths.end(), 0) - cfg.freq_bins;
    if (excess > 0) widths.back() -= excess;
    validate_widths(widths, cfg);
    return widths;
}

std::vector<float> make_vorbis_window(int fft_size) {
    if (fft_size <= 0) {
        throw std::invalid_argument("DeepFilterNet3: invalid Vorbis window size");
    }
    std::vector<float> window(static_cast<size_t>(fft_size));
    for (int i = 0; i < fft_size; ++i) {
        const double inner = std::sin(kPi * (static_cast<double>(i) + 0.5) /
                                      static_cast<double>(fft_size));
        window[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(0.5 * kPi * inner * inner));
    }
    return window;
}

std::vector<float> make_mean_norm_state(int erb_bands) {
    return linspace(-60.0f, -90.0f, erb_bands);
}

std::vector<float> make_unit_norm_state(int df_bins) {
    return linspace(0.001f, 0.0001f, df_bins);
}

float normalization_alpha(const Config& cfg) {
    validate_config(cfg);
    const double dt = static_cast<double>(cfg.hop_size) /
                      static_cast<double>(cfg.sample_rate);
    const double unrounded = std::exp(-dt / static_cast<double>(cfg.norm_tau));
    double scale = 1000.0;
    for (;;) {
        const double rounded = std::round(unrounded * scale) / scale;
        if (rounded < 1.0) return static_cast<float>(rounded);
        scale *= 10.0;
        if (!std::isfinite(scale)) return static_cast<float>(unrounded);
    }
}

int frame_count(size_t input_length, const Config& cfg) {
    validate_config(cfg);
    const size_t padded = input_length + static_cast<size_t>(cfg.fft_size);
    if (padded < input_length) {
        throw std::overflow_error("DeepFilterNet3: input length overflow");
    }
    const size_t frames = padded / static_cast<size_t>(cfg.hop_size);
    if (frames > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("DeepFilterNet3: too many STFT frames");
    }
    return static_cast<int>(frames);
}

void analyze(const float* audio, size_t length,
             const Config& cfg, const std::vector<float>& window,
             std::vector<float>& spec_real,
             std::vector<float>& spec_imag) {
    validate_config(cfg);
    if (length > 0 && !audio) {
        throw std::invalid_argument("DeepFilterNet3: null audio input");
    }
    if (window.size() != static_cast<size_t>(cfg.fft_size)) {
        throw std::invalid_argument("DeepFilterNet3: invalid analysis window");
    }

    const int frames = frame_count(length, cfg);
    const size_t spectrum_size = static_cast<size_t>(frames) * cfg.freq_bins;
    spec_real.assign(spectrum_size, 0.0f);
    spec_imag.assign(spectrum_size, 0.0f);

    const size_t history = static_cast<size_t>(cfg.fft_size - cfg.hop_size);
    const size_t padded_size = history + length + static_cast<size_t>(cfg.fft_size);
    if (padded_size < length) {
        throw std::overflow_error("DeepFilterNet3: padded input length overflow");
    }
    std::vector<float> padded(padded_size, 0.0f);
    if (length > 0) std::copy_n(audio, length, padded.data() + history);

    KissRealPlan plan(cfg.fft_size, false);
    std::vector<kiss_fft_scalar> frame(static_cast<size_t>(cfg.fft_size));
    std::vector<kiss_fft_cpx> spectrum(static_cast<size_t>(cfg.freq_bins));
    const float scale = (2.0f * static_cast<float>(cfg.hop_size)) /
                        (static_cast<float>(cfg.fft_size) *
                         static_cast<float>(cfg.fft_size));

    for (int t = 0; t < frames; ++t) {
        const size_t start = static_cast<size_t>(t) * cfg.hop_size;
        for (int i = 0; i < cfg.fft_size; ++i) {
            frame[static_cast<size_t>(i)] =
                padded[start + static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }
        kiss_fftr(plan.get(), frame.data(), spectrum.data());
        const size_t base = static_cast<size_t>(t) * cfg.freq_bins;
        for (int f = 0; f < cfg.freq_bins; ++f) {
            spec_real[base + static_cast<size_t>(f)] =
                spectrum[static_cast<size_t>(f)].r * scale;
            spec_imag[base + static_cast<size_t>(f)] =
                spectrum[static_cast<size_t>(f)].i * scale;
        }
    }
}

void compute_features(const std::vector<float>& spec_real,
                      const std::vector<float>& spec_imag,
                      int num_frames,
                      const Config& cfg,
                      const std::vector<int>& erb_widths,
                      std::vector<float>& feat_erb,
                      std::vector<float>& feat_spec) {
    validate_config(cfg);
    validate_widths(erb_widths, cfg);
    if (num_frames < 0 ||
        spec_real.size() != static_cast<size_t>(num_frames) * cfg.freq_bins ||
        spec_imag.size() != spec_real.size()) {
        throw std::invalid_argument("DeepFilterNet3: invalid spectrum shape");
    }

    feat_erb.assign(static_cast<size_t>(num_frames) * cfg.erb_bands, 0.0f);
    feat_spec.assign(static_cast<size_t>(2) * num_frames * cfg.df_bins, 0.0f);
    std::vector<float> mean_state = make_mean_norm_state(cfg.erb_bands);
    std::vector<float> unit_state = make_unit_norm_state(cfg.df_bins);
    const float alpha = normalization_alpha(cfg);
    const float one_minus_alpha = 1.0f - alpha;
    const size_t spec_channel_stride = static_cast<size_t>(num_frames) * cfg.df_bins;

    for (int t = 0; t < num_frames; ++t) {
        const size_t spectrum_base = static_cast<size_t>(t) * cfg.freq_bins;
        int frequency = 0;
        for (int band = 0; band < cfg.erb_bands; ++band) {
            const int width = erb_widths[static_cast<size_t>(band)];
            float power = 0.0f;
            for (int j = 0; j < width; ++j, ++frequency) {
                const float re = spec_real[spectrum_base + static_cast<size_t>(frequency)];
                const float im = spec_imag[spectrum_base + static_cast<size_t>(frequency)];
                power += re * re + im * im;
            }
            power /= static_cast<float>(width);
            const float x = 10.0f * std::log10(power + 1e-10f);
            float& state = mean_state[static_cast<size_t>(band)];
            state = x * one_minus_alpha + state * alpha;
            feat_erb[static_cast<size_t>(t) * cfg.erb_bands + band] =
                (x - state) / 40.0f;
        }

        const size_t feature_base = static_cast<size_t>(t) * cfg.df_bins;
        for (int f = 0; f < cfg.df_bins; ++f) {
            const float re = spec_real[spectrum_base + static_cast<size_t>(f)];
            const float im = spec_imag[spectrum_base + static_cast<size_t>(f)];
            const float magnitude = std::sqrt(re * re + im * im);
            float& state = unit_state[static_cast<size_t>(f)];
            state = magnitude * one_minus_alpha + state * alpha;
            const float divisor = std::sqrt(state);
            feat_spec[feature_base + static_cast<size_t>(f)] = re / divisor;
            feat_spec[spec_channel_stride + feature_base + static_cast<size_t>(f)] = im / divisor;
        }
    }
}

void apply_network_output(const std::vector<float>& spec_real,
                          const std::vector<float>& spec_imag,
                          const float* erb_mask,
                          const float* df_coefs,
                          int num_frames,
                          const Config& cfg,
                          const std::vector<int>& erb_widths,
                          std::vector<float>& enhanced_real,
                          std::vector<float>& enhanced_imag) {
    validate_config(cfg);
    validate_widths(erb_widths, cfg);
    if (!erb_mask || !df_coefs || num_frames < 0 ||
        spec_real.size() != static_cast<size_t>(num_frames) * cfg.freq_bins ||
        spec_imag.size() != spec_real.size()) {
        throw std::invalid_argument("DeepFilterNet3: invalid network output inputs");
    }

    enhanced_real.resize(spec_real.size());
    enhanced_imag.resize(spec_imag.size());

    // ERB mask uses a piecewise-constant inverse filterbank. Apply it to the
    // original spectrum; low bins are replaced by deep filtering below.
    for (int t = 0; t < num_frames; ++t) {
        const size_t base = static_cast<size_t>(t) * cfg.freq_bins;
        int frequency = 0;
        for (int band = 0; band < cfg.erb_bands; ++band) {
            const float gain = erb_mask[static_cast<size_t>(t) * cfg.erb_bands + band];
            const int width = erb_widths[static_cast<size_t>(band)];
            for (int j = 0; j < width; ++j, ++frequency) {
                const size_t index = base + static_cast<size_t>(frequency);
                enhanced_real[index] = spec_real[index] * gain;
                enhanced_imag[index] = spec_imag[index] * gain;
            }
        }
    }

    const int pad_before = cfg.df_order - 1 - cfg.df_lookahead;
    for (int t = 0; t < num_frames; ++t) {
        for (int f = 0; f < cfg.df_bins; ++f) {
            float out_re = 0.0f;
            float out_im = 0.0f;
            for (int tap = 0; tap < cfg.df_order; ++tap) {
                const int source_t = t + tap - pad_before;
                if (source_t < 0 || source_t >= num_frames) continue;

                const size_t source_index =
                    static_cast<size_t>(source_t) * cfg.freq_bins + f;
                const size_t coefficient_index =
                    ((static_cast<size_t>(tap) * num_frames + t) * cfg.df_bins + f) * 2;
                const float x_re = spec_real[source_index];
                const float x_im = spec_imag[source_index];
                const float w_re = df_coefs[coefficient_index];
                const float w_im = df_coefs[coefficient_index + 1];
                out_re += x_re * w_re - x_im * w_im;
                out_im += x_re * w_im + x_im * w_re;
            }
            const size_t output_index = static_cast<size_t>(t) * cfg.freq_bins + f;
            enhanced_real[output_index] = out_re;
            enhanced_imag[output_index] = out_im;
        }
    }
}

void synthesize(const std::vector<float>& spec_real,
                const std::vector<float>& spec_imag,
                int num_frames,
                const Config& cfg,
                const std::vector<float>& window,
                float* output,
                size_t output_length) {
    validate_config(cfg);
    if (output_length > 0 && !output) {
        throw std::invalid_argument("DeepFilterNet3: null audio output");
    }
    if (num_frames < 0 ||
        spec_real.size() != static_cast<size_t>(num_frames) * cfg.freq_bins ||
        spec_imag.size() != spec_real.size() ||
        window.size() != static_cast<size_t>(cfg.fft_size)) {
        throw std::invalid_argument("DeepFilterNet3: invalid synthesis inputs");
    }
    if (output_length == 0) return;

    const size_t delay = static_cast<size_t>(cfg.fft_size - cfg.hop_size);
    const size_t raw_length = static_cast<size_t>(num_frames) * cfg.hop_size;
    if (delay + output_length < output_length || delay + output_length > raw_length) {
        throw std::invalid_argument("DeepFilterNet3: insufficient synthesis frames");
    }

    std::fill_n(output, output_length, 0.0f);
    KissRealPlan plan(cfg.fft_size, true);
    std::vector<kiss_fft_cpx> spectrum(static_cast<size_t>(cfg.freq_bins));
    std::vector<kiss_fft_scalar> time(static_cast<size_t>(cfg.fft_size));
    std::vector<float> windowed(static_cast<size_t>(cfg.fft_size));
    std::vector<float> memory(delay, 0.0f);

    for (int t = 0; t < num_frames; ++t) {
        const size_t spectrum_base = static_cast<size_t>(t) * cfg.freq_bins;
        for (int f = 0; f < cfg.freq_bins; ++f) {
            spectrum[static_cast<size_t>(f)].r =
                spec_real[spectrum_base + static_cast<size_t>(f)];
            spectrum[static_cast<size_t>(f)].i =
                spec_imag[spectrum_base + static_cast<size_t>(f)];
        }
        kiss_fftri(plan.get(), spectrum.data(), time.data());
        for (int i = 0; i < cfg.fft_size; ++i) {
            // kissfft's inverse is intentionally left unnormalized: libdf
            // normalizes only the analysis transform.
            windowed[static_cast<size_t>(i)] =
                time[static_cast<size_t>(i)] * window[static_cast<size_t>(i)];
        }

        const size_t raw_base = static_cast<size_t>(t) * cfg.hop_size;
        for (int i = 0; i < cfg.hop_size; ++i) {
            const float sample = windowed[static_cast<size_t>(i)] +
                                 memory[static_cast<size_t>(i)];
            const size_t raw_index = raw_base + static_cast<size_t>(i);
            if (raw_index >= delay && raw_index < delay + output_length) {
                output[raw_index - delay] = sample;
            }
        }

        // Match libdf's synthesis-memory update for overlap ratios >= 50%.
        const size_t shift_remainder = delay - static_cast<size_t>(cfg.hop_size);
        if (shift_remainder > 0) {
            std::rotate(memory.begin(), memory.begin() + cfg.hop_size, memory.end());
        }
        const float* second_half = windowed.data() + cfg.hop_size;
        for (size_t i = 0; i < shift_remainder; ++i) memory[i] += second_half[i];
        for (int i = 0; i < cfg.hop_size; ++i) {
            memory[shift_remainder + static_cast<size_t>(i)] =
                second_half[shift_remainder + static_cast<size_t>(i)];
        }
    }
}

}  // namespace speech_core::deepfilter_dsp
