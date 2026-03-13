#include "speech_core/audio/resampler.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace speech_core;

// Generate a sine wave at a given frequency and sample rate
static std::vector<float> make_sine(float freq, int sample_rate, size_t samples) {
    std::vector<float> buf(samples);
    for (size_t i = 0; i < samples; i++) {
        buf[i] = std::sin(2.0f * static_cast<float>(M_PI) * freq
                          * static_cast<float>(i) / sample_rate);
    }
    return buf;
}

// RMS energy of a signal
static float rms(const std::vector<float>& v) {
    double sum = 0.0;
    for (float s : v) sum += static_cast<double>(s) * s;
    return static_cast<float>(std::sqrt(sum / v.size()));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_identity() {
    auto sine = make_sine(440.0f, 16000, 16000);
    auto out = Resampler::resample(sine.data(), sine.size(), 16000, 16000);
    assert(out.size() == sine.size());
    for (size_t i = 0; i < out.size(); i++) {
        assert(out[i] == sine[i]);
    }
    printf("  PASS: identity\n");
}

void test_empty_input() {
    auto out = Resampler::resample(nullptr, 0, 48000, 16000);
    assert(out.empty());
    printf("  PASS: empty_input\n");
}

void test_output_length() {
    // 48000 samples at 48kHz = 1 second → should produce ~16000 samples at 16kHz
    std::vector<float> buf(48000, 0.0f);
    auto out = Resampler::resample(buf.data(), buf.size(), 48000, 16000);
    assert(out.size() == 16000);

    // 16000 samples at 16kHz → ~24000 at 24kHz
    std::vector<float> buf2(16000, 0.0f);
    auto out2 = Resampler::resample(buf2.data(), buf2.size(), 16000, 24000);
    assert(out2.size() == 24000);

    printf("  PASS: output_length\n");
}

void test_downsample_preserves_low_freq() {
    // 1kHz sine at 48kHz, downsample to 16kHz
    // 1kHz is well below Nyquist (8kHz) — should pass through cleanly
    auto sine = make_sine(1000.0f, 48000, 48000);
    auto out = Resampler::resample(sine.data(), sine.size(), 48000, 16000);

    // Compare to reference 1kHz sine at 16kHz (skip edges where filter rings)
    auto ref = make_sine(1000.0f, 16000, 16000);
    int margin = 50;  // skip edge samples
    float max_err = 0.0f;
    for (size_t i = margin; i < out.size() - margin; i++) {
        float err = std::abs(out[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    // Should be very close — within 2% of full scale
    assert(max_err < 0.02f);
    printf("  PASS: downsample_preserves_low_freq (max_err=%.5f)\n", max_err);
}

void test_anti_aliasing() {
    // 12kHz sine at 48kHz, downsample to 16kHz
    // 12kHz > Nyquist/2 of 16kHz (8kHz) → should be heavily attenuated
    auto sine = make_sine(12000.0f, 48000, 48000);
    float input_rms = rms(sine);
    auto out = Resampler::resample(sine.data(), sine.size(), 48000, 16000);
    float output_rms = rms(out);

    // Should be attenuated by at least 30dB
    float attenuation_db = 20.0f * std::log10(output_rms / input_rms);
    assert(attenuation_db < -30.0f);
    printf("  PASS: anti_aliasing (attenuation=%.1fdB)\n", attenuation_db);
}

void test_upsample() {
    // 1kHz sine at 16kHz, upsample to 48kHz
    auto sine = make_sine(1000.0f, 16000, 16000);
    auto out = Resampler::resample(sine.data(), sine.size(), 16000, 48000);
    assert(out.size() == 48000);

    // Verify the tone is preserved
    auto ref = make_sine(1000.0f, 48000, 48000);
    int margin = 150;
    float max_err = 0.0f;
    for (size_t i = margin; i < out.size() - margin; i++) {
        float err = std::abs(out[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    assert(max_err < 0.05f);
    printf("  PASS: upsample (max_err=%.5f)\n", max_err);
}

void test_kernel_cache() {
    Resampler::clear_cache();

    // First call builds kernel, second should hit cache
    std::vector<float> buf(4800, 0.0f);
    auto out1 = Resampler::resample(buf.data(), buf.size(), 48000, 16000);
    auto out2 = Resampler::resample(buf.data(), buf.size(), 48000, 16000);
    assert(out1.size() == out2.size());
    for (size_t i = 0; i < out1.size(); i++) {
        assert(out1[i] == out2[i]);
    }
    printf("  PASS: kernel_cache\n");
}

void test_24k_to_16k() {
    // Common TTS→STT conversion: 24kHz to 16kHz
    auto sine = make_sine(2000.0f, 24000, 24000);
    auto out = Resampler::resample(sine.data(), sine.size(), 24000, 16000);
    assert(out.size() == 16000);

    auto ref = make_sine(2000.0f, 16000, 16000);
    int margin = 50;
    float max_err = 0.0f;
    for (size_t i = margin; i < out.size() - margin; i++) {
        float err = std::abs(out[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    assert(max_err < 0.02f);
    printf("  PASS: 24k_to_16k (max_err=%.5f)\n", max_err);
}

int main() {
    printf("test_resampler:\n");

    test_identity();
    test_empty_input();
    test_output_length();
    test_downsample_preserves_low_freq();
    test_anti_aliasing();
    test_upsample();
    test_kernel_cache();
    test_24k_to_16k();

    Resampler::clear_cache();
    printf("All resampler tests passed.\n");
    return 0;
}
