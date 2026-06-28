// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/audio/offline_spectral_de_esser.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kSampleRate = 48000;

double peak_abs(const std::vector<float>& samples) {
    double peak = 0.0;
    for (float sample : samples) {
        peak = std::max(peak, std::abs(static_cast<double>(sample)));
    }
    return peak;
}

double projected_band_rms(
    const std::vector<float>& samples,
    const std::vector<double>& frequencies,
    size_t begin,
    size_t end) {
    const size_t count = end - begin;
    double energy = 0.0;

    for (double frequency : frequencies) {
        double cosine_sum = 0.0;
        double sine_sum = 0.0;

        for (size_t i = begin; i < end; ++i) {
            const double phase = 2.0 * kPi * frequency
                               * static_cast<double>(i)
                               / static_cast<double>(kSampleRate);
            const double sample = static_cast<double>(samples[i]);
            cosine_sum += sample * std::cos(phase);
            sine_sum += sample * std::sin(phase);
        }

        const double amplitude = 2.0
            * std::sqrt(cosine_sum * cosine_sum + sine_sum * sine_sum)
            / static_cast<double>(count);
        energy += 0.5 * amplitude * amplitude;
    }

    return std::sqrt(energy);
}

double ratio_db(double after, double before) {
    return 20.0 * std::log10((after + 1.0e-12) / (before + 1.0e-12));
}

}  // namespace

void test_cli_defaults() {
    const auto params = audio::OfflineSpectralDeEsser::cli_default_parameters();
    assert(params.fft_size == 1024);
    assert(params.hop_size == 128);
    assert(std::abs(params.amount - 1.76) < 1.0e-12);
    assert(params.clip_output_to_input_peak);
    std::printf("  PASS: cli_defaults\n");
}

void test_silence_stays_silent() {
    std::vector<float> input(4096, 0.0f);

    const auto output = audio::OfflineSpectralDeEsser::process_mono(
        input.data(), input.size(), kSampleRate);

    assert(output.size() == input.size());
    for (float sample : output) {
        assert(std::isfinite(sample));
        assert(std::abs(sample) < 1.0e-6f);
    }

    std::printf("  PASS: silence_stays_silent\n");
}

void test_voxcpm2_chunk_contract() {
    std::vector<float> input(6144, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSampleRate);
        const double carrier = 0.16 * std::sin(2.0 * kPi * 1200.0 * t);
        const double sibilant = 0.08 * std::sin(2.0 * kPi * 9800.0 * t);
        const bool burst = i > 1200 && i < 3600;
        input[i] = static_cast<float>(carrier + (burst ? sibilant : 0.0));
    }

    const auto output = audio::OfflineSpectralDeEsser::process_mono(
        input.data(), input.size(), kSampleRate);

    assert(output.size() == input.size());
    for (float sample : output) {
        assert(std::isfinite(sample));
    }

    assert(peak_abs(output) <= peak_abs(input) + 1.0e-5);
    std::printf("  PASS: voxcpm2_chunk_contract\n");
}

void test_reduces_sibilance_without_collapsing_voice_band() {
    constexpr size_t kSamples = 48000;
    constexpr size_t kBurstBegin = 12000;
    constexpr size_t kBurstEnd = 32000;
    const std::vector<double> voice_frequencies = {700.0, 1400.0, 2400.0};
    const std::vector<double> sibilance_frequencies = {
        8906.25, 9187.5, 9468.75, 9750.0, 10031.25,
        10312.5, 10593.75, 10875.0, 11156.25, 11437.5
    };

    std::vector<float> input(kSamples, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(kSampleRate);
        double sample = 0.11 * std::sin(2.0 * kPi * voice_frequencies[0] * t)
                      + 0.08 * std::sin(2.0 * kPi * voice_frequencies[1] * t)
                      + 0.04 * std::sin(2.0 * kPi * voice_frequencies[2] * t);

        if (i >= kBurstBegin && i < kBurstEnd) {
            const double u = static_cast<double>(i - kBurstBegin)
                           / static_cast<double>(kBurstEnd - kBurstBegin);
            const double envelope = 0.5 - 0.5 * std::cos(2.0 * kPi * u);
            for (size_t j = 0; j < sibilance_frequencies.size(); ++j) {
                const double phase = 0.37 * static_cast<double>(j + 1);
                sample += envelope * 0.026
                    * std::sin(2.0 * kPi * sibilance_frequencies[j] * t + phase);
            }
        }

        input[i] = static_cast<float>(sample);
    }

    const auto output = audio::OfflineSpectralDeEsser::process_mono(
        input.data(), input.size(), kSampleRate);

    const double input_sibilance = projected_band_rms(
        input, sibilance_frequencies, kBurstBegin, kBurstEnd);
    const double output_sibilance = projected_band_rms(
        output, sibilance_frequencies, kBurstBegin, kBurstEnd);
    const double input_voice = projected_band_rms(
        input, voice_frequencies, kBurstBegin, kBurstEnd);
    const double output_voice = projected_band_rms(
        output, voice_frequencies, kBurstBegin, kBurstEnd);

    const double sibilance_change_db = ratio_db(output_sibilance, input_sibilance);
    const double voice_change_db = ratio_db(output_voice, input_voice);

    assert(sibilance_change_db < -1.0);
    assert(voice_change_db > -1.5);

    std::printf(
        "  PASS: reduces_sibilance_without_collapsing_voice_band "
        "(sib=%.2fdB, voice=%.2fdB)\n",
        sibilance_change_db,
        voice_change_db);
}

int main() {
    std::printf("test_deesser:\n");
    test_cli_defaults();
    test_silence_stays_silent();
    test_voxcpm2_chunk_contract();
    test_reduces_sibilance_without_collapsing_voice_band();
    std::printf("All deesser tests passed.\n");
    return 0;
}
