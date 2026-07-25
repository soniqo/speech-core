#include "speech_core/models/moss_whisper_features.h"

#include <kiss_fftr.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace speech_core {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct KissFftrDeleter {
    void operator()(kiss_fftr_state* value) const {
        if (value) kiss_fftr_free(value);
    }
};

using KissFftrPlan = std::unique_ptr<kiss_fftr_state, KissFftrDeleter>;

double hertz_to_slaney_mel(double hertz) {
    constexpr double kMinimumLogHertz = 1000.0;
    constexpr double kMinimumLogMel = 15.0;
    const double log_step = 27.0 / std::log(6.4);
    if (hertz < kMinimumLogHertz) return 3.0 * hertz / 200.0;
    return kMinimumLogMel
        + std::log(hertz / kMinimumLogHertz) * log_step;
}

double slaney_mel_to_hertz(double mel) {
    constexpr double kMinimumLogHertz = 1000.0;
    constexpr double kMinimumLogMel = 15.0;
    const double log_step = 27.0 / std::log(6.4);
    if (mel < kMinimumLogMel) return 200.0 * mel / 3.0;
    return kMinimumLogHertz
        * std::exp((mel - kMinimumLogMel) / log_step);
}

}  // namespace

MossWhisperFeatureExtractor::MossWhisperFeatureExtractor() {
    hann_window_.resize(kFftSize);
    for (int index = 0; index < kFftSize; ++index) {
        hann_window_[static_cast<std::size_t>(index)] = static_cast<float>(
            0.5 * (1.0 - std::cos(
                2.0 * kPi * static_cast<double>(index)
                / static_cast<double>(kFftSize))));
    }

    constexpr int kFrequencyBins = kFftSize / 2 + 1;
    std::vector<double> points(kMelBins + 2);
    const double mel_minimum = hertz_to_slaney_mel(0.0);
    const double mel_maximum =
        hertz_to_slaney_mel(static_cast<double>(kSampleRate) / 2.0);
    for (int index = 0; index < kMelBins + 2; ++index) {
        const double mel = mel_minimum
            + static_cast<double>(index) * (mel_maximum - mel_minimum)
                / static_cast<double>(kMelBins + 1);
        points[static_cast<std::size_t>(index)] =
            slaney_mel_to_hertz(mel);
    }

    mel_filterbank_.assign(
        static_cast<std::size_t>(kMelBins * kFrequencyBins), 0.0f);
    for (int mel = 0; mel < kMelBins; ++mel) {
        const double left = points[static_cast<std::size_t>(mel)];
        const double center = points[static_cast<std::size_t>(mel + 1)];
        const double right = points[static_cast<std::size_t>(mel + 2)];
        const double slaney = 2.0 / (right - left);
        for (int bin = 0; bin < kFrequencyBins; ++bin) {
            const double hertz =
                static_cast<double>(bin * kSampleRate) / kFftSize;
            const double rising = (hertz - left) / (center - left);
            const double falling = (right - hertz) / (right - center);
            const double triangle =
                std::max(0.0, std::min(rising, falling));
            mel_filterbank_[
                static_cast<std::size_t>(mel * kFrequencyBins + bin)] =
                static_cast<float>(triangle * slaney);
        }
    }
}

std::size_t MossWhisperFeatureExtractor::audio_token_count(
    std::size_t sample_count) {
    if (sample_count == 0) return 0;
    return (sample_count - 1)
        / static_cast<std::size_t>(kEncoderStrideSamples) + 1;
}

MossLogMelFeatures MossWhisperFeatureExtractor::extract_padded_chunk(
    const float* audio, std::size_t length) const {
    if (!audio || length == 0) {
        throw std::invalid_argument("MOSS audio chunk is empty");
    }
    if (length > static_cast<std::size_t>(kChunkSamples)) {
        throw std::invalid_argument(
            "MOSS audio chunk exceeds the 30-second encoder input");
    }

    std::vector<float> fixed_audio(
        static_cast<std::size_t>(kChunkSamples), 0.0f);
    std::copy_n(audio, length, fixed_audio.begin());

    constexpr int kPad = kFftSize / 2;
    std::vector<float> centered(
        fixed_audio.size() + static_cast<std::size_t>(2 * kPad), 0.0f);
    for (int index = 0; index < kPad; ++index) {
        centered[static_cast<std::size_t>(index)] =
            fixed_audio[static_cast<std::size_t>(kPad - index)];
    }
    std::copy(
        fixed_audio.begin(), fixed_audio.end(),
        centered.begin() + kPad);
    for (int index = 0; index < kPad; ++index) {
        centered[
            static_cast<std::size_t>(kPad + kChunkSamples + index)] =
            fixed_audio[
                static_cast<std::size_t>(kChunkSamples - 2 - index)];
    }

    KissFftrPlan plan(
        kiss_fftr_alloc(kFftSize, /*inverse_fft=*/0, nullptr, nullptr));
    if (!plan) {
        throw std::runtime_error("MOSS KISS FFT plan allocation failed");
    }

    constexpr int kFrequencyBins = kFftSize / 2 + 1;
    std::vector<float> frame(static_cast<std::size_t>(kFftSize));
    std::vector<kiss_fft_cpx> spectrum(
        static_cast<std::size_t>(kFrequencyBins));
    std::vector<float> power(static_cast<std::size_t>(kFrequencyBins));
    std::vector<float> mel_by_frame(
        static_cast<std::size_t>(kTimeFrames * kMelBins));

    for (int time = 0; time < kTimeFrames; ++time) {
        const std::size_t start =
            static_cast<std::size_t>(time * kHopLength);
        for (int index = 0; index < kFftSize; ++index) {
            frame[static_cast<std::size_t>(index)] =
                centered[start + static_cast<std::size_t>(index)]
                * hann_window_[static_cast<std::size_t>(index)];
        }
        kiss_fftr(plan.get(), frame.data(), spectrum.data());
        for (int bin = 0; bin < kFrequencyBins; ++bin) {
            const auto& value = spectrum[static_cast<std::size_t>(bin)];
            power[static_cast<std::size_t>(bin)] =
                value.r * value.r + value.i * value.i;
        }
        for (int mel = 0; mel < kMelBins; ++mel) {
            const float* filter = mel_filterbank_.data()
                + static_cast<std::size_t>(mel * kFrequencyBins);
            double sum = 0.0;
            for (int bin = 0; bin < kFrequencyBins; ++bin) {
                sum += static_cast<double>(
                    power[static_cast<std::size_t>(bin)])
                    * static_cast<double>(filter[bin]);
            }
            mel_by_frame[
                static_cast<std::size_t>(time * kMelBins + mel)] =
                static_cast<float>(std::max(sum, 1e-10));
        }
    }

    float peak = -std::numeric_limits<float>::infinity();
    for (float& value : mel_by_frame) {
        value = std::log10(value);
        peak = std::max(peak, value);
    }
    const float floor = peak - 8.0f;

    MossLogMelFeatures result;
    result.mel_bins = kMelBins;
    result.time_frames = kTimeFrames;
    result.data.resize(static_cast<std::size_t>(kMelBins * kTimeFrames));
    for (int time = 0; time < kTimeFrames; ++time) {
        for (int mel = 0; mel < kMelBins; ++mel) {
            const float value = mel_by_frame[
                static_cast<std::size_t>(time * kMelBins + mel)];
            result.data[
                static_cast<std::size_t>(mel * kTimeFrames + time)] =
                (std::max(value, floor) + 4.0f) * 0.25f;
        }
    }
    return result;
}

}  // namespace speech_core
