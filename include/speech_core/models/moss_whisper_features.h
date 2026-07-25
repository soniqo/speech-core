#pragma once

#include <cstddef>
#include <vector>

namespace speech_core {

/// Fixed-shape Whisper log-mel input consumed by the portable MOSS audio
/// encoder. Data is row-major `[mel_bins, time_frames]`.
struct MossLogMelFeatures {
    std::vector<float> data;
    int mel_bins = 0;
    int time_frames = 0;
};

/// Exact 400-point Whisper frontend used by MOSS-Transcribe-Diarize.
///
/// The ordinary speech-core FFT helper intentionally pads non-power-of-two
/// transforms. MOSS was trained and exported with a true 400-point DFT, so
/// this frontend uses the bundled KISS FFT implementation directly.
class MossWhisperFeatureExtractor {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr int kFftSize = 400;
    static constexpr int kHopLength = 160;
    static constexpr int kMelBins = 80;
    static constexpr int kChunkSamples = 480000;
    static constexpr int kTimeFrames = 3000;
    static constexpr int kEncoderStrideSamples = 1280;

    MossWhisperFeatureExtractor();

    static std::size_t audio_token_count(std::size_t sample_count);

    /// Extract exactly `[80, 3000]` features from one non-empty, at-most
    /// 30-second 16 kHz mono chunk. Short chunks are right-padded with zeroes.
    MossLogMelFeatures extract_padded_chunk(
        const float* audio, std::size_t length) const;

private:
    std::vector<float> hann_window_;
    /// Row-major `[mel_bins, fft_bins]`.
    std::vector<float> mel_filterbank_;
};

}  // namespace speech_core
