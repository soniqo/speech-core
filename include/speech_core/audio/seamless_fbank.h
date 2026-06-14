#pragma once

#include <cstddef>
#include <vector>

namespace speech_core::audio {

/// Wav2Vec2-BERT / SeamlessM4T log-mel feature extractor.
///
/// Reimplements `transformers.SeamlessM4TFeatureExtractor`
/// (`facebook/w2v-bert-2.0`), the DSP front-end consumed by the Sidon
/// speech-restoration predictor. The reference pipeline is a Kaldi-compatible
/// 80-bin log-mel fbank followed by per-utterance per-mel-bin mean/variance
/// normalisation (CMVN) and ×2 frame stacking (80 → 160, time halved).
///
/// Pipeline, per the upstream extractor (constants in seamless_fbank.cpp):
///   1. scale waveform by 2^15 (Kaldi int16 convention; cancels under CMVN but
///      kept for the pre-log floor),
///   2. frame at 25 ms / 10 ms hop (400 / 160 samples), snip_edges,
///   3. per frame: remove DC offset, pre-emphasis 0.97, Povey window,
///      right zero-pad to 512, power spectrum (257 bins),
///   4. Kaldi mel filterbank (80 bins, 20–8000 Hz, no norm) on the power
///      spectrum, natural log with a float32-eps floor,
///   5. CMVN: per mel bin over time, zero-mean unit-variance (sample variance,
///      ddof = 1),
///   6. drop the trailing odd frame, then reshape so each output row is two
///      consecutive 80-d frames concatenated (frame 2t ‖ frame 2t+1).
///
/// Output is row-major `[T, 160]` with `T = floor(num_frames / 2)`, suitable to
/// feed directly as the predictor's `input_features[1, T, 160]` (the batch
/// dimension is implicit / size 1).
///
/// @param audio        Input PCM Float32, mono, nominally in [-1, 1].
/// @param length       Number of samples.
/// @param sample_rate  Must be 16000; other rates throw (resample first).
/// @param out_frames   Receives T (the stacked-frame count).
/// @return             Flattened `[T, 160]` features, or empty if the clip is
///                     too short to yield a single stacked frame (< ~25 ms ×2).
std::vector<float> seamless_log_mel(
    const float* audio, size_t length, int sample_rate,
    int& out_frames);

}  // namespace speech_core::audio
