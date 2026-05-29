#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <string>
#include <vector>

namespace speech_core {

/// WeSpeaker ResNet34-LM speaker embedding via LiteRT.
///
/// Matches soniqo/WeSpeaker-ResNet34-LM-LiteRT:
///   Input 0:  fbank [1, 298, 80] f32 — precomputed kaldi-compatible log-mel
///             fbank (80 bins, 25 ms / 10 ms, Hamming, no dither, per-frame
///             DC removal). T=298 ≈ 3 s at 16 kHz.
///   Output 0: embedding [1, 256]
///
/// The wrapper computes the fbank from raw 16 kHz audio internally (pads or
/// tiles clips shorter than 3 s), so callers still pass raw float audio.
/// L2-normalises the embedding. Not thread-safe — one per worker.
class LiteRTWeSpeakerEmbedding : public EmbeddingInterface {
public:
    static constexpr int kEmbeddingDim = 256;
    static constexpr int kNumMelBins   = 80;
    static constexpr int kNumFrames    = 298;
    static constexpr int kFrameLenSamp = 400;  // 25 ms @ 16 kHz
    static constexpr int kHopSamp      = 160;  // 10 ms @ 16 kHz

    explicit LiteRTWeSpeakerEmbedding(const std::string& model_path, bool hw_accel = true);
    ~LiteRTWeSpeakerEmbedding() override;

    std::vector<float> embed(const float* audio, size_t length, int sample_rate) override;
    int embedding_dim() const override { return kEmbeddingDim; }
    int input_sample_rate() const override { return 16000; }

private:
    /// kaldi-compatible log-mel fbank [kNumFrames, kNumMelBins], row-major
    /// (frame-major). Pads or tiles the input to span kNumFrames.
    std::vector<float> compute_fbank(const float* audio, size_t length);

    LiteRtModel         model_    = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
};

}  // namespace speech_core
