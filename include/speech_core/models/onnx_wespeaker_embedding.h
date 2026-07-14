#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <string>
#include <vector>

namespace speech_core {

/// WeSpeaker ResNet34-LM speaker embedding via ONNX Runtime.
///
/// Matches soniqo/WeSpeaker-ResNet34-LM-ONNX:
///   Input:  fbank     [1, T, 80]  — precomputed kaldi log-mel (T dynamic)
///   Output: embedding [1, 256]
///
/// Same feature contract as LiteRTWeSpeakerEmbedding, deliberately: pyannote's
/// in-graph compute_fbank uses torch.fft.rfft, which has no ONNX lowering, so the
/// export takes features rather than audio — exactly as the LiteRT bundle already
/// does. Both wrappers call audio::wespeaker_fbank(), so the runtimes are fed
/// bit-identical features and the swap is a true drop-in (cosine 1.000000 against
/// the PyTorch reference).
///
/// The graph accepts a dynamic frame count, but this wrapper pins it to the same
/// 298 frames (~3 s) the LiteRT path uses, so behaviour is unchanged. Embedding
/// longer turns is a quality change, not a runtime change — it gets its own gate.
///
/// Not thread-safe — one per worker.
class OnnxWeSpeakerEmbedding : public EmbeddingInterface {
public:
    static constexpr int kEmbeddingDim = 256;
    static constexpr int kNumMelBins   = 80;
    static constexpr int kNumFrames    = 298;

    explicit OnnxWeSpeakerEmbedding(const std::string& model_path,
                                    bool hw_accel = true);
    ~OnnxWeSpeakerEmbedding() override;

    OnnxWeSpeakerEmbedding(const OnnxWeSpeakerEmbedding&) = delete;
    OnnxWeSpeakerEmbedding& operator=(const OnnxWeSpeakerEmbedding&) = delete;

    std::vector<float> embed(const float* audio, size_t length,
                             int sample_rate) override;

    int embedding_dim() const override { return kEmbeddingDim; }
    int input_sample_rate() const override { return 16000; }

private:
    const OrtApi* api_     = nullptr;
    OrtSession*   session_ = nullptr;
};

}  // namespace speech_core
