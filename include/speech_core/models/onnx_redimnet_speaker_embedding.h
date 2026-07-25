#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace speech_core {

/// ReDimNet2-B6 speaker-identity embeddings from the portable ONNX export.
///
/// This is the identity layer used after MOSS activity routing. It is not a
/// diarizer and its vectors must not be treated as authentication.
class OnnxReDimNetSpeakerEmbedding final : public EmbeddingInterface {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr std::size_t kInputSamples = 96000;
    static constexpr std::size_t kMinimumSamples = 32000;
    static constexpr std::size_t kMinimumShortSamples = 9600;
    static constexpr int kEmbeddingDimension = 192;

    explicit OnnxReDimNetSpeakerEmbedding(
        const std::string& model_path, bool hardware_acceleration = true);
    ~OnnxReDimNetSpeakerEmbedding() override;

    OnnxReDimNetSpeakerEmbedding(
        const OnnxReDimNetSpeakerEmbedding&) = delete;
    OnnxReDimNetSpeakerEmbedding& operator=(
        const OnnxReDimNetSpeakerEmbedding&) = delete;

    std::vector<float> embed(
        const float* audio, std::size_t length, int sample_rate) override;

    /// Conservative 0.6-to-2-second retrieval probe. Do not use its result to
    /// create, enroll, or update an identity.
    std::vector<float> embed_short_utterance(
        const float* audio, std::size_t length, int sample_rate) override;

    int embedding_dim() const override { return kEmbeddingDimension; }
    int input_sample_rate() const override { return kSampleRate; }

    static std::vector<float> prepare_audio(
        const float* samples, std::size_t length,
        std::size_t minimum_samples = kMinimumSamples);

private:
    std::vector<float> embed_with_minimum(
        const float* audio, std::size_t length, int sample_rate,
        std::size_t minimum_samples);
    void validate_contract();

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
};

}  // namespace speech_core
