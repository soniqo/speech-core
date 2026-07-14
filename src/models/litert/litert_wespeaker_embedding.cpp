#include "speech_core/models/litert_wespeaker_embedding.h"

#include "speech_core/audio/wespeaker_fbank.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace speech_core {

// Kaldi fbank now lives in audio::wespeaker_fbank — shared with the ONNX
// wrapper so both runtimes are fed bit-identical features.

LiteRTWeSpeakerEmbedding::LiteRTWeSpeakerEmbedding(const std::string& model_path, bool hw_accel) {
    LiteRTEngine::get().load(model_path, hw_accel, &model_, &compiled_);
}

LiteRTWeSpeakerEmbedding::~LiteRTWeSpeakerEmbedding() {
    if (compiled_) LiteRtDestroyCompiledModel(compiled_);
    if (model_)    LiteRtDestroyModel(model_);
}

std::vector<float> LiteRTWeSpeakerEmbedding::compute_fbank(const float* audio, size_t length) {
    return audio::wespeaker_fbank(audio, length, kNumFrames);
}

std::vector<float> LiteRTWeSpeakerEmbedding::embed(
    const float* audio, size_t length, int sample_rate)
{
    if (sample_rate != 16000) throw std::runtime_error("WeSpeaker expects 16kHz input");

    auto fbank = compute_fbank(audio, length);  // [kNumFrames * kNumMelBins]

    auto env     = LiteRTEngine::get().env();
    auto t_fbank = make_type(kLiteRtElementTypeFloat32, {1, kNumFrames, kNumMelBins});
    auto t_emb   = make_type(kLiteRtElementTypeFloat32, {1, kEmbeddingDim});

    LiteRtHostBuffer in_fbank(env, t_fbank, fbank.size() * sizeof(float), fbank.data());
    LiteRtHostBuffer out_emb (env, t_emb,   kEmbeddingDim * sizeof(float));

    LiteRtTensorBuffer ins [1] = { in_fbank.raw() };
    LiteRtTensorBuffer outs[1] = { out_emb.raw() };
    litert_check(LiteRtRunCompiledModel(compiled_, 0, 1, ins, 1, outs), "WeSpeaker Run");

    std::vector<float> embedding(kEmbeddingDim);
    out_emb.read(embedding.data(), kEmbeddingDim * sizeof(float));

    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) for (float& v : embedding) v /= norm;

    return embedding;
}

}  // namespace speech_core
