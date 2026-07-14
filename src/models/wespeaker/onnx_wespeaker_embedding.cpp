#include "speech_core/models/onnx_wespeaker_embedding.h"

#include "speech_core/audio/wespeaker_fbank.h"
#include "speech_core/models/onnx_engine.h"

#include <cmath>
#include <stdexcept>

namespace speech_core {

OnnxWeSpeakerEmbedding::OnnxWeSpeakerEmbedding(const std::string& model_path,
                                               bool hw_accel)
{
    api_     = OnnxEngine::get().api();
    session_ = OnnxEngine::get().load(model_path, hw_accel);
}

OnnxWeSpeakerEmbedding::~OnnxWeSpeakerEmbedding() {
    if (session_) api_->ReleaseSession(session_);
}

std::vector<float> OnnxWeSpeakerEmbedding::embed(
    const float* audio, size_t length, int sample_rate)
{
    if (sample_rate != 16000) {
        throw std::runtime_error("WeSpeaker expects 16kHz input");
    }

    // Same frontend the LiteRT wrapper uses — one implementation, so a runtime
    // comparison compares runtimes and not two subtly different fbanks.
    auto fbank = audio::wespeaker_fbank(audio, length, kNumFrames);

    auto* mem = OnnxEngine::get().cpu_memory();
    const int64_t in_shape[] = {1, kNumFrames, kNumMelBins};

    OrtValue* t_fbank = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, fbank.data(), fbank.size() * sizeof(float),
        in_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_fbank));

    const char* in_names[]  = {"fbank"};
    const char* out_names[] = {"embedding"};
    OrtValue* inputs[]  = {t_fbank};
    OrtValue* outputs[] = {nullptr};

    ort_check(api_, api_->Run(session_, nullptr, in_names, inputs, 1,
                              out_names, 1, outputs));

    float* out = nullptr;
    ort_check(api_, api_->GetTensorMutableData(outputs[0],
                                               reinterpret_cast<void**>(&out)));

    std::vector<float> embedding(out, out + kEmbeddingDim);

    api_->ReleaseValue(outputs[0]);
    api_->ReleaseValue(t_fbank);

    // The graph applies mean-centering internally but not L2 normalisation;
    // clustering compares embeddings by cosine, so normalise here — same as the
    // LiteRT wrapper.
    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) for (float& v : embedding) v /= norm;

    return embedding;
}

}  // namespace speech_core
