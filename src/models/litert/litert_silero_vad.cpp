#include "speech_core/models/litert_silero_vad.h"

#include "speech_core/models/litert_engine.h"

#include <cstring>
#include <stdexcept>

namespace speech_core {

LiteRTSileroVad::LiteRTSileroVad(const std::string& model_path, bool hw_accel) {
    interp_ = LiteRTEngine::get().load(model_path, hw_accel, &model_);

    // Sanity: model exports as forward(audio[1,576], state[2,1,128]) → (prob, state_out)
    const int32_t in_count  = TfLiteInterpreterGetInputTensorCount(interp_);
    const int32_t out_count = TfLiteInterpreterGetOutputTensorCount(interp_);
    if (in_count != 2 || out_count != 2) {
        throw std::runtime_error("LiteRT Silero: expected 2 inputs and 2 outputs, got "
                                 + std::to_string(in_count) + "/" + std::to_string(out_count));
    }

    // Resolve output indices by byte size — the TFLite converter does not
    // preserve the original output order, and the two outputs are unambiguously
    // distinguishable: prob is exactly one float, state_out is 256 floats.
    for (int i = 0; i < out_count; ++i) {
        const TfLiteTensor* t = TfLiteInterpreterGetOutputTensor(interp_, i);
        const size_t bytes = TfLiteTensorByteSize(t);
        if      (bytes == sizeof(float))               prob_idx_  = i;
        else if (bytes == kStateSize * sizeof(float))  state_idx_ = i;
    }
    if (prob_idx_ < 0 || state_idx_ < 0) {
        throw std::runtime_error("LiteRT Silero: could not identify prob/state_out outputs by size");
    }

    reset();
}

LiteRTSileroVad::~LiteRTSileroVad() {
    if (interp_) TfLiteInterpreterDelete(interp_);
    if (model_)  TfLiteModelDelete(model_);
}

void LiteRTSileroVad::reset() {
    context_.fill(0.0f);
    state_.fill(0.0f);
}

float LiteRTSileroVad::process_chunk(const float* samples, size_t length) {
    if (length != kChunkSamples) {
        throw std::runtime_error("LiteRT Silero: expected " + std::to_string(kChunkSamples)
                                 + " samples per chunk, got " + std::to_string(length));
    }

    // Assemble [context(64) | chunk(512)] for the model input.
    std::memcpy(input_buffer_.data(),                   context_.data(), kContextSamples * sizeof(float));
    std::memcpy(input_buffer_.data() + kContextSamples, samples,         kChunkSamples   * sizeof(float));

    TfLiteTensor* in_audio = TfLiteInterpreterGetInputTensor(interp_, 0);
    TfLiteTensor* in_state = TfLiteInterpreterGetInputTensor(interp_, 1);

    litert_check(TfLiteTensorCopyFromBuffer(in_audio, input_buffer_.data(),
                                            kTotalSamples * sizeof(float)),
                 "CopyFromBuffer(audio)");
    litert_check(TfLiteTensorCopyFromBuffer(in_state, state_.data(),
                                            kStateSize * sizeof(float)),
                 "CopyFromBuffer(state)");

    litert_check(TfLiteInterpreterInvoke(interp_), "Invoke");

    const TfLiteTensor* out_prob  = TfLiteInterpreterGetOutputTensor(interp_, prob_idx_);
    const TfLiteTensor* out_state = TfLiteInterpreterGetOutputTensor(interp_, state_idx_);

    float prob = 0.0f;
    litert_check(TfLiteTensorCopyToBuffer(out_prob, &prob, sizeof(float)),
                 "CopyToBuffer(prob)");
    litert_check(TfLiteTensorCopyToBuffer(out_state, state_.data(),
                                          kStateSize * sizeof(float)),
                 "CopyToBuffer(state_out)");

    // Keep the last 64 samples of the chunk as next call's left context.
    std::memcpy(context_.data(),
                samples + (kChunkSamples - kContextSamples),
                kContextSamples * sizeof(float));

    return prob;
}

}  // namespace speech_core
