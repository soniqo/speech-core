#include "speech_core/models/litert_silero_vad.h"

#include <cstring>
#include <stdexcept>

namespace speech_core {

LiteRTSileroVad::LiteRTSileroVad(const std::string& model_path, bool hw_accel) {
    LiteRTEngine::get().load(model_path, hw_accel, &model_, &compiled_);

    // Resolve the prob vs state_out output index by element count.
    LiteRtLayout outs[2]{};
    litert_check(LiteRtGetCompiledModelOutputTensorLayouts(
                     compiled_, 0, 2, outs, /*update_allocation=*/false),
                 "GetOutputTensorLayouts");
    if      (layout_element_count(outs[0]) == 1) { prob_idx_ = 0; state_idx_ = 1; }
    else if (layout_element_count(outs[1]) == 1) { prob_idx_ = 1; state_idx_ = 0; }
    else {
        throw std::runtime_error("LiteRT Silero: neither output is the scalar prob tensor");
    }

    reset();
}

LiteRTSileroVad::~LiteRTSileroVad() {
    if (compiled_) LiteRtDestroyCompiledModel(compiled_);
    if (model_)    LiteRtDestroyModel(model_);
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

    auto env     = LiteRTEngine::get().env();
    auto t_audio = make_type(kLiteRtElementTypeFloat32, {1, static_cast<int32_t>(kTotalSamples)});
    auto t_state = make_type(kLiteRtElementTypeFloat32, {2, 1, 128});
    auto t_prob  = make_type(kLiteRtElementTypeFloat32, {1, 1});

    LiteRtHostBuffer in_audio (env, t_audio, kTotalSamples * sizeof(float), input_buffer_.data());
    LiteRtHostBuffer in_state (env, t_state, kStateSize    * sizeof(float), state_.data());
    LiteRtHostBuffer out_prob (env, t_prob,  sizeof(float));
    LiteRtHostBuffer out_state(env, t_state, kStateSize    * sizeof(float));

    LiteRtTensorBuffer ins[2]  = { in_audio.raw(), in_state.raw() };
    LiteRtTensorBuffer outs[2];
    outs[prob_idx_]  = out_prob.raw();
    outs[state_idx_] = out_state.raw();
    litert_check(LiteRtRunCompiledModel(compiled_, 0, 2, ins, 2, outs), "Run");

    float prob = 0.0f;
    out_prob .read(&prob,         sizeof(float));
    out_state.read(state_.data(), kStateSize * sizeof(float));

    // Keep the last 64 samples of the chunk as next call's left context.
    std::memcpy(context_.data(),
                samples + (kChunkSamples - kContextSamples),
                kContextSamples * sizeof(float));
    return prob;
}

}  // namespace speech_core
