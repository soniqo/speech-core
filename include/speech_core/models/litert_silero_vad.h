#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <array>
#include <string>

namespace speech_core {

/// Silero VAD v5 — voice activity detection via LiteRT.
/// VADInterface::process_chunk consumes 512 samples per call; the LiteRT model
/// takes [1, 576] (64 left-context + 512 chunk). The wrapper keeps the 64-sample
/// tail of the previous chunk and prepends it transparently, so callers see the
/// same VADInterface as the ORT variant.
class LiteRTSileroVad : public VADInterface {
public:
    explicit LiteRTSileroVad(const std::string& model_path, bool hw_accel = false);
    ~LiteRTSileroVad() override;

    float process_chunk(const float* samples, size_t length) override;
    void  reset() override;

    int    input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }

private:
    LiteRtModel         model_    = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;

    static constexpr size_t kContextSamples = 64;
    static constexpr size_t kChunkSamples   = 512;
    static constexpr size_t kTotalSamples   = kContextSamples + kChunkSamples;  // 576
    static constexpr size_t kStateSize      = 2 * 1 * 128;

    // Resolved at construction by output element count — TFLite converter
    // doesn't preserve the source model's output order, and the two outputs
    // are unambiguously distinguishable (prob = 1 float vs state = 256 floats).
    int prob_idx_  = -1;
    int state_idx_ = -1;

    std::array<float, kContextSamples> context_{};
    std::array<float, kStateSize>      state_{};
    std::array<float, kTotalSamples>   input_buffer_{};
};

}  // namespace speech_core
