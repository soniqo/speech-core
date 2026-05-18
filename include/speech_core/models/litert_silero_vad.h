#pragma once

#include "speech_core/interfaces.h"

#include <tensorflow/lite/c/c_api.h>

#include <array>
#include <string>

namespace speech_core {

/// Silero VAD v5 — voice activity detection via LiteRT (TFLite).
/// VADInterface::process_chunk consumes 512 samples per call; the LiteRT model
/// itself takes [1, 576] (64 left-context + 512 chunk). The wrapper keeps the
/// 64-sample tail of the previous chunk and prepends it transparently, so the
/// public API matches the ORT variant.
class LiteRTSileroVad : public VADInterface {
public:
    explicit LiteRTSileroVad(const std::string& model_path, bool hw_accel = false);
    ~LiteRTSileroVad() override;

    float process_chunk(const float* samples, size_t length) override;
    void reset() override;

    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }

private:
    TfLiteModel*       model_   = nullptr;
    TfLiteInterpreter* interp_  = nullptr;

    static constexpr size_t kContextSamples = 64;
    static constexpr size_t kChunkSamples   = 512;
    static constexpr size_t kTotalSamples   = kContextSamples + kChunkSamples;  // 576
    static constexpr size_t kStateSize      = 2 * 1 * 128;

    std::array<float, kContextSamples> context_{};
    std::array<float, kStateSize>      state_{};
    std::array<float, kTotalSamples>   input_buffer_{};
};

}  // namespace speech_core
