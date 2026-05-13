#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <array>
#include <string>

namespace speech_core {

/// Silero VAD v5 — voice activity detection via ONNX Runtime.
/// Input: 512 samples (32 ms @ 16 kHz) per chunk.
/// Output: speech probability [0, 1].
class SileroVad : public VADInterface {
public:
    explicit SileroVad(const std::string& model_path, bool hw_accel = false);
    ~SileroVad() override;

    float process_chunk(const float* samples, size_t length) override;
    void reset() override;

    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }

private:
    const OrtApi* api_;
    OrtSession* session_ = nullptr;

    // LSTM state carried across chunks (Silero v5: [2, 1, 128])
    static constexpr size_t kStateSize = 2 * 1 * 128;
    std::array<float, kStateSize> state_{};
    int64_t sr_ = 16000;
};

}  // namespace speech_core
