#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <array>
#include <string>

namespace speech_core {

/// Silero VAD v5 — voice activity detection via ONNX Runtime.
/// Caller feeds 512 samples (32 ms @ 16 kHz) per chunk.
/// Output: speech probability [0, 1].
///
/// v5 is NOT stateless across the sample axis: the graph expects the previous
/// chunk's last 64 samples prepended to the current 512, so it actually sees a
/// 576-sample window with left context. The ONNX input axis is dynamic, so
/// feeding a bare 512 does not error — it silently returns degraded
/// probabilities. Measured on one FLEURS German clip: 16 of 808 frames cleared
/// 0.5 without the context, 488 of 808 with it. Downstream that read as "this
/// audio is mostly silence", the VAD handed the ASR ~2% of the speech, and the
/// transcripts came back correct but cut mid-sentence.
///
/// The context lives here, not in the caller: every caller would otherwise have
/// to know a detail of the graph, and the one that forgets gets no error.
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

    // Left context the v5 graph expects ahead of each chunk. Zero-filled on
    // reset(), which matches the reference implementation's first chunk.
    static constexpr size_t kContextSize = 64;
    static constexpr size_t kMaxChunk = 512;
    std::array<float, kContextSize> context_{};
    std::array<float, kContextSize + kMaxChunk> input_buf_{};

    int64_t sr_ = 16000;
};

}  // namespace speech_core
