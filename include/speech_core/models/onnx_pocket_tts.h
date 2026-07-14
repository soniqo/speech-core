#pragma once

#include "speech_core/interfaces.h"

#include <cstdint>
#include <memory>
#include <string>

namespace speech_core {

/// Runtime controls for the public, fixed-voice Pocket TTS ONNX bundle.
struct PocketTtsConfig {
    /// Euler flow-matching evaluations per generated 80 ms audio frame.
    int flow_steps = 4;
    /// Hard upper bound for generated frames (500 frames = 40 seconds).
    int max_frames = 500;
    /// Frames retained after the EOS score first crosses eos_threshold.
    int frames_after_eos = 3;
    float eos_threshold = -4.0f;
    float temperature = 0.7f;
    /// -1 selects a fresh random seed for each synthesis call.
    std::int32_t seed = -1;
    /// Per-session ONNX Runtime intra-op thread count.
    int intra_threads = 2;
    /// The exported recurrent graphs are CPU-oriented. Hardware execution is
    /// opt-in because Android NNAPI support varies by device and ORT build.
    bool hardware_acceleration = false;
};

/// Measurements from the most recent synthesis call.
struct PocketTtsMetrics {
    double conditioning_ms = 0.0;
    double first_audio_ms = 0.0;
    double total_ms = 0.0;
    int frames_generated = 0;
    int output_samples = 0;
    std::uint32_t seed_used = 0;
    bool stopped_on_eos = false;
    bool cancelled = false;
};

/// Pocket TTS streaming ONNX backend.
///
/// Published bundle: https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8
/// (pin revision `v1.0.0`).
///
/// The bundle contains five graphs (`lm_main.int8.onnx`,
/// `lm_flow.int8.onnx`, `decoder.int8.onnx`, `encoder.onnx`, and
/// `text_conditioner.onnx`) plus `vocab.json` and `token_scores.json`.
/// The public export bakes in Kyutai's `alba` voice preset and is English-only.
///
/// Unlike the sherpa-onnx compatibility runtime, this implementation
/// interleaves one autoregressive latent with one Mimi decoder invocation and
/// emits each 1,920-sample (80 ms at 24 kHz) frame immediately. TTFA therefore
/// does not include generation of the remainder of the utterance.
class OnnxPocketTts final : public TTSInterface {
public:
    explicit OnnxPocketTts(const std::string& bundle_directory,
                           PocketTtsConfig config = {});
    ~OnnxPocketTts() override;

    OnnxPocketTts(const OnnxPocketTts&) = delete;
    OnnxPocketTts& operator=(const OnnxPocketTts&) = delete;

    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    PocketTtsMetrics last_metrics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
