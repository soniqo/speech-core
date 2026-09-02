#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace speech_core {

/// Pipecat Smart Turn v3.2 end-of-turn classifier (ONNX Runtime backend).
///
/// Whisper-Tiny encoder + attention pooling + MLP head, 8 M parameters. Our
/// export embeds the Whisper log-mel front-end (including the zero-mean /
/// unit-variance waveform normalisation the model was trained with), so the
/// graph takes raw 16 kHz audio:
///
///     audio [1, 128000] float32  ->  probability [1, 1] float32
///
/// The window is the last 8 s of the user's turn, zero-padded at the front
/// when shorter. Run it on the whole current turn after the VAD reports a
/// pause. One call costs roughly 20-50 ms on a laptop CPU with ORT's default
/// two threads (the encoder dominates; the embedded front-end is ~4 ms).
///
/// Model files: https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX
class OnnxSmartTurn final : public TurnCompletionInterface {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr std::size_t kWindowSamples = 128000;  // 8 s

    explicit OnnxSmartTurn(
        const std::string& model_path, bool hardware_acceleration = false);
    ~OnnxSmartTurn() override;

    OnnxSmartTurn(const OnnxSmartTurn&) = delete;
    OnnxSmartTurn& operator=(const OnnxSmartTurn&) = delete;

    float turn_complete_probability(
        const float* samples, std::size_t length, int sample_rate) override;

    /// The last kWindowSamples of `samples` (already at kSampleRate), left-padded
    /// with zeros when the turn is shorter than 8 s.
    static std::vector<float> prepare_window(
        const float* samples, std::size_t length);

private:
    void validate_contract();

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
    std::vector<float> window_;
};

}  // namespace speech_core
