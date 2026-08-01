#pragma once

#include "speech_core/interfaces.h"

#include <cstddef>
#include <memory>
#include <string>

namespace speech_core {

/// Portable LocalVQE v1.4-AEC hybrid runtime.
///
/// The ONNX graph is only the recurrent neural residual mask. This class also
/// owns the exact released delay estimator, adaptive filter/controller,
/// 512/256 sqrt-Hann codec, and overlap-add state required around that graph.
class OnnxLocalVQEEchoCanceller final
    : public EchoCancellerInterface,
      public FrameEchoCancellerInterface {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr std::size_t kFrameSize = 256;
    static constexpr std::size_t kFftSize = 512;
    static constexpr std::size_t kAlgorithmicLatencySamples = kFrameSize;

    struct Config {
        bool hardware_acceleration = true;
        int intra_threads = 0;
        bool enable_prealignment = true;
    };

    struct Profile {
        double adaptive_filter_ms = 0.0;
        double neural_mask_ms = 0.0;
        double total_ms = 0.0;
    };

    explicit OnnxLocalVQEEchoCanceller(
        const std::string& bundle_directory);
    OnnxLocalVQEEchoCanceller(
        const std::string& bundle_directory, const Config& config);
    ~OnnxLocalVQEEchoCanceller() override;

    OnnxLocalVQEEchoCanceller(
        const OnnxLocalVQEEchoCanceller&) = delete;
    OnnxLocalVQEEchoCanceller& operator=(
        const OnnxLocalVQEEchoCanceller&) = delete;

    /// Process one synchronized 16 ms microphone/reference pair.
    ///
    /// The caller owns timestamp alignment. Both frames must represent the
    /// same capture interval; a gap in either stream requires reset().
    void process_frame(
        const float* microphone,
        const float* reference,
        float* output) override;

    /// Estimate and freeze bulk playback delay from one synchronized clip.
    /// The clip must contain the same number of microphone/reference samples.
    bool prime_delay(
        const float* microphone,
        const float* reference,
        std::size_t sample_count) override;

    int current_delay_samples() const override;
    float delay_confidence() const override;
    Profile last_profile() const;

    /// EchoCancellerInterface compatibility. feed_reference() queues exact
    /// 16 kHz reference samples; cancel_echo() consumes the same number and
    /// rejects underrun or non-256-aligned calls instead of passing raw audio.
    void feed_reference(
        const float* samples, std::size_t length) override;
    void cancel_echo(
        const float* input, std::size_t length, float* output) override;
    int input_sample_rate() const override { return kSampleRate; }
    std::size_t frame_size() const override { return kFrameSize; }
    void reset() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
