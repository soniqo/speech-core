#pragma once

#include "speech_core/interfaces.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace speech_core {

/// One timestamp-aligned frame published after mandatory echo cancellation.
///
/// The three buffers always describe the same capture interval and retain only
/// `sample_count` real samples; the internal model-only padding is not exposed.
struct EchoCancelledFrame {
    std::int64_t start_time_ns = 0;
    std::size_t sample_count = 0;
    bool discontinuity = false;
    std::vector<float> raw_microphone;
    std::vector<float> playback_reference;
    std::vector<float> cleaned_microphone;
};

struct EchoCancellationPrimingEvent {
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 0;
    std::string reason;
    int delay_samples = 0;
    float delay_confidence = 0.0f;
};

/// Timestamp-aligns independently captured microphone and playback PCM before
/// sending still-separate frames through a FrameEchoCancellerInterface.
///
/// Capture callbacks only copy bounded timestamped segments. A private worker
/// performs model inference and publishes cleaned microphone frames. Missing
/// timestamps, backwards clocks, buffer overrun, or model failures stop this
/// stream; raw microphone PCM is never emitted as a fallback.
class TimestampedEchoCancellationStream {
public:
    struct Config {
        int sample_rate = 16000;
        std::size_t frame_size = 256;
        std::size_t capacity_samples = 16000 * 30;
        std::int64_t reference_wait_ns = 100000000;
        std::int64_t clock_skew_tolerance_ns = 10000000;
        std::int64_t timestamp_discontinuity_ns = 100000000;
        std::size_t playback_priming_samples = 16000 * 2;
        std::size_t playback_repriming_silence_samples = 16000 * 2;
        float playback_activation_rms = 0.001f;
        /// Must use the same monotonic epoch as input start_time_ns.
        std::function<std::int64_t()> current_time_ns;
    };

    using OutputCallback =
        std::function<void(const EchoCancelledFrame&)>;
    using FailureCallback =
        std::function<void(const std::string&)>;
    using PrimingCallback =
        std::function<void(const EchoCancellationPrimingEvent&)>;

    TimestampedEchoCancellationStream(
        FrameEchoCancellerInterface& canceller,
        Config config,
        OutputCallback output,
        FailureCallback failure = {},
        PrimingCallback priming = {});
    ~TimestampedEchoCancellationStream();

    TimestampedEchoCancellationStream(
        const TimestampedEchoCancellationStream&) = delete;
    TimestampedEchoCancellationStream& operator=(
        const TimestampedEchoCancellationStream&) = delete;

    void push_microphone(
        const float* samples,
        std::size_t count,
        std::int64_t start_time_ns,
        bool discontinuity = false);

    void push_reference(
        const float* samples,
        std::size_t count,
        std::int64_t start_time_ns,
        bool discontinuity = false);

    /// Mark both capture sources stopped, drain the final partial model frame,
    /// and wait for all already-captured microphone PCM to be published.
    /// Throws the fail-closed reason if synchronization or inference failed.
    void finish();

    /// Discard queued audio and stop without publishing a trailing frame.
    void cancel();

    std::optional<std::string> failure() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
