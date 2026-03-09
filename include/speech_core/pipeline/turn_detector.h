#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/vad/streaming_vad.h"

namespace speech_core {

/// Turn detection events.
struct TurnEvent {
    enum Type {
        /// User started speaking.
        UserSpeechStarted,
        /// User finished speaking — full utterance audio is available.
        UserSpeechEnded,
        /// User interrupted agent speech (barge-in).
        Interruption,
        /// Brief interruption ended — agent speech can resume.
        InterruptionRecovered
    };

    Type type;
    float time;

    /// Accumulated user audio for this turn (populated on UserSpeechEnded).
    std::vector<float> audio;
};

/// Wraps StreamingVAD with interruption handling and utterance accumulation.
///
/// Accepts raw audio from the mic, runs it through VADInterface + StreamingVAD
/// hysteresis, accumulates speech audio, and emits TurnEvents with interruption
/// logic based on the current pipeline state.
class TurnDetector {
public:
    using EventCallback = std::function<void(const TurnEvent&)>;

    /// @param vad     Platform VAD model (Silero, etc.)
    /// @param config  Agent configuration
    /// @param on_event  Callback for turn events
    TurnDetector(VADInterface& vad, const AgentConfig& config, EventCallback on_event);

    /// Feed audio samples from the microphone.
    /// @param samples  PCM Float32 at vad.input_sample_rate()
    /// @param count    Number of samples
    void push_audio(const float* samples, size_t count);

    /// Notify the turn detector that the agent is currently speaking.
    /// This enables interruption detection.
    void set_agent_speaking(bool speaking);

    /// Flush any pending turn at end of stream.
    void flush();

    /// Reset all state.
    void reset();

private:
    VADInterface& vad_;
    AgentConfig config_;
    StreamingVAD streaming_vad_;
    EventCallback on_event_;

    bool agent_speaking_ = false;
    bool in_speech_ = false;
    std::vector<float> utterance_buffer_;
    float utterance_start_ = 0.0f;
    float interruption_time_ = -1.0f;

    /// Ring buffer keeping recent audio for pre-speech capture.
    std::vector<float> pre_speech_ring_;
    size_t pre_speech_capacity_ = 0;  // max samples in ring buffer

    void force_end_utterance(float time);
};

}  // namespace speech_core
