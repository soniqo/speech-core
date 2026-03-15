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

    /// True if this UserSpeechStarted was emitted because the user resumed
    /// speaking after an eager STT utterance was already dispatched.
    bool eager_resumed = false;

    /// True if this UserSpeechEnded was emitted eagerly (on first silence
    /// frame) rather than after silence was confirmed.
    bool eager = false;

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

    /// Suppress VAD events for the given duration (seconds).
    /// Audio still feeds through the VAD model to keep its RNN warm,
    /// but no turn events fire until the guard expires. Used after
    /// agent playback to let AEC residual echo settle.
    void set_post_playback_guard(float seconds);

    /// Flush any pending turn at end of stream.
    void flush();

    /// Reset all state (clears any active post-playback guard).
    void reset();

    /// Whether the user is currently speaking.
    bool in_speech() const { return in_speech_; }

    /// Get a snapshot of the current utterance buffer (for partial transcription).
    std::vector<float> utterance_snapshot() const { return utterance_buffer_; }

private:
    VADInterface& vad_;
    AgentConfig config_;
    StreamingVAD streaming_vad_;
    EventCallback on_event_;

    bool agent_speaking_ = false;
    bool in_speech_ = false;
    bool interruption_confirmed_ = false;  // true once min_interruption_duration met
    bool eager_utterance_sent_ = false;    // true when eager STT fired early
    bool eager_pending_ = false;           // waiting for eager_stt_delay to elapse
    float eager_pending_time_ = 0.0f;     // time when SpeechPaused triggered pending eager
    std::vector<float> utterance_buffer_;
    float utterance_start_ = 0.0f;
    float interruption_time_ = -1.0f;

    /// Ring buffer keeping recent audio for pre-speech capture.
    std::vector<float> pre_speech_ring_;
    size_t pre_speech_capacity_ = 0;  // max samples in ring buffer

    /// Post-playback guard: remaining samples to suppress before resuming VAD.
    size_t guard_remaining_samples_ = 0;

    void force_end_utterance(float time);
};

}  // namespace speech_core
