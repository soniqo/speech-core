#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/pipeline/conversation_context.h"
#include "speech_core/pipeline/speech_queue.h"
#include "speech_core/pipeline/turn_detector.h"
#include "speech_core/protocol/events.h"
#include "speech_core/tools/tool_executor.h"
#include "speech_core/tools/tool_registry.h"

namespace speech_core {

/// Main orchestrator for the STT -> LLM -> TTS voice agent pipeline.
///
/// Connects platform-provided STT, TTS, LLM, and VAD implementations
/// through turn detection, conversation tracking, and speech queuing.
/// Does NOT own audio I/O or network transport — the platform layer
/// feeds audio in and receives events via callbacks.
///
/// Thread model:
///   - push_audio() called from mic/input thread
///   - Events emitted on the calling thread (platform dispatches as needed)
///   - Internal state protected by pipeline mutex
class VoicePipeline {
public:
    /// Callback for pipeline events (transcription, audio output, errors).
    using EventCallback = std::function<void(const PipelineEvent&)>;

    /// @param stt       Speech-to-text implementation
    /// @param tts       Text-to-speech implementation
    /// @param llm       Language model implementation (nullable for Echo mode)
    /// @param vad       Voice activity detection implementation
    /// @param config    Pipeline configuration
    /// @param on_event  Event callback
    VoicePipeline(
        STTInterface& stt,
        TTSInterface& tts,
        LLMInterface* llm,
        VADInterface& vad,
        AgentConfig config,
        EventCallback on_event);

    ~VoicePipeline();

    /// Feed audio samples from the microphone.
    /// @param samples   PCM Float32 at VAD sample rate
    /// @param count     Number of samples
    void push_audio(const float* samples, size_t count);

    /// Inject a text message (bypasses STT, sent directly to LLM).
    void push_text(const std::string& text);

    /// Current pipeline state.
    enum class State {
        Idle,
        Listening,
        Transcribing,
        Thinking,
        Speaking  // stays in Speaking after TTS until resume_listening()
    };

    State state() const { return state_.load(); }

    /// Start the pipeline (enables audio processing).
    void start();

    /// Stop the pipeline (cancels any in-progress work).
    void stop();

    /// Signal that response playback has finished.
    /// Transitions from Speaking back to Idle.
    /// Call this from the platform layer after speaker output ends.
    void resume_listening();

    /// Whether the pipeline is running.
    bool is_running() const { return running_.load(); }

    /// Block until the worker thread has processed all pending utterances.
    /// Useful for testing — in production, events arrive asynchronously.
    void wait_idle();

    /// Access the tool registry for adding tools.
    ToolRegistry& tool_registry() { return tool_registry_; }

private:
    STTInterface& stt_;
    TTSInterface& tts_;
    LLMInterface* llm_;
    AgentConfig config_;
    EventCallback on_event_;

    TurnDetector turn_detector_;
    SpeechQueue speech_queue_;
    ConversationContext context_;

    ToolRegistry tool_registry_;
    ToolExecutor tool_executor_;

    std::atomic<State> state_{State::Idle};
    std::atomic<bool> running_{false};
    mutable std::mutex mutex_;  // protects turn_detector_ and push_audio

    // Worker thread for STT/LLM/TTS — keeps push_audio non-blocking
    struct PendingUtterance {
        std::vector<float> audio;
        float time;
    };
    std::thread worker_thread_;
    std::mutex worker_mutex_;
    std::condition_variable worker_cv_;
    std::condition_variable worker_idle_cv_;  // signaled when worker finishes processing
    std::vector<PendingUtterance> pending_utterances_;
    std::atomic<bool> worker_busy_{false};

    void worker_loop();
    void on_turn_event(const TurnEvent& event);
    void process_utterance(const std::string& transcript, const std::string& language = "");
    void speak(const std::string& text, const std::string& language = "");
    void emit_error(const std::string& message);
    std::string call_llm_with_tools();
};

}  // namespace speech_core
