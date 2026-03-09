#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/audio/pcm_codec.h"

#include <stdexcept>

namespace speech_core {

VoicePipeline::VoicePipeline(
    STTInterface& stt,
    TTSInterface& tts,
    LLMInterface* llm,
    VADInterface& vad,
    AgentConfig config,
    EventCallback on_event)
    : stt_(stt),
      tts_(tts),
      llm_(llm),
      config_(config),
      on_event_(std::move(on_event)),
      turn_detector_(vad, config,
                     [this](const TurnEvent& e) { on_turn_event(e); }),
      context_(/* system_prompt */ "", /* max_messages */ 50) {}

VoicePipeline::~VoicePipeline() {
    stop();
}

void VoicePipeline::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_.store(true);
    state_.store(State::Idle);
}

void VoicePipeline::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_.store(false);
    tts_.cancel();
    if (llm_) llm_->cancel();
    speech_queue_.cancel_all();
    state_.store(State::Idle);
}

void VoicePipeline::resume_listening() {
    if (!running_.load()) return;
    if (state_.load() == State::Cooldown) {
        std::lock_guard<std::mutex> lock(mutex_);
        turn_detector_.reset();
        state_.store(State::Idle);
    }
}

void VoicePipeline::push_audio(const float* samples, size_t count) {
    if (!running_.load()) return;
    // Suppress input during cooldown (waiting for playback to finish)
    if (state_.load() == State::Cooldown) return;
    std::lock_guard<std::mutex> lock(mutex_);
    turn_detector_.push_audio(samples, count);
}

void VoicePipeline::push_text(const std::string& text) {
    if (!running_.load()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    process_utterance(text);
}

void VoicePipeline::on_turn_event(const TurnEvent& event) {
    // Called from push_audio with mutex already held
    switch (event.type) {
    case TurnEvent::UserSpeechStarted: {
        state_.store(State::Listening);
        PipelineEvent e;
        e.type = EventType::SpeechStarted;
        e.start_time = event.time;
        on_event_(e);
        break;
    }

    case TurnEvent::UserSpeechEnded: {
        state_.store(State::Transcribing);

        // Transcribe the utterance
        try {
            auto result = stt_.transcribe(
                event.audio.data(), event.audio.size(),
                stt_.input_sample_rate());

            PipelineEvent transcript_event;
            transcript_event.type = EventType::TranscriptionCompleted;
            transcript_event.text = result.text;
            transcript_event.start_time = event.time;
            on_event_(transcript_event);

            if (!result.text.empty()) {
                process_utterance(result.text);
            } else {
                state_.store(State::Idle);
            }
        } catch (const std::exception& ex) {
            emit_error(std::string("STT failed: ") + ex.what());
            state_.store(State::Idle);
        }
        break;
    }

    case TurnEvent::Interruption: {
        // Cancel current TTS and clear queue
        tts_.cancel();
        speech_queue_.cancel_all();
        turn_detector_.set_agent_speaking(false);
        state_.store(State::Listening);
        break;
    }

    case TurnEvent::InterruptionRecovered:
        // Brief interruption — user stopped quickly, could resume playback
        // For now, pipeline stays in current state; platform layer handles
        // resuming audio playback.
        break;
    }
}

void VoicePipeline::process_utterance(const std::string& transcript) {
    context_.add_user_message(transcript);

    std::string response_text;

    switch (config_.mode) {
    case AgentConfig::Mode::Echo:
        response_text = transcript;
        break;

    case AgentConfig::Mode::TranscribeOnly:
        state_.store(State::Idle);
        return;

    case AgentConfig::Mode::Pipeline:
        if (!llm_) {
            response_text = transcript;
        } else {
            state_.store(State::Thinking);

            try {
                response_text = call_llm_with_tools();
            } catch (const std::exception& ex) {
                emit_error(std::string("LLM failed: ") + ex.what());
                state_.store(State::Idle);
                return;
            }
        }
        break;
    }

    if (!response_text.empty()) {
        context_.add_assistant_message(response_text);
        speak(response_text);
    } else {
        state_.store(State::Idle);
    }
}

std::string VoicePipeline::call_llm_with_tools() {
    // Pass tool definitions to LLM if tools are registered
    if (tool_registry_.size() > 0) {
        llm_->set_tools(tool_registry_.tools());
    }

    std::string accumulated;
    auto response = llm_->chat(context_.messages(),
        [&accumulated](const std::string& token, bool /*is_final*/) {
            accumulated += token;
        });

    // Handle tool calls from LLM
    if (!response.tool_calls.empty()) {
        for (const auto& tc : response.tool_calls) {
            // Emit tool call started
            PipelineEvent tool_started;
            tool_started.type = EventType::ToolCallStarted;
            tool_started.text = tc.name;
            on_event_(tool_started);

            // Find and execute the tool
            const auto* tool = tool_registry_.find(tc.name);
            if (tool) {
                auto result = tool_executor_.execute(*tool);

                PipelineEvent tool_completed;
                tool_completed.type = EventType::ToolCallCompleted;
                tool_completed.text = result.output;
                on_event_(tool_completed);

                // Inject tool result into conversation
                context_.add_tool_message(tc.name,
                    result.success ? result.output : "Tool execution failed");
            } else {
                context_.add_tool_message(tc.name, "Unknown tool");
            }
        }

        // Call LLM again with tool results in context
        accumulated.clear();
        response = llm_->chat(context_.messages(),
            [&accumulated](const std::string& token, bool /*is_final*/) {
                accumulated += token;
            });
    }

    return response.text.empty() ? accumulated : response.text;
}

void VoicePipeline::speak(const std::string& text) {
    state_.store(State::Speaking);
    turn_detector_.set_agent_speaking(true);

    uint64_t speech_id = speech_queue_.enqueue(text);
    speech_queue_.next();  // mark as playing

    PipelineEvent response_created;
    response_created.type = EventType::ResponseCreated;
    on_event_(response_created);

    try {
        tts_.synthesize(text, config_.language,
            [this, speech_id](const float* samples, size_t length, bool is_final) {
                auto pcm = PCMCodec::float_to_pcm16(samples, length);

                PipelineEvent audio_event;
                audio_event.type = EventType::ResponseAudioDelta;
                audio_event.audio_data = std::move(pcm);
                on_event_(audio_event);

                if (is_final) {
                    speech_queue_.mark_done(speech_id);
                    turn_detector_.set_agent_speaking(false);

                    PipelineEvent done;
                    done.type = EventType::ResponseDone;
                    on_event_(done);

                    // Enter cooldown — audio suppressed until platform calls resume_listening()
                    state_.store(State::Cooldown);
                }
            });
    } catch (const std::exception& ex) {
        speech_queue_.mark_done(speech_id);
        turn_detector_.set_agent_speaking(false);
        emit_error(std::string("TTS failed: ") + ex.what());
        state_.store(State::Idle);
    }
}

void VoicePipeline::emit_error(const std::string& message) {
    PipelineEvent error;
    error.type = EventType::Error;
    error.text = message;
    on_event_(error);
}

}  // namespace speech_core
