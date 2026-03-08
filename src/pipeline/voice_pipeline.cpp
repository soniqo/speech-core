#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/audio/pcm_codec.h"

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
      vad_(vad),
      config_(config),
      on_event_(std::move(on_event)),
      turn_detector_(vad, config,
                     [this](const TurnEvent& e) { on_turn_event(e); }),
      context_(/* system_prompt */ "", /* max_messages */ 50) {}

VoicePipeline::~VoicePipeline() {
    stop();
}

void VoicePipeline::start() {
    running_.store(true);
    state_.store(State::Idle);
}

void VoicePipeline::stop() {
    running_.store(false);
    tts_.cancel();
    if (llm_) llm_->cancel();
    speech_queue_.cancel_all();
    state_.store(State::Idle);
}

void VoicePipeline::push_audio(const float* samples, size_t count) {
    if (!running_.load()) return;
    turn_detector_.push_audio(samples, count);
}

void VoicePipeline::push_text(const std::string& text) {
    if (!running_.load()) return;
    process_utterance(text);
}

void VoicePipeline::on_turn_event(const TurnEvent& event) {
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
        break;
    }

    case TurnEvent::Interruption: {
        // Cancel current TTS and clear queue
        tts_.cancel();
        speech_queue_.cancel_all();
        turn_detector_.set_agent_speaking(false);
        break;
    }

    case TurnEvent::InterruptionRecovered:
        // Resume paused speech (not yet implemented — needs timer)
        break;
    }
}

void VoicePipeline::process_utterance(const std::string& transcript) {
    context_.add_user_message(transcript);

    std::string response_text;

    switch (config_.mode) {
    case AgentConfig::Mode::Echo:
        // Echo mode: speak back what the user said
        response_text = transcript;
        break;

    case AgentConfig::Mode::TranscribeOnly:
        // No response, just transcription
        state_.store(State::Idle);
        return;

    case AgentConfig::Mode::Pipeline:
        if (!llm_) {
            // No LLM configured, fall back to echo
            response_text = transcript;
        } else {
            state_.store(State::Thinking);

            // Collect LLM response
            std::string accumulated;
            llm_->chat(context_.messages(),
                [&accumulated](const std::string& token, bool is_final) {
                    accumulated += token;
                });
            response_text = accumulated;
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

void VoicePipeline::speak(const std::string& text) {
    state_.store(State::Speaking);
    turn_detector_.set_agent_speaking(true);

    uint64_t speech_id = speech_queue_.enqueue(text);
    speech_queue_.next();  // mark as playing

    PipelineEvent response_created;
    response_created.type = EventType::ResponseCreated;
    on_event_(response_created);

    tts_.synthesize(text, config_.language,
        [this, speech_id](const float* samples, size_t length, bool is_final) {
            // Emit audio chunks as events
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

                state_.store(State::Idle);
            }
        });
}

}  // namespace speech_core
