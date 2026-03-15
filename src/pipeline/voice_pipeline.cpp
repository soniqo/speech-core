#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/audio/pcm_codec.h"

#include <chrono>
#include <stdexcept>

namespace speech_core {

VoicePipeline::VoicePipeline(
    STTInterface& stt,
    TTSInterface& tts,
    LLMInterface* llm,
    VADInterface& vad,
    AgentConfig config,
    EventCallback on_event,
    EnhancerInterface* enhancer)
    : stt_(stt),
      tts_(tts),
      llm_(llm),
      enhancer_(enhancer),
      config_(config),
      on_event_(std::move(on_event)),
      turn_detector_(vad, config,
                     [this](const TurnEvent& e) { on_turn_event(e); }),
      context_(/* system_prompt */ "",
               config.max_history_messages > 0 ? config.max_history_messages : 0,
               config.max_history_tokens > 0 ? config.max_history_tokens : 0,
               config.mask_tool_results) {}

VoicePipeline::~VoicePipeline() {
    stop();
}

void VoicePipeline::start() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(true);
        state_.store(State::Idle);
    }
    worker_thread_ = std::thread(&VoicePipeline::worker_loop, this);
}

void VoicePipeline::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_.store(false);
        tts_.cancel();
        if (llm_) llm_->cancel();
        speech_queue_.cancel_all();
        state_.store(State::Idle);
    }
    // Wake and join the worker thread
    worker_cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void VoicePipeline::resume_listening() {
    if (!running_.load()) return;
    auto s = state_.load();
    if (s == State::Speaking) {
        std::lock_guard<std::mutex> lock(mutex_);
        turn_detector_.set_agent_speaking(false);
        turn_detector_.reset();
        // Post-playback guard: suppress VAD events for a short window
        // to let AEC residual echo settle. Non-blocking — the guard
        // counts down in push_audio as samples flow through.
        if (config_.post_playback_guard > 0) {
            turn_detector_.set_post_playback_guard(config_.post_playback_guard);
        }
        state_.store(State::Idle);
    }
}

void VoicePipeline::push_audio(const float* samples, size_t count) {
    if (!running_.load()) return;
    std::lock_guard<std::mutex> lock(mutex_);

    if (enhancer_ && count > 0) {
        enhance_buf_.resize(count);
        enhancer_->enhance(samples, count, enhancer_->input_sample_rate(),
                           enhance_buf_.data());
        turn_detector_.push_audio(enhance_buf_.data(), count);
    } else {
        turn_detector_.push_audio(samples, count);
    }
}

void VoicePipeline::worker_loop() {
    // Warm up STT model — first inference is slow due to Neural Engine/GPU
    // cold start. Running a dummy transcription brings latency from ~3s to <1s.
    if (config_.warmup_stt) {
        std::vector<float> silence(stt_.input_sample_rate() / 2, 0.0f);
        stt_.transcribe(silence.data(), silence.size(), stt_.input_sample_rate());
    }

    bool streaming_active = false;
    size_t stream_offset = 0;  // samples already sent to push_chunk

    while (running_.load()) {
        PendingUtterance utterance;
        bool have_utterance = false;
        {
            std::unique_lock<std::mutex> lock(worker_mutex_);
            bool use_timed_wait = config_.emit_partial_transcriptions
                                  && stt_.supports_streaming()
                                  && !streaming_active;
            bool poll_stream = streaming_active;

            if (use_timed_wait || poll_stream) {
                auto timeout = std::chrono::milliseconds(
                    static_cast<int>(config_.partial_transcription_interval * 1000));
                worker_cv_.wait_for(lock, timeout, [this] {
                    return !pending_utterances_.empty() || !running_.load();
                });
            } else {
                worker_cv_.wait(lock, [this] {
                    return !pending_utterances_.empty() || !running_.load();
                });
            }
            if (!running_.load()) {
                if (streaming_active) {
                    stt_.cancel_stream();
                    streaming_active = false;
                }
                return;
            }

            if (!pending_utterances_.empty()) {
                worker_busy_.store(true);
                utterance = std::move(pending_utterances_.front());
                pending_utterances_.erase(pending_utterances_.begin());
                have_utterance = true;
            }
        }

        // Streaming STT: feed new audio chunks during speech
        if (!have_utterance && streaming_active) {
            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (turn_detector_.in_speech()) {
                    snapshot = turn_detector_.utterance_snapshot();
                }
            }
            if (snapshot.size() > stream_offset) {
                try {
                    auto partial_text = stt_.push_chunk(
                        snapshot.data() + stream_offset,
                        snapshot.size() - stream_offset);
                    stream_offset = snapshot.size();
                    if (!partial_text.empty()) {
                        PipelineEvent partial;
                        partial.type = EventType::PartialTranscription;
                        partial.text = partial_text;
                        on_event_(partial);
                    }
                } catch (...) {}
            }
            continue;
        }

        // Start streaming when speech begins (no utterance yet, speech active)
        if (!have_utterance && !streaming_active
            && config_.emit_partial_transcriptions
            && stt_.supports_streaming()) {
            bool speech_active;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                speech_active = turn_detector_.in_speech();
            }
            if (speech_active) {
                stt_.begin_stream(stt_.input_sample_rate());
                streaming_active = true;
                stream_offset = 0;
            }
            continue;
        }

        if (!have_utterance) continue;

        // Emit SpeechEnded before starting STT
        {
            PipelineEvent ended;
            ended.type = EventType::SpeechEnded;
            ended.start_time = utterance.time;
            on_event_(ended);
        }

        // Run STT (no pipeline mutex held — push_audio continues to flow)
        try {
            auto stt_start = std::chrono::steady_clock::now();
            TranscriptionResult result;

            if (streaming_active) {
                // Feed any remaining audio, then finalize stream
                if (utterance.audio.size() > stream_offset) {
                    stt_.push_chunk(
                        utterance.audio.data() + stream_offset,
                        utterance.audio.size() - stream_offset);
                }
                result = stt_.end_stream();
                streaming_active = false;
                stream_offset = 0;
            } else {
                result = stt_.transcribe(
                    utterance.audio.data(), utterance.audio.size(),
                    stt_.input_sample_rate());
            }

            float stt_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - stt_start).count();

            // Check if this eager utterance was invalidated (user resumed speaking).
            // agent_speaking_ is NOT set during STT, so new speech during
            // transcription queues as a new utterance instead of interrupting.
            bool invalidated = eager_invalidated_.exchange(false);

            if (invalidated) {
                // Eager utterance discarded — user resumed speaking.
            } else {
                PipelineEvent transcript_event;
                transcript_event.type = EventType::TranscriptionCompleted;
                transcript_event.text = result.text;
                transcript_event.start_time = utterance.time;
                transcript_event.stt_duration_ms = stt_ms;
                on_event_(transcript_event);
            }

            if (!invalidated && !result.text.empty()) {
                process_utterance(result.text, result.language, stt_ms);
            } else if (invalidated) {
                // Eager utterance discarded — turn detector is tracking
                // active speech, state is already Listening.
            } else {
                // Empty transcription (noise/breath) — resume idle
                state_.store(State::Idle);
                std::lock_guard<std::mutex> lock(mutex_);
                turn_detector_.reset();
            }
        } catch (const std::exception& ex) {
            emit_error(std::string("STT failed: ") + ex.what());
            state_.store(State::Idle);
        }

        // Signal idle
        {
            std::lock_guard<std::mutex> lock(worker_mutex_);
            if (pending_utterances_.empty()) {
                worker_busy_.store(false);
                worker_idle_cv_.notify_all();
            }
        }
    }
}

void VoicePipeline::wait_idle() {
    std::unique_lock<std::mutex> lock(worker_mutex_);
    worker_idle_cv_.wait(lock, [this] {
        return (pending_utterances_.empty() && !worker_busy_.load()) || !running_.load();
    });
}

void VoicePipeline::push_text(const std::string& text) {
    if (!running_.load()) return;
    // push_text bypasses STT — called by user, not audio thread.
    // Safe to process inline (caller expects blocking).
    process_utterance(text);
}

void VoicePipeline::on_turn_event(const TurnEvent& event) {
    // Called from push_audio with mutex already held
    switch (event.type) {
    case TurnEvent::UserSpeechStarted: {
        // If user resumed speaking after an eager STT utterance was dispatched,
        // signal the worker to discard its result (it's a partial utterance).
        if (event.eager_resumed) {
            eager_invalidated_.store(true);
            turn_detector_.set_agent_speaking(false);
        }
        state_.store(State::Listening);
        PipelineEvent e;
        e.type = EventType::SpeechStarted;
        e.start_time = event.time;
        on_event_(e);
        break;
    }

    case TurnEvent::UserSpeechEnded: {
        state_.store(State::Transcribing);
        // Don't set agent_speaking_ here — the agent isn't responding yet.
        // New speech during STT should queue as a new utterance, not interrupt.
        // agent_speaking_ is set later when the agent actually starts
        // responding (Thinking state or TTS playback).
        // Enqueue audio for the worker thread — don't block push_audio
        // with STT/TTS which can take seconds.
        {
            std::lock_guard<std::mutex> wlock(worker_mutex_);
            pending_utterances_.push_back({event.audio, event.time, event.eager});
        }
        worker_cv_.notify_one();
        break;
    }

    case TurnEvent::Interruption: {
        // Cancel current TTS, LLM, and clear queue.
        // LLM cancel is needed when user speaks during Thinking state —
        // the worker thread is blocked in llm_->chat() and won't pick up
        // new utterances until the current call returns.
        tts_.cancel();
        if (llm_) llm_->cancel();
        speech_queue_.cancel_all();
        turn_detector_.set_agent_speaking(false);
        state_.store(State::Listening);

        PipelineEvent interrupted;
        interrupted.type = EventType::ResponseInterrupted;
        interrupted.start_time = event.time;
        on_event_(interrupted);
        break;
    }

    case TurnEvent::InterruptionRecovered:
        // Brief interruption — user stopped quickly, could resume playback
        // For now, pipeline stays in current state; platform layer handles
        // resuming audio playback.
        break;
    }
}

void VoicePipeline::process_utterance(const std::string& transcript,
                                      const std::string& language,
                                      float stt_duration_ms) {
    context_.add_user_message(transcript);

    std::string response_text;
    float llm_ms = 0.0f;

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
            // Now the agent is actively responding — mark as speaking so
            // new speech triggers interruption (cancels LLM).
            {
                std::lock_guard<std::mutex> lock(mutex_);
                turn_detector_.set_agent_speaking(true);
            }

            try {
                auto llm_start = std::chrono::steady_clock::now();
                response_text = call_llm_with_tools();
                llm_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - llm_start).count();
            } catch (const std::exception& ex) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    turn_detector_.set_agent_speaking(false);
                }
                emit_error(std::string("LLM failed: ") + ex.what());
                state_.store(State::Idle);
                return;
            }

            // If interrupted during LLM generation (state changed from
            // Thinking to Listening), discard the partial response.
            // agent_speaking_ was already reset by the interruption handler.
            if (state_.load() != State::Thinking) {
                return;
            }
        }
        break;
    }

    if (!response_text.empty()) {
        context_.add_assistant_message(response_text);
        speak(response_text, language, stt_duration_ms, llm_ms);
    } else {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            turn_detector_.set_agent_speaking(false);
        }
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

void VoicePipeline::speak(const std::string& text, const std::string& language,
                          float stt_duration_ms, float llm_duration_ms) {
    state_.store(State::Speaking);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        turn_detector_.set_agent_speaking(true);
    }

    uint64_t speech_id = speech_queue_.enqueue(text);
    speech_queue_.next();  // mark as playing

    PipelineEvent response_created;
    response_created.type = EventType::ResponseCreated;
    response_created.llm_duration_ms = llm_duration_ms;
    on_event_(response_created);

    // Use detected language from STT if available, otherwise fall back to config
    const auto& tts_language = !language.empty() ? language : config_.language;

    try {
        size_t total_samples = 0;
        size_t max_samples = config_.max_response_duration > 0
            ? static_cast<size_t>(config_.max_response_duration * tts_.output_sample_rate())
            : 0;

        auto tts_start = std::chrono::steady_clock::now();

        tts_.synthesize(text, tts_language,
            [this, speech_id, &total_samples, max_samples,
             tts_start, stt_duration_ms, llm_duration_ms](
                const float* samples, size_t length, bool is_final) {
                // Enforce max response duration to prevent TTS hallucination
                size_t emit_length = length;
                bool force_final = false;
                if (max_samples > 0 && total_samples + length > max_samples) {
                    emit_length = max_samples - total_samples;
                    force_final = true;
                }
                total_samples += emit_length;

                if (emit_length > 0) {
                    auto pcm = PCMCodec::float_to_pcm16(samples, emit_length);
                    PipelineEvent audio_event;
                    audio_event.type = EventType::ResponseAudioDelta;
                    audio_event.audio_data = std::move(pcm);
                    on_event_(audio_event);
                }

                if (is_final || force_final) {
                    speech_queue_.mark_done(speech_id);

                    float tts_ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - tts_start).count();

                    PipelineEvent done;
                    done.type = EventType::ResponseDone;
                    done.stt_duration_ms = stt_duration_ms;
                    done.llm_duration_ms = llm_duration_ms;
                    done.tts_duration_ms = tts_ms;
                    on_event_(done);

                    // Stay in Speaking — platform owns playback timing
                    if (force_final && !is_final) {
                        tts_.cancel();  // Stop TTS if we hit the cap
                    }
                }
            });
    } catch (const std::exception& ex) {
        speech_queue_.mark_done(speech_id);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            turn_detector_.set_agent_speaking(false);
        }
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
