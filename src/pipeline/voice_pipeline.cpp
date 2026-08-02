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
    if (running_.load()) return;

    // A completed stop() leaves both thread objects non-joinable. Join any
    // already-finished objects defensively before marking the new session
    // running; setting running_ first could revive an old worker and deadlock.
    if (worker_thread_.joinable()) worker_thread_.join();
    if (cancel_thread_.joinable()) cancel_thread_.join();

    // Reset dispatcher flags BEFORE spawning so a previous stop() cycle
    // doesn't leave shutdown=true (which would make the new dispatcher
    // exit immediately, silently dropping every Interruption).
    {
        std::lock_guard<std::mutex> clk(cancel_mutex_);
        cancel_shutdown_ = false;
        cancel_pending_ = false;
    }

    // start() is an independent input-stream boundary even if a caller reached
    // it after an incomplete shutdown path. Match on_turn_event's lock order:
    // pipeline mutex first, worker mutex second.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> wlock(worker_mutex_);
        turn_generation_.fetch_add(1, std::memory_order_acq_rel);
        eager_invalidated_generation_.store(0, std::memory_order_release);
        pending_utterances_.clear();
        worker_reset_requested_ = false;
        worker_busy_.store(false);
        turn_detector_.reset_for_new_stream();
        speech_queue_.cancel_all();
        running_.store(true);
        state_.store(State::Idle);
        if (echo_canceller_) echo_canceller_->reset();
    }

    worker_thread_ = std::thread(&VoicePipeline::worker_loop, this);
    cancel_thread_ = std::thread(&VoicePipeline::cancel_loop, this);
}

void VoicePipeline::stop() {
    // Close the input boundary and invalidate all work before asking backends
    // to cancel. on_turn_event uses the same mutex_ -> worker_mutex_ order.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> wlock(worker_mutex_);
        turn_generation_.fetch_add(1, std::memory_order_acq_rel);
        eager_invalidated_generation_.store(0, std::memory_order_release);
        running_.store(false);
        pending_utterances_.clear();
        worker_reset_requested_ = false;
        turn_detector_.reset_for_new_stream();
        speech_queue_.cancel_all();
        state_.store(State::Idle);
    }

    worker_cv_.notify_all();
    worker_idle_cv_.notify_all();

    // Cancellation hooks are thread-safe by interface contract. They run with
    // no pipeline locks held so a backend can finish its worker callback.
    stt_.cancel();
    tts_.cancel();
    if (llm_) llm_->cancel();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    {
        std::lock_guard<std::mutex> wlock(worker_mutex_);
        pending_utterances_.clear();
        worker_reset_requested_ = false;
        worker_busy_.store(false);
    }
    worker_idle_cv_.notify_all();

    // Tear down the cancel dispatcher AFTER the worker so the worker's
    // force_final speak() path (which calls tts_.cancel() on the worker
    // thread) cannot race with the dispatcher's own cancel call. Same
    // store-under-mutex pattern as the worker shutdown above so the
    // dispatcher's cv predicate (cancel_pending_ || cancel_shutdown_)
    // observes the shutdown flag before it sleeps.
    {
        std::lock_guard<std::mutex> clk(cancel_mutex_);
        cancel_shutdown_ = true;
    }
    cancel_cv_.notify_all();
    if (cancel_thread_.joinable()) {
        cancel_thread_.join();
    }
}

void VoicePipeline::cancel_current_turn() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::lock_guard<std::mutex> wlock(worker_mutex_);
        turn_generation_.fetch_add(1, std::memory_order_acq_rel);
        eager_invalidated_generation_.store(0, std::memory_order_release);
        pending_utterances_.clear();
        worker_reset_requested_ = running_.load();
        turn_detector_.reset_for_new_stream();
        speech_queue_.cancel_all();
        state_.store(State::Idle);
        if (echo_canceller_) echo_canceller_->reset();
    }

    worker_cv_.notify_all();

    // Batch cancellation is best-effort; generation checks remain the source
    // of correctness for implementations whose cancel() is a no-op.
    stt_.cancel();
    if (llm_) llm_->cancel();
    tts_.cancel();
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
    // Close the check/lock race with stop(): a producer may have observed
    // running=true immediately before stop() acquired this mutex.
    if (!running_.load()) return;

    const float* audio = samples;

    // Echo cancellation (if set) — removes TTS playback from mic signal
    if (echo_canceller_ && count > 0) {
        aec_buf_.resize(count);
        echo_canceller_->cancel_echo(audio, count, aec_buf_.data());
        audio = aec_buf_.data();
    }

    // Speech enhancement (if set) — denoising after AEC
    if (enhancer_ && count > 0) {
        enhance_buf_.resize(count);
        enhancer_->enhance(audio, count, enhancer_->input_sample_rate(),
                           enhance_buf_.data());
        audio = enhance_buf_.data();
    }

    turn_detector_.push_audio(audio, count);
}

void VoicePipeline::worker_loop() {
    // Every backend call below reaches third-party runtimes that report
    // failure by throwing — ORT does this for a graph whose inputs do not
    // match what the wrapper feeds it. Letting that escape a std::thread
    // entry point calls std::terminate, so an SDK consumer loses its whole
    // process to a model problem it could otherwise have handled.
    try {
        worker_loop_impl();
    } catch (const std::exception& e) {
        fail_worker(std::string("speech worker stopped: ") + e.what());
    } catch (...) {
        fail_worker("speech worker stopped: unknown error");
    }
}

void VoicePipeline::fail_worker(const std::string& message) {
    // Deliberately not emit_error: that drops anything whose generation is no
    // longer current, and a worker that has died is worth reporting whichever
    // turn it belonged to. The worker is gone once this returns, so the state
    // has to say so — otherwise is_running() keeps claiming a live pipeline.
    running_.store(false);
    state_.store(State::Idle);

    PipelineEvent error;
    error.type = EventType::Error;
    error.text = message;
    if (on_event_) on_event_(error);
}

void VoicePipeline::worker_loop_impl() {
    // Warm up STT model — first inference is slow due to Neural Engine/GPU
    // cold start. Running a dummy transcription brings latency from ~3s to <1s.
    if (config_.warmup_stt) {
        std::vector<float> silence(stt_.input_sample_rate() / 2, 0.0f);
        stt_.transcribe(silence.data(), silence.size(), stt_.input_sample_rate());
    }

    bool streaming_active = false;
    size_t stream_offset = 0;  // samples already sent to push_chunk
    uint64_t streaming_generation = 0;

    // Cancel any active streaming STT on exit from any path — the inner
    // !running_ check at the top of the loop only fires when the worker
    // wakes via the cv. If stop() is called while push_chunk is blocking,
    // the worker returns to the while-condition check and exits without
    // touching the stream. This guard catches that.
    struct StreamGuard {
        STTInterface& stt;
        bool& active;
        ~StreamGuard() {
            if (active) {
                try { stt.cancel_stream(); } catch (...) {}
                active = false;
            }
        }
    } stream_guard{stt_, streaming_active};

    auto finish_utterance = [this] {
        std::lock_guard<std::mutex> lock(worker_mutex_);
        if (pending_utterances_.empty()) {
            worker_busy_.store(false);
            worker_idle_cv_.notify_all();
        }
    };

    while (running_.load()) {
        PendingUtterance utterance;
        bool have_utterance = false;
        bool reset_worker_state = false;
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
                    return !pending_utterances_.empty()
                        || worker_reset_requested_
                        || !running_.load();
                });
            } else {
                worker_cv_.wait(lock, [this] {
                    return !pending_utterances_.empty()
                        || worker_reset_requested_
                        || !running_.load();
                });
            }
            if (!running_.load()) {
                if (streaming_active) {
                    stt_.cancel_stream();
                    streaming_active = false;
                    streaming_generation = 0;
                    stream_offset = 0;
                }
                return;
            }

            // cancel_current_turn() cannot directly touch the worker-owned
            // streaming session: cancel_stream() implementations are not
            // required to be thread-safe. Handle and acknowledge the reset
            // here before considering an utterance from the new generation.
            if (worker_reset_requested_) {
                reset_worker_state = true;
            } else if (!pending_utterances_.empty()) {
                worker_busy_.store(true);
                utterance = std::move(pending_utterances_.front());
                pending_utterances_.erase(pending_utterances_.begin());
                have_utterance = true;
            }
        }

        if (reset_worker_state) {
            if (streaming_active) {
                try { stt_.cancel_stream(); } catch (...) {}
                streaming_active = false;
                streaming_generation = 0;
                stream_offset = 0;
            }
            {
                std::lock_guard<std::mutex> lock(worker_mutex_);
                worker_reset_requested_ = false;
                if (pending_utterances_.empty() && !worker_busy_.load()) {
                    worker_idle_cv_.notify_all();
                }
            }
            continue;
        }

        // Streaming STT: feed new audio chunks during speech
        if (!have_utterance && streaming_active) {
            if (!is_current_turn(streaming_generation)) {
                try { stt_.cancel_stream(); } catch (...) {}
                streaming_active = false;
                streaming_generation = 0;
                stream_offset = 0;
                continue;
            }

            std::vector<float> snapshot;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (is_current_turn(streaming_generation)
                    && turn_detector_.in_speech()) {
                    snapshot = turn_detector_.utterance_snapshot();
                }
            }
            if (snapshot.size() > stream_offset) {
                try {
                    auto partial = stt_.push_chunk(
                        snapshot.data() + stream_offset,
                        snapshot.size() - stream_offset);
                    stream_offset = snapshot.size();
                    if (!partial.text.empty()
                        && is_current_turn(streaming_generation)) {
                        PipelineEvent event;
                        event.type = EventType::PartialTranscription;
                        event.text = partial.text;
                        event.confidence = partial.confidence;
                        on_event_(event);
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
            uint64_t generation;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                speech_active = turn_detector_.in_speech();
                generation = turn_generation_.load(std::memory_order_acquire);
            }
            if (speech_active && is_current_turn(generation)) {
                stt_.begin_stream(stt_.input_sample_rate());
                streaming_active = true;
                streaming_generation = generation;
                stream_offset = 0;
                if (!is_current_turn(generation)) {
                    try { stt_.cancel_stream(); } catch (...) {}
                    streaming_active = false;
                    streaming_generation = 0;
                    stream_offset = 0;
                }
            }
            continue;
        }

        if (!have_utterance) continue;

        if (!is_current_turn(utterance.generation)) {
            finish_utterance();
            continue;
        }

        // Emit SpeechEnded before starting STT
        {
            PipelineEvent ended;
            ended.type = EventType::SpeechEnded;
            ended.start_time = utterance.time;
            on_event_(ended);
        }
        if (!is_current_turn(utterance.generation)) {
            finish_utterance();
            continue;
        }

        // Run STT (no pipeline mutex held — push_audio continues to flow)
        try {
            auto stt_start = std::chrono::steady_clock::now();
            TranscriptionResult result;

            if (streaming_active
                && streaming_generation != utterance.generation) {
                try { stt_.cancel_stream(); } catch (...) {}
                streaming_active = false;
                streaming_generation = 0;
                stream_offset = 0;
            }

            if (streaming_active) {
                // Feed any remaining audio, then finalize stream
                if (utterance.audio.size() > stream_offset) {
                    stt_.push_chunk(
                        utterance.audio.data() + stream_offset,
                        utterance.audio.size() - stream_offset);
                }
                result = stt_.end_stream();
                streaming_active = false;
                streaming_generation = 0;
                stream_offset = 0;
            } else {
                result = stt_.transcribe(
                    utterance.audio.data(), utterance.audio.size(),
                    stt_.input_sample_rate());
            }

            float stt_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - stt_start).count();

            if (!is_current_turn(utterance.generation)) {
                finish_utterance();
                continue;
            }

            // Check if this eager utterance was invalidated (user resumed speaking).
            // agent_speaking_ is NOT set during STT, so new speech during
            // transcription queues as a new utterance instead of interrupting.
            uint64_t expected_generation = utterance.generation;
            bool invalidated =
                eager_invalidated_generation_.compare_exchange_strong(
                    expected_generation, 0, std::memory_order_acq_rel);

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

            // Filter low-confidence transcriptions (noise, coughs, mic bumps)
            bool low_confidence = config_.min_transcription_confidence > 0 &&
                                  result.confidence < config_.min_transcription_confidence;

            if (!invalidated && !result.text.empty() && !low_confidence) {
                process_utterance(result.text, result.language, stt_ms,
                                  utterance.generation);
            } else if (invalidated) {
                // Eager utterance discarded — turn detector is tracking
                // active speech, state is already Listening.
            } else {
                // Empty/low-confidence transcription — resume idle.
                // Reset turn detector and agent_speaking so VAD
                // can detect new speech immediately.
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (is_current_turn(utterance.generation)) {
                        turn_detector_.set_agent_speaking(false);
                        turn_detector_.reset();
                        state_.store(State::Idle);
                    }
                }
            }
        } catch (const std::exception& ex) {
            if (streaming_active) {
                try { stt_.cancel_stream(); } catch (...) {}
                streaming_active = false;
                streaming_generation = 0;
                stream_offset = 0;
            }
            if (is_current_turn(utterance.generation)) {
                emit_error(std::string("STT failed: ") + ex.what(),
                           utterance.generation);
                state_.store(State::Idle);
            }
        }

        finish_utterance();
    }
}

void VoicePipeline::wait_idle() {
    std::unique_lock<std::mutex> lock(worker_mutex_);
    worker_idle_cv_.wait(lock, [this] {
        return (pending_utterances_.empty()
                && !worker_busy_.load()
                && !worker_reset_requested_)
            || !running_.load();
    });
}

void VoicePipeline::push_text(const std::string& text) {
    if (!running_.load()) return;
    const uint64_t generation =
        turn_generation_.load(std::memory_order_acquire);
    if (!running_.load()) return;
    // push_text bypasses STT — called by user, not audio thread.
    // Safe to process inline (caller expects blocking).
    process_utterance(text, "", 0.0f, generation);
}

void VoicePipeline::on_turn_event(const TurnEvent& event) {
    // Called from push_audio with mutex already held
    switch (event.type) {
    case TurnEvent::UserSpeechStarted: {
        // If user resumed speaking after an eager STT utterance was dispatched,
        // signal the worker to discard its result (it's a partial utterance).
        if (event.eager_resumed) {
            eager_invalidated_generation_.store(
                turn_generation_.load(std::memory_order_acquire),
                std::memory_order_release);
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
            pending_utterances_.push_back({
                event.audio,
                event.time,
                event.eager,
                turn_generation_.load(std::memory_order_acquire)
            });
        }
        worker_cv_.notify_one();
        break;
    }

    case TurnEvent::Interruption: {
        // Audio thread (push_audio holds mutex_). Keep this path cheap.
        //
        // Synchronous, in-process work — bounded, microseconds:
        //   - queue drain   : speech_queue_.cancel_all() must run before
        //                     on_event_(ResponseInterrupted) so consumers
        //                     that re-enter from the callback observe an
        //                     empty queue.
        //   - turn detector : set_agent_speaking(false), already under
        //                     the mutex_ held by push_audio.
        //   - state flip    : the per-chunk speak() lambda guard drops
        //                     any further TTS chunks emitted before the
        //                     dispatched cancel actually takes effect.
        //
        // Deferred, potentially slow work — posted to cancel_thread_:
        //   - tts_.cancel(), llm_->cancel(): may block ~150ms on Ollama
        //     HTTP socket close, WebSocket TTS, etc. Running them inline
        //     here would stall mic frame delivery (push_audio holds
        //     mutex_). The dispatcher runs them with no pipeline locks.
        //
        // Coalescing: cancel_pending_ is a bool, not a queue. Back-to-
        // back Interruptions collapse to at most one in-flight cancel +
        // one pending re-run; cancel() is documented thread-safe and
        // idempotent on every shipped implementation.
        speech_queue_.cancel_all();
        turn_detector_.set_agent_speaking(false);
        state_.store(State::Listening);

        {
            std::lock_guard<std::mutex> clk(cancel_mutex_);
            cancel_pending_ = true;
        }
        cancel_cv_.notify_one();

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
                                      float stt_duration_ms,
                                      uint64_t generation) {
    if (!is_current_turn(generation)) return;

    context_.add_user_message(transcript);

    std::string response_text;
    float llm_ms = 0.0f;

    switch (config_.mode) {
    case AgentConfig::Mode::Echo:
        response_text = transcript;
        break;

    case AgentConfig::Mode::TranscribeOnly:
        if (is_current_turn(generation)) {
            state_.store(State::Idle);
        }
        return;

    case AgentConfig::Mode::Pipeline:
        if (!llm_) {
            response_text = transcript;
        } else {
            if (!is_current_turn(generation)) return;
            state_.store(State::Thinking);
            // Now the agent is actively responding — mark as speaking so
            // new speech triggers interruption (cancels LLM).
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!is_current_turn(generation)) return;
                turn_detector_.set_agent_speaking(true);
            }

            try {
                auto llm_start = std::chrono::steady_clock::now();
                response_text = call_llm_with_tools(generation);
                llm_ms = std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - llm_start).count();
            } catch (const std::exception& ex) {
                if (is_current_turn(generation)) {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (is_current_turn(generation)) {
                        turn_detector_.set_agent_speaking(false);
                    }
                }
                emit_error(std::string("LLM failed: ") + ex.what(), generation);
                if (is_current_turn(generation)) {
                    state_.store(State::Idle);
                }
                return;
            }

            // If interrupted during LLM generation (state changed from
            // Thinking to Listening), discard the partial response.
            // agent_speaking_ was already reset by the interruption handler.
            if (!is_current_turn(generation)
                || state_.load() != State::Thinking) {
                return;
            }
        }
        break;
    }

    if (!is_current_turn(generation)) return;

    if (!response_text.empty()) {
        context_.add_assistant_message(response_text);
        speak(response_text, language, stt_duration_ms, llm_ms, generation);
    } else {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_current_turn(generation)) {
                turn_detector_.set_agent_speaking(false);
            }
        }
        if (is_current_turn(generation)) {
            state_.store(State::Idle);
        }
    }
}

std::string VoicePipeline::call_llm_with_tools(uint64_t generation) {
    if (!is_current_turn(generation)) return std::string();

    // Pass tool definitions to LLM if tools are registered
    if (tool_registry_.size() > 0) {
        llm_->set_tools(tool_registry_.tools());
    }

    std::string accumulated;
    auto response = llm_->chat(context_.messages(),
        [&accumulated](const std::string& token, bool /*is_final*/) {
            accumulated += token;
        });

    if (!is_current_turn(generation)) return std::string();

    // Handle tool calls from LLM
    if (!response.tool_calls.empty()) {
        for (const auto& tc : response.tool_calls) {
            if (!is_current_turn(generation)) return std::string();

            // Emit tool call started
            PipelineEvent tool_started;
            tool_started.type = EventType::ToolCallStarted;
            tool_started.text = tc.name;
            on_event_(tool_started);
            if (!is_current_turn(generation)) return std::string();

            // Find and execute the tool
            const auto* tool = tool_registry_.find(tc.name);
            if (tool) {
                auto result = tool_executor_.execute(*tool);
                if (!is_current_turn(generation)) return std::string();

                PipelineEvent tool_completed;
                tool_completed.type = EventType::ToolCallCompleted;
                tool_completed.text = result.output;
                on_event_(tool_completed);
                if (!is_current_turn(generation)) return std::string();

                // Inject tool result into conversation
                context_.add_tool_message(tc.name,
                    result.success ? result.output : "Tool execution failed");
            } else {
                context_.add_tool_message(tc.name, "Unknown tool");
            }
        }

        // Don't fire the follow-up chat() if the user interrupted during
        // tool execution. Without this check, the second chat() runs to
        // completion and its response would race with the now-cancelled
        // turn. process_utterance does check state_ post-return but only
        // AFTER the second chat() has already consumed an LLM call.
        if (!is_current_turn(generation)
            || state_.load() != State::Thinking) {
            return std::string();
        }

        // Call LLM again with tool results in context
        accumulated.clear();
        response = llm_->chat(context_.messages(),
            [&accumulated](const std::string& token, bool /*is_final*/) {
                accumulated += token;
            });
    }

    if (!is_current_turn(generation)) return std::string();
    return response.text.empty() ? accumulated : response.text;
}

void VoicePipeline::speak(const std::string& text, const std::string& language,
                          float stt_duration_ms, float llm_duration_ms,
                          uint64_t generation) {
    if (!is_current_turn(generation)) return;

    uint64_t speech_id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!is_current_turn(generation)) return;
        state_.store(State::Speaking);
        turn_detector_.set_agent_speaking(true);
        speech_id = speech_queue_.enqueue(text);
        speech_queue_.next();  // mark as playing
    }

    if (!is_current_turn(generation)) return;

    PipelineEvent response_created;
    response_created.type = EventType::ResponseCreated;
    response_created.llm_duration_ms = llm_duration_ms;
    on_event_(response_created);
    if (!is_current_turn(generation)) return;

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
             tts_start, stt_duration_ms, llm_duration_ms, generation](
                const float* samples, size_t length, bool is_final) {
                // Drop chunks if we've left Speaking (interruption, stop()).
                // Some TTS impls don't honor cancel() promptly between chunks;
                // without this guard, ResponseAudioDelta events keep firing
                // after ResponseInterrupted, and AEC reference is fed audio
                // the speaker never actually played.
                if (!is_current_turn(generation)
                    || state_.load() != State::Speaking) {
                    return;
                }

                // Enforce max response duration to prevent TTS hallucination
                size_t emit_length = length;
                bool force_final = false;
                if (max_samples > 0 && total_samples + length > max_samples) {
                    emit_length = max_samples - total_samples;
                    force_final = true;
                }
                total_samples += emit_length;

                if (emit_length > 0) {
                    // Feed TTS audio as far-end reference for echo cancellation
                    if (echo_canceller_) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!is_current_turn(generation)
                            || state_.load() != State::Speaking) {
                            return;
                        }
                        echo_canceller_->feed_reference(samples, emit_length);
                    }
                    if (!is_current_turn(generation)) return;

                    auto pcm = PCMCodec::float_to_pcm16(samples, emit_length);
                    PipelineEvent audio_event;
                    audio_event.type = EventType::ResponseAudioDelta;
                    audio_event.audio_data = std::move(pcm);
                    on_event_(audio_event);
                    if (!is_current_turn(generation)) return;
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
                    if (force_final && !is_final
                        && is_current_turn(generation)) {
                        tts_.cancel();  // Stop TTS if we hit the cap
                    }
                }
            });
    } catch (const std::exception& ex) {
        speech_queue_.mark_done(speech_id);
        if (is_current_turn(generation)) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_current_turn(generation)) {
                turn_detector_.set_agent_speaking(false);
            }
        }
        emit_error(std::string("TTS failed: ") + ex.what(), generation);
        if (is_current_turn(generation)) {
            state_.store(State::Idle);
        }
    }
}

void VoicePipeline::emit_error(const std::string& message,
                               uint64_t generation) {
    if (!is_current_turn(generation)) return;

    PipelineEvent error;
    error.type = EventType::Error;
    error.text = message;
    on_event_(error);
}

bool VoicePipeline::is_current_turn(uint64_t generation) const {
    return generation != 0
        && turn_generation_.load(std::memory_order_acquire) == generation;
}

void VoicePipeline::cancel_loop() {
    // Long-lived dispatcher for off-thread cancel() calls. Owns no
    // pipeline locks across the third-party calls — that is the entire
    // point. Order: LLM first, then TTS. LLM cancel is the user-visible
    // latency driver because it unblocks the worker inside llm_->chat()
    // so process_utterance can observe state_ != Thinking and return.
    // TTS cancel is best-effort follow-up; the per-chunk state guard in
    // speak()'s lambda already prevents stale chunks from reaching the
    // platform. Exceptions out of cancel() are not contractually
    // defined, but we must keep the loop alive — swallow and continue.
    for (;;) {
        std::unique_lock<std::mutex> lock(cancel_mutex_);
        cancel_cv_.wait(lock, [this] {
            return cancel_pending_ || cancel_shutdown_;
        });
        if (cancel_shutdown_) return;
        cancel_pending_ = false;
        lock.unlock();

        if (llm_) {
            try { llm_->cancel(); } catch (...) {}
        }
        try { tts_.cancel(); } catch (...) {}
    }
}

}  // namespace speech_core
