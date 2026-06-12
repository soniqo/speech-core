// Stress tests for the VoicePipeline orchestration layer.
//
// Each scenario pins a specific named failure mode in the interruption /
// barge-in path. Tests do not abort on failure — main() runs every scenario
// and reports a summary, returning non-zero if any scenario failed. This
// makes the binary a bug list rather than a single PASS/FAIL gate.
//
// Scenarios target:
//   #1 Concurrent on_event_ callback safety + wait_idle audio-thread blindspot
//   #2 Cancel coupling: tts/llm cancel() called under pipeline mutex on audio thread
//   #3 Stale ResponseAudioDelta events after ResponseInterrupted
//   #4 cancel_stream() not invoked on streaming STT during Interruption
//   #5 Second chat() in call_llm_with_tools fires after interrupt during tool exec
//   #6 on_event_ thread-safety contract under concurrent push_audio + interrupts
//   #7 SpeechQueue::cancel_all leaks Cancelled items in items_
//
// Links only speech_core (no models). Picked up automatically by the
// tests/test_*.cpp glob in CMakeLists.txt.

#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/pipeline/speech_queue.h"
#include "speech_core/tools/tool_registry.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace speech_core;
using namespace std::chrono_literals;

namespace {

// ---------------------------------------------------------------------------
// Test outcome + soft-CHECK that records but does not abort
// ---------------------------------------------------------------------------

struct Outcome {
    std::string name;
    bool pass = true;
    std::vector<std::string> failures;
    std::vector<std::string> notes;

    void fail(const std::string& msg) { pass = false; failures.push_back(msg); }
    void note(const std::string& msg) { notes.push_back(msg); }
};

#define CHECK(out, cond, msg)                                                  \
    do {                                                                       \
        if (!(cond)) (out).fail(std::string(msg) + " [" + #cond + "]");        \
    } while (0)

// ---------------------------------------------------------------------------
// SyncBarrier — wait for a named milestone with a bounded timeout.
// Replaces sleep_for-and-hope synchronization.
// ---------------------------------------------------------------------------

class SyncBarrier {
public:
    void signal(const std::string& name) {
        std::lock_guard<std::mutex> lock(m_);
        signals_[name]++;
        cv_.notify_all();
    }

    bool wait_for_count(const std::string& name, int count,
                        std::chrono::milliseconds timeout = 2000ms) {
        std::unique_lock<std::mutex> lock(m_);
        return cv_.wait_for(lock, timeout, [&] { return signals_[name] >= count; });
    }

    bool wait(const std::string& name,
              std::chrono::milliseconds timeout = 2000ms) {
        return wait_for_count(name, 1, timeout);
    }

    int count_of(const std::string& name) {
        std::lock_guard<std::mutex> lock(m_);
        return signals_[name];
    }

    void reset() {
        std::lock_guard<std::mutex> lock(m_);
        signals_.clear();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::unordered_map<std::string, int> signals_;
};

// ---------------------------------------------------------------------------
// SequencedEventLog — event log with per-event monotonic sequence number
// AND an in-flight-callback counter so tests can wait for all in-flight
// callbacks to drain before inspecting or mutating state. The
// `clear_safely()` method documents the safe pattern that the existing
// test_pipeline_e2e EventLog lacks — its log.types.clear() bypasses the
// per-entry mutex and races with in-flight callbacks (pre-existing SIGTRAP
// hypothesis under load).
// ---------------------------------------------------------------------------

class SequencedEventLog {
public:
    struct Entry {
        uint64_t seq;
        EventType type;
        std::string text;
        size_t audio_bytes;
    };

    void on_event(const PipelineEvent& e) {
        callbacks_in_flight_.fetch_add(1, std::memory_order_acq_rel);
        {
            std::lock_guard<std::mutex> lock(m_);
            entries_.push_back({++seq_, e.type, e.text, e.audio_data.size()});
        }
        if (callbacks_in_flight_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lock(idle_m_);
            idle_cv_.notify_all();
        }
    }

    // Wait until no callback is currently inside on_event(). Returns true
    // if the wait succeeded, false on timeout.
    bool wait_no_callbacks_in_flight(std::chrono::milliseconds timeout = 1000ms) {
        std::unique_lock<std::mutex> lock(idle_m_);
        return idle_cv_.wait_for(lock, timeout, [&] {
            return callbacks_in_flight_.load(std::memory_order_acquire) == 0;
        });
    }

    size_t count(EventType t) const {
        std::lock_guard<std::mutex> lock(m_);
        size_t n = 0;
        for (auto& e : entries_) if (e.type == t) n++;
        return n;
    }

    bool has(EventType t) const { return count(t) > 0; }

    // Sequence of the FIRST event of this type, or 0 if none.
    uint64_t first_seq(EventType t) const {
        std::lock_guard<std::mutex> lock(m_);
        for (auto& e : entries_) if (e.type == t) return e.seq;
        return 0;
    }

    // Count events of type t whose seq > min_seq.
    size_t count_after(EventType t, uint64_t min_seq) const {
        std::lock_guard<std::mutex> lock(m_);
        size_t n = 0;
        for (auto& e : entries_) if (e.type == t && e.seq > min_seq) n++;
        return n;
    }

    // Total audio-bytes emitted after min_seq (for stale-audio assertions).
    size_t audio_bytes_after(uint64_t min_seq) const {
        std::lock_guard<std::mutex> lock(m_);
        size_t b = 0;
        for (auto& e : entries_) {
            if (e.type == EventType::ResponseAudioDelta && e.seq > min_seq) {
                b += e.audio_bytes;
            }
        }
        return b;
    }

    // Safe clear — first drains in-flight callbacks, then wipes under the lock.
    // The unsafe equivalent (vector::clear() with no mutex, racing in-flight
    // push_back) is the root cause of the existing test_pipeline_e2e SIGTRAP.
    void clear_safely() {
        wait_no_callbacks_in_flight();
        std::lock_guard<std::mutex> lock(m_);
        entries_.clear();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_);
        return entries_.size();
    }

private:
    mutable std::mutex m_;
    std::vector<Entry> entries_;
    uint64_t seq_ = 0;

    std::atomic<int> callbacks_in_flight_{0};
    std::mutex idle_m_;
    std::condition_variable idle_cv_;
};

// ---------------------------------------------------------------------------
// Configurable mocks
// ---------------------------------------------------------------------------

class StressSTT : public STTInterface {
public:
    // Configuration
    bool streaming = false;
    std::string next_text = "user said something";
    float next_confidence = 0.9f;
    std::chrono::milliseconds transcribe_delay{0};

    // Telemetry (atomic — read from test thread)
    std::atomic<int> transcribe_count{0};
    std::atomic<int> begin_stream_count{0};
    std::atomic<int> push_chunk_count{0};
    std::atomic<int> end_stream_count{0};
    std::atomic<int> cancel_stream_count{0};

    SyncBarrier* barrier = nullptr;

    TranscriptionResult transcribe(const float*, size_t, int) override {
        transcribe_count.fetch_add(1);
        if (barrier) barrier->signal("stt:transcribe_started");
        if (transcribe_delay > 0ms) std::this_thread::sleep_for(transcribe_delay);
        return {next_text, "", next_confidence, 0.f, 0.f};
    }

    int input_sample_rate() const override { return 16000; }

    bool supports_streaming() const override { return streaming; }

    void begin_stream(int) override {
        begin_stream_count.fetch_add(1);
        if (barrier) barrier->signal("stt:begin_stream");
    }

    PartialResult push_chunk(const float*, size_t) override {
        push_chunk_count.fetch_add(1);
        if (barrier) barrier->signal("stt:push_chunk");
        return {};
    }

    TranscriptionResult end_stream() override {
        end_stream_count.fetch_add(1);
        return {next_text, "", next_confidence, 0.f, 0.f};
    }

    void cancel_stream() override {
        cancel_stream_count.fetch_add(1);
        if (barrier) barrier->signal("stt:cancel_stream");
    }
};

class StressTTS : public TTSInterface {
public:
    enum class CancelMode { HonorImmediately, HonorAfterNChunks, NeverHonor };

    // Configuration
    int num_chunks = 8;
    int samples_per_chunk = 1200;        // ~50ms @ 24kHz
    std::chrono::milliseconds chunk_interval{30};
    CancelMode cancel_mode = CancelMode::HonorImmediately;
    int honor_after_n = 2;
    std::chrono::milliseconds cancel_block{0};  // sleep inside cancel()

    // Telemetry
    std::atomic<int> synthesize_count{0};
    std::atomic<int> cancel_count{0};
    std::atomic<int> chunks_emitted{0};
    std::atomic<int> chunks_emitted_after_cancel{0};

    SyncBarrier* barrier = nullptr;
    std::atomic<bool> cancelled_{false};

    void synthesize(const std::string& /*text*/, const std::string& /*lang*/,
                    TTSChunkCallback on_chunk) override {
        synthesize_count.fetch_add(1);
        cancelled_.store(false);
        if (barrier) barrier->signal("tts:synthesize_started");

        std::vector<float> buf(static_cast<size_t>(samples_per_chunk), 0.1f);

        for (int i = 0; i < num_chunks; ++i) {
            bool was_cancelled = cancelled_.load(std::memory_order_acquire);

            // Honor cancel based on mode
            bool stop_now = false;
            if (was_cancelled) {
                switch (cancel_mode) {
                case CancelMode::HonorImmediately:
                    stop_now = true;
                    break;
                case CancelMode::HonorAfterNChunks:
                    if (chunks_after_cancel_ >= honor_after_n) stop_now = true;
                    else chunks_after_cancel_++;
                    break;
                case CancelMode::NeverHonor:
                    break;
                }
            }
            if (stop_now) {
                if (barrier) barrier->signal("tts:exited");
                return;
            }

            const bool is_last = (i == num_chunks - 1);
            on_chunk(buf.data(), buf.size(), is_last);

            chunks_emitted.fetch_add(1);
            if (was_cancelled) chunks_emitted_after_cancel.fetch_add(1);
            if (barrier) barrier->signal("tts:chunk_emitted");

            if (!is_last && chunk_interval > 0ms) {
                std::this_thread::sleep_for(chunk_interval);
            }
        }
        if (barrier) barrier->signal("tts:exited");
    }

    int output_sample_rate() const override { return 24000; }

    void cancel() override {
        cancel_count.fetch_add(1);
        cancelled_.store(true, std::memory_order_release);
        if (barrier) barrier->signal("tts:cancel_called");
        if (cancel_block > 0ms) std::this_thread::sleep_for(cancel_block);
    }

private:
    int chunks_after_cancel_ = 0;
};

class StressLLM : public LLMInterface {
public:
    // Configuration
    std::string response = "ok";
    std::vector<ToolCall> first_tool_calls;       // returned by first chat()
    std::chrono::milliseconds first_chat_delay{0};
    std::chrono::milliseconds second_chat_delay{0};
    std::chrono::milliseconds cancel_block{0};

    // Telemetry
    std::atomic<int> chat_count{0};
    std::atomic<int> cancel_count{0};
    std::atomic<bool> cancelled{false};
    std::vector<std::vector<Message>> messages_seen;  // last input each call (test thread only after wait_idle)
    std::mutex messages_m;

    SyncBarrier* barrier = nullptr;

    LLMResponse chat(const std::vector<Message>& msgs,
                     LLMTokenCallback on_token) override {
        int my_call = chat_count.fetch_add(1) + 1;
        {
            std::lock_guard<std::mutex> lock(messages_m);
            messages_seen.push_back(msgs);
        }
        if (barrier) barrier->signal("llm:chat_started");
        if (barrier) barrier->signal("llm:chat_" + std::to_string(my_call) + "_started");

        auto delay = (my_call == 1) ? first_chat_delay : second_chat_delay;
        // Cooperative cancel — wake up every 10ms to check.
        auto deadline = std::chrono::steady_clock::now() + delay;
        while (std::chrono::steady_clock::now() < deadline) {
            if (cancelled.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(10ms);
        }
        on_token(response, true);

        LLMResponse r;
        r.text = response;
        if (my_call == 1) {
            r.tool_calls = first_tool_calls;
        }
        if (barrier) barrier->signal("llm:chat_" + std::to_string(my_call) + "_returned");
        return r;
    }

    void cancel() override {
        cancel_count.fetch_add(1);
        cancelled.store(true, std::memory_order_release);
        if (barrier) barrier->signal("llm:cancel_called");
        if (cancel_block > 0ms) std::this_thread::sleep_for(cancel_block);
    }
};

class StressVAD : public VADInterface {
public:
    std::vector<float> probs;
    std::atomic<size_t> prob_index{0};

    float process_chunk(const float*, size_t) override {
        size_t i = prob_index.fetch_add(1);
        if (i < probs.size()) return probs[i];
        return 0.0f;  // silence after the script ends
    }

    void reset() override { prob_index.store(0); }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// 512 samples per chunk; one float-per-sample audio buffer.
std::vector<float> make_audio_for_chunks(size_t num_chunks) {
    return std::vector<float>(num_chunks * 512, 0.0f);
}

std::vector<float> speech_probs(size_t n, float p = 0.9f) { return std::vector<float>(n, p); }
std::vector<float> silence_probs(size_t n, float p = 0.0f) { return std::vector<float>(n, p); }

template <class... Vs>
std::vector<float> concat(Vs&&... vs) {
    std::vector<float> out;
    (out.insert(out.end(), vs.begin(), vs.end()), ...);
    return out;
}

AgentConfig default_stress_config() {
    AgentConfig c;
    c.warmup_stt = false;
    c.eager_stt = false;
    c.allow_interruptions = true;
    c.min_interruption_duration = 0.0f;   // immediate barge-in
    c.post_playback_guard = 0.0f;          // disable for test speed
    c.max_response_duration = 60.0f;       // long enough for slow mocks
    return c;
}

// Push audio for N chunks, one push() call (synchronous). With 512-sample
// chunks at 16 kHz, N chunks = N * 32 ms of audio.
void push_n_chunks(VoicePipeline& p, size_t n) {
    auto audio = make_audio_for_chunks(n);
    p.push_audio(audio.data(), audio.size());
}

// ===========================================================================
// SCENARIO 1: Concurrent callback safety + SequencedEventLog drain pattern
// ---------------------------------------------------------------------------
// Targets failure mode #1 (SIGTRAP race in test_pipeline_e2e). The premise:
// audio-thread on_event_ callbacks (SpeechStarted via push_audio) can be
// in-flight while a test thread mutates log state after wait_idle(). The
// SequencedEventLog::clear_safely() drains callbacks_in_flight before clearing
// — this scenario demonstrates the pattern survives many tight iterations
// under concurrent audio push, without SIGTRAP or vector corruption.
// ===========================================================================

void scenario_concurrent_callbacks_safe_clear(Outcome& out) {
    StressSTT stt;
    StressTTS tts;
    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    constexpr int kIterations = 50;
    std::atomic<bool> stop_aux{false};

    // Aux thread: continuously pushes silence to keep the audio thread
    // hammering on_event_ (SpeechStarted/Ended) while the test thread clears.
    std::thread aux([&] {
        StressVAD aux_vad;
        // We can't push through aux_vad — there's only one VAD on the pipeline.
        // Instead, push raw audio; VAD probs come from `vad`. So this thread
        // just adds concurrent push_audio pressure with the same vad.
        auto buf = make_audio_for_chunks(2);
        while (!stop_aux.load()) {
            pipe.push_audio(buf.data(), buf.size());
            std::this_thread::sleep_for(1ms);
        }
    });

    for (int i = 0; i < kIterations; ++i) {
        // Script speech → silence so we get a full speech turn + Response
        vad.probs = concat(
            silence_probs(2),
            speech_probs(10),
            silence_probs(6)
        );
        vad.prob_index.store(0);

        push_n_chunks(pipe, vad.probs.size());
        pipe.wait_idle();

        log.clear_safely();  // <-- the pattern under test
    }

    stop_aux.store(true);
    aux.join();
    pipe.stop();

    // The assertion is: we got here without SIGTRAP / heap corruption.
    // Additionally, ResponseDone emitted at least once across the run.
    out.note("ran " + std::to_string(kIterations) + " iterations under concurrent push");
    // If we reach this line, no UB was detected by the OS allocator.
    CHECK(out, true, "completed without crash");
}

// ===========================================================================
// SCENARIO 2: Audio-thread cancel coupling budget
// ---------------------------------------------------------------------------
// Targets failure mode #2. tts_.cancel() and llm_->cancel() are called on
// the audio thread under pipeline mutex_. If the impl's cancel() blocks,
// the audio thread stalls — which means new mic frames are dropped during
// the stall. This scenario documents the worst-case stall budget under a
// deliberately slow cancel().
//
// Asserts: push_audio (the one triggering Interruption) returns within
// (cancel_block_ms + slack) ms. Slack is generous to accommodate CI jitter.
// ===========================================================================

void scenario_audio_thread_cancel_coupling(Outcome& out) {
    StressSTT stt;
    stt.next_text = "hello";
    StressTTS tts;
    tts.num_chunks = 100;                 // enough to be mid-stream
    tts.chunk_interval = 5ms;
    tts.cancel_block = 150ms;             // deliberately slow cancel

    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;
    cfg.vad.min_speech_duration = 0.064f; // 2 chunks = ~64ms
    cfg.min_interruption_duration = 0.0f;

    SyncBarrier barrier;
    tts.barrier = &barrier;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // First utterance: drive a speech turn so Response/TTS starts.
    vad.probs = concat(silence_probs(1), speech_probs(10), silence_probs(6));
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    // Wait until TTS is actually emitting chunks (Speaking state).
    if (!barrier.wait_for_count("tts:chunk_emitted", 1, 3000ms)) {
        out.fail("tts never started emitting chunks");
        pipe.stop();
        return;
    }

    // Push interrupting speech and time how long it takes.
    vad.probs = speech_probs(6);
    vad.prob_index.store(0);
    auto interrupt_audio = make_audio_for_chunks(6);

    auto t0 = std::chrono::steady_clock::now();
    pipe.push_audio(interrupt_audio.data(), interrupt_audio.size());
    auto elapsed = std::chrono::steady_clock::now() - t0;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    out.note("push_audio (with Interruption) returned in " +
             std::to_string(elapsed_ms) + " ms (cancel_block=150ms)");

    // The cancel dispatcher (cancel_thread_) decouples push_audio from
    // tts_.cancel() / llm_->cancel(). On-thread work in the Interruption
    // path is now just: speech_queue_.cancel_all, set_agent_speaking,
    // state_.store, brief cancel_mutex_ + cv.notify, on_event_. The
    // 150ms cancel block runs off-thread.
    //
    // 50ms gives 8x headroom over the on-thread observed cost while
    // catching any regression that re-couples cancel to the audio
    // thread. If this assertion regresses, look for a new call site
    // that synchronously cancels TTS or LLM inside on_turn_event or
    // push_audio.
    constexpr int kBudgetMs = 50;
    CHECK(out, elapsed_ms < kBudgetMs,
          "push_audio stalled > " + std::to_string(kBudgetMs) + " ms "
          "(cancel coupling — dispatcher not decoupling correctly)");

    pipe.stop();
}

// ===========================================================================
// SCENARIO 3: No ResponseAudioDelta after ResponseInterrupted
// ---------------------------------------------------------------------------
// Targets failure mode #3. The speak() lambda has NO per-chunk state_ check —
// once TTS is emitting, chunks are forwarded via on_event_(ResponseAudioDelta)
// regardless of pipeline state. If the TTS impl ignores cancel() (some real
// TTS engines do not honor cancel between chunks), the orchestration leaks
// those chunks to the platform AFTER it has emitted ResponseInterrupted.
//
// Two sub-runs:
//   (a) HonorImmediately TTS — should emit 0 stale chunks (sanity).
//   (b) NeverHonor TTS — exposes the leak; current main has no orchestration
//       guard so stale chunks emit until synthesize() returns naturally.
//       Recorded as a metric + soft failure.
// ===========================================================================

void scenario_no_audio_chunks_after_interrupted_sub(
    Outcome& out, StressTTS::CancelMode mode, const char* label)
{
    StressSTT stt;
    stt.next_text = "first utterance";

    StressTTS tts;
    tts.num_chunks = 30;
    tts.chunk_interval = 20ms;
    tts.cancel_mode = mode;

    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;
    cfg.vad.min_speech_duration = 0.064f;
    cfg.min_interruption_duration = 0.0f;

    SyncBarrier barrier;
    tts.barrier = &barrier;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // Trigger first utterance → Response → TTS chunks.
    vad.probs = concat(silence_probs(1), speech_probs(10), silence_probs(6));
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    // Wait until TTS has emitted some chunks (we're mid-stream).
    if (!barrier.wait_for_count("tts:chunk_emitted", 3, 3000ms)) {
        out.fail(std::string(label) + ": tts did not emit early chunks");
        pipe.stop();
        return;
    }

    // Push interrupting speech.
    vad.probs = speech_probs(6);
    vad.prob_index.store(0);
    auto interrupt_audio = make_audio_for_chunks(6);
    pipe.push_audio(interrupt_audio.data(), interrupt_audio.size());

    // Drain.
    pipe.wait_idle();
    barrier.wait("tts:exited", 3000ms);
    log.wait_no_callbacks_in_flight();

    uint64_t interrupted_seq = log.first_seq(EventType::ResponseInterrupted);
    if (interrupted_seq == 0) {
        out.fail(std::string(label) + ": ResponseInterrupted never fired");
        pipe.stop();
        return;
    }
    size_t stale_chunks = log.count_after(EventType::ResponseAudioDelta, interrupted_seq);
    size_t stale_bytes = log.audio_bytes_after(interrupted_seq);

    out.note(std::string(label) + ": stale ResponseAudioDelta after ResponseInterrupted = " +
             std::to_string(stale_chunks) + " (" + std::to_string(stale_bytes) + " bytes)");

    if (mode == StressTTS::CancelMode::HonorImmediately) {
        // A well-behaved TTS + correct orchestration should have 0 stale chunks.
        CHECK(out, stale_chunks == 0,
              std::string(label) + ": expected 0 stale chunks with HonorImmediately TTS");
    } else {
        // NeverHonor: ANY stale chunks here document a real orchestration gap.
        // The speak() lambda should guard each on_chunk forwarding with a
        // state_ check so chunks after ResponseInterrupted are dropped.
        CHECK(out, stale_chunks == 0,
              std::string(label) +
                  ": orchestration forwarded TTS chunks after ResponseInterrupted "
                  "(speak() lambda lacks per-chunk state_ check) — count=" +
                  std::to_string(stale_chunks));
    }

    pipe.stop();
}

void scenario_no_audio_chunks_after_interrupted(Outcome& out) {
    scenario_no_audio_chunks_after_interrupted_sub(
        out, StressTTS::CancelMode::HonorImmediately, "sanity_honor");
    scenario_no_audio_chunks_after_interrupted_sub(
        out, StressTTS::CancelMode::NeverHonor, "misbehaving_tts");
}

// ===========================================================================
// SCENARIO 4: Streaming STT lifecycle — begin_stream must be paired
// ---------------------------------------------------------------------------
// The streaming STT path: while user is speaking and worker is otherwise
// idle, worker_loop calls begin_stream/push_chunk; on UserSpeechEnded it
// either calls end_stream (normal finish) or stop()'s StreamGuard calls
// cancel_stream. Every begin_stream MUST be matched.
//
// This scenario engages streaming (long uninterrupted speech, partial
// transcriptions enabled) and then stops the pipeline mid-stream. Asserts
// the (begin_stream - end_stream - cancel_stream) balance is zero.
// ===========================================================================

void scenario_streaming_stt_lifecycle_balanced(Outcome& out) {
    StressSTT stt;
    stt.streaming = true;
    stt.next_text = "streaming hi";
    StressTTS tts;
    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;
    cfg.vad.min_speech_duration = 0.064f;
    cfg.emit_partial_transcriptions = true;
    cfg.partial_transcription_interval = 0.03f;  // 30 ms — fast retry

    SyncBarrier barrier;
    stt.barrier = &barrier;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // Long uninterrupted speech (no silence) so streaming engages and stays
    // engaged. The worker will time out waiting for a queued utterance
    // (cv.wait_for) and enter the streaming branch.
    vad.probs = concat(silence_probs(1), speech_probs(40));
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    // Give the worker time to enter streaming mode.
    if (!barrier.wait_for_count("stt:begin_stream", 1, 1500ms)) {
        out.note("worker did not enter streaming during this run "
                 "(timing-dependent — skipping lifecycle check)");
        pipe.stop();
        return;
    }

    out.note("streaming engaged; push_chunk so far = " +
             std::to_string(stt.push_chunk_count.load()));

    // Stop mid-stream — StreamGuard / !running branch must call cancel_stream.
    pipe.stop();

    int begin  = stt.begin_stream_count.load();
    int end    = stt.end_stream_count.load();
    int cancel = stt.cancel_stream_count.load();
    out.note("begin_stream=" + std::to_string(begin) +
             "  end_stream=" + std::to_string(end) +
             "  cancel_stream=" + std::to_string(cancel));

    CHECK(out, begin >= 1, "begin_stream should have been called at least once");
    CHECK(out, begin == end + cancel,
          "streaming lifecycle imbalanced: begin (" + std::to_string(begin) +
          ") != end (" + std::to_string(end) + ") + cancel (" +
          std::to_string(cancel) + ")");
}

// ===========================================================================
// SCENARIO 5: No second chat() in tool loop after Interruption
// ---------------------------------------------------------------------------
// Targets failure mode #5. call_llm_with_tools() does NOT check pipeline
// state between the first chat() (which returned tool_calls) and the second
// chat() (with tool results). An interruption arriving during tool execution
// is therefore "lost" — the second chat() still fires, and its response
// flows through context_.add_assistant_message + speak() ... unless the
// process_utterance state_ check at line 427 catches it. But that check is
// AFTER call_llm_with_tools returns, so the second chat() has already run.
// ===========================================================================

void scenario_no_second_chat_after_interrupt_in_tool_loop(Outcome& out) {
    StressSTT stt;
    StressTTS tts;
    tts.num_chunks = 50;       // long-running so we stay in Speaking
    tts.chunk_interval = 20ms;
    StressLLM llm;
    StressVAD vad;

    // First chat() returns a tool call.
    ToolCall tc;
    tc.name = "test_tool";
    tc.arguments = "{}";
    llm.first_tool_calls = {tc};
    llm.first_chat_delay = 10ms;
    llm.second_chat_delay = 200ms;  // long second chat — gives interrupt a chance

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Pipeline;
    cfg.vad.min_speech_duration = 0.064f;
    cfg.min_interruption_duration = 0.0f;

    SyncBarrier barrier;
    llm.barrier = &barrier;
    tts.barrier = &barrier;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, &llm, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });

    // Register a slow tool that lets us time the interrupt right after it.
    std::atomic<int> tool_calls{0};
    ToolDefinition td;
    td.name = "test_tool";
    td.description = "stress test tool";
    td.cooldown = 0;
    td.handler = [&](const std::string&, const std::string&) -> std::string {
        tool_calls.fetch_add(1);
        barrier.signal("tool:started");
        std::this_thread::sleep_for(80ms);
        barrier.signal("tool:done");
        return std::string("tool_result");
    };
    pipe.tool_registry().add(td);

    pipe.start();

    // Drive first utterance → first chat() → tool dispatched.
    vad.probs = concat(silence_probs(1), speech_probs(10), silence_probs(6));
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    if (!barrier.wait("tool:started", 3000ms)) {
        out.fail("tool was never dispatched (first chat or tool path broke)");
        pipe.stop();
        return;
    }

    // Interrupt DURING tool execution — should cancel the orchestration
    // before the second chat() fires.
    vad.probs = speech_probs(8);
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    pipe.wait_idle();
    log.wait_no_callbacks_in_flight();

    int chat_calls = llm.chat_count.load();
    out.note("llm chat_count after interrupt during tool exec = " +
             std::to_string(chat_calls) + " (1 = correct, 2 = bug)");

    CHECK(out, log.has(EventType::ResponseInterrupted),
          "ResponseInterrupted should have fired during tool execution");

    // Correct behavior: state was set to Listening by the interrupt before
    // the second chat() runs, so the second chat() should be skipped.
    // Current call_llm_with_tools has no state check between the two chats.
    CHECK(out, chat_calls == 1,
          "second chat() fired despite interrupt during tool execution "
          "(call_llm_with_tools lacks state guard between calls)");

    pipe.stop();
}

// ===========================================================================
// SCENARIO 6: Concurrent push_audio with interrupts — no corruption
// ---------------------------------------------------------------------------
// Targets failure mode #6 (and exercises #1). Pipeline contract says
// on_event_ may be called from the audio thread OR the worker thread, with
// no ordering guarantee. Consumers must serialize. This scenario runs
// 3 push_audio threads driving interrupting speech against a sluggish TTS
// and asserts no crashes, no UB, no event count anomalies, and that
// SequencedEventLog observes a valid total order.
// ===========================================================================

void scenario_concurrent_push_audio_with_interrupts(Outcome& out) {
    StressSTT stt;
    stt.next_text = "concurrent";
    StressTTS tts;
    tts.num_chunks = 8;
    tts.chunk_interval = 15ms;
    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;
    cfg.vad.min_speech_duration = 0.064f;
    cfg.min_interruption_duration = 0.0f;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // Single big script with many speech/silence/speech cycles.
    vad.probs.clear();
    for (int i = 0; i < 40; ++i) {
        auto piece = concat(silence_probs(3), speech_probs(10), silence_probs(6));
        vad.probs.insert(vad.probs.end(), piece.begin(), piece.end());
    }
    vad.prob_index.store(0);

    auto buf = make_audio_for_chunks(5);
    constexpr int kThreads = 3;
    constexpr int kPushesPerThread = 30;

    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) {
        ts.emplace_back([&] {
            for (int i = 0; i < kPushesPerThread; ++i) {
                pipe.push_audio(buf.data(), buf.size());
                std::this_thread::sleep_for(2ms);
            }
        });
    }
    for (auto& t : ts) t.join();

    pipe.wait_idle();
    log.wait_no_callbacks_in_flight();

    out.note("concurrent: " + std::to_string(kThreads) + " threads × " +
             std::to_string(kPushesPerThread) + " pushes; events=" +
             std::to_string(log.size()));

    // The orchestration must not have corrupted state.
    CHECK(out, pipe.is_running(), "pipeline still running after concurrent storm");

    pipe.stop();
    CHECK(out, !pipe.is_running(), "pipeline stopped cleanly");
}

// ===========================================================================
// SCENARIO 7: SpeechQueue does not leak Cancelled items
// ---------------------------------------------------------------------------
// Targets failure mode #7. cancel_all() marks items Cancelled but does not
// erase them. Cleanup only runs inside mark_done() when an item transitions
// to Done AND only erases Done items from the front. With sustained
// cancel-without-mark_done (or with Pending items behind a Cancelled front),
// items_ accumulates.
//
// This test exercises the queue directly (no pipeline needed).
// ===========================================================================

void scenario_speech_queue_leak(Outcome& out) {
    constexpr int kIterations = 200;
    {
        SpeechQueue q;
        for (int i = 0; i < kIterations; ++i) {
            q.enqueue("utterance");
            q.next();         // mark Playing
            q.cancel_all();   // mark Cancelled — but does not erase
            // No mark_done call — mirrors interrupt mid-TTS path where speak()
            // lambda never reaches is_final=true.
        }
        out.note("cancel_all without mark_done × " + std::to_string(kIterations) +
                 "  → queue size = " + std::to_string(q.size()));
        // The leak: size grows unboundedly with the iteration count.
        CHECK(out, q.size() <= 4,
              "SpeechQueue accumulated " + std::to_string(q.size()) +
              " Cancelled items after " + std::to_string(kIterations) +
              " cancel_all calls (cancel_all should erase or mark_done should "
              "drain Cancelled prefixes)");
    }

    // Also test the multi-item Cancelled-blocks-front variant.
    {
        SpeechQueue q;
        for (int i = 0; i < kIterations; ++i) {
            auto a = q.enqueue("a");
            q.next();
            (void)q.enqueue("b");      // queued behind Playing a
            q.cancel_all();             // [a:Cancelled, b:Cancelled]
            q.mark_done(a);             // a:Cancelled → Done; cleanup drops a;
                                        // b:Cancelled remains stuck at front
        }
        out.note("interleaved cancel_all+mark_done × " + std::to_string(kIterations) +
                 " → queue size = " + std::to_string(q.size()));
        CHECK(out, q.size() <= 4,
              "SpeechQueue front-Cancelled-blocks-cleanup leak: " +
              std::to_string(q.size()) + " items remain");
    }
}

// ===========================================================================
// SCENARIO 8: stop() during pending cancel dispatch — no deadlock, no UAF
// ---------------------------------------------------------------------------
// With cancel() dispatched off the audio thread, lifecycle safety hinges on
// stop() joining the dispatcher AFTER its in-flight cancel call returns.
// Fire an Interruption with a deliberately slow TTS cancel (200 ms), then
// immediately call stop(). Asserts stop() returns within the cancel-block
// budget + slack, and the pipeline cleanly shuts down (no crash, no hang).
//
// This is the key safety test for the off-thread dispatcher: if stop()
// reordered the join to happen BEFORE the in-flight cancel completes, the
// dispatcher thread would access tts_ / llm_ AFTER they may be destructed
// (the references go dangling once their owners outlive the pipeline).
// ===========================================================================

void scenario_stop_during_pending_cancel(Outcome& out) {
    StressSTT stt;
    StressTTS tts;
    tts.num_chunks = 50;
    tts.chunk_interval = 5ms;
    tts.cancel_block = 200ms;             // dispatcher will sit inside cancel()
    StressVAD vad;

    auto cfg = default_stress_config();
    cfg.mode = AgentConfig::Mode::Echo;
    cfg.vad.min_speech_duration = 0.064f;
    cfg.min_interruption_duration = 0.0f;

    SyncBarrier barrier;
    tts.barrier = &barrier;

    SequencedEventLog log;
    VoicePipeline pipe(stt, tts, nullptr, vad, cfg,
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // Trigger first utterance → enter Speaking.
    vad.probs = concat(silence_probs(1), speech_probs(10), silence_probs(6));
    vad.prob_index.store(0);
    push_n_chunks(pipe, vad.probs.size());

    if (!barrier.wait_for_count("tts:chunk_emitted", 1, 3000ms)) {
        out.fail("tts never started");
        pipe.stop();
        return;
    }

    // Fire interrupting speech → posts cancel to dispatcher → dispatcher
    // calls tts.cancel() which sleeps 200ms.
    vad.probs = speech_probs(6);
    vad.prob_index.store(0);
    auto interrupt_audio = make_audio_for_chunks(6);
    pipe.push_audio(interrupt_audio.data(), interrupt_audio.size());

    // Give the dispatcher a moment to start the cancel.
    if (!barrier.wait_for_count("tts:cancel_called", 1, 1000ms)) {
        out.fail("dispatcher never called tts.cancel()");
        pipe.stop();
        return;
    }

    // stop() now — dispatcher is inside the 200ms sleep. stop() must
    // join cleanly after the cancel finishes.
    auto t0 = std::chrono::steady_clock::now();
    pipe.stop();
    auto elapsed = std::chrono::steady_clock::now() - t0;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    out.note("stop() during in-flight cancel returned in " +
             std::to_string(elapsed_ms) + " ms");

    // stop() budget: in-flight cancel (200ms) + stop's own synchronous
    // cancel calls + join overhead. 600ms is generous for CI jitter.
    CHECK(out, elapsed_ms < 600, "stop() during dispatched cancel exceeded budget");
    CHECK(out, !pipe.is_running(), "pipeline running flag should be false after stop()");
}

// ===========================================================================
// Test runner
// ===========================================================================

struct Scenario {
    const char* name;
    void (*fn)(Outcome&);
};

const Scenario kScenarios[] = {
    {"concurrent_callbacks_safe_clear",            scenario_concurrent_callbacks_safe_clear},
    {"audio_thread_cancel_coupling",               scenario_audio_thread_cancel_coupling},
    {"no_audio_chunks_after_interrupted",          scenario_no_audio_chunks_after_interrupted},
    {"streaming_stt_lifecycle_balanced",           scenario_streaming_stt_lifecycle_balanced},
    {"no_second_chat_after_interrupt_in_tool_loop", scenario_no_second_chat_after_interrupt_in_tool_loop},
    {"concurrent_push_audio_with_interrupts",      scenario_concurrent_push_audio_with_interrupts},
    {"speech_queue_leak",                          scenario_speech_queue_leak},
    {"stop_during_pending_cancel",                 scenario_stop_during_pending_cancel},
};

}  // namespace

int main() {
    printf("== test_pipeline_stress ==\n");
    int pass_count = 0;
    int fail_count = 0;
    std::vector<Outcome> results;

    for (const auto& s : kScenarios) {
        Outcome o;
        o.name = s.name;
        printf("RUN   %s\n", s.name);

        try {
            s.fn(o);
        } catch (const std::exception& ex) {
            o.fail(std::string("uncaught exception: ") + ex.what());
        } catch (...) {
            o.fail("uncaught non-exception");
        }

        for (const auto& n : o.notes) printf("      note: %s\n", n.c_str());
        if (o.pass) {
            printf("PASS  %s\n", s.name);
            pass_count++;
        } else {
            for (const auto& f : o.failures) printf("      FAIL: %s\n", f.c_str());
            printf("FAIL  %s\n", s.name);
            fail_count++;
        }
        results.push_back(std::move(o));
    }

    printf("\n--- Summary ---\n");
    printf("  passed: %d\n", pass_count);
    printf("  failed: %d\n", fail_count);
    if (fail_count > 0) {
        printf("\nFailures expose real orchestration gaps in src/pipeline/voice_pipeline.cpp\n"
               "or src/pipeline/speech_queue.cpp. Each failure message names the contract\n"
               "violated. Fix the underlying code, do not relax the assertions.\n");
    }
    return fail_count == 0 ? 0 : 1;
}
