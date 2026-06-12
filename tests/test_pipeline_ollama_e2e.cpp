// FDB M0 — End-to-end test that wires the real OllamaLLM adapter into
// VoicePipeline. The whole point is to prove the integration path works,
// not just the OllamaLLM unit (test_ollama_llm.cpp) and not just the
// pipeline orchestration with a MockLLM (test_pipeline_e2e.cpp).
//
// Mocks STT / TTS / VAD (copied verbatim from test_pipeline_e2e.cpp's
// mocks for self-containment) and stands up an in-process MockOllama
// httplib::Server on an ephemeral port (copied from test_ollama_llm.cpp).
// Each scenario drives one full barge-in turn through the pipeline and
// asserts the event ordering plus light TTFT instrumentation printed
// for human inspection.
//
// This is the M0 foundation for the full FDB harness (M1: corpus loader,
// M2: real model swap-in, M3: CSV summary, M4: Python scorer + CI).

#include "speech_core/pipeline/voice_pipeline.h"
#include "speech_core/llm/ollama_llm.h"
#include "speech_core/tools/tool_types.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace speech_core;
using namespace std::chrono_literals;

namespace {

// ---------------------------------------------------------------------------
// Mocks — copied verbatim from test_pipeline_e2e.cpp so this file is
// self-contained. The pattern across all test files in this project is
// inline mock copies, not a shared header — matches the existing style.
// ---------------------------------------------------------------------------

class MockSTT : public STTInterface {
public:
    std::string next_text = "hello world";
    float next_confidence = 0.95f;
    int call_count = 0;

    TranscriptionResult transcribe(
        const float*, size_t, int) override
    {
        call_count++;
        return {next_text, "", next_confidence, 0.0f, 1.0f};
    }
    int input_sample_rate() const override { return 16000; }
};

class MockTTS : public TTSInterface {
public:
    std::string last_text;
    int call_count = 0;
    std::atomic<bool> cancelled{false};

    void synthesize(const std::string& text, const std::string&,
                    TTSChunkCallback on_chunk) override
    {
        call_count++;
        last_text = text;
        cancelled.store(false);
        // Emit a handful of chunks with a small inter-chunk sleep so a
        // mid-stream cancel actually has a chance to land. Honour the
        // cancel flag between chunks.
        for (int i = 0; i < 5; ++i) {
            if (cancelled.load()) break;
            float samples[8] = {0.0f, 0.05f, 0.0f, -0.05f,
                                0.0f, 0.05f, 0.0f, -0.05f};
            on_chunk(samples, 8, i == 4);
            std::this_thread::sleep_for(20ms);
        }
    }
    int output_sample_rate() const override { return 24000; }
    void cancel() override { cancelled.store(true); }
};

class MockVAD : public VADInterface {
public:
    std::vector<float> probs;
    std::atomic<size_t> prob_index{0};

    float process_chunk(const float*, size_t) override {
        size_t i = prob_index.fetch_add(1);
        return (i < probs.size()) ? probs[i] : 0.0f;
    }
    void reset() override { prob_index.store(0); }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }
};

// EventLog — thread-safe collector. on_event fires from BOTH push_audio
// (audio thread) and the worker thread, so every mutator/query takes the
// mutex. Adds steady_clock timestamps so scenarios can compute TTFTs.
struct EventLog {
    struct Entry {
        EventType type;
        std::string text;
        std::chrono::steady_clock::time_point at;
    };
    mutable std::mutex m;
    std::vector<Entry> entries;

    void on_event(const PipelineEvent& e) {
        std::lock_guard<std::mutex> lock(m);
        entries.push_back({e.type, e.text,
                           std::chrono::steady_clock::now()});
    }
    bool has(EventType t) const {
        std::lock_guard<std::mutex> lock(m);
        for (auto& e : entries) if (e.type == t) return true;
        return false;
    }
    size_t count(EventType t) const {
        std::lock_guard<std::mutex> lock(m);
        size_t n = 0;
        for (auto& e : entries) if (e.type == t) n++;
        return n;
    }
    std::string text_for(EventType t) const {
        std::lock_guard<std::mutex> lock(m);
        for (auto& e : entries) if (e.type == t) return e.text;
        return "";
    }
    std::chrono::steady_clock::time_point first_at(EventType t) const {
        std::lock_guard<std::mutex> lock(m);
        for (auto& e : entries) if (e.type == t) return e.at;
        return {};
    }
};

// ---------------------------------------------------------------------------
// MockOllama — in-process httplib::Server (copied from
// test_ollama_llm.cpp). Routes POST /api/chat to a user-supplied handler
// that emits NDJSON via DataSink. Counts requests so scenarios can assert
// against retry behaviour and tool round-trips.
// ---------------------------------------------------------------------------

using ChatHandler =
    std::function<void(const nlohmann::json& req, httplib::DataSink& sink,
                       int request_index /* 1-based */)>;

class MockOllama {
public:
    MockOllama(ChatHandler chat_handler,
               int status = 200,
               std::string error_body = {})
        : chat_handler_(std::move(chat_handler)),
          status_(status),
          error_body_(std::move(error_body))
    {
        server_.Post("/api/chat",
            [this](const httplib::Request& req, httplib::Response& res) {
                int idx = ++request_count_;
                {
                    std::lock_guard<std::mutex> lk(bodies_mu_);
                    request_bodies_.push_back(req.body);
                }
                if (status_ != 200) {
                    res.status = status_;
                    res.set_content(error_body_, "application/json");
                    return;
                }
                nlohmann::json parsed;
                try {
                    parsed = nlohmann::json::parse(req.body);
                } catch (const std::exception&) {
                    res.status = 400;
                    res.set_content("{\"error\":\"bad request body\"}",
                                    "application/json");
                    return;
                }
                res.set_chunked_content_provider(
                    "application/x-ndjson",
                    [parsed, idx, h = chat_handler_]
                    (size_t, httplib::DataSink& sink) {
                        h(parsed, sink, idx);
                        sink.done();
                        return true;
                    });
            });

        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ < 0) throw std::runtime_error("MockOllama: bind failed");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        while (!server_.is_running()) std::this_thread::sleep_for(2ms);
    }

    ~MockOllama() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }
    int request_count() const { return request_count_.load(); }
    std::vector<std::string> request_bodies() const {
        std::lock_guard<std::mutex> lk(bodies_mu_);
        return request_bodies_;
    }

private:
    httplib::Server  server_;
    ChatHandler      chat_handler_;
    int              status_;
    std::string      error_body_;
    int              port_ = -1;
    std::thread      thread_;
    std::atomic<int> request_count_{0};
    mutable std::mutex bodies_mu_;
    std::vector<std::string> request_bodies_;
};

void write_json_line(httplib::DataSink& sink, const nlohmann::json& j) {
    std::string s = j.dump();
    s.push_back('\n');
    sink.write(s.data(), s.size());
}

// ---------------------------------------------------------------------------
// Helpers — VAD probability scripts and audio buffer construction.
// ---------------------------------------------------------------------------

std::vector<float> make_audio_for_chunks(size_t n) {
    return std::vector<float>(n * 512, 0.0f);
}

AgentConfig pipeline_config() {
    AgentConfig c;
    c.mode = AgentConfig::Mode::Pipeline;
    c.warmup_stt = false;
    c.eager_stt = false;
    c.allow_interruptions = true;
    c.min_interruption_duration = 0.0f;
    c.post_playback_guard = 0.0f;
    c.max_response_duration = 60.0f;
    return c;
}

long long ms_between(std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

// Speech onset = 8 high probs (>=250ms @ ~32ms/chunk), then 5 silence
// (>=100ms) to confirm end-of-speech.
std::vector<float> standard_turn_probs() {
    std::vector<float> probs;
    probs.push_back(0.0f);
    for (int i = 0; i < 8; ++i) probs.push_back(0.8f);
    for (int i = 0; i < 5; ++i) probs.push_back(0.1f);
    return probs;
}

// ===========================================================================
// SCENARIO 1: Happy path — full STT -> OllamaLLM -> TTS round trip.
// ===========================================================================
void test_happy_path_full_round_trip() {
    MockOllama server([](const nlohmann::json& /*req*/,
                         httplib::DataSink& sink, int /*idx*/) {
        // Three content deltas, then the terminator.
        for (const char* d : {"The", " answer", " is 4."}) {
            write_json_line(sink, {
                {"model", "test"},
                {"message", {{"role", "assistant"}, {"content", d}}},
                {"done", false}
            });
        }
        write_json_line(sink, {
            {"model", "test"},
            {"message", {{"role", "assistant"}, {"content", ""}}},
            {"done", true}, {"done_reason", "stop"}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "test";
    OllamaLLM llm(opts);

    MockSTT stt; stt.next_text = "what is two plus two";
    MockTTS tts;
    MockVAD vad; vad.probs = standard_turn_probs();

    EventLog log;
    VoicePipeline pipe(stt, tts, &llm, vad, pipeline_config(),
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    auto t0 = std::chrono::steady_clock::now();
    auto audio = make_audio_for_chunks(vad.probs.size());
    pipe.push_audio(audio.data(), audio.size());
    pipe.wait_idle();

    // Give TTS time to emit all chunks (synthesize loop's inter-chunk
    // sleeps run on the worker thread after wait_idle returns? No: speak()
    // runs inside worker_loop synchronously, so wait_idle covers it).
    // The 5 chunks @ 20ms each = ~100ms; wait_idle blocks until the
    // worker is idle, which is after speak() returns.

    assert(log.has(EventType::SpeechStarted));
    assert(log.has(EventType::TranscriptionCompleted));
    assert(log.text_for(EventType::TranscriptionCompleted)
           == "what is two plus two");
    assert(log.has(EventType::ResponseCreated));
    assert(log.has(EventType::ResponseAudioDelta));
    assert(log.has(EventType::ResponseDone));
    assert(log.count(EventType::Error) == 0);
    assert(stt.call_count == 1);
    assert(tts.call_count == 1);
    assert(tts.last_text == "The answer is 4.");
    assert(server.request_count() == 1);

    // Verify the request body sent to Ollama contains the user message.
    auto bodies = server.request_bodies();
    assert(bodies.size() == 1);
    auto sent = nlohmann::json::parse(bodies[0]);
    assert(sent["model"] == "test");
    assert(sent["stream"] == true);
    bool found_user_msg = false;
    for (const auto& m : sent["messages"]) {
        if (m["role"] == "user" &&
            m["content"] == "what is two plus two") found_user_msg = true;
    }
    assert(found_user_msg);

    // TTFT instrumentation (lower-bound checks only — no upper bounds
    // here, CI loopback varies too much to assert tight latency).
    auto speech_end_at = log.first_at(EventType::SpeechEnded);
    auto txn_done_at   = log.first_at(EventType::TranscriptionCompleted);
    auto resp_created  = log.first_at(EventType::ResponseCreated);
    auto resp_audio    = log.first_at(EventType::ResponseAudioDelta);
    auto resp_done     = log.first_at(EventType::ResponseDone);
    printf("  TTFT happy_path: stt_ms=%lld llm_ms=%lld tts_ms=%lld total_ms=%lld\n",
           ms_between(speech_end_at, txn_done_at),
           ms_between(txn_done_at, resp_created),
           ms_between(resp_created, resp_audio),
           ms_between(t0, resp_done));

    pipe.stop();
    printf("  PASS: happy_path_full_round_trip\n");
}

// ===========================================================================
// SCENARIO 2: Cancel mid-LLM via barge-in.
// ===========================================================================
void test_cancel_mid_llm_via_barge_in() {
    std::atomic<bool> server_should_block{true};
    MockOllama server([&](const nlohmann::json& /*req*/,
                          httplib::DataSink& sink, int /*idx*/) {
        // One delta, then long sleep so cancel arrives mid-stream.
        write_json_line(sink, {
            {"model", "test"},
            {"message", {{"role", "assistant"}, {"content", "I think the"}}},
            {"done", false}
        });
        for (int i = 0; i < 100 && server_should_block.load(); ++i) {
            std::this_thread::sleep_for(20ms);
        }
        write_json_line(sink, {
            {"model", "test"},
            {"message", {{"role", "assistant"}, {"content", ""}}},
            {"done", true}, {"done_reason", "stop"}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "test";
    opts.request_timeout = 10s;
    OllamaLLM llm(opts);

    MockSTT stt; stt.next_text = "first prompt";
    MockTTS tts;
    MockVAD vad;

    EventLog log;
    VoicePipeline pipe(stt, tts, &llm, vad, pipeline_config(),
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    // First utterance — push and wait for the LLM to begin streaming.
    vad.probs = standard_turn_probs();
    vad.prob_index.store(0);
    auto audio = make_audio_for_chunks(vad.probs.size());
    pipe.push_audio(audio.data(), audio.size());

    // Wait for the first delta to land (so we know LLM is mid-stream).
    for (int i = 0; i < 200; ++i) {
        if (log.count(EventType::ResponseAudioDelta) > 0 ||
            tts.call_count > 0) break;
        std::this_thread::sleep_for(10ms);
    }

    // Barge-in. Push enough speech to fire SpeechStarted -> Interruption.
    std::vector<float> barge_probs;
    for (int i = 0; i < 8; ++i) barge_probs.push_back(0.8f);
    for (int i = 0; i < 5; ++i) barge_probs.push_back(0.1f);
    vad.probs = barge_probs;
    vad.prob_index.store(0);
    auto barge_audio = make_audio_for_chunks(barge_probs.size());

    auto barge_t0 = std::chrono::steady_clock::now();
    pipe.push_audio(barge_audio.data(), barge_audio.size());

    // Wait until ResponseInterrupted fires, with a generous timeout.
    bool saw_interrupted = false;
    for (int i = 0; i < 200; ++i) {
        if (log.has(EventType::ResponseInterrupted)) {
            saw_interrupted = true;
            break;
        }
        std::this_thread::sleep_for(10ms);
    }

    server_should_block.store(false);
    pipe.wait_idle();

    auto interrupted_at = log.first_at(EventType::ResponseInterrupted);
    long long cancel_ms = ms_between(barge_t0, interrupted_at);

    printf("  TTFT cancel_mid_llm: cancel_latency_ms=%lld\n", cancel_ms);

    assert(saw_interrupted && "ResponseInterrupted should have fired");
    // Cancel propagated quickly — we did NOT wait for the full 2 s server
    // sleep. Generous CI budget.
    assert(cancel_ms < 1500 &&
           "cancel did not short-circuit the LLM stream end-to-end");
    // No spurious retry — pipeline should not have started a second LLM
    // call for the cancelled prompt.
    assert(server.request_count() <= 2);

    pipe.stop();
    printf("  PASS: cancel_mid_llm_via_barge_in\n");
}

// ===========================================================================
// SCENARIO 3: Ollama returns HTTP 500 — propagates as Error event.
// ===========================================================================
void test_ollama_returns_http_500() {
    MockOllama server(
        [](const nlohmann::json&, httplib::DataSink&, int) {},
        500,
        "{\"error\":\"model not found\"}");

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "missing:latest";
    OllamaLLM llm(opts);

    MockSTT stt; stt.next_text = "hello";
    MockTTS tts;
    MockVAD vad; vad.probs = standard_turn_probs();

    EventLog log;
    VoicePipeline pipe(stt, tts, &llm, vad, pipeline_config(),
                       [&log](const PipelineEvent& e) { log.on_event(e); });
    pipe.start();

    auto audio = make_audio_for_chunks(vad.probs.size());
    pipe.push_audio(audio.data(), audio.size());
    pipe.wait_idle();

    assert(log.has(EventType::Error));
    std::string err = log.text_for(EventType::Error);
    assert(err.find("LLM") != std::string::npos);
    assert(err.find("500") != std::string::npos);
    assert(tts.call_count == 0 && "TTS must not run after LLM error");
    assert(!log.has(EventType::ResponseDone));
    assert(server.request_count() == 1 && "no retry expected");

    pipe.stop();
    printf("  PASS: ollama_returns_http_500\n");
}

// ===========================================================================
// SCENARIO 4: Tool call round trip — Ollama -> ToolRegistry -> Ollama again.
// ===========================================================================
void test_tool_call_round_trip() {
    MockOllama server([](const nlohmann::json& /*req*/,
                         httplib::DataSink& sink, int idx) {
        if (idx == 1) {
            // First request — return a tool_calls payload only.
            nlohmann::json tc = {
                {"function", {
                    {"name", "tell_time"},
                    {"arguments", nlohmann::json::object()}
                }}
            };
            write_json_line(sink, {
                {"model", "test"},
                {"message", {
                    {"role", "assistant"},
                    {"content", ""},
                    {"tool_calls", nlohmann::json::array({tc})}
                }},
                {"done", true}, {"done_reason", "tool_call"}
            });
        } else {
            // Second request — content reply using the tool result.
            write_json_line(sink, {
                {"model", "test"},
                {"message", {{"role", "assistant"},
                             {"content", "It is 3:30 PM."}}},
                {"done", false}
            });
            write_json_line(sink, {
                {"model", "test"},
                {"message", {{"role", "assistant"}, {"content", ""}}},
                {"done", true}, {"done_reason", "stop"}
            });
        }
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "test";
    OllamaLLM llm(opts);

    MockSTT stt; stt.next_text = "what time is it";
    MockTTS tts;
    MockVAD vad; vad.probs = standard_turn_probs();

    EventLog log;
    VoicePipeline pipe(stt, tts, &llm, vad, pipeline_config(),
                       [&log](const PipelineEvent& e) { log.on_event(e); });

    ToolDefinition td;
    td.name = "tell_time";
    td.description = "Get the current time";
    td.cooldown = 0;
    td.handler = [](const std::string&, const std::string&) {
        return std::string("3:30 PM");
    };
    pipe.tool_registry().add(td);

    pipe.start();

    auto audio = make_audio_for_chunks(vad.probs.size());
    pipe.push_audio(audio.data(), audio.size());
    pipe.wait_idle();

    assert(log.has(EventType::ToolCallStarted));
    assert(log.has(EventType::ToolCallCompleted));
    assert(log.text_for(EventType::ToolCallCompleted) == "3:30 PM");
    assert(server.request_count() == 2 && "expected 2 LLM calls (tool round trip)");

    // The second request body must include a tool result message.
    // ConversationContext::add_tool_message formats content as
    // "[<tool_name>] <output>" so we match on that pattern.
    auto bodies = server.request_bodies();
    assert(bodies.size() == 2);
    auto second = nlohmann::json::parse(bodies[1]);
    bool found_tool_msg = false;
    for (const auto& m : second["messages"]) {
        if (m["role"] == "tool") {
            const std::string& c = m["content"].get_ref<const std::string&>();
            if (c.find("tell_time") != std::string::npos &&
                c.find("3:30 PM")   != std::string::npos) {
                found_tool_msg = true;
                break;
            }
        }
    }
    assert(found_tool_msg &&
           "second LLM request must include the tool result as a tool message");

    assert(tts.call_count == 1);
    assert(tts.last_text == "It is 3:30 PM.");
    assert(log.has(EventType::ResponseDone));

    pipe.stop();
    printf("  PASS: tool_call_round_trip\n");
}

}  // namespace

int main() {
    printf("test_pipeline_ollama_e2e:\n");
    test_happy_path_full_round_trip();
    test_cancel_mid_llm_via_barge_in();
    test_ollama_returns_http_500();
    test_tool_call_round_trip();
    printf("All pipeline-Ollama e2e tests passed.\n");
    return 0;
}
