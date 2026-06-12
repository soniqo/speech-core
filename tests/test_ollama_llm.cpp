// Tests for the OllamaLLM adapter.
//
// Runs without a real Ollama server: an in-process httplib::Server is bound
// to 127.0.0.1 on an OS-assigned ephemeral port and serves canned NDJSON
// responses keyed off the request body. Each scenario tears its own server
// down via RAII; no flaky port collisions.
//
// An opt-in integration block at the end pings a real Ollama at
// localhost:11434 when SPEECH_CORE_OLLAMA_INTEGRATION=1 is set in the
// environment. CI does not set the env, so the block is skipped by default.

#include "speech_core/llm/ollama_llm.h"
#include "speech_core/tools/tool_types.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
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
// Mock Ollama server — RAII. Routes POST /api/chat to a user-supplied
// handler that receives the parsed request JSON and writes NDJSON lines
// via the provided sink. The sink writes to httplib's chunked transfer
// path via DataSink::write.
// ---------------------------------------------------------------------------

using ChatHandler =
    std::function<void(const nlohmann::json& req, httplib::DataSink& sink)>;

class MockOllama {
public:
    MockOllama(ChatHandler chat_handler,
               int status = 200,
               std::string error_body = {})
        : chat_handler_(std::move(chat_handler)),
          status_(status),
          error_body_(std::move(error_body)) {
        server_.Post("/api/chat",
            [this](const httplib::Request& req, httplib::Response& res) {
                last_request_body_ = req.body;
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
                    [parsed, h = chat_handler_](size_t /*offset*/,
                                                httplib::DataSink& sink) {
                        h(parsed, sink);
                        sink.done();
                        return true;
                    });
            });

        // Bind to 127.0.0.1 + any free port. bind_to_any_port returns the
        // assigned port; listen_after_bind runs in a worker thread.
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ < 0) throw std::runtime_error("MockOllama: bind failed");
        thread_ = std::thread([this] { server_.listen_after_bind(); });

        // Block until the server is actually accepting; otherwise the
        // first client request races the listen call.
        while (!server_.is_running()) std::this_thread::sleep_for(2ms);
    }

    ~MockOllama() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    std::string last_request_body() const { return last_request_body_; }

private:
    httplib::Server server_;
    ChatHandler     chat_handler_;
    int             status_;
    std::string     error_body_;
    int             port_ = -1;
    std::thread     thread_;
    std::string     last_request_body_;
};

// Write a JSON value as one NDJSON line.
void write_json_line(httplib::DataSink& sink, const nlohmann::json& j) {
    std::string s = j.dump();
    s.push_back('\n');
    sink.write(s.data(), s.size());
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_construct_rejects_empty_model() {
    OllamaLLM::Options opts;
    opts.base_url = "http://localhost:11434";
    opts.model = "";
    bool threw = false;
    try { OllamaLLM llm(opts); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw && "expected invalid_argument on empty model");
    printf("  PASS: construct_rejects_empty_model\n");
}

void test_construct_rejects_bad_scheme() {
    OllamaLLM::Options opts;
    opts.base_url = "ftp://x";
    opts.model = "llama3.2:1b";
    bool threw = false;
    try { OllamaLLM llm(opts); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw && "expected invalid_argument on non-http scheme");
    printf("  PASS: construct_rejects_bad_scheme\n");
}

void test_streaming_three_chunks() {
    MockOllama server([](const nlohmann::json& /*req*/,
                         httplib::DataSink& sink) {
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", "Hello"}}},
            {"done", false}
        });
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", " world"}}},
            {"done", false}
        });
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", "!"}}},
            {"done", false}
        });
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", ""}}},
            {"done", true}, {"done_reason", "stop"}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    OllamaLLM llm(opts);

    std::vector<std::string> deltas;
    int final_count = 0;
    auto resp = llm.chat(
        {{MessageRole::User, "hi"}},
        [&](const std::string& t, bool is_final) {
            if (is_final) final_count++;
            else deltas.push_back(t);
        });

    assert(deltas.size() == 3);
    assert(deltas[0] == "Hello");
    assert(deltas[1] == " world");
    assert(deltas[2] == "!");
    assert(final_count == 1);
    assert(resp.text == "Hello world!");
    assert(resp.tool_calls.empty());
    printf("  PASS: streaming_three_chunks\n");
}

void test_tool_call_response() {
    MockOllama server([](const nlohmann::json& /*req*/,
                         httplib::DataSink& sink) {
        nlohmann::json tc = {
            {"function", {
                {"name", "tell_time"},
                {"arguments", nlohmann::json::object()}
            }}
        };
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {
                {"role", "assistant"},
                {"content", ""},
                {"tool_calls", nlohmann::json::array({tc})}
            }},
            {"done", true}, {"done_reason", "tool_call"}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    OllamaLLM llm(opts);

    auto resp = llm.chat(
        {{MessageRole::User, "what time is it"}},
        [](const std::string&, bool) {});

    assert(resp.tool_calls.size() == 1);
    assert(resp.tool_calls[0].name == "tell_time");
    assert(resp.tool_calls[0].arguments == "{}");
    printf("  PASS: tool_call_response\n");
}

void test_cancel_from_other_thread() {
    std::atomic<bool> server_should_block{true};
    MockOllama server([&](const nlohmann::json& /*req*/,
                          httplib::DataSink& sink) {
        // One chunk, then sleep for 2s — long enough for cancel to
        // arrive. The sink will start returning false (write failure)
        // once httplib detects the client-closed connection from
        // Client::stop, ending the handler.
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", "Hi"}}},
            {"done", false}
        });
        for (int i = 0; i < 100 && server_should_block.load(); ++i) {
            std::this_thread::sleep_for(20ms);
        }
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    opts.request_timeout = 10s;
    OllamaLLM llm(opts);

    std::atomic<bool> got_final{false};
    std::thread canceller([&] {
        std::this_thread::sleep_for(100ms);
        llm.cancel();
    });

    auto t0 = std::chrono::steady_clock::now();
    auto resp = llm.chat(
        {{MessageRole::User, "wait"}},
        [&](const std::string&, bool is_final) {
            if (is_final) got_final.store(true);
        });
    auto elapsed = std::chrono::steady_clock::now() - t0;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    canceller.join();
    server_should_block.store(false);

    // cancel arrived ~100 ms in; chat should return well before the
    // 10 s timeout. Give CI generous slack but still much less than the
    // server's 2 s sleep budget.
    assert(elapsed_ms < 1500 && "cancel did not short-circuit the stream");
    assert(got_final.load() && "final on_token(is_final=true) not delivered");
    // Partial response is preserved.
    assert(resp.text == "Hi");
    printf("  PASS: cancel_from_other_thread (elapsed=%lldms)\n",
           static_cast<long long>(elapsed_ms));
}

void test_http_error_surfaces_as_runtime_error() {
    MockOllama server(
        [](const nlohmann::json&, httplib::DataSink&) {},
        500,
        "{\"error\":\"model not found\"}");

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "missing:latest";
    OllamaLLM llm(opts);

    bool threw = false;
    std::string what;
    try {
        llm.chat({{MessageRole::User, "x"}},
                 [](const std::string&, bool) {});
    } catch (const std::runtime_error& ex) {
        threw = true;
        what = ex.what();
    }
    assert(threw);
    assert(what.find("500") != std::string::npos);
    printf("  PASS: http_error_surfaces_as_runtime_error\n");
}

void test_transport_failure_surfaces_as_runtime_error() {
    // No server — point at an unused port.
    OllamaLLM::Options opts;
    opts.base_url = "http://127.0.0.1:1";  // privileged + unused
    opts.model = "llama3.2:1b";
    opts.connect_timeout = 200ms;
    OllamaLLM llm(opts);

    bool threw = false;
    try {
        llm.chat({{MessageRole::User, "x"}},
                 [](const std::string&, bool) {});
    } catch (const std::runtime_error&) { threw = true; }
    assert(threw && "transport failure should throw");
    printf("  PASS: transport_failure_surfaces_as_runtime_error\n");
}

void test_set_tools_sends_tools_array() {
    std::string captured_body;
    MockOllama server([&](const nlohmann::json& req,
                          httplib::DataSink& sink) {
        captured_body = req.dump();
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", "ok"}}},
            {"done", true}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    OllamaLLM llm(opts);

    ToolDefinition td;
    td.name = "tell_time";
    td.description = "Get the current time";
    llm.set_tools({td});

    llm.chat({{MessageRole::User, "hi"}},
             [](const std::string&, bool) {});

    auto sent = nlohmann::json::parse(captured_body);
    assert(sent.contains("tools"));
    assert(sent["tools"].is_array());
    assert(sent["tools"].size() == 1);
    assert(sent["tools"][0]["function"]["name"] == "tell_time");
    printf("  PASS: set_tools_sends_tools_array\n");
}

void test_options_forwarded_to_ollama() {
    std::string captured_body;
    MockOllama server([&](const nlohmann::json& req,
                          httplib::DataSink& sink) {
        captured_body = req.dump();
        write_json_line(sink, {
            {"model", "llama3.2:1b"},
            {"message", {{"role", "assistant"}, {"content", "ok"}}},
            {"done", true}
        });
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    opts.keep_alive = "30m";
    opts.temperature = 0.7;
    opts.num_ctx = 4096;
    opts.num_predict = 128;
    OllamaLLM llm(opts);
    llm.chat({{MessageRole::User, "hi"}},
             [](const std::string&, bool) {});

    auto sent = nlohmann::json::parse(captured_body);
    assert(sent["keep_alive"] == "30m");
    assert(sent.contains("options"));
    assert(sent["options"]["temperature"].get<double>() == 0.7);
    assert(sent["options"]["num_ctx"] == 4096);
    assert(sent["options"]["num_predict"] == 128);
    printf("  PASS: options_forwarded_to_ollama\n");
}

void test_malformed_ndjson_surfaces_as_runtime_error() {
    MockOllama server([](const nlohmann::json&,
                         httplib::DataSink& sink) {
        const char* garbage = "not even json\n";
        sink.write(garbage, std::strlen(garbage));
    });

    OllamaLLM::Options opts;
    opts.base_url = server.base_url();
    opts.model = "llama3.2:1b";
    OllamaLLM llm(opts);

    bool threw = false;
    std::string what;
    try {
        llm.chat({{MessageRole::User, "x"}},
                 [](const std::string&, bool) {});
    } catch (const std::runtime_error& ex) {
        threw = true;
        what = ex.what();
    }
    assert(threw);
    assert(what.find("parse") != std::string::npos ||
           what.find("json")  != std::string::npos);
    printf("  PASS: malformed_ndjson_surfaces_as_runtime_error\n");
}

// ---------------------------------------------------------------------------
// Opt-in integration smoke — pings a real Ollama at localhost:11434.
// Enable with SPEECH_CORE_OLLAMA_INTEGRATION=1 in the environment; set
// SPEECH_CORE_OLLAMA_MODEL to override the model (default "llama3.2:1b").
// ---------------------------------------------------------------------------

void test_real_ollama_integration() {
    const char* enabled = std::getenv("SPEECH_CORE_OLLAMA_INTEGRATION");
    if (!enabled || std::string(enabled) != "1") {
        printf("  SKIP: real_ollama_integration "
               "(set SPEECH_CORE_OLLAMA_INTEGRATION=1 to enable)\n");
        return;
    }
    const char* model_env = std::getenv("SPEECH_CORE_OLLAMA_MODEL");
    OllamaLLM::Options opts;
    opts.base_url = "http://localhost:11434";
    opts.model = (model_env && *model_env) ? model_env : "llama3.2:1b";
    opts.request_timeout = 60s;
    OllamaLLM llm(opts);

    std::string out;
    auto resp = llm.chat(
        {{MessageRole::System, "Respond with only the word: pong"},
         {MessageRole::User,   "ping"}},
        [&](const std::string& t, bool) { out += t; });
    // We don't pin the exact reply (models vary); just confirm we got
    // streaming + a final response.
    assert(!resp.text.empty() && "expected non-empty response from Ollama");
    assert(!out.empty() && "expected at least one streamed token");
    printf("  PASS: real_ollama_integration (model=%s, reply=\"%.*s\")\n",
           opts.model.c_str(),
           static_cast<int>(std::min<size_t>(resp.text.size(), 60)),
           resp.text.c_str());
}

}  // namespace

int main() {
    printf("test_ollama_llm:\n");
    test_construct_rejects_empty_model();
    test_construct_rejects_bad_scheme();
    test_streaming_three_chunks();
    test_tool_call_response();
    test_cancel_from_other_thread();
    test_http_error_surfaces_as_runtime_error();
    test_transport_failure_surfaces_as_runtime_error();
    test_set_tools_sends_tools_array();
    test_options_forwarded_to_ollama();
    test_malformed_ndjson_surfaces_as_runtime_error();
    test_real_ollama_integration();
    printf("All OllamaLLM tests passed.\n");
    return 0;
}
