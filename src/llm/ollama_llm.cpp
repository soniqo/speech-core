#include "speech_core/llm/ollama_llm.h"
#include "speech_core/tools/tool_types.h"

// cpp-httplib pulls in the OS sockets layer. We do NOT define
// CPPHTTPLIB_OPENSSL_SUPPORT — Ollama is a local service and we don't
// link OpenSSL. Plaintext HTTP only.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace speech_core {

namespace {

struct ParsedUrl {
    std::string scheme;
    std::string host;
    int         port;
};

// Minimal http://host[:port] parser. We don't need full URL semantics —
// Ollama runs locally with a simple base URL. Rejects anything we don't
// understand at construction time so misconfig surfaces immediately
// rather than as a stream error mid-turn.
ParsedUrl split_host_port(const std::string& base_url) {
    constexpr const char* kPrefix = "http://";
    constexpr size_t kPrefixLen = 7;
    if (base_url.size() < kPrefixLen ||
        base_url.compare(0, kPrefixLen, kPrefix) != 0) {
        throw std::invalid_argument(
            "OllamaLLM: base_url must start with http:// (got '" +
            base_url + "')");
    }
    std::string rest = base_url.substr(kPrefixLen);
    // Strip trailing slash if present (consistent with documented format).
    while (!rest.empty() && rest.back() == '/') rest.pop_back();
    if (rest.empty()) {
        throw std::invalid_argument(
            "OllamaLLM: base_url is empty after scheme");
    }
    auto colon = rest.find(':');
    ParsedUrl out;
    out.scheme = "http";
    if (colon == std::string::npos) {
        out.host = rest;
        out.port = 80;
    } else {
        out.host = rest.substr(0, colon);
        try {
            out.port = std::stoi(rest.substr(colon + 1));
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "OllamaLLM: invalid port in base_url '" + base_url + "'");
        }
        if (out.port <= 0 || out.port > 65535) {
            throw std::invalid_argument(
                "OllamaLLM: port out of range in base_url '" + base_url + "'");
        }
    }
    if (out.host.empty()) {
        throw std::invalid_argument(
            "OllamaLLM: empty host in base_url '" + base_url + "'");
    }
    return out;
}

const char* role_to_string(MessageRole role) {
    switch (role) {
    case MessageRole::System:    return "system";
    case MessageRole::User:      return "user";
    case MessageRole::Assistant: return "assistant";
    case MessageRole::Tool:      return "tool";
    }
    return "user";  // unreachable; default to user
}

// Convert a ToolDefinition into the OpenAI-style function tool shape that
// Ollama accepts on /api/chat. ToolDefinition does not yet carry a JSON
// schema for arguments, so we emit a permissive open-object schema. The
// model can still call the tool with any arguments; we round-trip the
// arguments as a string per the LLMResponse contract.
nlohmann::json tool_def_to_json(const ToolDefinition& def) {
    nlohmann::json fn;
    fn["name"] = def.name;
    fn["description"] = def.description;
    fn["parameters"] = {
        {"type", "object"},
        {"properties", nlohmann::json::object()},
        {"required", nlohmann::json::array()},
    };
    return nlohmann::json{{"type", "function"}, {"function", std::move(fn)}};
}

nlohmann::json build_request_body(
    const std::string& model,
    const std::vector<Message>& messages,
    const std::vector<ToolDefinition>& tools,
    const OllamaLLM::Options& opts)
{
    nlohmann::json body;
    body["model"]  = model;
    body["stream"] = true;

    nlohmann::json msgs = nlohmann::json::array();
    for (const auto& m : messages) {
        msgs.push_back({{"role", role_to_string(m.role)},
                        {"content", m.content}});
    }
    body["messages"] = std::move(msgs);

    if (!tools.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& t : tools) arr.push_back(tool_def_to_json(t));
        body["tools"] = std::move(arr);
    }

    if (!opts.keep_alive.empty()) body["keep_alive"] = opts.keep_alive;

    nlohmann::json options_obj = nlohmann::json::object();
    if (opts.temperature >= 0.0) options_obj["temperature"] = opts.temperature;
    if (opts.num_ctx > 0)        options_obj["num_ctx"]     = opts.num_ctx;
    if (opts.num_predict > 0)    options_obj["num_predict"] = opts.num_predict;
    if (!options_obj.empty())    body["options"]            = std::move(options_obj);

    return body;
}

}  // namespace

struct OllamaLLM::Impl {
    Options opts;
    std::vector<ToolDefinition> tools;

    std::mutex client_mu;                    // guards active_client
    std::atomic<bool> cancelled{false};
    httplib::Client* active_client = nullptr;  // raw, owned by chat()'s stack

    // Per-stream parser state (reset at the top of chat()).
    std::string      line_buffer;
    LLMResponse      response;
    LLMTokenCallback on_token;
    bool             stream_failed = false;
    std::string      stream_error;

    void reset_stream_state(LLMTokenCallback cb) {
        line_buffer.clear();
        response.text.clear();
        response.tool_calls.clear();
        on_token = std::move(cb);
        stream_failed = false;
        stream_error.clear();
    }

    void feed_bytes(const char* data, size_t len) {
        line_buffer.append(data, len);
        size_t pos = 0;
        while (true) {
            size_t nl = line_buffer.find('\n', pos);
            if (nl == std::string::npos) break;
            std::string line = line_buffer.substr(pos, nl - pos);
            pos = nl + 1;
            // Strip a single trailing CR for CRLF servers.
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) handle_line(line);
            if (stream_failed) break;
        }
        line_buffer.erase(0, pos);
    }

    void handle_line(const std::string& line) {
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (const std::exception& ex) {
            stream_failed = true;
            stream_error = std::string("bad json: ") + ex.what();
            return;
        }
        if (j.contains("error")) {
            stream_failed = true;
            if (j["error"].is_string()) {
                stream_error = j["error"].get<std::string>();
            } else {
                stream_error = j["error"].dump();
            }
            return;
        }
        if (!j.contains("message")) return;
        const auto& msg = j["message"];

        if (msg.contains("content") && msg["content"].is_string()) {
            const std::string& delta = msg["content"].get_ref<const std::string&>();
            if (!delta.empty()) {
                response.text += delta;
                if (on_token) on_token(delta, false);
            }
        }
        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            for (const auto& tc : msg["tool_calls"]) {
                if (!tc.contains("function")) continue;
                const auto& fn = tc["function"];
                ToolCall call;
                if (fn.contains("name") && fn["name"].is_string()) {
                    call.name = fn["name"].get<std::string>();
                }
                if (fn.contains("arguments")) {
                    // Ollama returns arguments as a JSON object. Our
                    // ToolCall.arguments is a string per LLMInterface, so
                    // re-serialise. If a server happens to send a string
                    // (older versions), preserve it as-is.
                    if (fn["arguments"].is_string()) {
                        call.arguments = fn["arguments"].get<std::string>();
                    } else {
                        call.arguments = fn["arguments"].dump();
                    }
                }
                response.tool_calls.push_back(std::move(call));
            }
        }
    }
};

OllamaLLM::OllamaLLM(Options options)
    : impl_(std::make_unique<Impl>())
{
    impl_->opts = std::move(options);
    if (impl_->opts.model.empty()) {
        throw std::invalid_argument("OllamaLLM: Options.model is required");
    }
    // Validate base_url early so misconfig surfaces at construction.
    (void)split_host_port(impl_->opts.base_url);
}

OllamaLLM::~OllamaLLM() = default;

void OllamaLLM::set_tools(const std::vector<ToolDefinition>& tools) {
    impl_->tools = tools;
}

void OllamaLLM::cancel() {
    impl_->cancelled.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lk(impl_->client_mu);
    if (impl_->active_client) impl_->active_client->stop();
}

LLMResponse OllamaLLM::chat(
    const std::vector<Message>& messages,
    LLMTokenCallback on_token)
{
    impl_->reset_stream_state(std::move(on_token));
    impl_->cancelled.store(false, std::memory_order_release);

    auto parsed = split_host_port(impl_->opts.base_url);
    httplib::Client cli(parsed.host, parsed.port);
    cli.set_connection_timeout(impl_->opts.connect_timeout);
    cli.set_read_timeout(impl_->opts.request_timeout);
    cli.set_write_timeout(impl_->opts.request_timeout);
    cli.set_keep_alive(true);

    httplib::Headers headers = {{"Content-Type", "application/json"}};
    for (const auto& [k, v] : impl_->opts.extra_headers) {
        headers.emplace(k, v);
    }

    nlohmann::json body = build_request_body(
        impl_->opts.model, messages, impl_->tools, impl_->opts);
    std::string body_str = body.dump();

    // Register the client so cancel() can interrupt it. If a cancel
    // already arrived between reset and now, bail without making a
    // request.
    {
        std::lock_guard<std::mutex> lk(impl_->client_mu);
        if (impl_->cancelled.load(std::memory_order_acquire)) {
            if (impl_->on_token) impl_->on_token("", true);
            return impl_->response;
        }
        impl_->active_client = &cli;
    }

    Impl* impl = impl_.get();
    auto res = cli.Post(
        "/api/chat", headers, body_str, "application/json",
        [impl](const char* data, size_t len) -> bool {
            if (impl->cancelled.load(std::memory_order_acquire)) return false;
            impl->feed_bytes(data, len);
            return !impl->stream_failed;
        });

    {
        std::lock_guard<std::mutex> lk(impl_->client_mu);
        impl_->active_client = nullptr;
    }

    // Cancellation path — return the partial response we accumulated and
    // signal the caller via on_token(is_final=true). The pipeline's
    // interruption logic expects to see the terminal callback.
    if (impl_->cancelled.load(std::memory_order_acquire)) {
        if (impl_->on_token) impl_->on_token("", true);
        return impl_->response;
    }

    // Check our own stream-parse error BEFORE the httplib transport error.
    // When handle_line sets stream_failed=true the receiver returns false,
    // which makes httplib mark the Result as Canceled. We must surface the
    // upstream parse error, not the downstream transport effect.
    if (impl_->stream_failed) {
        throw std::runtime_error(
            "OllamaLLM: stream parse error: " + impl_->stream_error);
    }
    if (!res) {
        throw std::runtime_error(
            "OllamaLLM: HTTP transport failed: " +
            httplib::to_string(res.error()));
    }
    if (res->status != 200) {
        throw std::runtime_error(
            "OllamaLLM: HTTP " + std::to_string(res->status) +
            ": " + res->body);
    }

    // Flush any trailing line without newline (some servers don't emit a
    // final '\n' before closing).
    if (!impl_->line_buffer.empty()) {
        impl_->handle_line(impl_->line_buffer);
        impl_->line_buffer.clear();
        if (impl_->stream_failed) {
            throw std::runtime_error(
                "OllamaLLM: stream parse error (trailing): " +
                impl_->stream_error);
        }
    }

    if (impl_->on_token) impl_->on_token("", true);
    return impl_->response;
}

}  // namespace speech_core
