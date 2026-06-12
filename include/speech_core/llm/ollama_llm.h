#pragma once

#include "speech_core/interfaces.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Ollama LLM adapter — implements LLMInterface against a local Ollama HTTP
/// server (default http://localhost:11434). Streams /api/chat NDJSON, parses
/// per-line deltas into on_token callbacks, and surfaces tool_calls in the
/// returned LLMResponse.
///
/// Thread safety:
///   - chat() is not safe to call concurrently on the same instance (one
///     in-flight request per OllamaLLM object).
///   - cancel() is safe to call from any thread while chat() is running on
///     another. It flips an atomic flag and forcibly closes the in-flight
///     httplib client so the streaming loop returns promptly.
///   - set_tools() is not synchronised; call it before chat() or between
///     turns.
///
/// Implementation is PIMPL — cpp-httplib and nlohmann/json are vendored
/// under third_party/ and never appear in this header. Consumers link
/// speech_core_llm_ollama and include only this file.
class OllamaLLM : public LLMInterface {
public:
    struct Options {
        /// Base URL of the Ollama server. Must include scheme and port; no
        /// trailing slash. Example: "http://localhost:11434".
        std::string base_url = "http://localhost:11434";

        /// Ollama model tag (e.g. "llama3.2:3b", "qwen2.5:7b-instruct").
        /// Must already be pulled on the server — we do not call /api/pull.
        std::string model;

        /// Total request timeout for a single chat() call. The streaming
        /// connection is held open for this long; cancel() short-circuits it.
        std::chrono::milliseconds request_timeout{120'000};

        /// Connection timeout for the initial TCP handshake to Ollama.
        std::chrono::milliseconds connect_timeout{5'000};

        /// Forwarded as the "keep_alive" field on /api/chat. Empty = use
        /// the server default ("5m"). Accepts Ollama's duration strings
        /// ("10m", "0" to unload immediately, "-1" to keep loaded forever).
        std::string keep_alive;

        /// Sampling temperature. Negative = leave the field out of the
        /// request and let Ollama use the model's default.
        double temperature = -1.0;

        /// Context window size in tokens. 0 = leave out of the request.
        int num_ctx = 0;

        /// Hard cap on output tokens for a single response. 0 = leave out
        /// of the request (Ollama uses its own default).
        int num_predict = 0;

        /// Extra HTTP headers to attach to every request (auth proxies,
        /// tracing). Keys are sent verbatim.
        std::unordered_map<std::string, std::string> extra_headers;
    };

    explicit OllamaLLM(Options options);
    ~OllamaLLM() override;

    OllamaLLM(const OllamaLLM&)            = delete;
    OllamaLLM& operator=(const OllamaLLM&) = delete;

    /// Run one /api/chat round trip with streaming NDJSON. Fires on_token
    /// once per non-empty content delta with is_final=false, then once
    /// with an empty token and is_final=true when the server emits
    /// done:true OR when cancel() short-circuits the stream.
    /// Throws std::runtime_error on transport errors, HTTP != 200, or
    /// malformed JSON.
    LLMResponse chat(
        const std::vector<Message>& messages,
        LLMTokenCallback on_token) override;

    /// Register tools for subsequent chat() calls. Stored by value; sent
    /// on every chat() request via the "tools" array. ToolDefinition
    /// carries no JSON schema yet — we send a permissive stub
    /// {type:"object", properties:{}, required:[]} for each tool's
    /// parameters; typed schemas are a follow-up.
    void set_tools(const std::vector<ToolDefinition>& tools) override;

    /// Thread-safe: flips an atomic cancel flag and closes the in-flight
    /// HTTP connection so the streaming loop in chat() returns promptly.
    /// chat() resets the flag at entry, so cancel() before chat() is a
    /// no-op for the next request (the prior cancel is consumed).
    void cancel() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
