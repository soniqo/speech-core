#include "speech_core/models/litert_functiongemma_llm.h"

#include "litert_lm/c/litert_lm.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace speech_core {

namespace {

constexpr const char* kFunctionCallStart        = "<start_function_call>";
constexpr const char* kFunctionCallEnd          = "<end_function_call>";
constexpr const char* kEscape                   = "<escape>";

constexpr const char* kRoleUser      = "user";
constexpr const char* kRoleAssistant = "assistant";
constexpr const char* kRoleSystem    = "system";

// Serialise tools into the JSON shape liblitert-lm expects on
// conversation_config_set_tools(...). We mirror the OpenAI / Ollama
// function-calling envelope ("type":"function","function":{...}) and leave
// `properties` empty when the speech_core ToolDefinition does not carry a
// parameter schema. The runtime treats this as "model decides argument
// names from the prompt" — fine for our voice-assistant use case.
std::string serialize_tools(const std::vector<ToolDefinition>& tools) {
    if (tools.empty()) return "";
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& t : tools) {
        nlohmann::json fn;
        fn["name"]        = t.name;
        fn["description"] = t.description;
        fn["parameters"]  = {
            {"type",       "object"},
            {"properties", nlohmann::json::object()},
            {"required",   nlohmann::json::array()},
        };
        arr.push_back({{"type", "function"}, {"function", std::move(fn)}});
    }
    return arr.dump();
}

const char* message_role_string(MessageRole role) {
    switch (role) {
        case MessageRole::System:    return kRoleSystem;
        case MessageRole::User:      return kRoleUser;
        case MessageRole::Assistant: return kRoleAssistant;
        case MessageRole::Tool:      return "tool";
    }
    return kRoleUser;
}

// ---------------------------------------------------------------------------
// FunctionGemma call-grammar parser
//
// Grammar:
//   call:NAME{KEY:VALUE,...}
//   - strings  : <escape>...<escape>
//   - ints     : bare integer literals
//   - bools    : true / false
//   - null     : null
//   - object   : {KEY:VALUE,...}
//   - array    : [VALUE,...]
//
// We don't strongly type the arguments downstream — the LLMInterface contract
// is `std::vector<ToolCall>` where each call carries a `std::string arguments`.
// The cleanest shape for downstream consumers is the same JSON object the
// model emitted, so we round-trip the parse into nlohmann::json::dump().
// ---------------------------------------------------------------------------

struct CallParser {
    const std::string& src;
    size_t pos = 0;

    explicit CallParser(const std::string& s) : src(s) {}

    bool eof() const { return pos >= src.size(); }
    char peek() const { return src[pos]; }

    void skip_ws() {
        while (!eof() && (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r')) ++pos;
    }

    bool match(const char* literal) {
        size_t n = std::strlen(literal);
        if (pos + n > src.size()) return false;
        if (std::strncmp(src.c_str() + pos, literal, n) != 0) return false;
        pos += n;
        return true;
    }

    nlohmann::json parse_value() {
        skip_ws();
        if (eof()) return nullptr;
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '<' && match(kEscape)) {
            std::string s;
            while (!eof() && !match(kEscape)) { s.push_back(src[pos]); ++pos; }
            return s;
        }
        if (c == 't' && match("true"))   return true;
        if (c == 'f' && match("false"))  return false;
        if (c == 'n' && match("null"))   return nullptr;
        return parse_number();
    }

    nlohmann::json parse_object() {
        nlohmann::json obj = nlohmann::json::object();
        if (eof() || peek() != '{') return obj;
        ++pos;  // {
        skip_ws();
        if (!eof() && peek() == '}') { ++pos; return obj; }
        while (!eof()) {
            skip_ws();
            std::string key;
            while (!eof()) {
                char c = peek();
                if (c == ':' || c == ',' || c == '}') break;
                key.push_back(c); ++pos;
            }
            // Trim trailing space from key
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
            skip_ws();
            if (!eof() && peek() == ':') ++pos;
            auto v = parse_value();
            obj[key] = std::move(v);
            skip_ws();
            if (!eof() && peek() == ',') { ++pos; continue; }
            if (!eof() && peek() == '}') { ++pos; break; }
            break;
        }
        return obj;
    }

    nlohmann::json parse_array() {
        nlohmann::json arr = nlohmann::json::array();
        if (eof() || peek() != '[') return arr;
        ++pos;
        skip_ws();
        if (!eof() && peek() == ']') { ++pos; return arr; }
        while (!eof()) {
            arr.push_back(parse_value());
            skip_ws();
            if (!eof() && peek() == ',') { ++pos; continue; }
            if (!eof() && peek() == ']') { ++pos; break; }
            break;
        }
        return arr;
    }

    nlohmann::json parse_number() {
        std::string raw;
        bool sawDot = false;
        while (!eof()) {
            char c = peek();
            bool isDigit = (c >= '0' && c <= '9');
            if (isDigit || c == '-' || c == '+') { raw.push_back(c); ++pos; }
            else if (c == '.')                    { sawDot = true; raw.push_back(c); ++pos; }
            else if (c == 'e' || c == 'E')        { sawDot = true; raw.push_back(c); ++pos; }
            else break;
        }
        if (raw.empty()) return nullptr;
        try {
            if (sawDot) return std::stod(raw);
            return static_cast<long long>(std::stoll(raw));
        } catch (...) {
            return raw;  // fall back to string if neither parse succeeds
        }
    }
};

std::vector<ToolCall> parse_function_calls(const std::string& text) {
    std::vector<ToolCall> calls;
    const std::string start = kFunctionCallStart;
    const std::string end   = kFunctionCallEnd;
    size_t cursor = 0;
    while (true) {
        size_t s = text.find(start, cursor);
        if (s == std::string::npos) break;
        size_t e = text.find(end, s + start.size());
        if (e == std::string::npos) break;
        std::string body = text.substr(s + start.size(), e - (s + start.size()));
        cursor = e + end.size();
        // Body shape: "call:NAME{...}"
        const std::string prefix = "call:";
        if (body.rfind(prefix, 0) != 0) continue;
        size_t brace = body.find('{', prefix.size());
        if (brace == std::string::npos) continue;
        std::string name = body.substr(prefix.size(), brace - prefix.size());
        // trim
        while (!name.empty() && (name.back() == ' ' || name.back() == '\t' || name.back() == '\n')) name.pop_back();
        std::string rest = body.substr(brace);
        CallParser parser(rest);
        auto args_json = parser.parse_object();
        ToolCall tc;
        tc.name      = std::move(name);
        tc.arguments = args_json.dump();
        calls.push_back(std::move(tc));
    }
    return calls;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct LiteRTFunctionGemmaLLM::Impl {
    Config            cfg;
    LiteRtLmEngine    engine = nullptr;
    std::string       tools_json;

    // Per-chat() state, accessed from the streaming callback.
    std::string       accumulated;
    LLMTokenCallback  on_token;
    std::atomic<bool> cancelled{false};

    std::mutex                conv_mu;  // guards active_conv during cancel()
    LiteRtLmConversation      active_conv = nullptr;

    Impl(const std::string& model_path, Config cfg_)
        : cfg(std::move(cfg_))
    {
        auto* settings = litert_lm_engine_settings_create(
            model_path.c_str(),
            cfg.backend.empty() ? "cpu" : cfg.backend.c_str(),
            nullptr, nullptr);
        if (!settings) {
            throw std::runtime_error(
                "LiteRTFunctionGemmaLLM: engine_settings_create failed for " + model_path);
        }
        if (cfg.max_num_tokens > 0) {
            litert_lm_engine_settings_set_max_num_tokens(settings, cfg.max_num_tokens);
        }
        if (!cfg.cache_dir.empty()) {
            litert_lm_engine_settings_set_cache_dir(settings, cfg.cache_dir.c_str());
        }
        if (!cfg.litert_dispatch_lib_dir.empty()) {
            litert_lm_engine_settings_set_litert_dispatch_lib_dir(
                settings, cfg.litert_dispatch_lib_dir.c_str());
        }
        engine = litert_lm_engine_create(settings);
        litert_lm_engine_settings_delete(settings);
        if (!engine) {
            throw std::runtime_error(
                "LiteRTFunctionGemmaLLM: engine_create failed (check that the "
                "model path points at a valid .litertlm bundle): " + model_path);
        }
    }

    ~Impl() {
        if (engine) litert_lm_engine_delete(engine);
    }

    LiteRtLmConversation make_conversation() {
        // Mirror the Python wrapper's create_conversation path EXACTLY —
        // any extra setter we push onto the session_config attached here
        // crashes the upstream runtime inside its execution-thread startup
        // (see commit message for the diagnostic trail). Specifically:
        //   - DO NOT call set_max_output_tokens / set_apply_prompt_template
        //     on the session_config attached to a Conversation (those are
        //     only valid via the Engine.create_session path).
        //   - Only call set_sampler_params when the caller opts in via
        //     cfg.use_explicit_sampler, and use TOP_P even when the caller
        //     would prefer greedy — sending Greedy (=3) from the
        //     conversation path segfaults upstream.
        //   - Delete session_config BEFORE conversation_create returns
        //     (Python deletes the wrapper immediately after attaching).
        //   - Only call set_enable_constrained_decoding when it would be
        //     true; setting it to false explicitly walks an untested path.
        auto* session_cfg = litert_lm_session_config_create();
        if (cfg.use_explicit_sampler) {
            LiteRtLmSamplerParams sp{};
            sp.type        = kLiteRtLmSamplerTopP;
            sp.top_k       = cfg.top_k       > 0    ? cfg.top_k       : 40;
            sp.top_p       = cfg.top_p       > 0.0f ? cfg.top_p       : 0.95f;
            sp.temperature = cfg.temperature > 0.0f ? cfg.temperature : 1.0f;
            sp.seed        = 0;
            litert_lm_session_config_set_sampler_params(session_cfg, &sp);
        }

        auto* conv_cfg = litert_lm_conversation_config_create();
        litert_lm_conversation_config_set_session_config(conv_cfg, session_cfg);
        // Matches Python: the wrapper deletes session_cfg immediately after
        // set_session_config. The conversation now owns the underlying state.
        litert_lm_session_config_delete(session_cfg);

        if (!cfg.system_message.empty()) {
            litert_lm_conversation_config_set_system_message(conv_cfg, cfg.system_message.c_str());
        }
        if (!tools_json.empty()) {
            litert_lm_conversation_config_set_tools(conv_cfg, tools_json.c_str());
        }
        if (cfg.enable_constrained_decoding) {
            litert_lm_conversation_config_set_enable_constrained_decoding(conv_cfg, true);
        }
        auto* conv = litert_lm_conversation_create(engine, conv_cfg);
        litert_lm_conversation_config_delete(conv_cfg);
        return conv;
    }
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

LiteRTFunctionGemmaLLM::LiteRTFunctionGemmaLLM(const std::string& model_path)
    : impl_(std::make_unique<Impl>(model_path, Config{}))
{}

LiteRTFunctionGemmaLLM::LiteRTFunctionGemmaLLM(
    const std::string& model_path, Config cfg)
    : impl_(std::make_unique<Impl>(model_path, std::move(cfg)))
{}

LiteRTFunctionGemmaLLM::~LiteRTFunctionGemmaLLM() = default;

void LiteRTFunctionGemmaLLM::set_tools(const std::vector<ToolDefinition>& tools) {
    impl_->tools_json = serialize_tools(tools);
}

void LiteRTFunctionGemmaLLM::cancel() {
    impl_->cancelled.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lk(impl_->conv_mu);
    if (impl_->active_conv) {
        litert_lm_conversation_cancel_process(impl_->active_conv);
    }
}

namespace {

// Kept for the streaming path (currently unused — we drive the synchronous
// send_message instead because the streaming variant is async in this build
// and would need a wait gate to be safe). Left in place so a future streaming
// impl has a working trampoline shape to start from.
[[maybe_unused]] void stream_trampoline(void* user_data,
                                        const char* chunk,
                                        bool is_final,
                                        const char* error_msg)
{
    auto* impl = static_cast<LiteRTFunctionGemmaLLM::Impl*>(user_data);
    if (!impl) return;
    if (impl->cancelled.load(std::memory_order_acquire)) return;
    if (error_msg && *error_msg) {
        if (impl->on_token) impl->on_token("", /*is_final=*/true);
        return;
    }
    std::string t = chunk ? chunk : "";
    impl->accumulated += t;
    if (impl->on_token) impl->on_token(t, is_final);
}

}  // namespace

LLMResponse LiteRTFunctionGemmaLLM::chat(
    const std::vector<Message>& messages,
    LLMTokenCallback on_token)
{
    impl_->accumulated.clear();
    impl_->on_token = std::move(on_token);
    impl_->cancelled.store(false, std::memory_order_release);

    // FunctionGemma is single-turn for tool calls. Concatenate any system /
    // tool messages into the prompt, then send the last user turn through.
    // The conversation config already carries the system_message and tools
    // list, so here we only need to assemble the visible user content.
    std::string content;
    for (const auto& msg : messages) {
        if (msg.role == MessageRole::User) {
            if (!content.empty()) content.push_back('\n');
            content += msg.content;
        }
    }

    LiteRtLmConversation conv = impl_->make_conversation();
    if (!conv) {
        return LLMResponse{};
    }
    {
        std::lock_guard<std::mutex> lk(impl_->conv_mu);
        impl_->active_conv = conv;
    }

    // liblitert-lm wants JSON-encoded message + context strings, not bare
    // role/content. The minimal shape matches the Python wrapper's
    // `normalize_message(str)` output.
    nlohmann::json msg = {{"role", kRoleUser}, {"content", content}};
    std::string msg_json = msg.dump();
    static const char* kEmptyCtxJson = "{}";

    // Use the synchronous send_message. It returns a JsonResponse handle
    // wrapping a JSON-encoded reply (NOT a multi-candidate Responses). The
    // shape mirrors the Python wrapper's Conversation.send_message path.
    int rc = 0;
    LiteRtLmJsonResponse response = litert_lm_conversation_send_message(
        conv,
        msg_json.c_str(),
        kEmptyCtxJson,
        /*extra=*/nullptr);
    if (response) {
        const char* raw = litert_lm_json_response_get_string(response);
        if (raw) {
            // The JSON object looks like:
            //   {"role":"model","content":[{"type":"text","text":"<...>"}], ...}
            // Pull out the assistant text — it carries the
            // <start_function_call>...<end_function_call> grammar.
            try {
                auto j = nlohmann::json::parse(raw);
                std::string assistant_text;
                if (j.contains("content")) {
                    const auto& c = j["content"];
                    if (c.is_string()) {
                        assistant_text = c.get<std::string>();
                    } else if (c.is_array()) {
                        for (const auto& part : c) {
                            if (part.is_object() && part.value("type", "") == "text" &&
                                part.contains("text") && part["text"].is_string()) {
                                assistant_text += part["text"].get<std::string>();
                            }
                        }
                    }
                }
                impl_->accumulated = assistant_text;
                if (impl_->on_token) {
                    impl_->on_token(assistant_text, /*is_final=*/true);
                }
            } catch (const std::exception& e) {
                // Fall back: surface the raw JSON so callers can debug.
                impl_->accumulated = raw;
                if (impl_->on_token) impl_->on_token(raw, /*is_final=*/true);
            }
        }
        litert_lm_json_response_delete(response);
    } else {
        rc = 1;
    }

    {
        std::lock_guard<std::mutex> lk(impl_->conv_mu);
        impl_->active_conv = nullptr;
    }
    litert_lm_conversation_delete(conv);

    LLMResponse out;
    out.text = impl_->accumulated;
    if (rc == 0) {
        out.tool_calls = parse_function_calls(impl_->accumulated);
    }
    return out;
}

}  // namespace speech_core
