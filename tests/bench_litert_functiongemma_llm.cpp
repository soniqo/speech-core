// Bench harness for LiteRTFunctionGemmaLLM.
//
// Loads the .litertlm bundle from SPEECH_FUNCTIONGEMMA_LITERTLM_PATH, runs a
// warmup pass + N timed runs of a tool-call prompt, and prints per-run
// wall-clock + an approximate tok/s (using a 4 chars/token proxy on the
// returned assistant text, which matches what production decoders see).
//
// Defaults:
//   SPEECH_FUNCTIONGEMMA_BENCH_WARMUP=1
//   SPEECH_FUNCTIONGEMMA_BENCH_RUNS=3
//   SPEECH_FUNCTIONGEMMA_MAX_NUM_TOKENS=256

#include "speech_core/models/litert_functiongemma_llm.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

bool file_exists(const std::string& path) {
    if (auto* f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return true; }
    return false;
}

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    try { return std::stoi(v); } catch (...) { return dflt; }
}

// Approximate tokens by counting whitespace-separated words + non-alphanum
// punctuation chunks. Underestimates BPE tokens (Gemma SentencePiece ~1.3x
// words) but is closer to reality than a byte-divided constant.
size_t approx_tokens(const std::string& s) {
    size_t toks = 0;
    bool in_word = false;
    for (char c : s) {
        bool is_word = (std::isalnum(static_cast<unsigned char>(c)) != 0);
        if (is_word) {
            if (!in_word) ++toks;
            in_word = true;
        } else {
            in_word = false;
            if (c == '<' || c == '>' || c == '{' || c == '}' ||
                c == '(' || c == ')' || c == ',' || c == ':' ||
                c == '_' || c == '"' || c == '\'') {
                ++toks;
            }
        }
    }
    return toks;
}

}  // namespace

int main() {
    const char* model_env = std::getenv("SPEECH_FUNCTIONGEMMA_LITERTLM_PATH");
    if (!model_env || !*model_env) {
        std::fprintf(stderr, "[bench] SPEECH_FUNCTIONGEMMA_LITERTLM_PATH not set\n");
        return 2;
    }
    std::string model_path = model_env;
    if (!file_exists(model_path)) {
        std::fprintf(stderr, "[bench] model file not present at %s\n", model_path.c_str());
        return 2;
    }

    const int warmup_runs = env_int("SPEECH_FUNCTIONGEMMA_BENCH_WARMUP", 1);
    const int timed_runs  = env_int("SPEECH_FUNCTIONGEMMA_BENCH_RUNS", 3);
    const int max_tokens  = env_int("SPEECH_FUNCTIONGEMMA_MAX_NUM_TOKENS", 256);

    using namespace speech_core;
    LiteRTFunctionGemmaLLM::Config cfg;
    cfg.max_num_tokens = max_tokens;
    cfg.system_message =
        "You are a helpful assistant that calls tools using the function-call grammar.";
    if (const char* be = std::getenv("SPEECH_FUNCTIONGEMMA_BACKEND")) {
        if (*be) {
            cfg.backend = be;
            std::printf("[backend] %s (from env)\n", be);
        } else {
            std::printf("[backend] %s (default)\n", cfg.backend.c_str());
        }
    } else {
        std::printf("[backend] %s (default)\n", cfg.backend.c_str());
    }

    LiteRTFunctionGemmaLLM llm(model_path, cfg);

    std::vector<ToolDefinition> tools;
    {
        ToolDefinition t;
        t.name        = "get_current_weather";
        t.description = "Get the current weather in a given location.";
        tools.push_back(std::move(t));
    }
    llm.set_tools(tools);

    std::vector<Message> messages;
    {
        Message m;
        m.role    = MessageRole::User;
        m.content = "What is the current weather in Tokyo?";
        messages.push_back(std::move(m));
    }

    std::printf("[bench] warmup=%d runs=%d max_num_tokens=%d\n",
                warmup_runs, timed_runs, max_tokens);

    for (int i = 0; i < warmup_runs; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto resp = llm.chat(messages, /*on_token=*/{});
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        std::printf("[warmup %d] wall=%.3fs chars=%zu approx_tokens=%zu\n",
                    i, s, resp.text.size(), approx_tokens(resp.text));
    }

    double sum_s = 0.0;
    size_t sum_tokens = 0;
    size_t last_chars = 0;
    std::string last_text;
    for (int i = 0; i < timed_runs; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto resp = llm.chat(messages, /*on_token=*/{});
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        size_t toks = approx_tokens(resp.text);
        sum_s += s;
        sum_tokens += toks;
        last_chars = resp.text.size();
        last_text = resp.text;
        double tps = (s > 0.0) ? (double)toks / s : 0.0;
        std::printf("[run %d] wall=%.3fs chars=%zu approx_tokens=%zu tok/s=%.2f\n",
                    i, s, resp.text.size(), toks, tps);
    }

    double mean_s = sum_s / std::max(1, timed_runs);
    double mean_tps = (sum_s > 0.0) ? (double)sum_tokens / sum_s : 0.0;
    std::printf("[summary] mean_wall=%.3fs total_tokens=%zu total_wall=%.3fs mean_tok_per_s=%.2f\n",
                mean_s, sum_tokens, sum_s, mean_tps);
    std::printf("[last_text] %s\n", last_text.c_str());
    (void)last_chars;
    return 0;
}
