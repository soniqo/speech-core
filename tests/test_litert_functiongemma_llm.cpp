// Smoke test for LiteRTFunctionGemmaLLM.
//
// The test loads the .litertlm bundle whose path is passed via the
// SPEECH_FUNCTIONGEMMA_LITERTLM_PATH environment variable, then runs a single
// tool-call prompt and asserts the response parses into one FunctionCall.
//
// If the env var is unset (or the file doesn't exist) the test exits 0 —
// CI keeps the smoke green by default, and on a machine staged with the
// bundle the runtime path is exercised end-to-end.

#include "speech_core/models/litert_functiongemma_llm.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

int print_skip(const std::string& msg) {
    std::printf("[skip] %s\n", msg.c_str());
    return 0;
}

bool file_exists(const std::string& path) {
    if (auto* f = std::fopen(path.c_str(), "rb")) { std::fclose(f); return true; }
    return false;
}

}  // namespace

int main() {
    const char* model_env = std::getenv("SPEECH_FUNCTIONGEMMA_LITERTLM_PATH");
    if (!model_env || !*model_env) {
        return print_skip(
            "SPEECH_FUNCTIONGEMMA_LITERTLM_PATH not set; runtime check skipped. "
            "Stage soniqo/FunctionGemma-270M-LiteRT-LM/model.litertlm and re-run.");
    }
    std::string model_path = model_env;
    if (!file_exists(model_path)) {
        return print_skip("model file not present at " + model_path);
    }

    using namespace speech_core;
    LiteRTFunctionGemmaLLM::Config cfg;
    // Match the Python smoke exactly: max_num_tokens=256 on the engine,
    // no explicit sampler, no constrained decoding.
    cfg.max_num_tokens = 256;
    cfg.system_message =
        "You are a helpful assistant that calls tools using the function-call grammar.";
    // Optional backend override: SPEECH_FUNCTIONGEMMA_BACKEND={cpu,gpu,npu}.
    // Useful on Apple-silicon Android emulators where XNNPack's CPU detection
    // picks Apple-specific kernels that don't survive HVF passthrough — flip
    // to gpu and the GPU accelerator runs the graph instead.
    if (const char* be = std::getenv("SPEECH_FUNCTIONGEMMA_BACKEND")) {
        if (*be) {
            cfg.backend = be;
            std::printf("[backend] %s (from env)\n", be);
        }
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

    auto response = llm.chat(messages, /*on_token=*/{});

    std::printf("[response.text] %s\n", response.text.c_str());
    std::printf("[response.tool_calls] %zu\n", response.tool_calls.size());

    if (response.tool_calls.empty()) {
        std::fprintf(stderr,
            "expected at least one tool call but got an empty list\n");
        return 1;
    }
    const auto& tc = response.tool_calls.front();
    std::printf("[tool_call] name=%s args=%s\n", tc.name.c_str(), tc.arguments.c_str());
    if (tc.name != "get_current_weather") {
        std::fprintf(stderr,
            "expected tool call name=get_current_weather, got name=%s\n",
            tc.name.c_str());
        return 1;
    }
    return 0;
}
