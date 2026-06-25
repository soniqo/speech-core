#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/tools/tool_types.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace speech_core {

/// FunctionGemma 270M tool-calling LLM via Google's liblitert-lm runtime.
///
/// Loads a `.litertlm` bundle (e.g.
/// https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) and exposes
/// it through the abstract `LLMInterface` so it plugs straight into
/// `VoicePipeline` next to `OllamaLLM` and any future on-device LLMs.
///
/// Construction is the only step that touches the filesystem; once built,
/// the engine handle stays alive for the lifetime of the object and is
/// reused for every `chat()` call. Tools registered through `set_tools()`
/// are embedded into the prompt as a `<start_function_declarations>...
/// <end_function_declarations>` block plus a JSON list passed to the runtime's
/// constraint provider when `enable_constrained_decoding` is requested. The
/// model emits `<start_function_call>...<end_function_call>` blocks; those are
/// parsed into `LLMResponse.tool_calls` after generation returns.
///
/// Thread safety:
///   - `chat()` is not safe to call concurrently on the same instance.
///   - `cancel()` is safe to call from another thread while `chat()` is
///     running — it flips an atomic flag and forwards to
///     `litert_lm_conversation_cancel_process()`.
///   - `set_tools()` is not synchronised; call it before `chat()` or between
///     turns.
class LiteRTFunctionGemmaLLM : public LLMInterface {
public:
    struct Config {
        /// Backend name string passed to the runtime. CPU is the safest
        /// default; the runtime also accepts "gpu" and "npu" but those
        /// require platform-specific dispatch libraries.
        std::string backend = "cpu";

        /// Optional dispatch lib dir for NPU backends. Empty = unused.
        std::string litert_dispatch_lib_dir;

        /// Cache dir for any disk-backed runtime state. Empty = runtime
        /// default. The bundled XNNPACK cache is unrelated and lives next
        /// to the .litertlm file.
        std::string cache_dir;

        /// Maximum number of tokens (prompt + completion). 0 = runtime
        /// default (taken from the .litertlm metadata).
        int max_num_tokens = 0;

        /// Cap on tokens generated per turn. Ignored by ``chat()`` — the
        /// upstream `liblitert-lm` runtime crashes when this is pushed onto
        /// the session_config attached to a Conversation. The runtime keeps
        /// its own default cap from the .litertlm metadata.
        int max_output_tokens = 0;

        /// Sampler: by default we never call ``set_sampler_params`` and let
        /// the C runtime use its built-in sampler. Set ``use_explicit_sampler
        /// = true`` to force TOP_P with the parameters below (mirrors the
        /// Python wrapper's behaviour). Greedy is **not** supported through
        /// this path — sending ``LiteRtLmSamplerType::Greedy`` from the
        /// conversation path crashes upstream.
        bool  use_explicit_sampler = false;
        int   top_k = 40;
        float top_p = 0.95f;
        float temperature = 1.0f;

        /// Apply the bundled chat template. Ignored by ``chat()`` — same
        /// reason as `max_output_tokens` above. The C runtime applies its
        /// own chat template from the bundle.
        bool apply_prompt_template = true;

        /// Constrain decoding to the FunctionGemma tool-call grammar.
        /// Strongly recommended once `set_tools()` has registered tools.
        bool enable_constrained_decoding = true;

        /// Optional system message inserted before the first user turn.
        std::string system_message;
    };

    /// Load a `.litertlm` bundle with default config (CPU backend, runtime
    /// default sampler + chat template). Throws `std::runtime_error` on
    /// failure.
    explicit LiteRTFunctionGemmaLLM(const std::string& model_path);

    /// Load a `.litertlm` bundle with a custom config.
    LiteRTFunctionGemmaLLM(const std::string& model_path, Config config);

    ~LiteRTFunctionGemmaLLM() override;

    LiteRTFunctionGemmaLLM(const LiteRTFunctionGemmaLLM&) = delete;
    LiteRTFunctionGemmaLLM& operator=(const LiteRTFunctionGemmaLLM&) = delete;

    // ----- LLMInterface --------------------------------------------------

    LLMResponse chat(const std::vector<Message>& messages,
                     LLMTokenCallback on_token) override;

    void set_tools(const std::vector<ToolDefinition>& tools) override;

    void cancel() override;

    // Implementation detail. Exposed publicly so the streaming trampoline
    // can `static_cast` from `void*` back to this type — Core ML / LiteRT C
    // APIs all take `void* user_data` callbacks.
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
