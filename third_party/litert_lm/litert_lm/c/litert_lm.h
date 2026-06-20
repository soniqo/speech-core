// Vendored C declarations for liblitert-lm extracted from Google's
// litert-lm-api wheel. We only declare the symbols speech-core actually calls
// so this stays small and reviewable; if a future model needs more of the C
// API, copy the matching signatures from `litert_lm/_ffi.py` in the wheel.
//
// The shared library itself is fetched at build time by
// `scripts/fetch_litert_lm.sh` and exposed to CMake via `LITERT_LM_DIR`.

#ifndef LITERT_LM_C_LITERT_LM_H_
#define LITERT_LM_C_LITERT_LM_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

typedef enum {
    kLiteRtLmLogVerbose = 0,
    kLiteRtLmLogInfo = 1,
    kLiteRtLmLogWarning = 2,
    kLiteRtLmLogError = 3,
} LiteRtLmLogSeverity;

void litert_lm_set_min_log_level(int level);

// ---------------------------------------------------------------------------
// Engine settings + engine
// ---------------------------------------------------------------------------

// All `void*` handles are opaque. Created handles must be freed with the
// matching `*_delete` function.
typedef void* LiteRtLmEngineSettings;
typedef void* LiteRtLmEngine;

// Backend names (passed as strings to engine_settings_create):
//   "cpu", "gpu", "npu"
LiteRtLmEngineSettings litert_lm_engine_settings_create(
    const char* model_path,
    const char* backend_name,
    const char* vision_backend_name,
    const char* audio_backend_name);
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings settings);

void litert_lm_engine_settings_set_max_num_tokens(
    LiteRtLmEngineSettings settings, int max_num_tokens);
void litert_lm_engine_settings_set_cache_dir(
    LiteRtLmEngineSettings settings, const char* cache_dir);
void litert_lm_engine_settings_set_litert_dispatch_lib_dir(
    LiteRtLmEngineSettings settings, const char* dispatch_lib_dir);
void litert_lm_engine_settings_set_enable_speculative_decoding(
    LiteRtLmEngineSettings settings, bool enable);
void litert_lm_engine_settings_set_num_prefill_tokens(
    LiteRtLmEngineSettings settings, int n);
void litert_lm_engine_settings_set_num_decode_tokens(
    LiteRtLmEngineSettings settings, int n);

LiteRtLmEngine litert_lm_engine_create(LiteRtLmEngineSettings settings);
void litert_lm_engine_delete(LiteRtLmEngine engine);

// ---------------------------------------------------------------------------
// Session config (referenced by Conversation; not used standalone here)
// ---------------------------------------------------------------------------

typedef void* LiteRtLmSessionConfig;

typedef enum {
    kLiteRtLmSamplerUnspecified = 0,
    kLiteRtLmSamplerTopK = 1,
    kLiteRtLmSamplerTopP = 2,
    kLiteRtLmSamplerGreedy = 3,
} LiteRtLmSamplerType;

typedef struct {
    int   type;        // LiteRtLmSamplerType
    int   top_k;
    float top_p;
    float temperature;
    int   seed;
} LiteRtLmSamplerParams;

LiteRtLmSessionConfig litert_lm_session_config_create(void);
void litert_lm_session_config_delete(LiteRtLmSessionConfig config);
void litert_lm_session_config_set_max_output_tokens(
    LiteRtLmSessionConfig config, int max);
void litert_lm_session_config_set_apply_prompt_template(
    LiteRtLmSessionConfig config, bool apply);
void litert_lm_session_config_set_sampler_params(
    LiteRtLmSessionConfig config, const LiteRtLmSamplerParams* params);

// ---------------------------------------------------------------------------
// Conversation config + conversation (high-level path we use for tool calls)
// ---------------------------------------------------------------------------

typedef void* LiteRtLmConversationConfig;
typedef void* LiteRtLmConversation;
typedef void* LiteRtLmResponses;     // multi-candidate result, used by run_text_scoring etc.
typedef void* LiteRtLmJsonResponse;  // single JSON-encoded reply, returned by conversation_send_message

LiteRtLmConversationConfig litert_lm_conversation_config_create(void);
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig config);

void litert_lm_conversation_config_set_session_config(
    LiteRtLmConversationConfig config, LiteRtLmSessionConfig session);
void litert_lm_conversation_config_set_system_message(
    LiteRtLmConversationConfig config, const char* system_message);
void litert_lm_conversation_config_set_tools(
    LiteRtLmConversationConfig config, const char* tools_json);
void litert_lm_conversation_config_set_messages(
    LiteRtLmConversationConfig config, const char* messages_json);
void litert_lm_conversation_config_set_enable_constrained_decoding(
    LiteRtLmConversationConfig config, bool enable);
void litert_lm_conversation_config_set_filter_channel_content_from_kv_cache(
    LiteRtLmConversationConfig config, bool filter);

LiteRtLmConversation litert_lm_conversation_create(
    LiteRtLmEngine engine, LiteRtLmConversationConfig config);
void litert_lm_conversation_delete(LiteRtLmConversation conv);
void litert_lm_conversation_cancel_process(LiteRtLmConversation conv);

// Stream callback signature (matches engine.h LiteRtLmStreamCallback):
//   void cb(void* user_data, const char* token, bool is_final, const char* role)
typedef void (*LiteRtLmStreamCallback)(void* user_data,
                                       const char* token,
                                       bool is_final,
                                       const char* role);

// Streaming send_message — returns 0 on success, non-zero on error.
//
// `msg_json` and `ctx_json` are JSON-encoded strings. The simplest valid
// message is `{"role":"user","content":"<text>"}`; pass `{}` for ctx when
// no extra context is needed.
int litert_lm_conversation_send_message_stream(
    LiteRtLmConversation conv,
    const char* msg_json,
    const char* ctx_json,
    void* extra,                                  // reserved; pass NULL
    LiteRtLmStreamCallback callback,
    void* user_data);

// Non-streaming send_message — returns a JsonResponse handle owned by the caller.
// The handle wraps a JSON-encoded reply that you read with
// litert_lm_json_response_get_string() and free with litert_lm_json_response_delete().
LiteRtLmJsonResponse litert_lm_conversation_send_message(
    LiteRtLmConversation conv,
    const char* msg_json,
    const char* ctx_json,
    void* extra);                                 // reserved; pass NULL

// Read the JSON-encoded text from a JsonResponse handle. Returned pointer is
// owned by the handle and remains valid until litert_lm_json_response_delete().
const char* litert_lm_json_response_get_string(LiteRtLmJsonResponse response);
void litert_lm_json_response_delete(LiteRtLmJsonResponse response);

// ---------------------------------------------------------------------------
// Responses — read accumulated text after a non-streaming send_message
// ---------------------------------------------------------------------------

void litert_lm_responses_delete(LiteRtLmResponses responses);
int  litert_lm_responses_get_num_candidates(LiteRtLmResponses responses);
const char* litert_lm_responses_get_response_text_at(
    LiteRtLmResponses responses, int idx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LITERT_LM_C_LITERT_LM_H_
