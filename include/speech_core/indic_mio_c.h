#ifndef SPEECH_CORE_INDIC_MIO_C_H
#define SPEECH_CORE_INDIC_MIO_C_H

// C ABI for Indic-Mio Hindi/Indic emotion TTS (LiteRT backend).
//
// Like voxcpm2_c.h, this is a standalone one-shot synthesis surface for
// non-C++ hosts (the speech-studio Tauri app via Rust FFI) on Linux/Windows.
// The model is an autoregressive Qwen3-0.6B speech-token LM plus the MioCodec
// wave decoder; the graph bundle is `soniqo/Indic-Mio-LiteRT` and its model
// card documents the full host contract this runtime implements.
//
// Emotion is controlled inline in the text with end-of-utterance suffix tags
// (`<happy> <sad> <angry> <disgust> <fear> <surprise>`); there is no separate
// style API. Voice cloning conditions on a reference clip via
// sc_indic_mio_set_reference(); without one, output uses the model's default
// (uncloned) voice.
//
// Symbols live in the speech_core_models_litert library; available only when
// speech-core is built with SPEECH_CORE_WITH_LITERT=ON.
//
// Typical use:
//   sc_indic_mio_t m = sc_indic_mio_create_from_pretrained(
//       "soniqo/Indic-Mio-LiteRT", NULL, NULL, NULL, NULL);
//   sc_indic_mio_set_reference(m, ref_pcm, ref_len, 24000);
//   sc_indic_mio_synthesize(m, "नमस्ते, आप कैसे हैं? <happy>", on_chunk, ctx);
//   sc_indic_mio_destroy(m);

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_indic_mio_s* sc_indic_mio_t;

/// Buffered output callback: mono float PCM in [-1, 1] at 24 kHz. The runtime
/// synthesizes whole utterances (the AR loop plus one decoder pass), so
/// `on_chunk` is invoked once per synthesize() with is_final=true. Do not
/// retain `samples` past the call.
typedef void (*sc_indic_mio_chunk_fn)(const float* samples, size_t length,
                                      bool is_final, void* context);

typedef void (*sc_indic_mio_progress_fn)(const char* filename, int file_index,
                                         int file_count, uint64_t downloaded,
                                         uint64_t total, void* context);

/// Create from a local bundle directory holding
/// indicmio-{text-prefill,token-step,audio-decoder,ref-encoder}.tflite,
/// config.json, and tokenizer.json. Returns NULL on failure (logged to stderr).
sc_indic_mio_t sc_indic_mio_create(const char* bundle_dir);

/// Like sc_indic_mio_create, but ensures the bundle for `model_id` (default
/// use: "soniqo/Indic-Mio-LiteRT") is cached locally first, downloading any
/// missing files (resumable; HF_ENDPOINT overrides the host). Semantics match
/// sc_voxcpm2_create_from_pretrained. Returns NULL on failure, including when
/// built without SPEECH_CORE_WITH_HF_DOWNLOAD.
sc_indic_mio_t sc_indic_mio_create_from_pretrained(
    const char* model_id, const char* revision, const char* cache_dir,
    sc_indic_mio_progress_fn on_progress, void* progress_context);

bool sc_indic_mio_has_download_support(void);

void sc_indic_mio_destroy(sc_indic_mio_t synth);

/// Condition subsequent synthesize() calls on a reference speaker clip.
/// `pcm` is mono float in [-1, 1] at `sample_rate`; internally resampled to
/// 24 kHz and center-cropped (preferred) or zero-padded to the ref encoder's
/// 10 s window, then encoded ONCE to the 128-dim global speaker embedding and
/// cached on the handle. Returns 0 on success, non-zero on failure.
int sc_indic_mio_set_reference(sc_indic_mio_t synth, const float* pcm,
                               size_t length, int sample_rate);

/// Drop the cached reference embedding (back to the default voice).
void sc_indic_mio_clear_reference(sc_indic_mio_t synth);

/// Sampling controls. Defaults follow the upstream reference: temperature 0.9,
/// top_k 50, top_p 0.9, repetition_penalty 1.0. EOS is suppressed until the
/// first speech token so every take produces audio.
void sc_indic_mio_set_temperature(sc_indic_mio_t synth, float temperature);
void sc_indic_mio_set_top_k(sc_indic_mio_t synth, int top_k);
void sc_indic_mio_set_top_p(sc_indic_mio_t synth, float top_p);
void sc_indic_mio_set_repetition_penalty(sc_indic_mio_t synth, float penalty);

/// Cap generated speech tokens per synthesize(). Each token is 40 ms of audio
/// (25 Hz at 24 kHz). The hard ceiling is the KV-cache budget minus the prompt
/// bucket (512 - 64 = 448) and the decoder bucket (384); values above the
/// ceiling are clamped. Default 384 (~15.4 s).
void sc_indic_mio_set_max_new_tokens(sc_indic_mio_t synth, int max_new_tokens);

/// Seed the sampler. A fixed non-zero seed makes synthesis reproducible; 0
/// (default) draws a fresh seed each synthesize(), reported by
/// sc_indic_mio_seed_used().
void sc_indic_mio_set_seed(sc_indic_mio_t synth, uint32_t seed);

/// Output sample rate in Hz (24000).
int sc_indic_mio_output_sample_rate(sc_indic_mio_t synth);

/// Maximum text tokens the prompt bucket accepts (64 including the chat
/// template). Longer prompts fail synthesize() with an error rather than
/// truncating mid-sentence — split the text upstream.
int sc_indic_mio_max_text_tokens(sc_indic_mio_t synth);

/// Synthesize `text` (UTF-8; optional trailing emotion tag), delivering PCM
/// via `on_chunk` (buffered: one final call). Returns 0 on success, non-zero
/// on failure (see sc_indic_mio_last_error).
int sc_indic_mio_synthesize(sc_indic_mio_t synth, const char* text,
                            sc_indic_mio_chunk_fn on_chunk, void* context);

/// Cancel an in-progress synthesize() (thread-safe; checked between AR steps).
void sc_indic_mio_cancel(sc_indic_mio_t synth);

/// Speech tokens emitted by the most recent synthesize(). 0 before the first.
int sc_indic_mio_tokens_generated(sc_indic_mio_t synth);

/// Whether the most recent synthesize() stopped on the model's EOS (true)
/// versus the max-new-tokens budget or a cancel() (false).
bool sc_indic_mio_stopped_on_eos(sc_indic_mio_t synth);

/// The seed actually used by the most recent synthesize().
uint32_t sc_indic_mio_seed_used(sc_indic_mio_t synth);

/// Last error message for this handle ("" = none; valid until the next call).
const char* sc_indic_mio_last_error(sc_indic_mio_t synth);

#ifdef __cplusplus
}
#endif

#endif  // SPEECH_CORE_INDIC_MIO_C_H
