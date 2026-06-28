#ifndef SPEECH_CORE_CHATTERBOX_C_H
#define SPEECH_CORE_CHATTERBOX_C_H

// C ABI for Chatterbox Multilingual TTS (LiteRT backend).
//
// One-shot, zero-shot (default-voice) text-to-speech across 23 languages at
// 24 kHz. Symbols live in speech_core_models_litert; available only when
// speech-core is built with SPEECH_CORE_WITH_LITERT=ON.
//
//   sc_chatterbox_t c = sc_chatterbox_create("/path/to/bundle");
//   sc_chatterbox_synthesize(c, "Hello there", "en", on_chunk, ctx);
//   sc_chatterbox_destroy(c);

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "speech_core/tts_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_chatterbox_s* sc_chatterbox_t;

/// Streaming output callback. `samples` is mono float PCM in [-1, 1] at 24 kHz.
/// `is_final` marks the last call (may carry length 0). Do not retain `samples`.
typedef void (*sc_chatterbox_chunk_fn)(const float* samples, size_t length,
                                       bool is_final, void* context);

/// Create from a Chatterbox LiteRT bundle directory (chatterbox-*.tflite,
/// grapheme_mtl_merged_expanded_v1.json, Cangjie5_TC.json, cond_emb.bin,
/// spks.bin, prompt_token.bin, prompt_feat.bin). NULL on failure (logged).
sc_chatterbox_t sc_chatterbox_create(const char* bundle_dir);

/// Destroy a synthesizer and free its resources.
void sc_chatterbox_destroy(sc_chatterbox_t synth);

// --- sampling parameters (apply to subsequent synthesize() calls) ---
void sc_chatterbox_set_temperature(sc_chatterbox_t synth, float temperature);
void sc_chatterbox_set_top_p(sc_chatterbox_t synth, float top_p);
void sc_chatterbox_set_min_p(sc_chatterbox_t synth, float min_p);
void sc_chatterbox_set_repetition_penalty(sc_chatterbox_t synth, float penalty);
/// Cap AR speech-token steps per synthesize() (default 1000).
void sc_chatterbox_set_max_tokens(sc_chatterbox_t synth, int max_tokens);
/// Fixed non-zero seed makes synthesis reproducible; 0 (default) draws fresh.
void sc_chatterbox_set_seed(sc_chatterbox_t synth, uint32_t seed);

/// Output sample rate in Hz (24000).
int sc_chatterbox_output_sample_rate(sc_chatterbox_t synth);

/// Synthesize `text` in `language` (ISO code, e.g. "en", "ar", "zh"), streaming
/// audio via `on_chunk`. Returns 0 on success, non-zero on failure
/// (see sc_chatterbox_last_error).
int sc_chatterbox_synthesize(sc_chatterbox_t synth, const char* text,
                             const char* language,
                             sc_chatterbox_chunk_fn on_chunk, void* context);

/// Synthesize with explicit delivery mode and offline postprocess flags.
/// NULL options preserve the legacy Streaming + SC_TTS_POSTPROCESS_NONE behavior.
int sc_chatterbox_synthesize_with_options(
    sc_chatterbox_t synth,
    const char* text,
    const char* language,
    const sc_tts_synthesis_options_t* options,
    sc_chatterbox_chunk_fn on_chunk,
    void* context);

/// Cancel an in-progress synthesize() (thread-safe; checked between AR steps).
void sc_chatterbox_cancel(sc_chatterbox_t synth);

/// Speech tokens emitted by the most recent synthesize(). 0 before the first.
int sc_chatterbox_tokens_generated(sc_chatterbox_t synth);

/// Seed actually used by the most recent synthesize().
uint32_t sc_chatterbox_seed_used(sc_chatterbox_t synth);

/// Last error message for the most recent failed call (valid until the next call
/// on the same handle; never NULL, "" = none).
const char* sc_chatterbox_last_error(sc_chatterbox_t synth);

#ifdef __cplusplus
}
#endif

#endif  // SPEECH_CORE_CHATTERBOX_C_H
