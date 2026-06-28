// C ABI for SupertonicTTS-3 — the non-autoregressive flow-matching multilingual TTS
// (44.1 kHz, 31 languages, G2P-free) running on LiteRT through speech-core's TTSInterface.
//
// Mirrors voxcpm2_c.h. Example:
//
//   sc_supertonic_t s = sc_supertonic_create("/path/to/Supertonic-3-LiteRT");
//   sc_supertonic_set_voice(s, "F1");
//   sc_supertonic_synthesize(s, "Hello there", "en", on_chunk, user_ctx);
//   sc_supertonic_destroy(s);
//
// The bundle dir holds the four graphs (`duration_predictor.tflite`, `text_encoder.tflite`,
// `vector_estimator.tflite`, `vocoder.tflite`), the tokenizer assets (`unicode_indexer.json`,
// `tts.json`), and a `voice_styles/` directory of `<id>.json` files — i.e. the layout of
// https://huggingface.co/soniqo/Supertonic-3-LiteRT .

#ifndef SPEECH_CORE_SUPERTONIC_C_H
#define SPEECH_CORE_SUPERTONIC_C_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "speech_core/tts_c.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_supertonic_s* sc_supertonic_t;

/// Streamed PCM callback: `samples` is `length` mono float32 at the output sample rate;
/// `is_final` is true on the last chunk of the utterance. `user` is passed through verbatim.
typedef void (*sc_supertonic_chunk_fn)(const float* samples, size_t length,
                                       bool is_final, void* user);

/// Create from a bundle directory (see the file header for the expected layout). Returns NULL on
/// failure; call sc_supertonic_last_error(NULL) for a message.
sc_supertonic_t sc_supertonic_create(const char* bundle_dir);

/// Create from explicit asset paths (graphs + tokenizer dir + voice-styles dir). `hw_accel`
/// requests a hardware delegate where available. Returns NULL on failure.
sc_supertonic_t sc_supertonic_create_from_paths(const char* duration_path,
                                                const char* text_encoder_path,
                                                const char* vector_estimator_path,
                                                const char* vocoder_path,
                                                const char* tokenizer_dir,
                                                const char* voice_styles_dir,
                                                bool hw_accel);

void sc_supertonic_destroy(sc_supertonic_t synth);

/// Select the voice (e.g. "F1", "M3"). No-op + error string on unknown id.
void sc_supertonic_set_voice(sc_supertonic_t synth, const char* voice_id);

/// Flow-matching ODE steps: 5 (fast) · 8 (default) · 12 (quality).
void sc_supertonic_set_total_step(sc_supertonic_t synth, int total_step);

/// Speech rate; divides predicted duration. Default 1.05.
void sc_supertonic_set_speed(sc_supertonic_t synth, float speed);

/// Latent-noise RNG seed; 0 ⇒ a fresh seed per synthesize() (default).
void sc_supertonic_set_seed(sc_supertonic_t synth, uint32_t seed);

/// Output sample rate (44100).
int sc_supertonic_output_sample_rate(sc_supertonic_t synth);

/// Synthesize `text` in ISO `language`, streaming PCM to `on_chunk`. Returns 0 on success,
/// non-zero on failure (see sc_supertonic_last_error).
int sc_supertonic_synthesize(sc_supertonic_t synth, const char* text, const char* language,
                             sc_supertonic_chunk_fn on_chunk, void* user);

/// Synthesize with explicit delivery mode and offline postprocess flags.
/// NULL options preserve the legacy Streaming + SC_TTS_POSTPROCESS_NONE behavior.
int sc_supertonic_synthesize_with_options(
    sc_supertonic_t synth,
    const char* text,
    const char* language,
    const sc_tts_synthesis_options_t* options,
    sc_supertonic_chunk_fn on_chunk,
    void* user);

/// Request cancellation of an in-flight synthesize() (callable from another thread).
void sc_supertonic_cancel(sc_supertonic_t synth);

/// Last error for `synth`, or the last creation error if `synth` is NULL. Never NULL.
const char* sc_supertonic_last_error(sc_supertonic_t synth);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // SPEECH_CORE_SUPERTONIC_C_H
