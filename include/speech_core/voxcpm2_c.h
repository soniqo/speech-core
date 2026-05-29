#ifndef SPEECH_CORE_VOXCPM2_C_H
#define SPEECH_CORE_VOXCPM2_C_H

// C ABI for VoxCPM2 voice cloning (LiteRT backend).
//
// This is a standalone, one-shot synthesis surface — distinct from the
// real-time pipeline in speech_core_c.h. It exists so non-C++ hosts (e.g. the
// speech-studio Tauri app, via Rust FFI) can clone a speaker from a reference
// clip and synthesize text in that voice on Linux/Windows, where the macOS
// MLX sidecar isn't available.
//
// Symbols live in the speech_core_models_litert library; link that plus the
// imported `litert` runtime. Available only when speech-core is built with
// SPEECH_CORE_WITH_LITERT=ON.
//
// Typical use:
//   sc_voxcpm2_t v = sc_voxcpm2_create("/path/to/bundle");
//   sc_voxcpm2_set_instruction(v, "calm, clear delivery");
//   sc_voxcpm2_set_reference(v, ref_pcm, ref_len, 16000);
//   sc_voxcpm2_synthesize(v, "Hello there", on_chunk, user_ctx);
//   sc_voxcpm2_destroy(v);

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sc_voxcpm2_s* sc_voxcpm2_t;

/// Streaming output callback. `samples` is mono float PCM in [-1, 1] at the
/// model's output rate (48 kHz). `is_final` marks the last call of a synthesis
/// (and may carry length 0). Do not retain `samples` past the call.
typedef void (*sc_voxcpm2_chunk_fn)(const float* samples, size_t length,
                                    bool is_final, void* context);

/// Create a synthesizer from a VoxCPM2 LiteRT bundle directory (holding
/// voxcpm2-{text-prefill,token-step,audio-encoder,audio-decoder}.tflite and
/// tokenizer.json). Returns NULL on failure (reason logged to stderr).
sc_voxcpm2_t sc_voxcpm2_create(const char* bundle_dir);

/// Destroy a synthesizer and free its resources.
void sc_voxcpm2_destroy(sc_voxcpm2_t synth);

/// Style prefix prepended as "({instruction})text" (VoxCPM2's training format).
/// Pass NULL or "" to clear. Persists across synthesize() calls.
void sc_voxcpm2_set_instruction(sc_voxcpm2_t synth, const char* instruction);

/// Cap AR steps per synthesize() (default 2048 = model max). Each step is
/// ~160 ms of audio; lowering it bounds latency.
void sc_voxcpm2_set_max_steps(sc_voxcpm2_t synth, int max_steps);

/// Ignore the model's stop signal for this many initial steps (default 8).
void sc_voxcpm2_set_min_steps_before_stop(sc_voxcpm2_t synth, int min_steps);

/// Honour the model's stop signal at all (default true). When false,
/// synthesize() always runs the full max_steps budget and never stops early.
void sc_voxcpm2_set_stop_on_stop_token(sc_voxcpm2_t synth, bool stop_on_stop_token);

/// Seed the per-step noise RNG that drives the diffusion-style AR sampler. A
/// fixed non-zero seed makes synthesis reproducible; 0 (default) draws a fresh
/// random seed at the next synthesize(), reported by sc_voxcpm2_seed_used().
void sc_voxcpm2_set_seed(sc_voxcpm2_t synth, uint32_t seed);

/// Maximum text tokens the model accepts (context window; 512 on the deployed
/// bundle). Prompts longer than this are trimmed from the front.
int sc_voxcpm2_max_text_tokens(sc_voxcpm2_t synth);

/// Condition subsequent synthesize() calls on a reference speaker clip.
/// `pcm` is mono float in [-1, 1] at `sample_rate` (resampled to 16 kHz and
/// trimmed/padded to the encoder's 6.4 s window internally). Returns 0 on
/// success, non-zero on failure (see sc_voxcpm2_last_error). Cleared by
/// sc_voxcpm2_clear_reference() — without a reference, output is uncloned.
int sc_voxcpm2_set_reference(sc_voxcpm2_t synth, const float* pcm,
                             size_t length, int sample_rate);

/// Drop any reference clip set by sc_voxcpm2_set_reference().
void sc_voxcpm2_clear_reference(sc_voxcpm2_t synth);

/// Output sample rate in Hz (48000).
int sc_voxcpm2_output_sample_rate(sc_voxcpm2_t synth);

/// Synthesize `text`, streaming audio via `on_chunk`. Returns 0 on success,
/// non-zero on failure (see sc_voxcpm2_last_error).
int sc_voxcpm2_synthesize(sc_voxcpm2_t synth, const char* text,
                          sc_voxcpm2_chunk_fn on_chunk, void* context);

/// Cancel an in-progress synthesize() (thread-safe; checked between AR steps).
void sc_voxcpm2_cancel(sc_voxcpm2_t synth);

/// Number of acoustic tokens (AR steps) emitted by the most recent
/// synthesize() on this handle. 0 before the first call.
int sc_voxcpm2_tokens_generated(sc_voxcpm2_t synth);

/// Whether the most recent synthesize() stopped on the model's stop signal
/// (true) versus the max-steps budget or a cancel() (false).
bool sc_voxcpm2_stopped_on_stop_token(sc_voxcpm2_t synth);

/// The seed actually used by the most recent synthesize() (the random draw when
/// set_seed(0) was in effect, else the seed passed in). 0 before the first call.
uint32_t sc_voxcpm2_seed_used(sc_voxcpm2_t synth);

/// Last error message for the most recent failed call on this handle. Returns a
/// pointer valid until the next call on the same handle; never NULL ("" = none).
const char* sc_voxcpm2_last_error(sc_voxcpm2_t synth);

#ifdef __cplusplus
}
#endif

#endif  // SPEECH_CORE_VOXCPM2_C_H
