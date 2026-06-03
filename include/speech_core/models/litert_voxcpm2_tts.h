#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"
#include "speech_core/models/voxcpm2_tokenizer.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace speech_core {

/// VoxCPM2 — 2B-parameter multilingual TTS via LiteRT.
/// 48 kHz output. Plain TTS by default; voice cloning when a reference clip is
/// set via set_reference() — that runs the audio_encoder graph and conditions
/// the prefill on the encoded reference latents.
/// Model: https://huggingface.co/soniqo/VoxCPM2-LiteRT
///
/// Pipeline (four LiteRT graphs orchestrated here):
///   audio_encoder  : (audio [1, 102400] = 6.4 s @ 16 kHz)
///                    → (audio_feats [1, 40, 4, 64])    — cloning only
///   text_prefill   : (tokens, masks, audio_feats, audio_mask, context_len)
///                    → (lm_hidden, residual_hidden, prefix_feat_cond, base_cache, residual_cache)
///   token_step ×N  : (lm_hidden, residual_hidden, prefix_feat_cond, base_cache,
///                     residual_cache, position_id, noise)
///                    → (pred_feat, stop_logits, next_lm_hidden, next_residual_hidden,
///                       base_cache, residual_cache)
///   audio_decoder  : (latent [1, 64, 256]) → PCM [1, 491520] (10.24 s @ 48 kHz)
///
/// Cloning builds the prefill sequence as
///   [ <|ref_audio_start|>, refLatent×N, <|ref_audio_end|>, text…, <|audio_start|> ]
/// with text_mask on the boundary/text tokens and audio_mask on the N latent
/// slots — mirroring the MLX ICL clone path in speech-swift's VoxCPM2TTS.
///
/// Caches grow from text_prefill's 512-slot output to the 2560-slot view that
/// token_step expects (zero-padded along axis 4). Every 64 token_step calls
/// produce one decoder chunk that streams via the TTSChunkCallback. cancel()
/// short-circuits the AR loop between steps.
class LiteRTVoxCPM2Tts : public TTSInterface {
public:
    LiteRTVoxCPM2Tts(const std::string& text_prefill_path,
                     const std::string& token_step_path,
                     const std::string& audio_encoder_path,
                     const std::string& audio_decoder_path,
                     const std::string& tokenizer_path,
                     bool hw_accel = false);
    ~LiteRTVoxCPM2Tts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 48000; }
    void cancel() override;

    /// Override the per-utterance instruction prefix (defaults to empty).
    /// VoxCPM2 was trained on "({instruction}){text}" prompts.
    void set_instruction(std::string instruction) { instruction_ = std::move(instruction); }

    /// Cap the number of AR steps per synthesize() call. Defaults to 2048
    /// (model max). Reducing it bounds latency and is useful in tests.
    void set_max_steps(int max_steps) { max_steps_ = max_steps; }

    /// Treat the model's stop signal as authoritative after this many steps.
    /// Below this floor the stop flag is ignored — the model can flicker on
    /// the very first few steps. Default 8 mirrors the upstream smoke test.
    void set_min_steps_before_stop(int min_steps) { min_stop_steps_ = min_steps; }

    /// Honour the model's stop signal at all. When false, synthesize() always
    /// runs the full max_steps() budget and never short-circuits on the stop
    /// logits — useful for fixed-length renders. Default true.
    void set_stop_on_stop_token(bool stop_on_stop_token) { stop_on_stop_token_ = stop_on_stop_token; }

    /// Seed the noise RNG that drives the diffusion-style AR step. The
    /// token_step graph consumes a per-step Gaussian noise tensor, so a fixed
    /// seed makes synthesis bit-reproducible and different seeds give different
    /// renders of the same text. Pass 0 to draw a fresh non-deterministic seed
    /// at the next synthesize() (its value is then reported by seed_used()).
    /// Default is 0 → a fresh random seed per call.
    void set_seed(uint32_t seed) { seed_ = seed; }

    /// The seed actually used by the most recent synthesize() call. When
    /// set_seed(0) was in effect this is the randomly drawn value; otherwise it
    /// echoes the seed passed in. 0 before the first synthesize().
    uint32_t seed_used() const { return seed_used_; }

    /// Number of AR steps (acoustic tokens) emitted by the most recent
    /// synthesize() call. 0 before the first call.
    int tokens_generated() const { return tokens_generated_; }

    /// Whether the most recent synthesize() stopped because the model raised
    /// its stop signal (true) versus hitting the max-steps budget or being
    /// cancelled (false). Meaningless before the first call.
    bool stopped_on_stop_token() const { return stopped_on_stop_token_; }

    /// Maximum number of text tokens the prefill graph accepts (the context
    /// window). Prompts longer than this are trimmed from the front. Queried
    /// from the loaded graph; 512 on the deployed bundle.
    int max_text_tokens() const { return max_text_; }

    /// Configure the keep-warm policy for the text_prefill graph.
    ///
    /// 0 (default) → always release the graph after each synthesize() finishes
    /// its prefill stage. Lowest cold-RSS footprint; every call pays a
    /// ~3-5 second cold-load on the next request.
    ///
    /// >0 → hold the prefill graph loaded across synthesize() calls; an internal
    /// reaper thread releases it once no synthesize() has touched it for `ms`.
    /// Trades a steady-state ~1.9 GiB of resident memory for zero cold-load on
    /// the hot path. Recommended for realtime / voice-agent workloads where
    /// turns arrive within seconds of each other. Internally clamped to a
    /// minimum poll cadence so very small values still behave sanely.
    ///
    /// The reaper thread is always running; idle_release_ms_ == 0 just turns
    /// its eviction step into a no-op. Re-enable via a positive value at any
    /// time without restarting the wrapper.
    void set_idle_release_ms(int64_t ms);

    /// Condition subsequent synthesize() calls on a reference speaker clip.
    /// `pcm` is mono float samples in [-1, 1] at `sample_rate`; it's resampled
    /// to 16 kHz and padded/truncated to the encoder's fixed 6.4 s window, then
    /// run through audio_encoder. The encoded latents are cached and reused on
    /// every synthesize() until clear_reference() (or a new set_reference()).
    /// Throws if the bundle's tokenizer lacks the reference boundary tokens.
    void set_reference(const float* pcm, size_t length, int sample_rate);

    /// Drop any reference clip; synthesize() falls back to plain (uncloned) TTS.
    void clear_reference();

    /// Whether a reference clip is currently set.
    bool has_reference() const { return ref_frames_ > 0; }

private:
    // text_prefill is the largest graph in the bundle (~1.9 GiB INT8). It is
    // invoked exactly once per synthesize() call and never used between calls.
    // To keep idle RSS low (so the prod CCX23 has node headroom for inference
    // + the rest of the cluster) we lazy-load it: the constructor loads it
    // once to query max_text_, then frees it; synthesize() reloads it for
    // the prefill stage and frees it again before entering the AR loop. The
    // path is retained so we can reload without re-plumbing it through every
    // caller. The compiled + model handles stay nullptr while unloaded;
    // load_text_prefill() / release_text_prefill() are the canonical entry
    // points.
    std::string         text_prefill_path_;
    bool                text_prefill_hw_accel_ = false;
    LiteRtModel         text_prefill_model_    = nullptr;
    LiteRtCompiledModel text_prefill_compiled_ = nullptr;
    LiteRtModel         token_step_model_      = nullptr;
    LiteRtCompiledModel token_step_compiled_   = nullptr;
    LiteRtModel         audio_encoder_model_   = nullptr;
    LiteRtCompiledModel audio_encoder_compiled_= nullptr;
    LiteRtModel         audio_decoder_model_   = nullptr;
    LiteRtCompiledModel audio_decoder_compiled_= nullptr;

    void load_text_prefill();
    void release_text_prefill();

    // Keep-warm cache for text_prefill.
    //
    // When idle_release_ms_ > 0, synthesize() leaves the prefill graph loaded
    // after its prefill stage; the reaper thread frees it once no synthesize()
    // has touched it for the configured idle window. When 0, synthesize()
    // releases at the end of every prefill (the original lazy-load path).
    //
    // Serialisation: prefill_mutex_ guards both the load/release transitions
    // and the prefill-stage Run() against the reaper thread. The reaper uses
    // try_lock so an in-flight synthesize() is never interrupted -- if the
    // mutex is busy, the reaper just retries next tick. last_prefill_used_
    // is updated under prefill_mutex_ at the entry and exit of each synth's
    // prefill stage, and read by the reaper under the same lock.
    //
    // Shutdown: destructor sets reaper_stop_, notifies the CV, joins the
    // thread, then proceeds to destroy the remaining graphs. The reaper never
    // touches a destroyed handle because the destructor blocks on join.
    std::atomic<std::int64_t>          idle_release_ms_{0};
    std::chrono::steady_clock::time_point last_prefill_used_{};
    std::mutex                         prefill_mutex_;
    std::atomic<bool>                  reaper_stop_{false};
    std::condition_variable            reaper_cv_;
    std::mutex                         reaper_mutex_;
    std::thread                        reaper_thread_;

    void reaper_loop_();

    std::unique_ptr<VoxCPM2Tokenizer> tokenizer_;
    int audio_start_token_     = -1;
    int ref_audio_start_token_ = -1;   // <|ref_audio_start|>, cloning only
    int ref_audio_end_token_   = -1;   // <|ref_audio_end|>, cloning only

    // Cached reference latents from the last set_reference() call. Laid out as
    // ref_frames_ consecutive [patch, feat] frames (kPredFeatFloats each),
    // matching one audio_feats slot per frame. Empty ⇒ plain TTS.
    std::vector<float> ref_feats_;
    int                ref_frames_ = 0;

    // Bundle-invariant shape constants (same across every VoxCPM2 export).
    static constexpr int  kMaxGenerated        = 2048;
    static constexpr int  kHidden              = 2048;
    static constexpr int  kFeatDim             = 64;
    static constexpr int  kPatchSize           = 4;
    static constexpr int  kPredFeatFloats      = kPatchSize * kFeatDim;            // 256
    static constexpr int  kFramesPerChunk      = 64;
    static constexpr int  kDecoderInputFloats  = kFramesPerChunk * kPredFeatFloats; // 16384
    static constexpr int  kDecoderOutputFloats = 491520;     // 10.24 s @ 48 kHz
    static constexpr int  kSamplesPerStep      = 7680;       // 160 ms @ 48 kHz
    static constexpr int  kBaseLayers          = 28;
    static constexpr int  kResidualLayers      = 8;
    static constexpr int  kKvHeads             = 2;
    static constexpr int  kHeadDim             = 128;

    // Reference-audio conditioning (audio_encoder graph). The graph's input
    // length is baked at export (audio_conditioning_sample_rate × seconds =
    // 16 kHz × 6.4 s = 102400 samples) and emits one latent frame per 2560
    // input samples (= patch_size 4 × VAE downsample 640), so 40 frames max.
    static constexpr int  kCondSampleRate      = 16000;
    static constexpr int  kRefAudioSamples     = 102400;   // 6.4 s @ 16 kHz
    static constexpr int  kRefFrameStride      = 2560;     // samples per latent frame
    static constexpr int  kMaxRefFrames        = kRefAudioSamples / kRefFrameStride; // 40

    // Context geometry — queried from the text_prefill graph at load (its
    // audio_feats input is [1, max_text, patch, feat]) so 256- and 512-token
    // exports both work. token_step grows the cache to max_text + max_generated.
    int  max_text_               = 512;
    int  full_seq_               = 512 + kMaxGenerated;
    long base_cache_floats_      = 0;
    long residual_cache_floats_  = 0;
    long base_prefill_floats_    = 0;
    long residual_prefill_floats_= 0;

    std::string       instruction_;
    int               max_steps_          = kMaxGenerated;
    int               min_stop_steps_     = 8;
    bool              stop_on_stop_token_ = true;
    uint32_t          seed_               = 0;   // 0 ⇒ draw a fresh seed per call
    std::atomic<bool> cancelled_{false};

    // Per-call synthesis metadata, populated at the end of synthesize().
    uint32_t          seed_used_             = 0;
    int               tokens_generated_      = 0;
    bool              stopped_on_stop_token_ = false;
};

}  // namespace speech_core
