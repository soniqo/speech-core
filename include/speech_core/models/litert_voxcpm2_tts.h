#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"
#include "speech_core/models/voxcpm2_tokenizer.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// VoxCPM2 — 2B-parameter multilingual TTS via LiteRT.
/// 48 kHz output. Plain TTS by default; voice cloning when a reference clip is
/// set via set_reference() — that runs the audio_encoder graph and conditions
/// the prefill on the encoded reference latents.
/// Model: https://huggingface.co/aufklarer/VoxCPM2-LiteRT
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
    LiteRtModel         text_prefill_model_    = nullptr;
    LiteRtCompiledModel text_prefill_compiled_ = nullptr;
    LiteRtModel         token_step_model_      = nullptr;
    LiteRtCompiledModel token_step_compiled_   = nullptr;
    LiteRtModel         audio_encoder_model_   = nullptr;
    LiteRtCompiledModel audio_encoder_compiled_= nullptr;
    LiteRtModel         audio_decoder_model_   = nullptr;
    LiteRtCompiledModel audio_decoder_compiled_= nullptr;

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
    int               max_steps_       = kMaxGenerated;
    int               min_stop_steps_  = 8;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
