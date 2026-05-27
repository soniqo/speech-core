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
/// 48 kHz output. Voice cloning + voice design (cloning surface deferred —
/// `synthesize()` always feeds zero audio_feats today).
/// Model: https://huggingface.co/aufklarer/VoxCPM2-LiteRT
///
/// Pipeline (four LiteRT graphs orchestrated here):
///   text_prefill   : (tokens, masks, audio_feats, audio_mask, context_len)
///                    → (lm_hidden, residual_hidden, prefix_feat_cond, base_cache, residual_cache)
///   token_step ×N  : (lm_hidden, residual_hidden, prefix_feat_cond, base_cache,
///                     residual_cache, position_id, noise)
///                    → (pred_feat, stop_logits, next_lm_hidden, next_residual_hidden,
///                       base_cache, residual_cache)
///   audio_decoder  : (latent [1, 64, 256]) → PCM [1, 491520] (10.24 s @ 48 kHz)
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
    int audio_start_token_ = -1;

    // Shape constants pulled from config.json (max_text_tokens = 512,
    // max_generated_tokens = 2048, hidden = 2048, etc.). Kept here so the
    // implementation file is just allocations and tensor-copy plumbing.
    static constexpr int  kMaxText             = 512;
    static constexpr int  kMaxGenerated        = 2048;
    static constexpr int  kHidden              = 2048;
    static constexpr int  kFeatDim             = 64;
    static constexpr int  kPatchSize           = 4;
    static constexpr int  kPredFeatFloats      = kPatchSize * kFeatDim;            // 256
    static constexpr int  kFramesPerChunk      = 64;
    static constexpr int  kDecoderInputFloats  = kFramesPerChunk * kPredFeatFloats; // 16384
    static constexpr int  kDecoderOutputFloats = 491520;     // 10.24 s @ 48 kHz
    static constexpr int  kSamplesPerStep      = 7680;       // 160 ms @ 48 kHz
    static constexpr long kBaseCacheFloats     = 2L * 28 * 1 * 2 * 2560 * 128;     // ~36.7 M
    static constexpr long kResidualCacheFloats = 2L *  8 * 1 * 2 * 2560 * 128;     // ~10.5 M
    static constexpr long kBasePrefillFloats   = 2L * 28 * 1 * 2 * 512  * 128;     // ~7.3 M
    static constexpr long kResidualPrefillFloats = 2L * 8 * 1 * 2 * 512  * 128;    //  2.1 M

    std::string       instruction_;
    int               max_steps_       = kMaxGenerated;
    int               min_stop_steps_  = 8;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
