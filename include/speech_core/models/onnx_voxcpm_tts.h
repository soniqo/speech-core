#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/voxcpm2_tokenizer.h"

#include <onnxruntime_c_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// VoxCPM 0.5B bilingual TTS via ONNX Runtime.
/// 16 kHz output. Plain TTS by default; prompt-audio conditioning when a
/// reference clip is set via set_reference(). Supplying the exact transcript
/// for that clip via set_reference_transcript() improves clone fidelity.
/// Model: https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX.
///
/// Mirrors the upstream VoxCPM 0.5B inference loop at the AR-loop level. The
/// three ORT sessions are loaded through OnnxEngine, which routes hw_accel=true to the
/// hardware EP via the optional SessionOptionsHook (with automatic CPU fallback) and
/// to NNAPI/QNN on Android. Each per-Run call mints a fresh set of OrtValues
/// over caller-owned host memory via CreateTensorWithDataAsOrtValue, then
/// reads back from the returned OrtValues' GetTensorMutableData pointers —
/// the same idiom ParakeetStt uses.
///
/// Pipeline (three ONNX graphs orchestrated here):
///   audio_encoder  : (audio [1, 102400] = 6.4 s @ 16 kHz)
///                    → (audio_feats [1, 80, 2, 64])    — prompt conditioning only
///   decoder        : UNIFIED prefill + token-step graph (the merged export).
///     prefill mode  : (text_tokens, text_mask, audio_feats, audio_mask, ctx_len)
///                     → (lm_hidden, residual_hidden, prefix_feat_cond, base_cache, residual_cache)
///     token mode ×N : (ts_lm_hidden, ts_residual_hidden, ts_prefix_feat_cond, ts_base_cache,
///                      ts_residual_cache, ts_position_id, ts_noise)
///                     → (ts_pred_feat, ts_stop_logits, ts_next_lm, ts_next_resid,
///                        ts_next_base_cache, ts_next_resid_cache)
///   audio_decoder  : (latent [1, 64, 128]) → PCM [1, 81920] (5.12 s @ 16 kHz)
///
/// The prefill and token-step graphs share one transformer; exporting them as
/// one graph stores those weights once. Both modes live in `decoder_session_`;
/// each Run requests only its mode's outputs but must still supply every graph
/// input, so the idle mode's inputs are fed cheap zero dummies.
class OnnxVoxCPMTts : public TTSInterface {
public:
    /// `decoder_path` is the unified prefill+token-step graph
    /// (voxcpm-decoder.onnx). audio_encoder/decoder + tokenizer as before.
    OnnxVoxCPMTts(const std::string& decoder_path,
                   const std::string& audio_encoder_path,
                   const std::string& audio_decoder_path,
                   const std::string& tokenizer_path,
                   bool hw_accel = true);
    ~OnnxVoxCPMTts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 16000; }
    void cancel() override;

    /// Override the per-utterance instruction prefix (defaults to empty).
    void set_instruction(std::string instruction) { instruction_ = std::move(instruction); }

    /// Cap the number of AR steps per synthesize() call. Defaults to 2048
    /// (model max). Reducing it bounds latency and is useful in tests.
    void set_max_steps(int max_steps) { max_steps_ = max_steps; }

    /// Treat the model's stop signal as authoritative after this many steps.
    void set_min_steps_before_stop(int min_steps) { min_stop_steps_ = min_steps; }

    /// Honour the model's stop signal at all. When false, synthesize() always
    /// runs the full max_steps() budget.
    void set_stop_on_stop_token(bool stop_on_stop_token) { stop_on_stop_token_ = stop_on_stop_token; }

    /// Seed the noise RNG that drives the diffusion-style AR step. 0 ⇒ draw a
    /// fresh non-deterministic seed at the next synthesize() (its value is
    /// then reported by seed_used()). Default is 0.
    void set_seed(uint32_t seed) { seed_ = seed; }

    /// The seed actually used by the most recent synthesize() call.
    uint32_t seed_used() const { return seed_used_; }

    /// Number of AR steps (acoustic tokens) emitted by the most recent
    /// synthesize() call. 0 before the first call.
    int tokens_generated() const { return tokens_generated_; }

    /// Whether the most recent synthesize() stopped because the model raised
    /// its stop signal (true) vs hitting max-steps / being cancelled (false).
    bool stopped_on_stop_token() const { return stopped_on_stop_token_; }

    /// Wall-clock model stage timings from the most recent synthesize() call.
    /// -1 before the first call; otherwise includes wrapper/ORT overhead for
    /// the stage, which is the number operators need for production latency.
    int64_t prefill_ms() const { return prefill_ms_; }
    int64_t ar_ms() const { return ar_ms_; }
    int64_t audio_decode_ms() const { return audio_decode_ms_; }

    /// Maximum number of text tokens the prefill graph accepts. 512 on the
    /// deployed bundle.
    int max_text_tokens() const { return max_text_; }

    /// Condition subsequent synthesize() calls on a reference speaker clip.
    /// `pcm` is mono float samples in [-1, 1] at `sample_rate`; resampled to
    /// 16 kHz and padded/truncated to the encoder's fixed 6.4 s window.
    void set_reference(const float* pcm, size_t length, int sample_rate);

    /// Optional exact transcript of the current reference clip. Call after
    /// set_reference(); clear_reference() clears it with the audio prompt.
    void set_reference_transcript(std::string transcript) {
        ref_transcript_ = std::move(transcript);
    }

    /// Drop any reference clip; synthesize() falls back to plain TTS.
    void clear_reference();

    /// Whether a reference clip is currently set.
    bool has_reference() const { return ref_frames_ > 0; }

private:
    // Helpers populated at construction: each graph's input/output names as
    // queried via Session{Get,Get}{Input,Output}Name. Stored as owned
    // std::string and a parallel const char*[] for Run().
    struct IoNames {
        std::vector<std::string>  in_names_str;
        std::vector<std::string>  out_names_str;
        std::vector<const char*>  in_names;
        std::vector<const char*>  out_names;
    };
    void query_io_names(OrtSession* session, IoNames& names);

    // Query the prefill graph's max_text_tokens context window. The
    // audio_feats input is the rank-4 tensor [1, max_text, patch, feat].
    int query_prefill_context(OrtSession* session);

    const OrtApi* api_ = nullptr;

    // One session for the unified prefill+token graph (weights resident once).
    OrtSession* decoder_session_       = nullptr;
    OrtSession* audio_encoder_session_ = nullptr;
    OrtSession* audio_decoder_session_ = nullptr;

    // The unified graph's I/O split by the "ts_" prefix: prefill_io_ holds the
    // prefill names (text_tokens…→lm_hidden…), step_io_ the token-step names
    // (ts_lm_hidden…→ts_pred_feat…). Each is a subset of decoder_session_'s I/O.
    IoNames prefill_io_;
    IoNames step_io_;
    IoNames encoder_io_;
    IoNames decoder_io_;

    std::unique_ptr<VoxCPM2Tokenizer> tokenizer_;
    int audio_start_token_     = -1;

    // Cached reference latents from the last set_reference() call. Laid out as
    // ref_frames_ consecutive [patch, feat] frames (kPredFeatFloats each).
    std::vector<float> ref_feats_;
    std::string        ref_transcript_;
    int                ref_frames_ = 0;

    // Bundle-invariant shape constants (same across every VoxCPM export).
    static constexpr int  kMaxGenerated        = 2048;
    static constexpr int  kHidden              = 1024;
    static constexpr int  kFeatDim             = 64;
    static constexpr int  kPatchSize           = 2;
    static constexpr int  kPredFeatFloats      = kPatchSize * kFeatDim;            // 128
    static constexpr int  kFramesPerChunk      = 64;
    static constexpr int  kDecoderInputFloats  = kFramesPerChunk * kPredFeatFloats; // 8192
    static constexpr int  kDecoderOutputFloats = 81920;      // 5.12 s @ 16 kHz
    static constexpr int  kSamplesPerStep      = 1280;       // 80 ms @ 16 kHz
    static constexpr int  kBaseLayers          = 24;
    static constexpr int  kResidualLayers      = 6;
    static constexpr int  kKvHeads             = 2;
    static constexpr int  kHeadDim             = 64;

    // Reference-audio conditioning constants.
    static constexpr int  kCondSampleRate      = 16000;
    static constexpr int  kRefAudioSamples     = 102400;   // 6.4 s @ 16 kHz
    static constexpr int  kRefFrameStride      = 1280;     // samples per latent frame
    static constexpr int  kMaxRefFrames        = kRefAudioSamples / kRefFrameStride; // 80

    // Context geometry queried from the loaded prefill graph.
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
    uint32_t          seed_               = 0;
    std::atomic<bool> cancelled_{false};

    // Per-call metadata, populated at the end of synthesize().
    uint32_t          seed_used_             = 0;
    int               tokens_generated_      = 0;
    bool              stopped_on_stop_token_ = false;
    int64_t           prefill_ms_            = -1;
    int64_t           ar_ms_                 = -1;
    int64_t           audio_decode_ms_       = -1;
};

}  // namespace speech_core
