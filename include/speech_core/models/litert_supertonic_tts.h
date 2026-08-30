#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"
#include "speech_core/models/supertonic_tokenizer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// SupertonicTTS-3 — 99M non-autoregressive flow-matching multilingual TTS via LiteRT.
/// 44.1 kHz output, 31 languages, **G2P-free** (NFKD + unicode-index tokenizer — no phonemizer).
/// Models: https://huggingface.co/soniqo/Supertonic-3-LiteRT
///
/// Four LiteRT graphs orchestrated here (text padded to a fixed T = 128):
///   duration_predictor : (text_ids[1,T] i64, style_dp[1,8,16], text_mask[1,1,T]) → duration[1] (sec)
///   text_encoder       : (text_ids, style_ttl[1,50,256], text_mask)              → text_emb[1,256,T]
///   vector_estimator ×N: (noisy[1,144,L], text_emb, style_ttl, latent_mask[1,1,L],
///                         text_mask, current_step[1], total_step[1])              → denoised[1,144,L]
///   vocoder            : (latent[1,144,L])                                        → wav[1, 512*6*L]
///   L = ceil(duration*44100/(512*6)); noisy = randn([1,144,L])*latent_mask; speed divides duration.
///
/// Unlike VoxCPM2, this is **non-autoregressive**: a fixed `total_step` ODE loop (default 8), no stop
/// logits, no KV cache. The published graphs are fixed-shape (T = 128 text tokens, L = 64 latent
/// frames ≈ 4.5 s), so synthesize() packs sentences into chunks, runs the duration predictor on each
/// as a preflight, and bisects a chunk at its best sentence/clause/word boundary only when its
/// predicted audio overflows the window — a long sentence is never word-packed at a character
/// count. A piece that overflows by a small margin (≤ kWindowStretchMax) is spoken slightly faster
/// to fit instead of being cut (the same host-side mechanism as set_speed()). The four graphs run
/// per piece and the trimmed PCM streams through the TTSChunkCallback; the inter-chunk silence
/// (0.3 s) is inserted only where a sentence ends, and a piece that continues into the next one
/// is tokenized with a trailing "," rather than a sentence-final ".".
///
/// Validated end-to-end against the ONNX reference at 66–82 dB mag-STFT SNR (en/de/ko); see the
/// Runner repo's `speech-models/stmodels/controlled_ab.py`. Voice is a precomputed style pair
/// (`voice_styles/<id>.json`); on-device voice cloning is out of scope (the style-extractor isn't
/// released).
class LiteRTSupertonicTts : public TTSInterface {
public:
    /// @param tokenizer_dir   holds `unicode_indexer.json` + `tts.json`
    /// @param voice_styles_dir holds `<id>.json` style files (F1..F5, M1..M5)
    LiteRTSupertonicTts(const std::string& duration_path,
                        const std::string& text_encoder_path,
                        const std::string& vector_estimator_path,
                        const std::string& vocoder_path,
                        const std::string& tokenizer_dir,
                        const std::string& voice_styles_dir,
                        bool hw_accel = false);
    ~LiteRTSupertonicTts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 44100; }
    void cancel() override;

    /// Select the voice (e.g. "F1", "M3"). Default "F1". Throws if unknown.
    void set_voice(const std::string& voice_id);

    /// Flow-matching ODE steps: 5 (fast) · 8 (default) · 12 (quality).
    void set_total_step(int total_step) { total_step_ = total_step; }

    /// Speech rate; divides predicted duration. Default 1.05.
    void set_speed(float speed) { speed_ = speed; }

    /// Seed the latent-noise RNG. 0 ⇒ a fresh non-deterministic seed per call (default); a fixed seed
    /// makes synthesis reproducible. The value used is reported by seed_used().
    void set_seed(uint32_t seed) { seed_ = seed; }
    uint32_t seed_used() const { return seed_used_; }

    /// Silence inserted between chunks, in seconds. Default 0.3.
    void set_chunk_silence(float seconds) { chunk_silence_s_ = seconds; }

    /// Available voice ids loaded from `voice_styles_dir`.
    std::vector<std::string> voices() const;

private:
    struct VoiceStyle {
        std::vector<float> style_ttl;  // [1,50,256] → 12800
        std::vector<float> style_dp;   // [1,8,16]   → 128
    };

    // Tokenized piece + its predicted duration. The duration predictor is the first graph anyway,
    // so it doubles as the fixed-window preflight (see synthesize()).
    struct Prepared {
        SupertonicTokenizer::Tokens tok;
        float duration      = 0.0f;  // seconds after speed; 0 ⇒ nothing to synthesize
        int   latent_frames = 0;     // L_true = ceil(int(duration * SR) / 3072)
    };
    Prepared prepare_chunk(const std::string& text, const std::string& language, bool continuation);

    // text_encoder → vector_estimator × N → vocoder on a prepared piece → trimmed 44.1 kHz PCM.
    // piece_index decorrelates the latent noise between the pieces of one utterance.
    std::vector<float> synth_prepared(const Prepared& prepared, size_t piece_index);
    const VoiceStyle& current_voice() const;
    void destroy_graphs() noexcept;  // idempotent; used by the dtor and ctor-failure cleanup

    LiteRtModel         duration_model_  = nullptr;  LiteRtCompiledModel duration_compiled_  = nullptr;
    LiteRtModel         encoder_model_   = nullptr;  LiteRtCompiledModel encoder_compiled_   = nullptr;
    LiteRtModel         vector_model_    = nullptr;  LiteRtCompiledModel vector_compiled_    = nullptr;
    LiteRtModel         vocoder_model_   = nullptr;  LiteRtCompiledModel vocoder_compiled_   = nullptr;

    std::unique_ptr<SupertonicTokenizer>        tokenizer_;
    std::unordered_map<std::string, VoiceStyle> voices_;
    std::string                                 voice_id_ = "F1";

    static constexpr int kTextT          = 128;       // fixed text length (relpos attention)
    static constexpr int kLatentChannels = 144;       // latent_dim(24) * chunk_compress(6)
    static constexpr int kChunkSamples   = 512 * 6;   // 3072 samples per latent frame
    static constexpr int kVecEstLMin     = 17;        // exported vector_estimator floor
    static constexpr int kStyleTtlFloats = 50 * 256;  // 12800
    static constexpr int kStyleDpFloats  = 8 * 16;    // 128
    static constexpr int kSampleRate     = 44100;
    // A piece whose predicted latent length overflows the fixed window by at most this ratio is
    // tempo-fitted into the window (its duration is clamped, so it is spoken up to 10% faster)
    // rather than bisected mid-sentence. Beyond it the planner splits the text.
    static constexpr float kWindowStretchMax = 1.10f;

    int               total_step_      = 8;
    float             speed_           = 1.05f;
    uint32_t          seed_            = 0;     // 0 ⇒ fresh seed per call
    uint32_t          seed_used_       = 0;
    float             chunk_silence_s_ = 0.3f;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
