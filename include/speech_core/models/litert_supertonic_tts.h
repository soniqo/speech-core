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
/// (0.3 s) is inserted only where a sentence ends. A piece that continues into the next one is
/// tokenized with a trailing "," rather than a sentence-final ".", and the seam between the two
/// is trimmed to a comma-length pause.
///
/// **Latent buckets.** Because L is fixed per graph, a bundle may ship extra exports of the two
/// L-dependent graphs next to the base ones — `vector_estimator_L128.tflite` + `vocoder_L128.tflite`
/// (≈ 9 s), and so on. They are discovered at construction and loaded lazily; each piece runs on
/// the smallest bucket whose window holds its predicted duration, so a long sentence is generated
/// in one pass (one coherent prosodic contour) and a split only remains for text longer than the
/// largest bucket. Short pieces keep using the cheap base graph.
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

    /// Select the voice (e.g. "F1", "M3"). Default "F1" (or the first loaded
    /// style when F1 is absent); an empty id restores that default. Throws
    /// std::invalid_argument if unknown.
    void set_voice(const std::string& voice_id) override;

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

    /// Latent-window buckets in the bundle, in frames of 3072 samples, ascending; [0] is the base
    /// graph pair (L=64 for the published bundle), the rest are optional `*_L{N}.tflite` siblings.
    std::vector<int> latent_buckets() const;

    /// Index into `buckets` (ascending frames) of the smallest window that holds `frames`, or the
    /// last one when none does (the caller then tempo-fits or truncates). Pure; unit-tested.
    static size_t choose_latent_bucket(const std::vector<int>& buckets, int frames);

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

    // One fixed-L export of the two L-dependent graphs. Handles are raw (freed in destroy_graphs()).
    struct LatentBucket {
        int                 frames = 0;             // L
        std::string         vector_path, vocoder_path;
        LiteRtModel         vector_model  = nullptr; LiteRtCompiledModel vector_compiled  = nullptr;
        LiteRtModel         vocoder_model = nullptr; LiteRtCompiledModel vocoder_compiled = nullptr;
        bool                failed = false;         // load attempted and failed; skipped from then on
        bool loaded() const { return vector_compiled && vocoder_compiled; }
    };

    // text_encoder → vector_estimator × N → vocoder on a prepared piece → trimmed 44.1 kHz PCM.
    // piece_index decorrelates the latent noise between the pieces of one utterance.
    std::vector<float> synth_prepared(const Prepared& prepared, size_t piece_index,
                                      const LatentBucket& bucket);
    // Smallest loadable bucket holding `frames` (loads it on first use; falls back on failure).
    const LatentBucket& bucket_for(int frames);
    void load_bucket(LatentBucket& bucket);
    void discover_buckets(const std::string& vector_estimator_path, const std::string& vocoder_path);
    const VoiceStyle& current_voice() const;
    void destroy_graphs() noexcept;  // idempotent; used by the dtor and ctor-failure cleanup

    LiteRtModel         duration_model_  = nullptr;  LiteRtCompiledModel duration_compiled_  = nullptr;
    LiteRtModel         encoder_model_   = nullptr;  LiteRtCompiledModel encoder_compiled_   = nullptr;
    std::vector<LatentBucket> buckets_;              // ascending frames; [0] = base, loaded in the ctor
    bool                hw_accel_        = false;

    std::unique_ptr<SupertonicTokenizer>        tokenizer_;
    std::unordered_map<std::string, VoiceStyle> voices_;
    std::string                                 voice_id_ = "F1";
    std::string                                 default_voice_id_;  // resolved in the ctor

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
    // At a forced intra-sentence split the model's own utterance padding on both pieces would
    // leave a ~700 ms gap; the seam is trimmed to this much tail + head silence (150 ms total,
    // chosen by ear against 250 ms).
    static constexpr int kSeamTailMs = 100;
    static constexpr int kSeamHeadMs = 50;

    int               total_step_      = 8;
    float             speed_           = 1.05f;
    uint32_t          seed_            = 0;     // 0 ⇒ fresh seed per call
    uint32_t          seed_used_       = 0;
    float             chunk_silence_s_ = 0.3f;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
