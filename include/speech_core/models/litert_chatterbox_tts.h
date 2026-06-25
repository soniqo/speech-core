#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/chatterbox_tokenizer.h"
#include "speech_core/models/litert_engine.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace speech_core {

/// Chatterbox Multilingual TTS via LiteRT — 24 kHz, 23 languages, zero-shot
/// default voice. Model: https://huggingface.co/soniqo/Chatterbox-LiteRT
///
/// Pipeline (five LiteRT graphs + shipped default-voice conditioning):
///   tokenizer            : text -> BPE token IDs (+ sot/eot wrap)
///   t3_prefill           : (text_ids, cond_emb) -> first-step KV cache
///   t3_step  ×N (AR)     : (speech_token, pos, cache) -> next-token logits + cache
///                          sampled (temp/top-p/min-p/rep-pen); BOS is step 0
///   flow_encoder         : (prompt_token ++ speech_tokens) -> mu
///   flow_estimator ×10   : Euler CFM with CFG (batch-2) -> mel
///   hift                 : mel -> 24 kHz PCM
///
/// The T3 graphs are batch=1; T3 classifier-free guidance (cfg_weight) needs an
/// uncond prefill variant and is a follow-up — cfg_weight defaults to 0 here.
/// The flow side already runs CFG (estimator batch-2, cfg_rate 0.7).
class LiteRTChatterboxTts : public TTSInterface {
public:
    /// Construct from a bundle directory containing the five `chatterbox-*.tflite`
    /// graphs, the tokenizer JSONs, `cond_emb.bin`, and the flow conditioning
    /// (`spks.bin` / `prompt_token.bin` / `prompt_feat.bin`).
    explicit LiteRTChatterboxTts(const std::string& bundle_dir, bool hw_accel = false);
    ~LiteRTChatterboxTts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    void set_temperature(float t) { temperature_ = t; }
    void set_top_p(float p) { top_p_ = p; }
    void set_min_p(float p) { min_p_ = p; }
    void set_repetition_penalty(float r) { repetition_penalty_ = r; }
    void set_max_tokens(int n) { max_tokens_ = n; }
    /// 0 (default) -> a fresh random seed per call (reported by seed_used()).
    void set_seed(uint32_t s) { seed_ = s; }
    uint32_t seed_used() const { return seed_used_; }
    int tokens_generated() const { return tokens_generated_; }

private:
    struct Graph;  // signature-mapped LiteRT graph (impl detail)
    std::unique_ptr<Graph> prefill_, step_, enc_, est_, hift_;
    std::unique_ptr<ChatterboxTokenizer> tok_;

    std::vector<float>   cond_emb_;       // [34*1024] T3 conditioning
    std::vector<float>   spks_;           // [80] flow speaker embedding
    std::vector<int32_t> prompt_token_;   // [157] flow reference token prefix
    std::vector<float>   prompt_feat_;    // [314*80] flow reference mel
    int cond_len_ = 34, prompt_feat_frames_ = 314;

    std::vector<int> generate_speech_tokens(const std::vector<long>& text_ids, std::mt19937& rng);
    std::vector<float> flow_to_wav(const std::vector<int>& speech_tokens, std::mt19937& rng);

    // sampling / generation params. Default temperature 0 = greedy, which reliably
    // hits the stop token; sampling (temperature > 0) can over-generate until the
    // alignment-stream-analyzer (force-EOS on text-alignment completion) is ported.
    float temperature_ = 0.0f, top_p_ = 0.95f, min_p_ = 0.05f, repetition_penalty_ = 1.2f;
    int   max_tokens_ = 1000;
    uint32_t seed_ = 0, seed_used_ = 0;
    int   tokens_generated_ = 0;
    std::atomic<bool> cancelled_{false};

    // bundle-invariant shape constants (T3)
    static constexpr int kHidden = 1024, kLayers = 30, kKv = 16, kHeadDim = 64;
    static constexpr int kVocab = 8194, kPrefillT = 512, kTextPad = 478;
    static constexpr int kStepCache = 1536, kBos = 6561, kStop = 6562;
    static constexpr int kStartText = 255, kStopText = 0;
    // flow
    static constexpr int kMel = 80, kFlowT = 1024, kHop = 480;
    static constexpr float kCfgRate = 0.7f;
};

}  // namespace speech_core
