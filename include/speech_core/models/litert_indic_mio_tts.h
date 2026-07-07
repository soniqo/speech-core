#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/indic_mio_tokenizer.h"
#include "speech_core/models/litert_engine.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace speech_core {

class IndicMioIstft;

/// Indic-Mio — Hindi/Indic emotion TTS (Qwen3-0.6B speech-token LM +
/// MioCodec) via LiteRT. 24 kHz output. Emotion is inline text (suffix tags
/// like `<happy>`); voice cloning conditions on a 128-dim global speaker
/// embedding cached by set_reference().
/// Model: https://huggingface.co/soniqo/Indic-Mio-LiteRT — the model card
/// documents the host contract this class implements.
///
/// Pipeline (four LiteRT graphs orchestrated here):
///   text_prefill  : (ids [1,64], last_index)      → (logits [1,164480], KV [28,1,8,512,128] ×2)
///   token_step ×N : (id [1,1], pos [1,1], write_index, KV) → (logits, KV)
///   audio_decoder : (codes [1,384], global [1,128], valid_tokens)
///                   → (STFT real/imag [1,961,768])          — host ISTFT follows
///   ref_encoder   : (audio24k [1,240000])         → (global [1,128])  — cloning only
///
/// The host owns the AR loop: sample (temperature/top-k/top-p/repetition
/// penalty, EOS suppressed until the first speech token), stop on
/// <|im_end|>/<|endoftext|>, map speech codes as id-151669, decode with
/// valid_tokens masking, then run the "same"-padding ISTFT (IndicMioIstft)
/// and trim to tokens×960 samples.
///
/// Memory policy mirrors LiteRTVoxCPM2Tts: token_step / audio_decoder /
/// ref_encoder stay resident; the 1.1 GB text_prefill graph is loaded for the
/// one prefill call per synthesize() and released after.
class LiteRTIndicMioTts : public TTSInterface {
public:
    LiteRTIndicMioTts(const std::string& text_prefill_path,
                      const std::string& token_step_path,
                      const std::string& audio_decoder_path,
                      const std::string& ref_encoder_path,
                      const std::string& tokenizer_path,
                      bool hw_accel = false);
    ~LiteRTIndicMioTts() override;

    // --- TTSInterface (language is implicit in the text for this model) ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    /// Condition on a reference clip (mono float PCM at `sample_rate`).
    /// Resampled to 24 kHz and center-cropped (preferred) or symmetrically
    /// zero-padded to the encoder's 10 s window, then encoded once; the
    /// 128-float embedding is cached on the handle. Throws on encoder failure.
    void set_reference(const float* pcm, size_t length, int sample_rate);
    void clear_reference();

    // --- Sampling (defaults = the upstream reference) ---
    void set_temperature(float t) { temperature_ = t; }
    void set_top_k(int k) { top_k_ = k; }
    void set_top_p(float p) { top_p_ = p; }
    void set_repetition_penalty(float p) { repetition_penalty_ = p; }
    /// Clamped to the KV budget (512-64) and the decoder bucket (384).
    void set_max_new_tokens(int n);
    /// 0 → fresh random seed per synthesize() (reported by seed_used()).
    void set_seed(uint32_t seed) { seed_ = seed; }

    int max_text_tokens() const { return kPromptLen; }
    int tokens_generated() const { return tokens_generated_; }
    bool stopped_on_eos() const { return stopped_on_eos_; }
    uint32_t seed_used() const { return seed_used_; }

    // Geometry / vocabulary constants (the published bundle's manifest).
    static constexpr int kPromptLen        = 64;
    static constexpr int kMaxSeq           = 512;
    static constexpr int kLayers           = 28;
    static constexpr int kKvHeads          = 8;
    static constexpr int kHeadDim          = 128;
    static constexpr int kVocab            = 164480;
    static constexpr int kDecoderTokens    = 384;
    static constexpr int kNfft             = 1920;
    static constexpr int kHop              = 480;
    static constexpr int kSamplesPerToken  = 960;
    static constexpr int kGlobalDim        = 128;
    static constexpr int kRefSamples       = 240000;  // 10 s @ 24 kHz
    static constexpr int kImStartId        = 151644;
    static constexpr int kImEndId          = 151645;
    static constexpr int kEndOfTextId      = 151643;
    static constexpr int kSpeechOffset     = 151669;
    static constexpr int kSpeechCount      = 12800;

private:
    void load_text_prefill();
    void release_text_prefill();
    std::vector<int64_t> build_prompt(const std::string& text, int& real_len) const;

    // Resident graphs.
    LiteRtModel         token_step_model_      = nullptr;
    LiteRtCompiledModel token_step_compiled_   = nullptr;
    LiteRtModel         decoder_model_         = nullptr;
    LiteRtCompiledModel decoder_compiled_      = nullptr;
    LiteRtModel         ref_encoder_model_     = nullptr;
    LiteRtCompiledModel ref_encoder_compiled_  = nullptr;

    // Lazy-loaded per synthesize() (1.1 GB — one call per synthesis).
    std::string         text_prefill_path_;
    bool                text_prefill_hw_accel_ = false;
    LiteRtModel         text_prefill_model_    = nullptr;
    LiteRtCompiledModel text_prefill_compiled_ = nullptr;
    std::mutex          prefill_mutex_;

    std::unique_ptr<IndicMioTokenizer> tokenizer_;
    std::unique_ptr<IndicMioIstft>     istft_;

    std::vector<float> global_embedding_;  // empty → zeros (default voice)

    float    temperature_        = 0.9f;
    int      top_k_              = 50;
    float    top_p_              = 0.9f;
    float    repetition_penalty_ = 1.0f;
    int      max_new_tokens_     = kDecoderTokens;
    uint32_t seed_               = 0;

    std::atomic<bool> cancelled_{false};
    int      tokens_generated_ = 0;
    bool     stopped_on_eos_   = false;
    uint32_t seed_used_        = 0;
};

}  // namespace speech_core
