#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// NVIDIA Canary (NeMo Conformer-AED) speech recognition.
///
/// FastConformer encoder plus an autoregressive Transformer decoder, so unlike
/// the transducer paths this produces nothing until the utterance ends: the
/// encoder consumes the whole segment, then tokens are decoded one at a time.
/// There is no streaming mode and no partial-result hook.
///
/// The front end is the NeMo contract Parakeet also uses — pre-emphasis,
/// 128-bin log-mel at 512/160/400, centred constant-padded STFT with a
/// symmetric Hann window, Slaney bank, log(x + 2^-24), then per-feature
/// normalisation over the sample variance — because both are FastConformer
/// encoders trained by the same AudioToMelSpectrogramPreprocessor.
///
/// Expects the bundle published at soniqo/Canary-180M-Flash-ONNX:
///   encoder: audio_signal[1,128,T] float, length[1] int64
///            -> encoder_embeddings[1,T',H], encoder_mask[1,T']
///   decoder: input_ids, encoder_embeddings, encoder_mask, decoder_mems
///            -> logits (log-probabilities, newest position only),
///               decoder_hidden_states
///
/// The decode contract — prompt token ids, decoder cache dimensions, the
/// end-of-text id — comes from the decoder graph's ONNX metadata rather than
/// from token-string lookups or input-shape probing. Both of those are
/// silent-failure paths: a prompt token that resolves to -1 makes the decoder
/// emit fluent text that stops after two words or repeats a fragment, which
/// reads like a cache bug.
class OnnxCanaryStt : public STTInterface {
public:
    struct Config {
        int sample_rate     = 16000;
        int n_fft           = 512;
        int hop_length      = 160;
        int win_length      = 400;
        int num_mel_bins    = 128;
        float pre_emphasis  = 0.97f;

        /// Source language token, e.g. "en", "de", "es", "fr".
        std::string language = "en";
        /// Target language. Differing from [language] requests translation.
        std::string target_language = "en";

        /// Decoding stops here even if the model never emits end-of-text,
        /// which a damaged export or pathological audio can cause.
        int max_decode_tokens = 512;
    };

    OnnxCanaryStt(const std::string& encoder_path,
                  const std::string& decoder_path,
                  const std::string& vocab_path,
                  bool hw_accel = true);

    OnnxCanaryStt(const std::string& encoder_path,
                  const std::string& decoder_path,
                  const std::string& vocab_path,
                  const Config& config,
                  bool hw_accel = true);

    ~OnnxCanaryStt() override;

    TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) override;

    int input_sample_rate() const override { return cfg_.sample_rate; }

    /// Stop the decode loop at the next token. Thread-safe.
    void cancel() override;

    /// Change the source language between utterances. Returns false if the
    /// bundle has no prompt token for it.
    bool set_language(const std::string& language);

    /// Change the target language; differing from the source requests
    /// translation. Returns false if the bundle has no prompt token for it.
    bool set_target_language(const std::string& language);

private:
    void load_vocab(const std::string& path);
    void load_decode_contract();

    std::vector<float> compute_features(const float* audio, size_t length) const;

    /// The bundle's decode prompt with the configured language pair patched in.
    std::vector<int64_t> build_prompt() const;

    /// SentencePiece pieces to text: "▁" marks a word boundary.
    std::string detokenize(const std::vector<int64_t>& ids) const;

    Config cfg_;
    OrtSession* encoder_ = nullptr;
    OrtSession* decoder_ = nullptr;
    const OrtApi* api_ = nullptr;

    std::unordered_map<int64_t, std::string> vocab_;
    /// Language code to prompt token id, e.g. "de" -> 76.
    std::unordered_map<std::string, int64_t> language_tokens_;

    /// Decode contract, read from the decoder graph's metadata.
    std::vector<int64_t> prompt_template_;
    size_t prompt_source_index_ = 0;
    size_t prompt_target_index_ = 0;
    int64_t eos_id_ = -1;
    int64_t mem_layers_ = 0;
    int64_t mem_width_ = 0;
    bool logits_are_log_probs_ = false;

    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
