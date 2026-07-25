#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

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
/// The front end is the same NeMo contract Parakeet uses — pre-emphasis,
/// 128-bin log-mel at 512/160/400, then per-feature normalisation — because
/// both are FastConformer encoders trained by the same preprocessor.
///
/// Expects the ONNX export published for onnx-asr (istupakov/canary-*-onnx):
///   encoder: audio_signal[1,128,T] int64 length[1]
///            -> encoder_embeddings, encoder_mask
///   decoder: input_ids, encoder_embeddings, encoder_mask, decoder_mems
///            -> logits, decoder_hidden_states
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
        /// Emit punctuation and capitalisation.
        bool punctuation = true;

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

    /// Change the source language prompt token between utterances. Returns
    /// false if the token is not in the vocabulary.
    bool set_language(const std::string& language);

private:
    void load_vocab(const std::string& path);
    std::vector<float> compute_features(const float* audio, size_t length) const;

    /// The ten control tokens Canary expects before any transcript.
    std::vector<int64_t> build_prompt() const;

    /// Token id for a control token such as "<|en|>", or -1 if absent.
    int64_t token_id(const std::string& token) const;

    /// SentencePiece pieces to text: "▁" marks a word boundary.
    std::string detokenize(const std::vector<int64_t>& ids) const;

    Config cfg_;
    OrtSession* encoder_ = nullptr;
    OrtSession* decoder_ = nullptr;
    const OrtApi* api_ = nullptr;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    int64_t eos_id_ = -1;

    /// Layer count and hidden width of decoder_mems, read from the export so
    /// the initial empty cache matches whatever model was loaded.
    int64_t mem_layers_ = 0;
    int64_t mem_width_ = 0;
};

}  // namespace speech_core
