#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Parakeet TDT v3 (0.6B) — speech recognition via LiteRT.
/// FastConformer encoder + fused LSTM decoder/joint network.
/// Two .tflite files: parakeet-encoder.tflite + parakeet-decoder-joint.tflite.
/// Input: PCM Float32 audio at 16 kHz. Output: transcribed text + language.
class LiteRTParakeetStt : public STTInterface {
public:
    struct Config {
        int num_mel_bins    = 128;
        int sample_rate     = 16000;
        int n_fft           = 512;
        int hop_length      = 160;
        int win_length      = 400;
        float pre_emphasis  = 0.97f;
        int enc_mel_frames  = 500;  // fixed encoder time dim — soniqo export is [1,128,500]
        int encoder_hidden  = 1024;
        int decoder_hidden  = 640;
        int decoder_layers  = 2;
        int vocab_size      = 1024;
        int blank_id        = 1024;
        int num_dur_bins    = 5;
        int duration_bins[5] = {0, 1, 2, 3, 4};
        int total_logits    = 1030;
        int first_text_token = 0;
    };

    LiteRTParakeetStt(const std::string& encoder_path,
                      const std::string& decoder_joint_path,
                      const std::string& vocab_path,
                      bool hw_accel = true);
    ~LiteRTParakeetStt() override;

    // --- STTInterface (batch) ---
    TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }

    // --- STTInterface (streaming) ---
    bool supports_streaming() const override { return true; }
    void begin_stream(int sample_rate) override;
    PartialResult push_chunk(const float* audio, size_t length) override;
    void flush_stream() override;
    TranscriptionResult end_stream() override;
    void cancel_stream() override;

    /// Restrict Parakeet's language-token choice to one language.
    /// Accepts ISO codes or BCP-47 tags ("en", "en-US"). "auto" clears it.
    bool set_language(const std::string& language);

    /// Restrict Parakeet's language-token choice to this shortlist.
    /// Unknown languages are ignored; returns false if none resolve.
    bool set_allowed_languages(const std::vector<std::string>& languages);
    void clear_language_guidance();

private:
    struct DecodeResult {
        std::string text;
        std::string language;
        float confidence = 0.0f;
    };

    bool load_vocab(const std::string& path);
    std::vector<float> compute_mel(const float* audio, size_t length);
    DecodeResult decode(const float* audio, size_t length);
    DecodeResult tdt_decode(const float* encoded, int64_t enc_len, int64_t hidden);
    std::string decode_tokens(const std::vector<int>& token_ids);

    LiteRtModel         enc_model_    = nullptr;
    LiteRtCompiledModel enc_compiled_ = nullptr;
    LiteRtModel         dec_model_    = nullptr;
    LiteRtCompiledModel dec_compiled_ = nullptr;
    Config cfg_;

    std::unordered_map<int, std::string> vocab_;
    std::unordered_map<int, std::string> lang_tokens_;
    std::vector<int> guided_lang_tokens_;

    std::vector<float> stream_buffer_;
    int  stream_sample_rate_ = 16000;
    bool streaming_          = false;
};

}  // namespace speech_core
