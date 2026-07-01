#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Whisper speech recognition for sherpa-onnx style encoder/decoder exports.
class OnnxWhisperStt : public STTInterface {
public:
    struct Config {
        int sample_rate = 16000;
        int n_fft = 400;
        int hop_length = 160;
        int win_length = 400;

        // sherpa-onnx pads feature frames after the actual audio so Whisper can
        // emit EOT reliably. The encoder is capped at 3000 frames.
        int max_feature_frames = 3000;
        int reserve_tail_frames = 50;
        int tail_padding_frames = 1000;

        // Chunk long inputs before feature extraction. 0 uses
        // (max_feature_frames - reserve_tail_frames) * hop_length.
        size_t max_audio_samples = 0;

        // Empty means auto-detect for multilingual models. Values are language
        // codes from model metadata, e.g. "en", "de", "es", "zh".
        std::string language;
        std::string task = "transcribe";

        // 0 uses sherpa's heuristic: min(audio_seconds * 6, n_text_ctx / 2).
        int max_decode_tokens = 0;
    };

    OnnxWhisperStt(const std::string& encoder_path,
                   const std::string& decoder_path,
                   const std::string& tokens_path,
                   bool hw_accel = true);

    OnnxWhisperStt(const std::string& encoder_path,
                   const std::string& decoder_path,
                   const std::string& tokens_path,
                   const Config& config,
                   bool hw_accel = true);

    ~OnnxWhisperStt() override;

    TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) override;

    int input_sample_rate() const override { return cfg_.sample_rate; }

    /// Set a fixed language prompt. Passing an empty string re-enables
    /// auto-detection on multilingual models.
    bool set_language(const std::string& language);

private:
    struct Metadata {
        int n_mels = 80;
        int n_text_layer = 0;
        int n_text_ctx = 0;
        int n_text_state = 0;
        int n_vocab = 0;
        int sot = 50258;
        int eot = 50257;
        int transcribe = 50360;
        int translate = 50359;
        int no_timestamps = 50364;
        int is_multilingual = 0;
        std::vector<int64_t> sot_sequence;
        std::vector<int32_t> all_language_tokens;
        std::vector<std::string> all_language_codes;
        std::unordered_map<std::string, int32_t> lang2id;
        std::unordered_map<int32_t, std::string> id2lang;
    };

    struct DecodeResult {
        std::string text;
        std::string language;
        float confidence = 1.0f;
    };

    void load_metadata();
    void load_tokens(const std::string& path);

    std::vector<float> compute_features(
        const float* audio, size_t length, int* out_frames) const;
    DecodeResult decode_chunk(const float* audio, size_t length);
    int32_t detect_language(OrtValue* cross_k, OrtValue* cross_v);
    DecodeResult decode_greedy(OrtValue* cross_k, OrtValue* cross_v,
                               int num_feature_frames);

    std::vector<float> make_mel_filterbank() const;
    std::string decode_tokens(const std::vector<int32_t>& ids) const;

    const OrtApi* api_ = nullptr;
    OrtSession* encoder_ = nullptr;
    OrtSession* decoder_ = nullptr;
    Config cfg_;
    Metadata meta_;
    std::vector<std::string> token_table_;
};

}  // namespace speech_core
