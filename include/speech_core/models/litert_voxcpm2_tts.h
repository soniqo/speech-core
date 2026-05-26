#pragma once

#include "speech_core/interfaces.h"

#include <tensorflow/lite/c/c_api.h>

#include <string>

namespace speech_core {

/// VoxCPM2 — 2B-parameter multilingual TTS via LiteRT (TFLite).
/// 48 kHz studio-quality output, voice cloning, instruction-driven voice design.
/// Model: https://huggingface.co/aufklarer/VoxCPM2-LiteRT
///
/// Skeleton status (this file): the constructor loads the four LiteRT graphs
/// and verifies the tokenizer file exists, but `synthesize()` throws — the
/// orchestration loop (HF tokenizer → text_prefill → token_step ×N →
/// audio_decode, with explicit K/V cache management) is deferred to a
/// follow-up. The PR CI lanes prove that the wrapper compiles and links
/// cleanly against `libtensorflowlite_c`; the load smoke test (when models
/// are downloaded) proves the four graphs and tokenizer parse.
class LiteRTVoxCPM2Tts : public TTSInterface {
public:
    /// Construct from the four LiteRT graph files + a HuggingFace `tokenizer.json`.
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

private:
    TfLiteModel*       text_prefill_model_   = nullptr;
    TfLiteInterpreter* text_prefill_         = nullptr;
    TfLiteModel*       token_step_model_     = nullptr;
    TfLiteInterpreter* token_step_           = nullptr;
    TfLiteModel*       audio_encoder_model_  = nullptr;
    TfLiteInterpreter* audio_encoder_        = nullptr;
    TfLiteModel*       audio_decoder_model_  = nullptr;
    TfLiteInterpreter* audio_decoder_        = nullptr;

    std::string tokenizer_path_;
    bool        cancelled_ = false;
};

}  // namespace speech_core
