#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/moss_prompt_processor.h"
#include "speech_core/models/moss_whisper_features.h"

#include <onnxruntime_c_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

namespace speech_core {

/// Portable MOSS-Transcribe-Diarize runtime for the validated two-graph ONNX
/// bundle published by speech-models.
class OnnxMossTranscribeDiarize final
    : public TranscribeDiarizeInterface {
public:
    struct Config {
        int max_new_tokens = 512;
        int audio_intra_threads = 0;
        int decoder_intra_threads = 0;
        bool audio_hardware_acceleration = true;
        bool decoder_hardware_acceleration = false;
    };

    struct Profile {
        double total_ms = 0.0;
        double feature_ms = 0.0;
        double audio_encoder_ms = 0.0;
        double decoder_prefill_ms = 0.0;
        double decoder_generate_ms = 0.0;
        int prompt_tokens = 0;
        int generated_tokens = 0;
        int audio_chunks = 0;
    };

    explicit OnnxMossTranscribeDiarize(
        const std::string& bundle_directory);
    OnnxMossTranscribeDiarize(
        const std::string& bundle_directory, const Config& config);
    ~OnnxMossTranscribeDiarize() override;

    OnnxMossTranscribeDiarize(
        const OnnxMossTranscribeDiarize&) = delete;
    OnnxMossTranscribeDiarize& operator=(
        const OnnxMossTranscribeDiarize&) = delete;

    DiarizedTranscriptionResult transcribe_diarized(
        const float* audio, std::size_t length, int sample_rate) override;

    int input_sample_rate() const override {
        return MossWhisperFeatureExtractor::kSampleRate;
    }

    void cancel() override;
    Profile last_profile() const;

private:
    void validate_bundle(const std::string& bundle_directory) const;
    void validate_graph_contracts();

    const OrtApi* api_ = nullptr;
    OrtSession* audio_encoder_ = nullptr;
    OrtSession* decoder_ = nullptr;
    Config config_;
    MossWhisperFeatureExtractor feature_extractor_;
    MossTokenizerDecoder tokenizer_;
    ONNXTensorElementDataType audio_type_ =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ONNXTensorElementDataType decoder_type_ =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::atomic<bool> cancel_requested_{false};
    mutable std::mutex inference_mutex_;
    mutable std::mutex profile_mutex_;
    Profile last_profile_;
};

}  // namespace speech_core
