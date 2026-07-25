#include "speech_core/models/onnx_redimnet_speaker_embedding.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace speech_core {
namespace {

struct OrtValueHandle {
    const OrtApi* api = nullptr;
    OrtValue* value = nullptr;
    ~OrtValueHandle() {
        if (api && value) api->ReleaseValue(value);
    }
};

std::vector<std::string> io_names(
    const OrtApi* api, OrtSession* session, bool inputs) {
    OrtAllocator* allocator = nullptr;
    ort_check(api, api->GetAllocatorWithDefaultOptions(&allocator));
    std::size_t count = 0;
    if (inputs) {
        ort_check(api, api->SessionGetInputCount(session, &count));
    } else {
        ort_check(api, api->SessionGetOutputCount(session, &count));
    }
    std::vector<std::string> result;
    for (std::size_t index = 0; index < count; ++index) {
        char* name = nullptr;
        if (inputs) {
            ort_check(api, api->SessionGetInputName(
                session, index, allocator, &name));
        } else {
            ort_check(api, api->SessionGetOutputName(
                session, index, allocator, &name));
        }
        result.emplace_back(name);
        OrtStatus* status = api->AllocatorFree(allocator, name);
        if (status) api->ReleaseStatus(status);
    }
    return result;
}

std::pair<std::vector<int64_t>, ONNXTensorElementDataType> tensor_contract(
    const OrtApi* api, OrtSession* session, std::size_t index,
    bool input) {
    OrtTypeInfo* type_info = nullptr;
    if (input) {
        ort_check(api, api->SessionGetInputTypeInfo(
            session, index, &type_info));
    } else {
        ort_check(api, api->SessionGetOutputTypeInfo(
            session, index, &type_info));
    }
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    OrtStatus* cast =
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (cast || !tensor_info) {
        if (cast) api->ReleaseStatus(cast);
        api->ReleaseTypeInfo(type_info);
        throw std::runtime_error("ReDimNet graph I/O must be tensors");
    }
    ONNXTensorElementDataType type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::size_t rank = 0;
    ort_check(api, api->GetTensorElementType(tensor_info, &type));
    ort_check(api, api->GetDimensionsCount(tensor_info, &rank));
    std::vector<int64_t> shape(rank);
    ort_check(api, api->GetDimensions(tensor_info, shape.data(), rank));
    api->ReleaseTypeInfo(type_info);
    return {std::move(shape), type};
}

}  // namespace

OnnxReDimNetSpeakerEmbedding::OnnxReDimNetSpeakerEmbedding(
    const std::string& model_path, bool hardware_acceleration) {
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    session_ = engine.load(model_path, hardware_acceleration);
    try {
        validate_contract();
    } catch (...) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
        throw;
    }
}

OnnxReDimNetSpeakerEmbedding::~OnnxReDimNetSpeakerEmbedding() {
    if (session_) api_->ReleaseSession(session_);
}

void OnnxReDimNetSpeakerEmbedding::validate_contract() {
    if (io_names(api_, session_, true) != std::vector<std::string>{"audio"}
        || io_names(api_, session_, false)
            != std::vector<std::string>{"embedding"}) {
        throw std::runtime_error(
            "ReDimNet ONNX graph has incompatible I/O names");
    }
    const auto input = tensor_contract(api_, session_, 0, true);
    const auto output = tensor_contract(api_, session_, 0, false);
    if (input.first != std::vector<int64_t>{1, 96000}
        || output.first != std::vector<int64_t>{1, 192}
        || input.second != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output.second != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "ReDimNet ONNX graph has incompatible tensor shapes or types");
    }
}

std::vector<float> OnnxReDimNetSpeakerEmbedding::prepare_audio(
    const float* samples, std::size_t length,
    std::size_t minimum_samples) {
    if (!samples && length > 0) {
        throw std::invalid_argument("ReDimNet audio buffer is null");
    }
    if (minimum_samples == 0) {
        throw std::invalid_argument(
            "ReDimNet minimum audio duration must be positive");
    }
    if (length < minimum_samples) {
        throw std::invalid_argument(
            "ReDimNet speaker identity has insufficient clean audio");
    }
    for (std::size_t index = 0; index < length; ++index) {
        if (!std::isfinite(samples[index])) {
            throw std::invalid_argument(
                "ReDimNet audio contains a non-finite sample");
        }
    }

    std::vector<float> prepared(kInputSamples);
    if (length == kInputSamples) {
        std::copy_n(samples, length, prepared.begin());
    } else if (length > kInputSamples) {
        const std::size_t start = (length - kInputSamples) / 2;
        std::copy_n(samples + start, kInputSamples, prepared.begin());
    } else {
        for (std::size_t index = 0; index < kInputSamples; ++index) {
            prepared[index] = samples[index % length];
        }
    }
    return prepared;
}

std::vector<float> OnnxReDimNetSpeakerEmbedding::embed(
    const float* audio, std::size_t length, int sample_rate) {
    return embed_with_minimum(
        audio, length, sample_rate, kMinimumSamples);
}

std::vector<float>
OnnxReDimNetSpeakerEmbedding::embed_short_utterance(
    const float* audio, std::size_t length, int sample_rate) {
    return embed_with_minimum(
        audio, length, sample_rate, kMinimumShortSamples);
}

std::vector<float> OnnxReDimNetSpeakerEmbedding::embed_with_minimum(
    const float* audio, std::size_t length, int sample_rate,
    std::size_t minimum_samples) {
    if (sample_rate <= 0) {
        throw std::invalid_argument(
            "ReDimNet sample rate must be positive");
    }
    if (!audio && length > 0) {
        throw std::invalid_argument(
            "ReDimNet audio buffer is null");
    }
    if (length == 0) {
        throw std::invalid_argument(
            "ReDimNet speaker identity has insufficient clean audio");
    }
    std::vector<float> resampled;
    if (sample_rate != kSampleRate) {
        resampled = Resampler::resample(
            audio, length, sample_rate, kSampleRate);
        audio = resampled.data();
        length = resampled.size();
    }
    std::vector<float> prepared =
        prepare_audio(audio, length, minimum_samples);

    const int64_t input_shape[] = {1, 96000};
    OrtValue* input = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        OnnxEngine::get().cpu_memory(),
        prepared.data(), prepared.size() * sizeof(float),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
    OrtValueHandle input_handle{api_, input};
    const char* input_names[] = {"audio"};
    const char* output_names[] = {"embedding"};
    OrtValue* inputs[] = {input};
    OrtValue* outputs[] = {nullptr};
    ort_check(api_, api_->Run(
        session_, nullptr, input_names, inputs, 1,
        output_names, 1, outputs));
    OrtValueHandle output_handle{api_, outputs[0]};

    float* values = nullptr;
    ort_check(api_, api_->GetTensorMutableData(
        outputs[0], reinterpret_cast<void**>(&values)));
    std::vector<float> embedding(
        values, values + kEmbeddingDimension);
    float squared_norm = 0.0f;
    for (const float value : embedding) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "ReDimNet returned a non-finite embedding");
        }
        squared_norm += value * value;
    }
    const float norm = std::sqrt(squared_norm);
    if (!(norm > 1e-8f)) {
        throw std::runtime_error("ReDimNet returned a zero embedding");
    }
    for (float& value : embedding) value /= norm;
    return embedding;
}

}  // namespace speech_core
