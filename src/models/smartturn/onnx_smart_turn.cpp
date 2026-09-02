#include "speech_core/models/onnx_smart_turn.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
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
    const OrtApi* api, OrtSession* session, std::size_t index, bool input) {
    OrtTypeInfo* type_info = nullptr;
    if (input) {
        ort_check(api, api->SessionGetInputTypeInfo(
            session, index, &type_info));
    } else {
        ort_check(api, api->SessionGetOutputTypeInfo(
            session, index, &type_info));
    }
    const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
    OrtStatus* cast = api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (cast || !tensor_info) {
        if (cast) api->ReleaseStatus(cast);
        api->ReleaseTypeInfo(type_info);
        throw std::runtime_error("Smart Turn graph I/O must be tensors");
    }
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::size_t rank = 0;
    ort_check(api, api->GetTensorElementType(tensor_info, &type));
    ort_check(api, api->GetDimensionsCount(tensor_info, &rank));
    std::vector<int64_t> shape(rank);
    ort_check(api, api->GetDimensions(tensor_info, shape.data(), rank));
    api->ReleaseTypeInfo(type_info);
    return {std::move(shape), type};
}

}  // namespace

OnnxSmartTurn::OnnxSmartTurn(
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
    window_.assign(kWindowSamples, 0.0f);
}

OnnxSmartTurn::~OnnxSmartTurn() {
    if (session_) api_->ReleaseSession(session_);
}

void OnnxSmartTurn::validate_contract() {
    if (io_names(api_, session_, true) != std::vector<std::string>{"audio"}
        || io_names(api_, session_, false)
            != std::vector<std::string>{"probability"}) {
        throw std::runtime_error(
            "Smart Turn ONNX graph has incompatible I/O names "
            "(expected audio -> probability; use the soniqo audio-input export)");
    }
    const auto input = tensor_contract(api_, session_, 0, true);
    const auto output = tensor_contract(api_, session_, 0, false);
    if (input.first != std::vector<int64_t>{1, static_cast<int64_t>(kWindowSamples)}
        || output.first != std::vector<int64_t>{1, 1}
        || input.second != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || output.second != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "Smart Turn ONNX graph has incompatible tensor shapes or types");
    }
}

std::vector<float> OnnxSmartTurn::prepare_window(
    const float* samples, std::size_t length) {
    if (!samples && length > 0) {
        throw std::invalid_argument("Smart Turn audio buffer is null");
    }
    std::vector<float> window(kWindowSamples, 0.0f);
    if (length >= kWindowSamples) {
        std::copy_n(samples + (length - kWindowSamples), kWindowSamples,
                    window.begin());
    } else if (length > 0) {
        std::copy_n(samples, length, window.begin() + (kWindowSamples - length));
    }
    return window;
}

float OnnxSmartTurn::turn_complete_probability(
    const float* samples, std::size_t length, int sample_rate) {
    if (sample_rate <= 0) {
        throw std::invalid_argument("Smart Turn sample rate must be positive");
    }
    if (!samples && length > 0) {
        throw std::invalid_argument("Smart Turn audio buffer is null");
    }
    std::vector<float> resampled;
    if (sample_rate != kSampleRate) {
        resampled = Resampler::resample(samples, length, sample_rate, kSampleRate);
        samples = resampled.data();
        length = resampled.size();
    }
    // Only the last 8 s matter, so resampling a long turn first is wasteful but
    // rare (the pipeline caps utterances well below that on 16 kHz input).
    std::fill(window_.begin(), window_.end(), 0.0f);
    if (length >= kWindowSamples) {
        std::copy_n(samples + (length - kWindowSamples), kWindowSamples,
                    window_.begin());
    } else if (length > 0) {
        std::copy_n(samples, length, window_.begin() + (kWindowSamples - length));
    }
    for (const float value : window_) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "Smart Turn audio contains a non-finite sample");
        }
    }

    const int64_t input_shape[] = {1, static_cast<int64_t>(kWindowSamples)};
    OrtValue* input = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        OnnxEngine::get().cpu_memory(),
        window_.data(), window_.size() * sizeof(float),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
    OrtValueHandle input_handle{api_, input};
    const char* input_names[] = {"audio"};
    const char* output_names[] = {"probability"};
    OrtValue* inputs[] = {input};
    OrtValue* outputs[] = {nullptr};
    ort_check(api_, api_->Run(
        session_, nullptr, input_names, inputs, 1, output_names, 1, outputs));
    OrtValueHandle output_handle{api_, outputs[0]};

    float* values = nullptr;
    ort_check(api_, api_->GetTensorMutableData(
        outputs[0], reinterpret_cast<void**>(&values)));
    const float probability = values[0];
    if (!std::isfinite(probability)) {
        throw std::runtime_error("Smart Turn returned a non-finite probability");
    }
    return std::min(1.0f, std::max(0.0f, probability));
}

}  // namespace speech_core
