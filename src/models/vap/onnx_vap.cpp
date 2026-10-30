#include "speech_core/models/onnx_vap.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace speech_core {
namespace {

struct ValueGuard {
    const OrtApi* api;
    OrtValue** values;
    std::size_t count;
    ~ValueGuard() {
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i]) api->ReleaseValue(values[i]);
        }
    }
};

struct TypeGuard {
    const OrtApi* api;
    OrtTypeInfo* value;
    ~TypeGuard() { if (value) api->ReleaseTypeInfo(value); }
};

void check_tensor(const OrtApi* api, OrtSession* session, bool input,
                  std::size_t index, const char* expected_name,
                  const std::vector<int64_t>& expected_shape) {
    OrtAllocator* allocator = nullptr;
    ort_check(api, api->GetAllocatorWithDefaultOptions(&allocator));
    char* name = nullptr;
    ort_check(api, input ? api->SessionGetInputName(session, index, allocator, &name)
                         : api->SessionGetOutputName(session, index, allocator, &name));
    const std::string actual_name(name);
    ort_check(api, api->AllocatorFree(allocator, name));
    if (actual_name != expected_name) throw std::runtime_error("VAP graph has incompatible I/O names");
    OrtTypeInfo* info = nullptr;
    ort_check(api, input ? api->SessionGetInputTypeInfo(session, index, &info)
                         : api->SessionGetOutputTypeInfo(session, index, &info));
    TypeGuard guard{api, info};
    const OrtTensorTypeAndShapeInfo* tensor = nullptr;
    ort_check(api, api->CastTypeInfoToTensorInfo(info, &tensor));
    if (!tensor) throw std::runtime_error("VAP graph I/O must be tensors");
    ONNXTensorElementDataType type;
    std::size_t rank = 0;
    ort_check(api, api->GetTensorElementType(tensor, &type));
    ort_check(api, api->GetDimensionsCount(tensor, &rank));
    std::vector<int64_t> shape(rank);
    ort_check(api, api->GetDimensions(tensor, shape.data(), rank));
    if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape != expected_shape) {
        throw std::runtime_error("VAP graph has incompatible tensor shapes or types");
    }
}

}  // namespace

OnnxVap::OnnxVap(const std::string& model_path, bool hardware_acceleration) {
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    session_ = engine.load(model_path, hardware_acceleration);
    try { validate_contract(); }
    catch (...) {
        api_->ReleaseSession(session_);
        session_ = nullptr;
        throw;
    }
}

OnnxVap::~OnnxVap() {
    if (session_) api_->ReleaseSession(session_);
}

void OnnxVap::validate_contract() {
    std::size_t inputs = 0, outputs = 0;
    ort_check(api_, api_->SessionGetInputCount(session_, &inputs));
    ort_check(api_, api_->SessionGetOutputCount(session_, &outputs));
    if (inputs != 1 || outputs != 3) throw std::runtime_error("VAP graph needs one input and three outputs");
    check_tensor(api_, session_, true, 0, "audio", {1, 2, static_cast<int64_t>(kWindowSamples)});
    const char* names[] = {"p_now", "p_future", "vad"};
    for (std::size_t i = 0; i < 3; ++i) {
        check_tensor(api_, session_, false, i, names[i], {1, static_cast<int64_t>(kFrames), 2});
    }
}

std::vector<float> OnnxVap::prepare_window(
    const float* user, const float* agent, std::size_t length) {
    if (!user || !agent || length == 0) {
        throw std::invalid_argument("VAP requires two nonempty synchronized channels");
    }
    for (std::size_t i = 0; i < length; ++i) {
        if (!std::isfinite(user[i]) || !std::isfinite(agent[i])) {
            throw std::invalid_argument("VAP audio contains a non-finite sample");
        }
    }
    std::vector<float> result(2 * kWindowSamples, 0.0f);
    const auto count = std::min(length, kWindowSamples);
    const auto offset = kWindowSamples - count;
    std::copy_n(user + length - count, count, result.begin() + offset);
    std::copy_n(agent + length - count, count, result.begin() + kWindowSamples + offset);
    return result;
}

VapForecast OnnxVap::forecast(
    const float* user, const float* agent, std::size_t length, int sample_rate) {
    if (sample_rate != kSampleRate) {
        throw std::invalid_argument("VAP requires synchronized 16000 Hz audio");
    }
    auto window = prepare_window(user, agent, length);
    const int64_t shape[] = {1, 2, static_cast<int64_t>(kWindowSamples)};
    OrtValue* input = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        OnnxEngine::get().cpu_memory(), window.data(), window.size() * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));
    ValueGuard input_guard{api_, &input, 1};
    const char* input_names[] = {"audio"};
    const char* output_names[] = {"p_now", "p_future", "vad"};
    const OrtValue* inputs[] = {input};
    OrtValue* outputs[] = {nullptr, nullptr, nullptr};
    ValueGuard output_guard{api_, outputs, 3};
    ort_check(api_, api_->Run(session_, nullptr, input_names, inputs, 1,
                              output_names, 3, outputs));
    VapForecast forecast;
    std::array<float, 2>* targets[] = {&forecast.p_now, &forecast.p_future, &forecast.voice_activity};
    for (std::size_t output = 0; output < 3; ++output) {
        float* values = nullptr;
        ort_check(api_, api_->GetTensorMutableData(outputs[output], reinterpret_cast<void**>(&values)));
        for (std::size_t channel = 0; channel < 2; ++channel) {
            const float value = values[2 * (kFrames - 1) + channel];
            if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
                throw std::runtime_error("VAP returned an invalid probability");
            }
            (*targets[output])[channel] = value;
        }
    }
    return forecast;
}

}  // namespace speech_core
