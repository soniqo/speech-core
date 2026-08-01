#include "speech_core/models/onnx_localvqe_echo_canceller.h"

#include "localvqe_aec_frontend.h"
#include "speech_core/models/onnx_engine.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace speech_core {
namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr std::size_t kSpectrumBins = 256;
constexpr std::size_t kControllerWeights = 2742;

const std::array<const char*, 9> kInputNames = {
    "mic_spectrum",
    "reference_spectrum",
    "conv_state",
    "align_key_state",
    "align_reference_state",
    "align_smooth_state",
    "s4_real_state",
    "s4_imag_state",
    "ccm_state",
};

const std::array<const char*, 8> kOutputNames = {
    "enhanced_spectrum",
    "next_conv_state",
    "next_align_key_state",
    "next_align_reference_state",
    "next_align_smooth_state",
    "next_s4_real_state",
    "next_s4_imag_state",
    "next_ccm_state",
};

const std::array<std::vector<int64_t>, 7> kStateShapes = {
    std::vector<int64_t>{1, 75936},
    std::vector<int64_t>{1, 16, 15, 64},
    std::vector<int64_t>{1, 20, 15, 64},
    std::vector<int64_t>{1, 16, 4, 16},
    std::vector<int64_t>{1, 80},
    std::vector<int64_t>{1, 80},
    std::vector<int64_t>{1, 2, 2, 256},
};

double elapsed_ms(const Clock::time_point& started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

std::size_t element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::runtime_error(
                "LocalVQE state shape must be fully resolved");
        }
        const auto value = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("LocalVQE state tensor is too large");
        }
        count *= value;
    }
    return count;
}

struct OrtValueHandle {
    const OrtApi* api = nullptr;
    OrtValue* value = nullptr;

    OrtValueHandle() = default;
    OrtValueHandle(const OrtApi* api_value, OrtValue* value_value)
        : api(api_value), value(value_value) {}
    ~OrtValueHandle() {
        if (api && value) api->ReleaseValue(value);
    }
    OrtValueHandle(const OrtValueHandle&) = delete;
    OrtValueHandle& operator=(const OrtValueHandle&) = delete;
    OrtValueHandle(OrtValueHandle&& other) noexcept
        : api(other.api), value(other.value) {
        other.value = nullptr;
    }
    OrtValueHandle& operator=(OrtValueHandle&& other) noexcept {
        if (this != &other) {
            if (api && value) api->ReleaseValue(value);
            api = other.api;
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }
    OrtValue* get() const { return value; }
};

struct OrtStringHandle {
    const OrtApi* api = nullptr;
    OrtAllocator* allocator = nullptr;
    char* value = nullptr;
    ~OrtStringHandle() {
        if (api && allocator && value) {
            OrtStatus* status = api->AllocatorFree(allocator, value);
            if (status) api->ReleaseStatus(status);
        }
    }
};

struct TensorInfo {
    ONNXTensorElementDataType type =
        ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t> shape;
};

TensorInfo tensor_info(
    const OrtApi* api, OrtSession* session, std::size_t index,
    bool input) {
    OrtTypeInfo* raw = nullptr;
    if (input) {
        ort_check(api, api->SessionGetInputTypeInfo(session, index, &raw));
    } else {
        ort_check(api, api->SessionGetOutputTypeInfo(session, index, &raw));
    }
    const OrtTensorTypeAndShapeInfo* tensor = nullptr;
    OrtStatus* cast_status = api->CastTypeInfoToTensorInfo(raw, &tensor);
    if (cast_status || !tensor) {
        if (cast_status) api->ReleaseStatus(cast_status);
        api->ReleaseTypeInfo(raw);
        throw std::runtime_error("LocalVQE graph I/O must be tensors");
    }
    TensorInfo output;
    ort_check(api, api->GetTensorElementType(tensor, &output.type));
    std::size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(tensor, &rank));
    output.shape.resize(rank);
    ort_check(api, api->GetDimensions(
        tensor, output.shape.data(), output.shape.size()));
    api->ReleaseTypeInfo(raw);
    return output;
}

TensorInfo value_info(const OrtApi* api, OrtValue* value) {
    OrtTensorTypeAndShapeInfo* raw = nullptr;
    ort_check(api, api->GetTensorTypeAndShape(value, &raw));
    TensorInfo output;
    ort_check(api, api->GetTensorElementType(raw, &output.type));
    std::size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(raw, &rank));
    output.shape.resize(rank);
    ort_check(api, api->GetDimensions(
        raw, output.shape.data(), output.shape.size()));
    api->ReleaseTensorTypeAndShapeInfo(raw);
    return output;
}

std::vector<std::string> session_names(
    const OrtApi* api, OrtSession* session, bool input) {
    OrtAllocator* allocator = nullptr;
    ort_check(api, api->GetAllocatorWithDefaultOptions(&allocator));
    std::size_t count = 0;
    if (input) {
        ort_check(api, api->SessionGetInputCount(session, &count));
    } else {
        ort_check(api, api->SessionGetOutputCount(session, &count));
    }
    std::vector<std::string> names;
    names.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        OrtStringHandle name{api, allocator, nullptr};
        if (input) {
            ort_check(api, api->SessionGetInputName(
                session, index, allocator, &name.value));
        } else {
            ort_check(api, api->SessionGetOutputName(
                session, index, allocator, &name.value));
        }
        names.emplace_back(name.value);
    }
    return names;
}

void validate_shape(
    const TensorInfo& info,
    const std::vector<int64_t>& expected,
    const char* label,
    bool allow_symbolic) {
    if (info.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
        || info.shape.size() != expected.size()) {
        throw std::runtime_error(
            std::string("LocalVQE ") + label
            + " has an incompatible tensor contract");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (info.shape[index] == expected[index]) continue;
        if (allow_symbolic && info.shape[index] < 0) continue;
        throw std::runtime_error(
            std::string("LocalVQE ") + label
            + " has an incompatible shape");
    }
}

OrtValueHandle make_tensor(
    const OrtApi* api,
    OrtMemoryInfo* memory,
    std::vector<float>& values,
    const std::vector<int64_t>& shape) {
    if (values.size() != element_count(shape)) {
        throw std::runtime_error(
            "LocalVQE host tensor does not match its declared shape");
    }
    OrtValue* output = nullptr;
    ort_check(api, api->CreateTensorWithDataAsOrtValue(
        memory,
        values.data(),
        values.size() * sizeof(float),
        shape.data(),
        shape.size(),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &output));
    return OrtValueHandle(api, output);
}

std::vector<float> copy_float_tensor(
    const OrtApi* api,
    OrtValue* value,
    const std::vector<int64_t>& expected,
    const char* label) {
    const TensorInfo info = value_info(api, value);
    validate_shape(info, expected, label, false);
    float* data = nullptr;
    ort_check(api, api->GetTensorMutableData(
        value, reinterpret_cast<void**>(&data)));
    return std::vector<float>(
        data, data + static_cast<std::ptrdiff_t>(element_count(expected)));
}

void fft_in_place(
    std::vector<float>& real,
    std::vector<float>& imaginary,
    bool inverse) {
    const std::size_t count = real.size();
    for (std::size_t index = 1, reversed = 0;
         index < count; ++index) {
        std::size_t bit = count >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imaginary[index], imaginary[reversed]);
        }
    }
    for (std::size_t length = 2; length <= count; length <<= 1) {
        const double angle =
            (inverse ? 2.0 : -2.0) * kPi / static_cast<double>(length);
        const double root_real = std::cos(angle);
        const double root_imaginary = std::sin(angle);
        for (std::size_t offset = 0; offset < count; offset += length) {
            double current_real = 1.0;
            double current_imaginary = 0.0;
            for (std::size_t index = 0; index < length / 2; ++index) {
                const float upper_real = real[offset + index];
                const float upper_imaginary = imaginary[offset + index];
                const std::size_t lower =
                    offset + index + length / 2;
                const float lower_real = static_cast<float>(
                    real[lower] * current_real
                    - imaginary[lower] * current_imaginary);
                const float lower_imaginary = static_cast<float>(
                    real[lower] * current_imaginary
                    + imaginary[lower] * current_real);
                real[offset + index] = upper_real + lower_real;
                imaginary[offset + index] =
                    upper_imaginary + lower_imaginary;
                real[lower] = upper_real - lower_real;
                imaginary[lower] =
                    upper_imaginary - lower_imaginary;
                const double next_real =
                    current_real * root_real
                    - current_imaginary * root_imaginary;
                current_imaginary =
                    current_real * root_imaginary
                    + current_imaginary * root_real;
                current_real = next_real;
            }
        }
    }
    if (inverse) {
        const float scale = 1.0f / static_cast<float>(count);
        for (std::size_t index = 0; index < count; ++index) {
            real[index] *= scale;
            imaginary[index] *= scale;
        }
    }
}

class LocalVQECodec {
public:
    explicit LocalVQECodec(std::vector<float> window)
        : window_(std::move(window)) {
        if (window_.size() != OnnxLocalVQEEchoCanceller::kFftSize
            || !std::all_of(
                window_.begin(), window_.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                "LocalVQE analysis window is invalid");
        }
        reset();
    }

    void reset() {
        microphone_history_.fill(0.0f);
        reference_history_.fill(0.0f);
        overlap_.fill(0.0f);
    }

    void analyze(
        const float* residual,
        const float* echo_estimate,
        std::vector<float>& microphone,
        std::vector<float>& reference) {
        analyze_one(
            residual, microphone_history_, microphone);
        analyze_one(
            echo_estimate, reference_history_, reference);
    }

    void synthesize(
        const std::vector<float>& spectrum,
        float* output) {
        if (spectrum.size() != kSpectrumBins * 2) {
            throw std::runtime_error(
                "LocalVQE enhanced spectrum has the wrong size");
        }
        std::vector<float> real(
            OnnxLocalVQEEchoCanceller::kFftSize, 0.0f);
        std::vector<float> imaginary(
            OnnxLocalVQEEchoCanceller::kFftSize, 0.0f);
        for (std::size_t bin = 1; bin < kSpectrumBins; ++bin) {
            real[bin] = spectrum[bin - 1];
            imaginary[bin] =
                spectrum[kSpectrumBins + bin - 1];
            real[OnnxLocalVQEEchoCanceller::kFftSize - bin] =
                real[bin];
            imaginary[OnnxLocalVQEEchoCanceller::kFftSize - bin] =
                -imaginary[bin];
        }
        real[kSpectrumBins] = 2.0f * spectrum[kSpectrumBins - 1];
        fft_in_place(real, imaginary, true);
        for (std::size_t index = 0;
             index < OnnxLocalVQEEchoCanceller::kFftSize; ++index) {
            overlap_[index] += real[index] * window_[index];
        }
        std::copy_n(
            overlap_.begin(),
            OnnxLocalVQEEchoCanceller::kFrameSize,
            output);
        std::move(
            overlap_.begin()
                + OnnxLocalVQEEchoCanceller::kFrameSize,
            overlap_.end(),
            overlap_.begin());
        std::fill(
            overlap_.begin()
                + OnnxLocalVQEEchoCanceller::kFrameSize,
            overlap_.end(),
            0.0f);
    }

private:
    void analyze_one(
        const float* frame,
        std::array<float, OnnxLocalVQEEchoCanceller::kFrameSize>&
            history,
        std::vector<float>& output) {
        std::vector<float> real(
            OnnxLocalVQEEchoCanceller::kFftSize, 0.0f);
        std::vector<float> imaginary(
            OnnxLocalVQEEchoCanceller::kFftSize, 0.0f);
        std::copy(history.begin(), history.end(), real.begin());
        std::copy_n(
            frame,
            OnnxLocalVQEEchoCanceller::kFrameSize,
            real.begin() + OnnxLocalVQEEchoCanceller::kFrameSize);
        std::copy_n(
            frame,
            OnnxLocalVQEEchoCanceller::kFrameSize,
            history.begin());
        for (std::size_t index = 0;
             index < OnnxLocalVQEEchoCanceller::kFftSize; ++index) {
            real[index] *= window_[index];
        }
        fft_in_place(real, imaginary, false);
        output.assign(kSpectrumBins * 2, 0.0f);
        for (std::size_t bin = 0; bin < kSpectrumBins; ++bin) {
            output[bin] = real[bin + 1];
            output[kSpectrumBins + bin] = imaginary[bin + 1];
        }
    }

    std::vector<float> window_;
    std::array<float, OnnxLocalVQEEchoCanceller::kFrameSize>
        microphone_history_{};
    std::array<float, OnnxLocalVQEEchoCanceller::kFrameSize>
        reference_history_{};
    std::array<float, OnnxLocalVQEEchoCanceller::kFftSize> overlap_{};
};

nlohmann::json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(
            "Could not read LocalVQE bundle file " + path.string());
    }
    nlohmann::json value;
    stream >> value;
    return value;
}

std::vector<float> read_finite_array(
    const nlohmann::json& value,
    const char* key,
    std::size_t expected_count) {
    const auto found = value.find(key);
    if (found == value.end() || !found->is_array()
        || found->size() != expected_count) {
        throw std::runtime_error(
            std::string("LocalVQE frontend has invalid ") + key);
    }
    std::vector<float> output;
    output.reserve(expected_count);
    for (const auto& element : *found) {
        if (!element.is_number()) {
            throw std::runtime_error(
                std::string("LocalVQE frontend has non-numeric ") + key);
        }
        const float number = element.get<float>();
        if (!std::isfinite(number)) {
            throw std::runtime_error(
                std::string("LocalVQE frontend has non-finite ") + key);
        }
        output.push_back(number);
    }
    return output;
}

}  // namespace

class OnnxLocalVQEEchoCanceller::Impl {
public:
    Impl(const std::string& bundle_directory, const Config& config)
        : config_(config) {
        if (config.intra_threads < 0) {
            throw std::invalid_argument(
                "LocalVQE intra_threads must be non-negative");
        }
        const fs::path bundle(bundle_directory);
        const fs::path model_path =
            bundle / "LocalVQEAECResidualMask.onnx";
        const fs::path frontend_path =
            bundle / "LocalVQEAECFrontend.json";
        const fs::path config_path = bundle / "config.json";
        if (!fs::is_regular_file(model_path)
            || !fs::is_regular_file(frontend_path)
            || !fs::is_regular_file(config_path)) {
            throw std::runtime_error(
                "LocalVQE ONNX bundle is incomplete");
        }

        const nlohmann::json metadata = read_json(config_path);
        if (metadata.value("model_type", std::string{})
                != "localvqe-v1.4-aec-onnx"
            || metadata.value("source_run_id", std::string{})
                != "v1.4.r005"
            || metadata.value("source_gguf_sha256", std::string{})
                != "b6e43138588a83bfe903ab5e143b4020b91c1e1629f5a575ac5855ff0003c731"
            || metadata.value("sample_rate", 0) != kSampleRate
            || metadata.value("fft_size", 0)
                != static_cast<int>(kFftSize)
            || metadata.value("hop_size", 0)
                != static_cast<int>(kFrameSize)
            || metadata.value("precision", std::string{}) != "float32") {
            throw std::runtime_error(
                "LocalVQE bundle metadata does not match v1.4-AEC");
        }

        const nlohmann::json frontend = read_json(frontend_path);
        if (frontend.value("format_version", 0) != 1
            || frontend.value("source_run_id", std::string{})
                != "v1.4.r005"
            || frontend.value("sample_rate", 0) != kSampleRate
            || frontend.value("fft_size", 0)
                != static_cast<int>(kFftSize)
            || frontend.value("hop_size", 0)
                != static_cast<int>(kFrameSize)
            || frontend.value("daf_block_size", 0) != 128
            || frontend.value("daf_partitions", 0) != 128
            || frontend.value("daf_iterations", 0) != 2) {
            throw std::runtime_error(
                "LocalVQE frontend metadata is incompatible");
        }
        std::vector<float> controller = read_finite_array(
            frontend, "controller_weights", kControllerWeights);
        std::vector<float> window = read_finite_array(
            frontend, "analysis_window", kFftSize);
        frontend_ = localvqe_aec_daf_create(
            controller.data(), controller.size());
        if (!frontend_) {
            throw std::runtime_error(
                "Could not initialize LocalVQE adaptive frontend");
        }
        localvqe_aec_daf_set_prealignment(
            frontend_, config.enable_prealignment);
        codec_ = std::make_unique<LocalVQECodec>(std::move(window));

        auto& engine = OnnxEngine::get();
        api_ = engine.api();
        try {
            session_ = engine.load(
                model_path.string(),
                config.hardware_acceleration,
                false,
                config.intra_threads);
            validate_graph();
            reset_model_state();
        } catch (...) {
            if (session_ && api_) api_->ReleaseSession(session_);
            session_ = nullptr;
            localvqe_aec_daf_destroy(frontend_);
            frontend_ = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (session_ && api_) api_->ReleaseSession(session_);
        if (frontend_) localvqe_aec_daf_destroy(frontend_);
    }

    void process_frame(
        const float* microphone,
        const float* reference,
        float* output) {
        if (!microphone || !reference || !output) {
            throw std::invalid_argument(
                "LocalVQE frame pointers must not be null");
        }
        for (std::size_t index = 0; index < kFrameSize; ++index) {
            if (!std::isfinite(microphone[index])
                || !std::isfinite(reference[index])) {
                throw std::invalid_argument(
                    "LocalVQE input contains NaN or infinity");
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        process_frame_locked(microphone, reference, output);
    }

    bool prime_delay(
        const float* microphone,
        const float* reference,
        std::size_t sample_count) {
        if ((!microphone || !reference) && sample_count != 0) {
            throw std::invalid_argument(
                "LocalVQE delay-prime pointers must not be null");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return localvqe_aec_daf_prime_delay(
            frontend_, microphone, reference, sample_count);
    }

    int current_delay_samples() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(
            localvqe_aec_daf_current_delay_samples(frontend_));
    }

    float delay_confidence() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return localvqe_aec_daf_delay_confidence(frontend_);
    }

    Profile last_profile() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return profile_;
    }

    void feed_reference(
        const float* samples,
        std::size_t length) {
        if (!samples && length != 0) {
            throw std::invalid_argument(
                "LocalVQE reference pointer must not be null");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (reference_queue_.size() + length
            > kSampleRate * 10u) {
            throw std::runtime_error(
                "LocalVQE reference queue exceeded ten seconds");
        }
        for (std::size_t index = 0; index < length; ++index) {
            if (!std::isfinite(samples[index])) {
                throw std::invalid_argument(
                    "LocalVQE reference contains NaN or infinity");
            }
            reference_queue_.push_back(samples[index]);
        }
    }

    void cancel_echo(
        const float* input,
        std::size_t length,
        float* output) {
        if ((!input || !output) && length != 0) {
            throw std::invalid_argument(
                "LocalVQE audio pointers must not be null");
        }
        if (length % kFrameSize != 0) {
            throw std::invalid_argument(
                "LocalVQE cancel_echo requires complete 256-sample frames");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (reference_queue_.size() < length) {
            throw std::runtime_error(
                "LocalVQE reference underrun; raw microphone was not passed");
        }
        std::array<float, kFrameSize> reference{};
        for (std::size_t offset = 0; offset < length;
             offset += kFrameSize) {
            for (std::size_t index = 0; index < kFrameSize; ++index) {
                reference[index] = reference_queue_.front();
                reference_queue_.pop_front();
            }
            process_frame_locked(
                input + offset, reference.data(), output + offset);
        }
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        localvqe_aec_daf_reset(frontend_);
        localvqe_aec_daf_set_prealignment(
            frontend_, config_.enable_prealignment);
        codec_->reset();
        reset_model_state();
        reference_queue_.clear();
        profile_ = {};
    }

private:
    void validate_graph() {
        const std::vector<std::string> actual_inputs =
            session_names(api_, session_, true);
        const std::vector<std::string> actual_outputs =
            session_names(api_, session_, false);
        if (actual_inputs != std::vector<std::string>(
                kInputNames.begin(), kInputNames.end())
            || actual_outputs != std::vector<std::string>(
                kOutputNames.begin(), kOutputNames.end())) {
            throw std::runtime_error(
                "LocalVQE ONNX graph signature is incompatible");
        }
        validate_shape(
            tensor_info(api_, session_, 0, true),
            {1, 2, 1, 256}, "mic_spectrum", false);
        validate_shape(
            tensor_info(api_, session_, 1, true),
            {1, 2, 1, 256}, "reference_spectrum", false);
        for (std::size_t index = 0; index < kStateShapes.size(); ++index) {
            validate_shape(
                tensor_info(api_, session_, index + 2, true),
                kStateShapes[index], kInputNames[index + 2], false);
        }
        validate_shape(
            tensor_info(api_, session_, 0, false),
            {1, 2, 1, 256}, "enhanced_spectrum", true);
        for (std::size_t index = 0; index < kStateShapes.size(); ++index) {
            validate_shape(
                tensor_info(api_, session_, index + 1, false),
                kStateShapes[index], kOutputNames[index + 1], true);
        }
    }

    void reset_model_state() {
        states_.clear();
        states_.reserve(kStateShapes.size());
        for (const auto& shape : kStateShapes) {
            states_.emplace_back(element_count(shape), 0.0f);
        }
    }

    void process_frame_locked(
        const float* microphone,
        const float* reference,
        float* output) {
        const auto total_started = Clock::now();
        std::array<float, kFrameSize> residual{};
        std::array<float, kFrameSize> echo_estimate{};
        const auto adaptive_started = Clock::now();
        if (!localvqe_aec_daf_process(
                frontend_,
                microphone,
                reference,
                kFrameSize,
                residual.data(),
                echo_estimate.data())) {
            throw std::runtime_error(
                "LocalVQE adaptive frontend rejected a synchronized frame");
        }
        const double adaptive_ms = elapsed_ms(adaptive_started);

        std::vector<float> microphone_spectrum;
        std::vector<float> reference_spectrum;
        codec_->analyze(
            residual.data(),
            echo_estimate.data(),
            microphone_spectrum,
            reference_spectrum);
        OrtMemoryInfo* raw_memory = nullptr;
        ort_check(api_, api_->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &raw_memory));
        struct MemoryGuard {
            const OrtApi* api;
            OrtMemoryInfo* memory;
            ~MemoryGuard() {
                if (api && memory) api->ReleaseMemoryInfo(memory);
            }
        } memory{api_, raw_memory};

        std::vector<OrtValueHandle> inputs;
        inputs.reserve(kInputNames.size());
        inputs.emplace_back(make_tensor(
            api_, raw_memory, microphone_spectrum, {1, 2, 1, 256}));
        inputs.emplace_back(make_tensor(
            api_, raw_memory, reference_spectrum, {1, 2, 1, 256}));
        for (std::size_t index = 0; index < states_.size(); ++index) {
            inputs.emplace_back(make_tensor(
                api_, raw_memory, states_[index], kStateShapes[index]));
        }
        std::array<OrtValue*, kInputNames.size()> raw_inputs{};
        for (std::size_t index = 0; index < inputs.size(); ++index) {
            raw_inputs[index] = inputs[index].get();
        }
        std::array<OrtValue*, kOutputNames.size()> raw_outputs{};
        const auto neural_started = Clock::now();
        ort_check(api_, api_->Run(
            session_,
            nullptr,
            kInputNames.data(),
            raw_inputs.data(),
            raw_inputs.size(),
            kOutputNames.data(),
            kOutputNames.size(),
            raw_outputs.data()));
        std::vector<OrtValueHandle> outputs;
        outputs.reserve(raw_outputs.size());
        for (OrtValue* value : raw_outputs) {
            outputs.emplace_back(api_, value);
        }
        std::vector<float> enhanced = copy_float_tensor(
            api_, outputs[0].get(), {1, 2, 1, 256},
            "enhanced_spectrum output");
        std::vector<std::vector<float>> next_states;
        next_states.reserve(kStateShapes.size());
        for (std::size_t index = 0; index < kStateShapes.size(); ++index) {
            next_states.emplace_back(copy_float_tensor(
                api_,
                outputs[index + 1].get(),
                kStateShapes[index],
                kOutputNames[index + 1]));
        }
        const double neural_ms = elapsed_ms(neural_started);
        if (!std::all_of(
                enhanced.begin(), enhanced.end(),
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                "LocalVQE neural mask produced NaN or infinity");
        }
        codec_->synthesize(enhanced, output);
        if (!std::all_of(
                output, output + kFrameSize,
                [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error(
                "LocalVQE synthesis produced NaN or infinity");
        }
        states_.swap(next_states);
        profile_ = {
            adaptive_ms,
            neural_ms,
            elapsed_ms(total_started),
        };
    }

    Config config_;
    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
    localvqe_aec_daf* frontend_ = nullptr;
    std::unique_ptr<LocalVQECodec> codec_;
    std::vector<std::vector<float>> states_;
    std::deque<float> reference_queue_;
    mutable std::mutex mutex_;
    Profile profile_;
};

OnnxLocalVQEEchoCanceller::OnnxLocalVQEEchoCanceller(
    const std::string& bundle_directory)
    : OnnxLocalVQEEchoCanceller(bundle_directory, Config{}) {}

OnnxLocalVQEEchoCanceller::OnnxLocalVQEEchoCanceller(
    const std::string& bundle_directory,
    const Config& config)
    : impl_(std::make_unique<Impl>(bundle_directory, config)) {}

OnnxLocalVQEEchoCanceller::~OnnxLocalVQEEchoCanceller() = default;

void OnnxLocalVQEEchoCanceller::process_frame(
    const float* microphone,
    const float* reference,
    float* output) {
    impl_->process_frame(microphone, reference, output);
}

bool OnnxLocalVQEEchoCanceller::prime_delay(
    const float* microphone,
    const float* reference,
    std::size_t sample_count) {
    return impl_->prime_delay(microphone, reference, sample_count);
}

int OnnxLocalVQEEchoCanceller::current_delay_samples() const {
    return impl_->current_delay_samples();
}

float OnnxLocalVQEEchoCanceller::delay_confidence() const {
    return impl_->delay_confidence();
}

OnnxLocalVQEEchoCanceller::Profile
OnnxLocalVQEEchoCanceller::last_profile() const {
    return impl_->last_profile();
}

void OnnxLocalVQEEchoCanceller::feed_reference(
    const float* samples,
    std::size_t length) {
    impl_->feed_reference(samples, length);
}

void OnnxLocalVQEEchoCanceller::cancel_echo(
    const float* input,
    std::size_t length,
    float* output) {
    impl_->cancel_echo(input, length, output);
}

void OnnxLocalVQEEchoCanceller::reset() {
    impl_->reset();
}

}  // namespace speech_core
