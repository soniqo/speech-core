#include "speech_core/models/onnx_moss_transcribe_diarize.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/transcription/moss_transcript_parser.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace speech_core {
namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

constexpr int64_t kDecoderLayers = 28;
constexpr int64_t kDecoderKvHeads = 8;
constexpr int64_t kDecoderHeadDim = 128;
constexpr int64_t kDecoderHidden = 1024;
constexpr int64_t kDecoderVocabulary = 151936;

double milliseconds_since(const Clock::time_point& started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

struct OrtValueHandle {
    const OrtApi* api = nullptr;
    OrtValue* value = nullptr;

    OrtValueHandle() = default;
    OrtValueHandle(const OrtApi* api_value, OrtValue* ort_value)
        : api(api_value), value(ort_value) {}
    ~OrtValueHandle() {
        if (value && api) api->ReleaseValue(value);
    }
    OrtValueHandle(const OrtValueHandle&) = delete;
    OrtValueHandle& operator=(const OrtValueHandle&) = delete;
    OrtValueHandle(OrtValueHandle&& other) noexcept
        : api(other.api), value(other.value) {
        other.value = nullptr;
    }
    OrtValueHandle& operator=(OrtValueHandle&& other) noexcept {
        if (this != &other) {
            if (value && api) api->ReleaseValue(value);
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
        if (value && api && allocator) {
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

std::size_t checked_element_count(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error(
                "MOSS runtime received an unresolved tensor shape");
        }
        if (dimension == 0) return 0;
        const std::size_t value = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("MOSS tensor is too large");
        }
        count *= value;
    }
    return count;
}

std::size_t element_size(ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) return 2;
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return 4;
    throw std::runtime_error("MOSS graph must use Float16 or Float32 I/O");
}

uint16_t float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x800000u)
            >> static_cast<unsigned>(1 - exponent);
        if (mantissa & 0x1000u) mantissa += 0x2000u;
        return static_cast<uint16_t>(sign | (mantissa >> 13u));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(
            sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    if (mantissa & 0x1000u) {
        mantissa += 0x2000u;
        if (mantissa & 0x800000u) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10u)
        | (mantissa >> 13u));
}

float half_to_float(uint16_t value) {
    const uint32_t sign =
        static_cast<uint32_t>(value & 0x8000u) << 16u;
    int exponent = static_cast<int>((value >> 10u) & 0x1fu);
    uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1u;
                --exponent;
            }
            mantissa &= 0x3ffu;
            bits = sign
                | (static_cast<uint32_t>(exponent + 127 - 15) << 23u)
                | (mantissa << 13u);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign
            | (static_cast<uint32_t>(exponent + 127 - 15) << 23u)
            | (mantissa << 13u);
    }
    float output = 0.0f;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

void write_float(
    std::vector<uint8_t>& storage, std::size_t index, float value,
    ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const uint16_t converted = float_to_half(value);
        std::memcpy(
            storage.data() + index * sizeof(converted),
            &converted, sizeof(converted));
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::memcpy(
            storage.data() + index * sizeof(value), &value, sizeof(value));
    } else {
        throw std::runtime_error("Unsupported MOSS floating-point type");
    }
}

float read_float(
    const void* data, std::size_t index, ONNXTensorElementDataType type) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        uint16_t value = 0;
        std::memcpy(
            &value,
            static_cast<const uint8_t*>(data) + index * sizeof(value),
            sizeof(value));
        return half_to_float(value);
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float value = 0.0f;
        std::memcpy(
            &value,
            static_cast<const uint8_t*>(data) + index * sizeof(value),
            sizeof(value));
        return value;
    }
    throw std::runtime_error("Unsupported MOSS floating-point type");
}

OrtValueHandle make_raw_tensor(
    const OrtApi* api, OrtMemoryInfo* memory, void* data,
    std::size_t byte_count, const std::vector<int64_t>& shape,
    ONNXTensorElementDataType type) {
    OrtValue* value = nullptr;
    ort_check(api, api->CreateTensorWithDataAsOrtValue(
        memory, data, byte_count, shape.data(), shape.size(), type, &value));
    return OrtValueHandle(api, value);
}

template <typename T>
OrtValueHandle make_typed_tensor(
    const OrtApi* api, OrtMemoryInfo* memory, std::vector<T>& data,
    const std::vector<int64_t>& shape, ONNXTensorElementDataType type) {
    return make_raw_tensor(
        api, memory, data.data(), data.size() * sizeof(T), shape, type);
}

TensorInfo value_info(const OrtApi* api, OrtValue* value) {
    OrtTensorTypeAndShapeInfo* raw = nullptr;
    ort_check(api, api->GetTensorTypeAndShape(value, &raw));
    TensorInfo output;
    ort_check(api, api->GetTensorElementType(raw, &output.type));
    std::size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(raw, &rank));
    output.shape.resize(rank);
    ort_check(api, api->GetDimensions(raw, output.shape.data(), rank));
    api->ReleaseTensorTypeAndShapeInfo(raw);
    return output;
}

TensorInfo session_tensor_info(
    const OrtApi* api, OrtSession* session, std::size_t index,
    bool input) {
    OrtTypeInfo* raw = nullptr;
    if (input) {
        ort_check(api, api->SessionGetInputTypeInfo(session, index, &raw));
    } else {
        ort_check(api, api->SessionGetOutputTypeInfo(session, index, &raw));
    }
    const OrtTensorTypeAndShapeInfo* tensor = nullptr;
    OrtStatus* status = api->CastTypeInfoToTensorInfo(raw, &tensor);
    if (status || !tensor) {
        if (status) api->ReleaseStatus(status);
        api->ReleaseTypeInfo(raw);
        throw std::runtime_error("MOSS graph I/O must be tensors");
    }
    TensorInfo output;
    ort_check(api, api->GetTensorElementType(tensor, &output.type));
    std::size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(tensor, &rank));
    output.shape.resize(rank);
    ort_check(api, api->GetDimensions(tensor, output.shape.data(), rank));
    api->ReleaseTypeInfo(raw);
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

void require_names(
    const std::vector<std::string>& actual,
    const std::vector<std::string>& expected,
    const char* label) {
    if (actual != expected) {
        throw std::runtime_error(
            std::string("Incompatible MOSS ") + label + " graph signature");
    }
}

std::vector<uint8_t> copy_tensor_bytes(
    const OrtApi* api, OrtValue* value,
    ONNXTensorElementDataType expected_type) {
    const TensorInfo info = value_info(api, value);
    if (info.type != expected_type) {
        throw std::runtime_error("MOSS graph returned an unexpected type");
    }
    const std::size_t bytes =
        checked_element_count(info.shape) * element_size(info.type);
    void* data = nullptr;
    ort_check(api, api->GetTensorMutableData(value, &data));
    std::vector<uint8_t> output(bytes);
    if (bytes > 0) std::memcpy(output.data(), data, bytes);
    return output;
}

int64_t cache_sequence_length(
    const TensorInfo& info, std::size_t expected_delta = 0) {
    if (info.shape.size() != 5
        || info.shape[0] != kDecoderLayers
        || info.shape[1] != 1
        || info.shape[2] != kDecoderKvHeads
        || info.shape[4] != kDecoderHeadDim
        || info.shape[3] < 0
        || (expected_delta > 0
            && info.shape[3] != static_cast<int64_t>(expected_delta))) {
        throw std::runtime_error(
            "MOSS decoder returned an incompatible K/V cache");
    }
    return info.shape[3];
}

/// Restride the cache one step longer, into a buffer the caller keeps.
///
/// The sequence axis is not the outermost one, so growing it moves every
/// head's block and the copy itself cannot be avoided without changing the
/// graph's cache contract. What can be avoided is doing it into a *fresh*
/// allocation each step: the buffer reaches tens of megabytes, and asking the
/// OS for it again per token faults in every page again. Measured on a 29
/// token paragraph, the copies alone are 2.2 GB and the reallocations another
/// 2.3 GB of first-touch faults. The caller reserves both buffers once and
/// swaps them, so `resize` here never reallocates and the fault cost is paid
/// once per paragraph rather than once per token.
///
/// The cost is quadratic in generated tokens either way, so this shrinks the
/// constant and does not change the shape. Removing the quadratic needs the
/// decoder re-exported to return its whole cache, or a capacity-padded cache
/// with the attention mask covering the unused tail -- both of which change
/// numerics and must be measured against a transcript, not assumed.
void append_cache_into(
    std::vector<uint8_t>& output,
    const std::vector<uint8_t>& past, int64_t past_length,
    const std::vector<uint8_t>& delta, int64_t delta_length,
    std::size_t scalar_size) {
    const int64_t next_length = past_length + delta_length;
    const std::size_t head_width =
        static_cast<std::size_t>(kDecoderHeadDim) * scalar_size;
    const std::size_t past_head_bytes =
        static_cast<std::size_t>(past_length) * head_width;
    const std::size_t delta_head_bytes =
        static_cast<std::size_t>(delta_length) * head_width;
    const std::size_t next_head_bytes =
        static_cast<std::size_t>(next_length) * head_width;
    const std::size_t head_count =
        static_cast<std::size_t>(kDecoderLayers * kDecoderKvHeads);
    output.resize(head_count * next_head_bytes);
    // Backwards: head h's destination overlaps heads above it in the same
    // buffer only for h below the write head, so descending order keeps every
    // read ahead of the write when output and past are the same allocation.
    for (std::size_t index = head_count; index-- > 0;) {
        uint8_t* destination = output.data() + index * next_head_bytes;
        if (past_head_bytes > 0) {
            std::memmove(
                destination, past.data() + index * past_head_bytes,
                past_head_bytes);
        }
        if (delta_head_bytes > 0) {
            std::memcpy(
                destination + past_head_bytes,
                delta.data() + index * delta_head_bytes,
                delta_head_bytes);
        }
    }
}

int64_t argmax_logits(
    const OrtApi* api, OrtValue* logits,
    ONNXTensorElementDataType expected_type) {
    const TensorInfo info = value_info(api, logits);
    if (info.type != expected_type || info.shape.size() != 3
        || info.shape[0] != 1 || info.shape[1] != 1
        || info.shape[2] != kDecoderVocabulary) {
        throw std::runtime_error(
            "MOSS decoder returned incompatible logits");
    }
    void* data = nullptr;
    ort_check(api, api->GetTensorMutableData(logits, &data));
    int64_t best_index = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int64_t index = 0; index < kDecoderVocabulary; ++index) {
        const float value = read_float(
            data, static_cast<std::size_t>(index), expected_type);
        if (!std::isnan(value) && value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

void validate_expected_shape(
    const TensorInfo& info, const std::vector<int64_t>& expected,
    const char* label) {
    if (info.shape.size() != expected.size()) {
        throw std::runtime_error(
            std::string("MOSS ") + label + " has an incompatible rank");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        // ORT reports symbolic graph dimensions as -1. Accept those at load
        // time and validate every resolved runtime tensor below; the exported
        // decoder leaves even semantically fixed batch/head dimensions
        // symbolic on a few outputs.
        if (expected[index] >= 0 && info.shape[index] >= 0
            && info.shape[index] != expected[index]) {
            throw std::runtime_error(
                std::string("MOSS ") + label
                + " has an incompatible shape");
        }
    }
}

}  // namespace

OnnxMossTranscribeDiarize::OnnxMossTranscribeDiarize(
    const std::string& bundle_directory)
    : OnnxMossTranscribeDiarize(bundle_directory, Config{}) {}

OnnxMossTranscribeDiarize::OnnxMossTranscribeDiarize(
    const std::string& bundle_directory, const Config& config)
    : config_(config),
      tokenizer_((fs::path(bundle_directory) / "vocab.json").string()) {
    if (config_.max_new_tokens <= 0) {
        throw std::invalid_argument(
            "MOSS max_new_tokens must be positive");
    }
    validate_bundle(bundle_directory);

    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    const fs::path bundle(bundle_directory);
    audio_encoder_ = engine.load(
        (bundle / "audio_encoder.onnx").string(),
        config_.audio_hardware_acceleration, false,
        config_.audio_intra_threads);
    try {
        decoder_ = engine.load(
            (bundle / "decoder.onnx").string(),
            config_.decoder_hardware_acceleration, false,
            config_.decoder_intra_threads);
        validate_graph_contracts();
    } catch (...) {
        if (decoder_) api_->ReleaseSession(decoder_);
        decoder_ = nullptr;
        if (audio_encoder_) api_->ReleaseSession(audio_encoder_);
        audio_encoder_ = nullptr;
        throw;
    }
}

OnnxMossTranscribeDiarize::~OnnxMossTranscribeDiarize() {
    if (decoder_) api_->ReleaseSession(decoder_);
    if (audio_encoder_) api_->ReleaseSession(audio_encoder_);
}

void OnnxMossTranscribeDiarize::validate_bundle(
    const std::string& bundle_directory) const {
    const fs::path bundle(bundle_directory);
    const std::array<const char*, 6> required = {
        "audio_encoder.onnx",
        "decoder.onnx",
        "config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "vocab.json",
    };
    for (const char* name : required) {
        if (!fs::is_regular_file(bundle / name)) {
            throw std::runtime_error(
                std::string("MOSS bundle is missing ") + name);
        }
    }

    const auto config = json::parse_flat_object(
        json::read_file((bundle / "config.json").string()));
    const auto model_type = config.find("model_type");
    const auto source_revision = config.find("source_revision");
    if (model_type == config.end()
        || model_type->second != "moss-transcribe-diarize-onnx"
        || source_revision == config.end()
        || source_revision->second
            != "e6d68cdfcddbdad1a7e8454f0cb859cad76e2502") {
        throw std::runtime_error(
            "MOSS bundle metadata does not match the supported export");
    }

    const auto processor = json::parse_flat_object(
        json::read_file((bundle / "processor_config.json").string()));
    const auto preprocessor = json::parse_flat_object(
        json::read_file((bundle / "preprocessor_config.json").string()));
    auto require_value = [](const json::Dict& values, const char* key,
                            const char* expected, const char* label) {
        const auto found = values.find(key);
        if (found == values.end() || found->second != expected) {
            throw std::runtime_error(
                std::string("MOSS ") + label
                + " has an incompatible " + key);
        }
    };
    require_value(
        processor, "audio_tokens_per_second", "12.5",
        "processor_config.json");
    require_value(
        processor, "audio_merge_size", "4", "processor_config.json");
    require_value(
        processor, "time_marker_every_seconds", "5",
        "processor_config.json");
    require_value(
        processor, "enable_time_marker", "true",
        "processor_config.json");
    require_value(
        preprocessor, "feature_size", "80", "preprocessor_config.json");
    require_value(
        preprocessor, "hop_length", "160", "preprocessor_config.json");
    require_value(
        preprocessor, "n_fft", "400", "preprocessor_config.json");
    require_value(
        preprocessor, "n_samples", "480000",
        "preprocessor_config.json");
    require_value(
        preprocessor, "nb_max_frames", "3000",
        "preprocessor_config.json");
    require_value(
        preprocessor, "sampling_rate", "16000",
        "preprocessor_config.json");
}

void OnnxMossTranscribeDiarize::validate_graph_contracts() {
    require_names(
        session_names(api_, audio_encoder_, true),
        {"input_features"}, "audio-encoder input");
    require_names(
        session_names(api_, audio_encoder_, false),
        {"audio_embeds"}, "audio-encoder output");
    require_names(
        session_names(api_, decoder_, true),
        {
            "input_ids", "input_embeds", "input_embed_mask",
            "position_ids", "attention_mask", "past_keys", "past_values",
        },
        "decoder input");
    require_names(
        session_names(api_, decoder_, false),
        {"logits", "new_keys", "new_values"}, "decoder output");

    const TensorInfo audio_input =
        session_tensor_info(api_, audio_encoder_, 0, true);
    const TensorInfo audio_output =
        session_tensor_info(api_, audio_encoder_, 0, false);
    validate_expected_shape(
        audio_input, {1, 80, 3000}, "audio input");
    validate_expected_shape(
        audio_output, {1, 375, 1024}, "audio output");
    if (audio_input.type != audio_output.type
        || (audio_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            && audio_input.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
        throw std::runtime_error(
            "MOSS audio graph has incompatible numeric I/O");
    }
    audio_type_ = audio_input.type;

    const TensorInfo input_ids =
        session_tensor_info(api_, decoder_, 0, true);
    const TensorInfo input_embeds =
        session_tensor_info(api_, decoder_, 1, true);
    const TensorInfo embed_mask =
        session_tensor_info(api_, decoder_, 2, true);
    const TensorInfo position_ids =
        session_tensor_info(api_, decoder_, 3, true);
    const TensorInfo attention =
        session_tensor_info(api_, decoder_, 4, true);
    const TensorInfo past_keys =
        session_tensor_info(api_, decoder_, 5, true);
    const TensorInfo past_values =
        session_tensor_info(api_, decoder_, 6, true);
    const TensorInfo logits =
        session_tensor_info(api_, decoder_, 0, false);
    const TensorInfo new_keys =
        session_tensor_info(api_, decoder_, 1, false);
    const TensorInfo new_values =
        session_tensor_info(api_, decoder_, 2, false);

    if (input_ids.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
        || position_ids.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        throw std::runtime_error(
            "MOSS decoder token inputs must be Int64");
    }
    decoder_type_ = input_embeds.type;
    if ((decoder_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
         && decoder_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        || embed_mask.type != decoder_type_
        || attention.type != decoder_type_
        || past_keys.type != decoder_type_
        || past_values.type != decoder_type_
        || logits.type != decoder_type_
        || new_keys.type != decoder_type_
        || new_values.type != decoder_type_
        || audio_type_ != decoder_type_) {
        throw std::runtime_error(
            "MOSS graphs do not share one supported numeric type");
    }
    validate_expected_shape(input_ids, {1, -1}, "decoder input_ids");
    validate_expected_shape(
        input_embeds, {1, -1, kDecoderHidden},
        "decoder input_embeds");
    validate_expected_shape(
        embed_mask, {1, -1, 1}, "decoder input_embed_mask");
    validate_expected_shape(position_ids, {-1}, "decoder position_ids");
    validate_expected_shape(
        attention, {1, 1, -1, -1}, "decoder attention_mask");
    validate_expected_shape(
        past_keys,
        {kDecoderLayers, 1, kDecoderKvHeads, -1, kDecoderHeadDim},
        "decoder past_keys");
    validate_expected_shape(
        past_values,
        {kDecoderLayers, 1, kDecoderKvHeads, -1, kDecoderHeadDim},
        "decoder past_values");
    validate_expected_shape(
        logits, {1, 1, kDecoderVocabulary}, "decoder logits");
    validate_expected_shape(
        new_keys,
        {kDecoderLayers, 1, kDecoderKvHeads, -1, kDecoderHeadDim},
        "decoder new_keys");
    validate_expected_shape(
        new_values,
        {kDecoderLayers, 1, kDecoderKvHeads, -1, kDecoderHeadDim},
        "decoder new_values");
}

void OnnxMossTranscribeDiarize::cancel() {
    cancel_requested_.store(true, std::memory_order_release);
}

OnnxMossTranscribeDiarize::Profile
OnnxMossTranscribeDiarize::last_profile() const {
    std::lock_guard<std::mutex> lock(profile_mutex_);
    return last_profile_;
}

DiarizedTranscriptionResult
OnnxMossTranscribeDiarize::transcribe_diarized(
    const float* audio, std::size_t length, int sample_rate) {
    if (!audio && length > 0) {
        throw std::invalid_argument("MOSS audio buffer is null");
    }
    if (length == 0) return {};
    if (sample_rate <= 0) {
        throw std::invalid_argument("MOSS sample rate must be positive");
    }

    std::lock_guard<std::mutex> inference_lock(inference_mutex_);
    cancel_requested_.store(false, std::memory_order_release);
    Profile profile;
    const auto total_started = Clock::now();

    std::vector<float> resampled;
    if (sample_rate != input_sample_rate()) {
        resampled = Resampler::resample(
            audio, length, sample_rate, input_sample_rate());
        audio = resampled.data();
        length = resampled.size();
    }
    if (length == 0) return {};

    auto* memory = OnnxEngine::get().cpu_memory();
    const std::size_t scalar_size = element_size(audio_type_);
    std::vector<uint8_t> audio_embeddings;
    std::size_t total_audio_tokens = 0;
    const auto feature_started = Clock::now();
    std::vector<MossLogMelFeatures> chunk_features;
    std::vector<std::size_t> chunk_token_counts;
    for (std::size_t offset = 0; offset < length;
         offset += static_cast<std::size_t>(
             MossWhisperFeatureExtractor::kChunkSamples)) {
        const std::size_t chunk_length = std::min(
            length - offset,
            static_cast<std::size_t>(
                MossWhisperFeatureExtractor::kChunkSamples));
        chunk_features.push_back(
            feature_extractor_.extract_padded_chunk(
                audio + offset, chunk_length));
        const std::size_t tokens =
            MossWhisperFeatureExtractor::audio_token_count(chunk_length);
        chunk_token_counts.push_back(tokens);
        total_audio_tokens += tokens;
    }
    profile.feature_ms = milliseconds_since(feature_started);
    profile.audio_chunks = static_cast<int>(chunk_features.size());
    audio_embeddings.reserve(
        total_audio_tokens * static_cast<std::size_t>(kDecoderHidden)
        * scalar_size);

    const auto audio_started = Clock::now();
    const char* audio_input_names[] = {"input_features"};
    const char* audio_output_names[] = {"audio_embeds"};
    for (std::size_t chunk = 0; chunk < chunk_features.size(); ++chunk) {
        if (cancel_requested_.load(std::memory_order_acquire)) return {};
        const auto& features = chunk_features[chunk];
        std::vector<uint8_t> feature_storage(
            features.data.size() * scalar_size);
        for (std::size_t index = 0; index < features.data.size(); ++index) {
            write_float(
                feature_storage, index, features.data[index], audio_type_);
        }
        const std::vector<int64_t> feature_shape = {1, 80, 3000};
        auto feature_tensor = make_raw_tensor(
            api_, memory, feature_storage.data(), feature_storage.size(),
            feature_shape, audio_type_);
        OrtValue* inputs[] = {feature_tensor.get()};
        OrtValue* outputs[] = {nullptr};
        ort_check(api_, api_->Run(
            audio_encoder_, nullptr, audio_input_names, inputs, 1,
            audio_output_names, 1, outputs));
        OrtValueHandle output(api_, outputs[0]);
        const TensorInfo output_info = value_info(api_, output.get());
        validate_expected_shape(
            output_info, {1, 375, 1024}, "audio output");
        if (output_info.type != audio_type_) {
            throw std::runtime_error(
                "MOSS audio encoder changed numeric type");
        }
        void* output_data = nullptr;
        ort_check(api_, api_->GetTensorMutableData(
            output.get(), &output_data));
        const std::size_t bytes = chunk_token_counts[chunk]
            * static_cast<std::size_t>(kDecoderHidden) * scalar_size;
        const auto* begin = static_cast<const uint8_t*>(output_data);
        audio_embeddings.insert(
            audio_embeddings.end(), begin, begin + bytes);
    }
    profile.audio_encoder_ms = milliseconds_since(audio_started);

    MossPreparedPrompt prompt =
        MossPromptProcessor::prepare(total_audio_tokens);
    if (prompt.audio_placeholder_count != total_audio_tokens) {
        throw std::runtime_error(
            "MOSS prompt/audio placeholder count mismatch");
    }
    profile.prompt_tokens = static_cast<int>(prompt.input_ids.size());

    const std::size_t prompt_tokens = prompt.input_ids.size();
    std::vector<uint8_t> input_embeddings(
        prompt_tokens * static_cast<std::size_t>(kDecoderHidden)
        * scalar_size, 0);
    std::vector<uint8_t> input_embed_mask(
        prompt_tokens * scalar_size, 0);
    std::size_t audio_index = 0;
    for (std::size_t token = 0; token < prompt_tokens; ++token) {
        if (prompt.input_ids[token] != MossPromptProcessor::kAudioTokenId) {
            continue;
        }
        const std::size_t row_bytes =
            static_cast<std::size_t>(kDecoderHidden) * scalar_size;
        std::memcpy(
            input_embeddings.data() + token * row_bytes,
            audio_embeddings.data() + audio_index * row_bytes, row_bytes);
        write_float(
            input_embed_mask, token, 1.0f, decoder_type_);
        ++audio_index;
    }
    if (audio_index != total_audio_tokens) {
        throw std::runtime_error(
            "MOSS prompt did not consume all audio embeddings");
    }

    std::vector<int64_t> positions(prompt_tokens);
    for (std::size_t index = 0; index < prompt_tokens; ++index) {
        positions[index] = static_cast<int64_t>(index);
    }
    std::vector<uint8_t> attention(
        prompt_tokens * prompt_tokens * scalar_size);
    for (std::size_t row = 0; row < prompt_tokens; ++row) {
        for (std::size_t column = 0; column < prompt_tokens; ++column) {
            write_float(
                attention, row * prompt_tokens + column,
                column <= row ? 0.0f : -10000.0f, decoder_type_);
        }
    }
    uint8_t empty_sentinel = 0;
    const std::vector<int64_t> ids_shape = {
        1, static_cast<int64_t>(prompt_tokens)};
    const std::vector<int64_t> embeds_shape = {
        1, static_cast<int64_t>(prompt_tokens), kDecoderHidden};
    const std::vector<int64_t> mask_shape = {
        1, static_cast<int64_t>(prompt_tokens), 1};
    const std::vector<int64_t> positions_shape = {
        static_cast<int64_t>(prompt_tokens)};
    const std::vector<int64_t> attention_shape = {
        1, 1, static_cast<int64_t>(prompt_tokens),
        static_cast<int64_t>(prompt_tokens)};
    const std::vector<int64_t> empty_cache_shape = {
        kDecoderLayers, 1, kDecoderKvHeads, 0, kDecoderHeadDim};

    auto ids_tensor = make_typed_tensor(
        api_, memory, prompt.input_ids,
        ids_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto embeds_tensor = make_raw_tensor(
        api_, memory, input_embeddings.data(), input_embeddings.size(),
        embeds_shape, decoder_type_);
    auto mask_tensor = make_raw_tensor(
        api_, memory, input_embed_mask.data(), input_embed_mask.size(),
        mask_shape, decoder_type_);
    auto positions_tensor = make_typed_tensor(
        api_, memory, positions, positions_shape,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto attention_tensor = make_raw_tensor(
        api_, memory, attention.data(), attention.size(),
        attention_shape, decoder_type_);
    auto past_keys_tensor = make_raw_tensor(
        api_, memory, &empty_sentinel, 0, empty_cache_shape, decoder_type_);
    auto past_values_tensor = make_raw_tensor(
        api_, memory, &empty_sentinel, 0, empty_cache_shape, decoder_type_);

    const char* decoder_input_names[] = {
        "input_ids", "input_embeds", "input_embed_mask",
        "position_ids", "attention_mask", "past_keys", "past_values",
    };
    const char* decoder_output_names[] = {
        "logits", "new_keys", "new_values",
    };
    OrtValue* decoder_inputs[] = {
        ids_tensor.get(), embeds_tensor.get(), mask_tensor.get(),
        positions_tensor.get(), attention_tensor.get(),
        past_keys_tensor.get(), past_values_tensor.get(),
    };
    OrtValue* decoder_outputs[] = {nullptr, nullptr, nullptr};
    const auto prefill_started = Clock::now();
    ort_check(api_, api_->Run(
        decoder_, nullptr, decoder_input_names, decoder_inputs, 7,
        decoder_output_names, 3, decoder_outputs));
    profile.decoder_prefill_ms = milliseconds_since(prefill_started);
    OrtValueHandle logits(api_, decoder_outputs[0]);
    OrtValueHandle new_keys(api_, decoder_outputs[1]);
    OrtValueHandle new_values(api_, decoder_outputs[2]);

    const TensorInfo first_key_info = value_info(api_, new_keys.get());
    const TensorInfo first_value_info = value_info(api_, new_values.get());
    const int64_t first_key_length =
        cache_sequence_length(first_key_info, prompt_tokens);
    const int64_t first_value_length =
        cache_sequence_length(first_value_info, prompt_tokens);
    if (first_key_info.type != decoder_type_
        || first_value_info.type != decoder_type_
        || first_key_length != first_value_length) {
        throw std::runtime_error(
            "MOSS decoder prefill returned incompatible caches");
    }
    std::vector<uint8_t> past_keys =
        copy_tensor_bytes(api_, new_keys.get(), decoder_type_);
    std::vector<uint8_t> past_values =
        copy_tensor_bytes(api_, new_values.get(), decoder_type_);
    // Both buffers reach their final size once, here, instead of growing into
    // a new allocation on every generated token.
    const std::size_t cache_capacity =
        static_cast<std::size_t>(kDecoderLayers * kDecoderKvHeads)
        * static_cast<std::size_t>(
            first_key_length + config_.max_new_tokens)
        * static_cast<std::size_t>(kDecoderHeadDim) * scalar_size;
    std::vector<uint8_t> scratch_keys;
    std::vector<uint8_t> scratch_values;
    past_keys.reserve(cache_capacity);
    past_values.reserve(cache_capacity);
    scratch_keys.reserve(cache_capacity);
    scratch_values.reserve(cache_capacity);
    int64_t past_length = first_key_length;
    int64_t next_token =
        argmax_logits(api_, logits.get(), decoder_type_);
    std::vector<int64_t> generated;
    generated.reserve(static_cast<std::size_t>(config_.max_new_tokens));

    const auto generate_started = Clock::now();
    for (int step = 0; step < config_.max_new_tokens; ++step) {
        generated.push_back(next_token);
        if (next_token == prompt.eos_token_id) break;
        if (cancel_requested_.load(std::memory_order_acquire)) return {};

        std::vector<int64_t> token_ids = {next_token};
        std::vector<uint8_t> zero_embed(
            static_cast<std::size_t>(kDecoderHidden) * scalar_size, 0);
        std::vector<uint8_t> zero_mask(scalar_size, 0);
        std::vector<int64_t> token_position = {past_length};
        std::vector<uint8_t> zero_attention(
            static_cast<std::size_t>(past_length + 1) * scalar_size, 0);
        const std::vector<int64_t> token_ids_shape = {1, 1};
        const std::vector<int64_t> token_embed_shape = {
            1, 1, kDecoderHidden};
        const std::vector<int64_t> token_mask_shape = {1, 1, 1};
        const std::vector<int64_t> token_position_shape = {1};
        const std::vector<int64_t> token_attention_shape = {
            1, 1, 1, past_length + 1};
        const std::vector<int64_t> cache_shape = {
            kDecoderLayers, 1, kDecoderKvHeads,
            past_length, kDecoderHeadDim};

        auto token_ids_tensor = make_typed_tensor(
            api_, memory, token_ids, token_ids_shape,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        auto zero_embed_tensor = make_raw_tensor(
            api_, memory, zero_embed.data(), zero_embed.size(),
            token_embed_shape, decoder_type_);
        auto zero_mask_tensor = make_raw_tensor(
            api_, memory, zero_mask.data(), zero_mask.size(),
            token_mask_shape, decoder_type_);
        auto token_position_tensor = make_typed_tensor(
            api_, memory, token_position, token_position_shape,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        auto zero_attention_tensor = make_raw_tensor(
            api_, memory, zero_attention.data(), zero_attention.size(),
            token_attention_shape, decoder_type_);
        auto past_key_tensor = make_raw_tensor(
            api_, memory, past_keys.data(), past_keys.size(),
            cache_shape, decoder_type_);
        auto past_value_tensor = make_raw_tensor(
            api_, memory, past_values.data(), past_values.size(),
            cache_shape, decoder_type_);
        OrtValue* step_inputs[] = {
            token_ids_tensor.get(), zero_embed_tensor.get(),
            zero_mask_tensor.get(), token_position_tensor.get(),
            zero_attention_tensor.get(), past_key_tensor.get(),
            past_value_tensor.get(),
        };
        OrtValue* step_outputs[] = {nullptr, nullptr, nullptr};
        ort_check(api_, api_->Run(
            decoder_, nullptr, decoder_input_names, step_inputs, 7,
            decoder_output_names, 3, step_outputs));
        OrtValueHandle step_logits(api_, step_outputs[0]);
        OrtValueHandle step_keys(api_, step_outputs[1]);
        OrtValueHandle step_values(api_, step_outputs[2]);
        const TensorInfo key_info = value_info(api_, step_keys.get());
        const TensorInfo value_info_value =
            value_info(api_, step_values.get());
        const int64_t key_delta = cache_sequence_length(key_info, 1);
        const int64_t value_delta =
            cache_sequence_length(value_info_value, 1);
        if (key_info.type != decoder_type_
            || value_info_value.type != decoder_type_
            || key_delta != value_delta) {
            throw std::runtime_error(
                "MOSS decoder step returned incompatible caches");
        }
        const std::vector<uint8_t> delta_keys =
            copy_tensor_bytes(api_, step_keys.get(), decoder_type_);
        const std::vector<uint8_t> delta_values =
            copy_tensor_bytes(api_, step_values.get(), decoder_type_);
        append_cache_into(
            scratch_keys, past_keys, past_length,
            delta_keys, key_delta, scalar_size);
        past_keys.swap(scratch_keys);
        append_cache_into(
            scratch_values, past_values, past_length,
            delta_values, value_delta, scalar_size);
        past_values.swap(scratch_values);
        past_length += key_delta;
        next_token =
            argmax_logits(api_, step_logits.get(), decoder_type_);
    }
    profile.decoder_generate_ms = milliseconds_since(generate_started);
    profile.generated_tokens = static_cast<int>(generated.size());
    profile.total_ms = milliseconds_since(total_started);
    {
        std::lock_guard<std::mutex> profile_lock(profile_mutex_);
        last_profile_ = profile;
    }

    const std::string raw_text = tokenizer_.decode(generated);
    return parse_moss_transcript(raw_text);
}

}  // namespace speech_core
