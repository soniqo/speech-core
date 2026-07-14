#include "speech_core/models/onnx_pocket_tts.h"

#include "speech_core/models/onnx_engine.h"
#include "speech_core/models/pocket_tts_tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace speech_core {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string path_join(const std::string& directory, const char* filename) {
    if (directory.empty()) return filename;
    const char tail = directory.back();
    if (tail == '/' || tail == '\\') return directory + filename;
#ifdef _WIN32
    return directory + "\\" + filename;
#else
    return directory + "/" + filename;
#endif
}

bool is_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    return stream.good();
}

std::size_t element_size(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(bool);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(std::int64_t);
        default:
            throw std::runtime_error("Pocket TTS encountered an unsupported ONNX tensor type");
    }
}

std::size_t shape_element_count(const std::vector<std::int64_t>& shape) {
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("Pocket TTS encountered an unresolved tensor dimension");
        }
        if (dimension == 0) return 0;
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

struct ValueDeleter {
    const OrtApi* api = nullptr;
    void operator()(OrtValue* value) const {
        if (api && value) api->ReleaseValue(value);
    }
};
using ValuePtr = std::unique_ptr<OrtValue, ValueDeleter>;

struct SessionDeleter {
    const OrtApi* api = nullptr;
    void operator()(OrtSession* session) const {
        if (api && session) api->ReleaseSession(session);
    }
};
using SessionPtr = std::unique_ptr<OrtSession, SessionDeleter>;

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

}  // namespace

class OnnxPocketTts::Impl {
public:
    Impl(const std::string& bundle_directory, PocketTtsConfig config)
        : config_(config) {
        validate_config();

        const auto lm_main_path = required_path(bundle_directory, "lm_main.int8.onnx");
        const auto lm_flow_path = required_path(bundle_directory, "lm_flow.int8.onnx");
        const auto decoder_path = required_path(bundle_directory, "decoder.int8.onnx");
        const auto encoder_path = required_path(bundle_directory, "encoder.onnx");
        const auto conditioner_path = required_path(bundle_directory, "text_conditioner.onnx");
        const auto vocabulary_path = required_path(bundle_directory, "vocab.json");
        const auto scores_path = required_path(bundle_directory, "token_scores.json");

        auto& engine = OnnxEngine::get();
        api_ = engine.api();
        memory_ = engine.cpu_memory();
        ort_check(api_, api_->GetAllocatorWithDefaultOptions(&allocator_));

        lm_main_ = load_session(engine, lm_main_path, true);
        lm_flow_ = load_session(engine, lm_flow_path, true);
        decoder_ = load_session(engine, decoder_path, true);
        conditioner_ = load_session(engine, conditioner_path, false);
        encoder_ = load_session(engine, encoder_path, false);

        lm_main_io_ = query_io(lm_main_.get());
        lm_flow_io_ = query_io(lm_flow_.get());
        decoder_io_ = query_io(decoder_.get());
        conditioner_io_ = query_io(conditioner_.get());
        encoder_io_ = query_io(encoder_.get());
        validate_model_contract();

        tokenizer_ = std::make_unique<PocketTtsTokenizer>(vocabulary_path, scores_path);
        lm_cache_length_ = query_lm_cache_length();
        voice_embedding_ = create_voice_embedding();
        const auto voice_shape = tensor_shape(voice_embedding_.get());
        if (voice_shape.size() != 3 || voice_shape[0] != 1 || voice_shape[2] != 1024) {
            throw std::runtime_error("Pocket TTS fixed voice embedding has an unexpected shape");
        }
        voice_tokens_ = static_cast<int>(voice_shape[1]);

        // The public encoder is a tiny fixed-voice adapter. Its output owns
        // the cached Alba prompt, so the session itself is no longer needed.
        encoder_.reset();
    }

    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) {
        static_cast<void>(language);  // The public bundle is English-only.
        std::lock_guard<std::mutex> synthesis_lock(synthesis_mutex_);
        cancelled_.store(false);

        PocketTtsMetrics metrics;
        const auto started = Clock::now();

        if (!on_chunk) {
            throw std::invalid_argument("Pocket TTS requires a chunk callback");
        }
        if (text.empty()) {
            on_chunk(nullptr, 0, true);
            metrics.total_ms = elapsed_ms(started, Clock::now());
            publish_metrics(metrics);
            return;
        }

        const auto ids32 = tokenizer_->encode_ids(text);
        if (ids32.empty()) {
            on_chunk(nullptr, 0, true);
            metrics.total_ms = elapsed_ms(started, Clock::now());
            publish_metrics(metrics);
            return;
        }

        const int remaining_cache =
            lm_cache_length_ - voice_tokens_ - static_cast<int>(ids32.size());
        if (remaining_cache <= 0) {
            throw std::length_error(
                "Pocket TTS text and voice conditioning exceed the 1000-token LM cache");
        }
        const int frame_limit = std::min(config_.max_frames, remaining_cache);

        std::vector<std::int64_t> ids(ids32.begin(), ids32.end());
        const std::vector<std::int64_t> token_shape = {
            1, static_cast<std::int64_t>(ids.size())};
        auto token_tensor = tensor_view(ids.data(), ids.size(), token_shape,
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        const std::vector<const OrtValue*> conditioner_inputs = {token_tensor.get()};
        auto conditioner_outputs = run(conditioner_.get(), conditioner_io_, conditioner_inputs);
        auto text_embedding = std::move(conditioner_outputs[0]);

        State lm_state = initial_state(lm_main_.get(), 2);
        auto empty_sequence = owned_tensor(
            {1, 0, 32}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);

        run_lm(empty_sequence.get(), voice_embedding_.get(), lm_state);
        run_lm(empty_sequence.get(), text_embedding.get(), lm_state);
        text_embedding.reset();

        metrics.conditioning_ms = elapsed_ms(started, Clock::now());
        State decoder_state = initial_state(decoder_.get(), 1);
        auto empty_text = owned_tensor(
            {1, 0, 1024}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, true);

        std::vector<float> current(32, std::numeric_limits<float>::quiet_NaN());
        std::vector<float> noise(32, 0.0f);
        const auto seed = resolve_seed();
        metrics.seed_used = seed;
        std::mt19937 rng(seed);
        const float noise_stddev = std::sqrt(config_.temperature);

        int eos_frame = -1;
        bool first_audio = true;
        for (int frame = 0; frame < frame_limit; ++frame) {
            if (cancelled_.load()) break;

            auto sequence = tensor_view(
                current.data(), current.size(), {1, 1, 32},
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            auto lm_result = run_lm(sequence.get(), empty_text.get(), lm_state);
            auto conditioning = std::move(lm_result.first);
            auto eos_logit = std::move(lm_result.second);

            const float eos_score = first_float(eos_logit.get());
            if (eos_frame < 0 && eos_score > config_.eos_threshold) {
                eos_frame = frame;
            }
            if (eos_frame >= 0 && frame >= eos_frame + config_.frames_after_eos) {
                metrics.stopped_on_eos = true;
                break;
            }

            // Recreate the distribution on every frame to match sherpa-onnx's
            // deterministic seeded reference implementation.
            std::normal_distribution<float> distribution(0.0f, noise_stddev);
            for (float& value : noise) value = distribution(rng);

            current = run_flow(conditioning.get(), noise);
            auto latent = tensor_view(
                current.data(), current.size(), {1, 1, 32},
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            auto audio = run_decoder(latent.get(), decoder_state);
            const std::size_t sample_count = tensor_element_count(audio.get());
            if (sample_count != 1920) {
                throw std::runtime_error(
                    "Pocket TTS Mimi decoder did not return one 1,920-sample frame");
            }
            float* samples = nullptr;
            ort_check(api_, api_->GetTensorMutableData(audio.get(),
                                                       reinterpret_cast<void**>(&samples)));

            if (first_audio) {
                metrics.first_audio_ms = elapsed_ms(started, Clock::now());
                first_audio = false;
            }
            on_chunk(samples, sample_count, false);
            ++metrics.frames_generated;
            metrics.output_samples += static_cast<int>(sample_count);
        }

        metrics.cancelled = cancelled_.load();
        on_chunk(nullptr, 0, true);
        metrics.total_ms = elapsed_ms(started, Clock::now());
        publish_metrics(metrics);
    }

    void cancel() { cancelled_.store(true); }

    PocketTtsMetrics last_metrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        return last_metrics_;
    }

private:
    struct IoNames {
        std::vector<std::string> input_strings;
        std::vector<std::string> output_strings;
        std::vector<const char*> inputs;
        std::vector<const char*> outputs;
    };

    struct State {
        std::vector<ValuePtr> values;
    };

    void validate_config() const {
        if (config_.flow_steps < 1 || config_.flow_steps > 32) {
            throw std::invalid_argument("Pocket TTS flow_steps must be in [1, 32]");
        }
        if (config_.max_frames < 1 || config_.max_frames > 1000) {
            throw std::invalid_argument("Pocket TTS max_frames must be in [1, 1000]");
        }
        if (config_.frames_after_eos < 0 || config_.frames_after_eos > 50) {
            throw std::invalid_argument("Pocket TTS frames_after_eos must be in [0, 50]");
        }
        if (!std::isfinite(config_.temperature) || config_.temperature < 0.0f ||
            config_.temperature > 10.0f) {
            throw std::invalid_argument("Pocket TTS temperature must be finite and in [0, 10]");
        }
        if (!std::isfinite(config_.eos_threshold)) {
            throw std::invalid_argument("Pocket TTS eos_threshold must be finite");
        }
        if (config_.intra_threads < 1 || config_.intra_threads > 64) {
            throw std::invalid_argument("Pocket TTS intra_threads must be in [1, 64]");
        }
    }

    std::string required_path(const std::string& directory, const char* filename) const {
        const auto path = path_join(directory, filename);
        if (!is_file(path)) {
            throw std::runtime_error("Missing Pocket TTS bundle file: " + path);
        }
        return path;
    }

    SessionPtr load_session(OnnxEngine& engine,
                            const std::string& path,
                            bool capture_hint) const {
        OrtSession* raw = engine.load(
            path, config_.hardware_acceleration, capture_hint, config_.intra_threads);
        return SessionPtr(raw, SessionDeleter{api_});
    }

    ValuePtr adopt(OrtValue* value) const {
        return ValuePtr(value, ValueDeleter{api_});
    }

    IoNames query_io(OrtSession* session) const {
        IoNames names;
        size_t count = 0;
        ort_check(api_, api_->SessionGetInputCount(session, &count));
        names.input_strings.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            OrtStringHandle handle{api_, allocator_, nullptr};
            ort_check(api_, api_->SessionGetInputName(
                session, index, allocator_, &handle.value));
            names.input_strings.emplace_back(handle.value);
        }

        ort_check(api_, api_->SessionGetOutputCount(session, &count));
        names.output_strings.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            OrtStringHandle handle{api_, allocator_, nullptr};
            ort_check(api_, api_->SessionGetOutputName(
                session, index, allocator_, &handle.value));
            names.output_strings.emplace_back(handle.value);
        }

        names.inputs.reserve(names.input_strings.size());
        for (const auto& name : names.input_strings) names.inputs.push_back(name.c_str());
        names.outputs.reserve(names.output_strings.size());
        for (const auto& name : names.output_strings) names.outputs.push_back(name.c_str());
        return names;
    }

    void validate_model_contract() const {
        require_io(encoder_io_, 1, 1, "audio", "latents", "encoder");
        require_io(conditioner_io_, 1, 1, "token_ids", "embeddings", "text conditioner");
        require_io(lm_flow_io_, 4, 1, "c", "flow_dir", "flow model");
        require_io(lm_main_io_, 20, 20, "sequence", "conditioning", "LM main");
        require_io(decoder_io_, 57, 57, "latent", "audio_frame", "Mimi decoder");
    }

    static void require_io(const IoNames& names,
                           std::size_t inputs,
                           std::size_t outputs,
                           const char* first_input,
                           const char* first_output,
                           const char* label) {
        if (names.input_strings.size() != inputs ||
            names.output_strings.size() != outputs ||
            names.input_strings.empty() || names.output_strings.empty() ||
            names.input_strings.front() != first_input ||
            names.output_strings.front() != first_output) {
            throw std::runtime_error(
                std::string("Pocket TTS ") + label + " graph has an incompatible I/O contract");
        }
    }

    std::pair<std::vector<std::int64_t>, ONNXTensorElementDataType>
    input_shape_and_type(OrtSession* session, std::size_t input_index) const {
        OrtTypeInfo* type_info = nullptr;
        ort_check(api_, api_->SessionGetInputTypeInfo(session, input_index, &type_info));
        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        OrtStatus* cast_status = api_->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        if (cast_status != nullptr || tensor_info == nullptr) {
            if (cast_status) api_->ReleaseStatus(cast_status);
            api_->ReleaseTypeInfo(type_info);
            throw std::runtime_error("Pocket TTS model input is not a tensor");
        }

        ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        size_t rank = 0;
        ort_check(api_, api_->GetTensorElementType(tensor_info, &type));
        ort_check(api_, api_->GetDimensionsCount(tensor_info, &rank));
        std::vector<std::int64_t> shape(rank);
        ort_check(api_, api_->GetDimensions(tensor_info, shape.data(), rank));
        api_->ReleaseTypeInfo(type_info);
        return {std::move(shape), type};
    }

    int query_lm_cache_length() const {
        auto shape_and_type = input_shape_and_type(lm_main_.get(), 2);
        const auto& shape = shape_and_type.first;
        if (shape.size() != 5 || shape[2] <= 0) {
            throw std::runtime_error("Pocket TTS LM cache input has an unexpected shape");
        }
        return static_cast<int>(shape[2]);
    }

    State initial_state(OrtSession* session, std::size_t first_state_input) const {
        size_t input_count = 0;
        ort_check(api_, api_->SessionGetInputCount(session, &input_count));
        State state;
        state.values.reserve(input_count - first_state_input);
        for (std::size_t index = first_state_input; index < input_count; ++index) {
            auto shape_and_type = input_shape_and_type(session, index);
            for (auto& dimension : shape_and_type.first) {
                if (dimension < 0) dimension = 1;
            }
            state.values.push_back(owned_tensor(
                shape_and_type.first, shape_and_type.second, true));
        }
        return state;
    }

    ValuePtr owned_tensor(const std::vector<std::int64_t>& shape,
                          ONNXTensorElementDataType type,
                          bool zero) const {
        OrtValue* raw = nullptr;
        ort_check(api_, api_->CreateTensorAsOrtValue(
            allocator_, shape.data(), shape.size(), type, &raw));
        auto value = adopt(raw);
        if (zero) {
            const auto count = shape_element_count(shape);
            if (count > 0) {
                void* data = nullptr;
                ort_check(api_, api_->GetTensorMutableData(value.get(), &data));
                std::memset(data, 0, count * element_size(type));
            }
        }
        return value;
    }

    template <typename T>
    ValuePtr tensor_view(T* data,
                         std::size_t count,
                         const std::vector<std::int64_t>& shape,
                         ONNXTensorElementDataType type) const {
        static T empty_value{};
        OrtValue* raw = nullptr;
        T* pointer = count == 0 ? &empty_value : data;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            memory_, pointer, count * sizeof(T), shape.data(), shape.size(), type, &raw));
        return adopt(raw);
    }

    std::vector<ValuePtr> run(OrtSession* session,
                              const IoNames& names,
                              const std::vector<const OrtValue*>& inputs) const {
        if (inputs.size() != names.inputs.size()) {
            throw std::runtime_error("Pocket TTS internal input count mismatch");
        }
        std::vector<OrtValue*> raw_outputs(names.outputs.size(), nullptr);
        OrtStatus* status = api_->Run(
            session, nullptr, names.inputs.data(), inputs.data(), inputs.size(),
            names.outputs.data(), names.outputs.size(), raw_outputs.data());
        if (status != nullptr) {
            for (auto* output : raw_outputs) {
                if (output) api_->ReleaseValue(output);
            }
            ort_check(api_, status);
        }

        std::vector<ValuePtr> outputs;
        outputs.reserve(raw_outputs.size());
        for (auto* output : raw_outputs) outputs.push_back(adopt(output));
        return outputs;
    }

    std::pair<ValuePtr, ValuePtr> run_lm(const OrtValue* sequence,
                                         const OrtValue* embedding,
                                         State& state) const {
        std::vector<const OrtValue*> inputs;
        inputs.reserve(2 + state.values.size());
        inputs.push_back(sequence);
        inputs.push_back(embedding);
        for (const auto& value : state.values) inputs.push_back(value.get());

        auto outputs = run(lm_main_.get(), lm_main_io_, inputs);
        std::vector<ValuePtr> new_state;
        new_state.reserve(outputs.size() - 2);
        for (std::size_t index = 2; index < outputs.size(); ++index) {
            new_state.push_back(std::move(outputs[index]));
        }
        state.values = std::move(new_state);
        return {std::move(outputs[0]), std::move(outputs[1])};
    }

    std::vector<float> run_flow(const OrtValue* conditioning,
                                const std::vector<float>& noise) const {
        std::vector<float> latent = noise;
        float start = 0.0f;
        float end = 0.0f;
        const std::vector<std::int64_t> scalar_shape = {1, 1};
        auto start_tensor = tensor_view(
            &start, 1, scalar_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto end_tensor = tensor_view(
            &end, 1, scalar_shape, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto latent_tensor = tensor_view(
            latent.data(), latent.size(), {1, 32}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const float delta = 1.0f / static_cast<float>(config_.flow_steps);

        for (int step = 0; step < config_.flow_steps; ++step) {
            start = static_cast<float>(step) / static_cast<float>(config_.flow_steps);
            end = start + delta;
            const std::vector<const OrtValue*> inputs = {
                conditioning, start_tensor.get(), end_tensor.get(), latent_tensor.get()};
            auto outputs = run(lm_flow_.get(), lm_flow_io_, inputs);
            if (tensor_element_count(outputs[0].get()) != latent.size()) {
                throw std::runtime_error("Pocket TTS flow graph returned an unexpected shape");
            }
            float* direction = nullptr;
            ort_check(api_, api_->GetTensorMutableData(
                outputs[0].get(), reinterpret_cast<void**>(&direction)));
            for (std::size_t index = 0; index < latent.size(); ++index) {
                latent[index] += direction[index] * delta;
            }
        }
        return latent;
    }

    ValuePtr run_decoder(const OrtValue* latent, State& state) const {
        std::vector<const OrtValue*> inputs;
        inputs.reserve(1 + state.values.size());
        inputs.push_back(latent);
        for (const auto& value : state.values) inputs.push_back(value.get());

        auto outputs = run(decoder_.get(), decoder_io_, inputs);
        std::vector<ValuePtr> new_state;
        new_state.reserve(outputs.size() - 1);
        for (std::size_t index = 1; index < outputs.size(); ++index) {
            new_state.push_back(std::move(outputs[index]));
        }
        state.values = std::move(new_state);
        return std::move(outputs[0]);
    }

    ValuePtr create_voice_embedding() const {
        float placeholder = 0.0f;
        auto audio = tensor_view(
            &placeholder, 1, {1, 1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        const std::vector<const OrtValue*> inputs = {audio.get()};
        auto outputs = run(encoder_.get(), encoder_io_, inputs);
        return std::move(outputs[0]);
    }

    std::vector<std::int64_t> tensor_shape(const OrtValue* value) const {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(value, &info));
        size_t rank = 0;
        ort_check(api_, api_->GetDimensionsCount(info, &rank));
        std::vector<std::int64_t> shape(rank);
        ort_check(api_, api_->GetDimensions(info, shape.data(), rank));
        api_->ReleaseTensorTypeAndShapeInfo(info);
        return shape;
    }

    std::size_t tensor_element_count(const OrtValue* value) const {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(value, &info));
        size_t count = 0;
        ort_check(api_, api_->GetTensorShapeElementCount(info, &count));
        api_->ReleaseTensorTypeAndShapeInfo(info);
        return count;
    }

    float first_float(OrtValue* value) const {
        if (tensor_element_count(value) < 1) {
            throw std::runtime_error("Pocket TTS EOS output is empty");
        }
        float* data = nullptr;
        ort_check(api_, api_->GetTensorMutableData(
            value, reinterpret_cast<void**>(&data)));
        return data[0];
    }

    std::uint32_t resolve_seed() const {
        if (config_.seed >= 0) return static_cast<std::uint32_t>(config_.seed);
        std::random_device source;
        std::seed_seq sequence{source(), source(), source(), source()};
        std::vector<std::uint32_t> output(1);
        sequence.generate(output.begin(), output.end());
        return output[0];
    }

    void publish_metrics(const PocketTtsMetrics& metrics) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        last_metrics_ = metrics;
    }

    PocketTtsConfig config_;
    const OrtApi* api_ = nullptr;
    OrtMemoryInfo* memory_ = nullptr;
    OrtAllocator* allocator_ = nullptr;

    SessionPtr lm_main_{nullptr, SessionDeleter{}};
    SessionPtr lm_flow_{nullptr, SessionDeleter{}};
    SessionPtr decoder_{nullptr, SessionDeleter{}};
    SessionPtr conditioner_{nullptr, SessionDeleter{}};
    SessionPtr encoder_{nullptr, SessionDeleter{}};
    IoNames lm_main_io_;
    IoNames lm_flow_io_;
    IoNames decoder_io_;
    IoNames conditioner_io_;
    IoNames encoder_io_;

    std::unique_ptr<PocketTtsTokenizer> tokenizer_;
    ValuePtr voice_embedding_{nullptr, ValueDeleter{}};
    int voice_tokens_ = 0;
    int lm_cache_length_ = 0;

    std::mutex synthesis_mutex_;
    std::atomic<bool> cancelled_{false};
    mutable std::mutex metrics_mutex_;
    PocketTtsMetrics last_metrics_;
};

OnnxPocketTts::OnnxPocketTts(const std::string& bundle_directory,
                             PocketTtsConfig config)
    : impl_(std::make_unique<Impl>(bundle_directory, config)) {}

OnnxPocketTts::~OnnxPocketTts() = default;

void OnnxPocketTts::synthesize(const std::string& text,
                               const std::string& language,
                               TTSChunkCallback on_chunk) {
    impl_->synthesize(text, language, std::move(on_chunk));
}

void OnnxPocketTts::cancel() {
    impl_->cancel();
}

PocketTtsMetrics OnnxPocketTts::last_metrics() const {
    return impl_->last_metrics();
}

}  // namespace speech_core
