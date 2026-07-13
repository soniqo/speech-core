#include "speech_core/models/litert_kokoro_tts.h"

#include "tflite_c_api_minimal.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#ifdef __ANDROID__
#include <android/log.h>
#define KOKORO_LOGI(...) \
    __android_log_print(ANDROID_LOG_INFO, "Speech", __VA_ARGS__)
#else
#define KOKORO_LOGI(...)                    \
    do {                                    \
        std::fprintf(stderr, "[speech] ");  \
        std::fprintf(stderr, __VA_ARGS__);  \
        std::fprintf(stderr, "\n");        \
    } while (0)
#endif

namespace speech_core {
namespace {

std::string trim_copy(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

void tflite_check(TfLiteStatus status, const char* operation) {
    if (status != kTfLiteOk) {
        throw std::runtime_error(
            std::string("Kokoro LiteRT: ") + operation + " failed (status=" +
            std::to_string(static_cast<int>(status)) + ")");
    }
}

size_t tensor_element_count(const TfLiteTensor* tensor) {
    const int dimensions = TfLiteTensorNumDims(tensor);
    if (dimensions < 0) return 0;
    size_t count = 1;
    for (int i = 0; i < dimensions; ++i) {
        const int size = TfLiteTensorDim(tensor, i);
        if (size <= 0) return 0;
        count *= static_cast<size_t>(size);
    }
    return count;
}

bool tensor_has_shape(
    const TfLiteTensor* tensor,
    std::initializer_list<int> expected) {
    if (!tensor || TfLiteTensorNumDims(tensor) != static_cast<int>(expected.size())) {
        return false;
    }
    int dimension = 0;
    for (int value : expected) {
        if (TfLiteTensorDim(tensor, dimension++) != value) return false;
    }
    return true;
}

TfLiteTensor* find_input(
    TfLiteInterpreter* interpreter,
    std::initializer_list<int> shape) {
    TfLiteTensor* result = nullptr;
    const int count = TfLiteInterpreterGetInputTensorCount(interpreter);
    for (int i = 0; i < count; ++i) {
        TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
        if (!tensor_has_shape(tensor, shape)) continue;
        if (result) throw std::runtime_error("Kokoro LiteRT: ambiguous input shape");
        result = tensor;
    }
    if (!result) throw std::runtime_error("Kokoro LiteRT: required input shape missing");
    return result;
}

const TfLiteTensor* find_output(
    TfLiteInterpreter* interpreter,
    std::initializer_list<int> shape) {
    const TfLiteTensor* result = nullptr;
    const int count = TfLiteInterpreterGetOutputTensorCount(interpreter);
    for (int i = 0; i < count; ++i) {
        const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
        if (!tensor_has_shape(tensor, shape)) continue;
        if (result) throw std::runtime_error("Kokoro LiteRT: ambiguous output shape");
        result = tensor;
    }
    if (!result) throw std::runtime_error("Kokoro LiteRT: required output shape missing");
    return result;
}

void create_xnnpack_interpreter(
    const std::string& path,
    int num_threads,
    TfLiteModel** model,
    TfLiteOpaqueDelegate** delegate,
    TfLiteInterpreter** interpreter) {
    *model = TfLiteModelCreateFromFile(path.c_str());
    if (!*model) {
        throw std::runtime_error("Kokoro LiteRT: failed to load model: " + path);
    }
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    if (!options) {
        throw std::runtime_error("Kokoro LiteRT: failed to create interpreter options");
    }
    TfLiteInterpreterOptionsSetNumThreads(options, num_threads);
    TfLiteXNNPackDelegateOptions xnnpack_options =
        TfLiteXNNPackDelegateOptionsDefault();
    xnnpack_options.num_threads = num_threads;
    *delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
    if (!*delegate) {
        TfLiteInterpreterOptionsDelete(options);
        throw std::runtime_error("Kokoro LiteRT: failed to create XNNPACK delegate");
    }
    TfLiteInterpreterOptionsAddDelegate(options, *delegate);
    *interpreter = TfLiteInterpreterCreate(*model, options);
    TfLiteInterpreterOptionsDelete(options);
    if (!*interpreter) {
        throw std::runtime_error("Kokoro LiteRT: failed to create interpreter");
    }
    tflite_check(TfLiteInterpreterAllocateTensors(*interpreter), "AllocateTensors");
}

}  // namespace

LiteRTKokoroTts::LiteRTKokoroTts(const std::string& model_path,
                                 const std::string& voices_dir,
                                 const std::string& data_dir,
                                 bool hw_accel,
                                 int num_threads)
    : voices_dir_(voices_dir), num_threads_(num_threads) {
    if (num_threads_ <= 0 || num_threads_ > 64) {
        throw std::invalid_argument("Kokoro LiteRT: num_threads must be in [1, 64]");
    }

    try {
        // This fixed 339 MB graph can use the stable Interpreter C API that
        // libLiteRt exports. Unlike LiteRtCompiledModel 2.1.5, this path wires
        // InterpreterOptionsSetNumThreads through to the default XNNPACK
        // delegate on Windows and Android.
        model_ = TfLiteModelCreateFromFile(model_path.c_str());
        if (!model_) {
            throw std::runtime_error(
                "Kokoro LiteRT: failed to load model: " + model_path);
        }
        TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
        if (!options) {
            throw std::runtime_error("Kokoro LiteRT: failed to create interpreter options");
        }
        TfLiteInterpreterOptionsSetNumThreads(options, num_threads_);
        (void)hw_accel;
        TfLiteXNNPackDelegateOptions xnnpack_options =
            TfLiteXNNPackDelegateOptionsDefault();
        xnnpack_options.num_threads = num_threads_;
        xnnpack_delegate_ = TfLiteXNNPackDelegateCreate(&xnnpack_options);
        if (!xnnpack_delegate_) {
            TfLiteInterpreterOptionsDelete(options);
            throw std::runtime_error(
                "Kokoro LiteRT: failed to create XNNPACK delegate");
        }
        TfLiteInterpreterOptionsAddDelegate(options, xnnpack_delegate_);
        interpreter_ = TfLiteInterpreterCreate(model_, options);
        TfLiteInterpreterOptionsDelete(options);
        if (!interpreter_) {
            throw std::runtime_error("Kokoro LiteRT: failed to create interpreter");
        }
        tflite_check(TfLiteInterpreterAllocateTensors(interpreter_),
                     "AllocateTensors");

        if (TfLiteInterpreterGetInputTensorCount(interpreter_) != 5 ||
            TfLiteInterpreterGetOutputTensorCount(interpreter_) != 3) {
            throw std::runtime_error("Kokoro LiteRT: expected 5 inputs and 3 outputs");
        }
        const std::array<size_t, 5> expected_inputs = {128, 128, 256, 1, 9};
        for (size_t i = 0; i < expected_inputs.size(); ++i) {
            const TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(
                interpreter_, static_cast<int32_t>(i));
            if (!tensor || tensor_element_count(tensor) != expected_inputs[i]) {
                throw std::runtime_error(
                    "Kokoro LiteRT: unexpected input layout at index " +
                    std::to_string(i));
            }
        }

        for (int i = 0; i < 3; ++i) {
            const TfLiteTensor* tensor =
                TfLiteInterpreterGetOutputTensor(interpreter_, i);
            if (!tensor) continue;
            const size_t elements = tensor_element_count(tensor);
            if (elements == static_cast<size_t>(kOutputSamples)) {
                audio_output_idx_ = i;
            } else if (elements == 1) {
                length_output_idx_ = i;
            } else if (elements == static_cast<size_t>(kInputPhonemes)) {
                duration_output_idx_ = i;
            }
        }
        if (audio_output_idx_ < 0 || length_output_idx_ < 0 ||
            duration_output_idx_ < 0) {
            throw std::runtime_error(
                "Kokoro LiteRT: expected audio[36000], length[1], duration[128] outputs");
        }

        load_support_data(data_dir);
    } catch (...) {
        if (interpreter_) {
            TfLiteInterpreterDelete(interpreter_);
            interpreter_ = nullptr;
        }
        if (xnnpack_delegate_) {
            TfLiteXNNPackDelegateDelete(xnnpack_delegate_);
            xnnpack_delegate_ = nullptr;
        }
        if (model_) {
            TfLiteModelDelete(model_);
            model_ = nullptr;
        }
        throw;
    }
}

LiteRTKokoroTts::LiteRTKokoroTts(const std::string& encoder_path,
                                 const std::string& recurrent_path,
                                 const std::string& vocoder_path,
                                 const std::string& voices_dir,
                                 const std::string& data_dir,
                                 bool hw_accel,
                                 int num_threads)
    : staged_(true), voices_dir_(voices_dir), num_threads_(num_threads) {
    if (num_threads_ <= 0 || num_threads_ > 64) {
        throw std::invalid_argument("Kokoro LiteRT: num_threads must be in [1, 64]");
    }
    (void)hw_accel;
    try {
        const std::array<std::string, kStageCount> paths = {
            encoder_path, recurrent_path, vocoder_path};
        for (size_t i = 0; i < kStageCount; ++i) {
            create_xnnpack_interpreter(
                paths[i], num_threads_, &staged_models_[i],
                &staged_delegates_[i], &staged_interpreters_[i]);
        }

        // Fail during construction if a similarly named but incompatible graph
        // is supplied.  Shape-unique contracts also make input/output ordering
        // independent of converter-generated tensor names.
        auto* encoder = staged_interpreters_[0];
        find_input(encoder, {1, kInputPhonemes});
        find_input(encoder, {1, 1, kInputPhonemes});
        find_output(encoder, {1, 512, kInputPhonemes});
        find_output(encoder, {1, kInputPhonemes, 512});

        auto* recurrent = staged_interpreters_[1];
        find_input(recurrent, {1, 512, kInputPhonemes});
        find_input(recurrent, {1, kInputPhonemes, 512});
        find_input(recurrent, {1, kInputPhonemes});
        find_input(recurrent, {1, 1, 128});
        find_input(recurrent, {1});
        find_output(recurrent, {1, 512, kMaxFrames});
        find_output(recurrent, {1, kMaxFrames, 512});
        find_output(recurrent, {1, 1});
        find_output(recurrent, {1, kInputPhonemes});

        auto* vocoder = staged_interpreters_[2];
        find_input(vocoder, {1, 512, kMaxFrames});
        find_input(vocoder, {1, kMaxFrames, 512});
        find_input(vocoder, {1, 1});
        find_input(vocoder, {1, 256});
        find_input(vocoder, {1, 9});
        find_output(vocoder, {1, 1, kOutputSamples});
        find_output(vocoder, {1});

        duration_features_.resize(512 * kInputPhonemes);
        text_features_.resize(kInputPhonemes * 512);
        aligned_text_.resize(512 * kMaxFrames);
        shared_features_.resize(kMaxFrames * 512);
        load_support_data(data_dir);
    } catch (...) {
        for (size_t i = 0; i < kStageCount; ++i) {
            if (staged_interpreters_[i]) {
                TfLiteInterpreterDelete(staged_interpreters_[i]);
                staged_interpreters_[i] = nullptr;
            }
            if (staged_delegates_[i]) {
                TfLiteXNNPackDelegateDelete(staged_delegates_[i]);
                staged_delegates_[i] = nullptr;
            }
            if (staged_models_[i]) {
                TfLiteModelDelete(staged_models_[i]);
                staged_models_[i] = nullptr;
            }
        }
        throw;
    }
}

LiteRTKokoroTts::~LiteRTKokoroTts() {
    if (interpreter_) TfLiteInterpreterDelete(interpreter_);
    if (xnnpack_delegate_) TfLiteXNNPackDelegateDelete(xnnpack_delegate_);
    if (model_) TfLiteModelDelete(model_);
    for (size_t i = 0; i < kStageCount; ++i) {
        if (staged_interpreters_[i]) TfLiteInterpreterDelete(staged_interpreters_[i]);
        if (staged_delegates_[i]) TfLiteXNNPackDelegateDelete(staged_delegates_[i]);
        if (staged_models_[i]) TfLiteModelDelete(staged_models_[i]);
    }
}

void LiteRTKokoroTts::load_support_data(const std::string& data_dir) {
    if (!phonemizer_.load_vocab(data_dir + "/vocab_index.json")) {
        throw std::runtime_error("Kokoro LiteRT: failed to load vocab_index.json");
    }
    if (!phonemizer_.load_dictionaries(data_dir)) {
        throw std::runtime_error(
            "Kokoro LiteRT: failed to load us_gold.json/us_silver.json");
    }
    for (const char* lang : {"fr", "es", "it", "pt", "hi"}) {
        phonemizer_.load_language_dict(
            lang, data_dir + "/dict_" + lang + ".json");
    }
    set_voice("af_heart");
    current_lang_ = "en";
}

void LiteRTKokoroTts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

void LiteRTKokoroTts::set_speed(float speed) {
    if (!std::isfinite(speed) || speed <= 0.0f) {
        throw std::invalid_argument("Kokoro LiteRT: speed must be finite and positive");
    }
    speed_ = speed;
}

std::vector<float> LiteRTKokoroTts::load_voice_embedding(
    const std::string& name) const {
    const std::filesystem::path path =
        std::filesystem::path(voices_dir_) / (name + ".bin");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(
            "Kokoro LiteRT: voice file not found: " + path.string());
    }
    const std::streamsize expected = 256 * static_cast<std::streamsize>(sizeof(float));
    if (file.tellg() != expected) {
        throw std::runtime_error(
            "Kokoro LiteRT: voice embedding must contain exactly 256 float32 values: " +
            path.string());
    }
    file.seekg(0);
    std::vector<float> embedding(256);
    file.read(reinterpret_cast<char*>(embedding.data()), expected);
    if (!file) {
        throw std::runtime_error(
            "Kokoro LiteRT: failed to read voice embedding: " + path.string());
    }
    if (!std::all_of(embedding.begin(), embedding.end(),
                     [](float v) { return std::isfinite(v); })) {
        throw std::runtime_error(
            "Kokoro LiteRT: voice embedding contains non-finite values: " +
            path.string());
    }
    return embedding;
}

void LiteRTKokoroTts::set_voice(const std::string& name) {
    voice_embedding_ = load_voice_embedding(name);
}

void LiteRTKokoroTts::auto_switch_voice(const std::string& language) {
    if (language == current_lang_) return;
    current_lang_ = language;

    struct LangVoice {
        const char* language;
        const char* voice;
    };
    static const LangVoice voices[] = {
        {"en", "af_heart"}, {"fr", "ff_siwis"}, {"es", "ef_dora"},
        {"it", "if_sara"}, {"pt", "pf_dora"},  {"hi", "hf_alpha"},
        {"ja", "jf_alpha"}, {"zh", "zf_xiaobei"}, {"ko", "kf_somi"},
    };

    for (const auto& entry : voices) {
        if (language != entry.language) continue;
        const std::filesystem::path path =
            std::filesystem::path(voices_dir_) / (std::string(entry.voice) + ".bin");
        if (std::filesystem::exists(path)) {
            voice_embedding_ = load_voice_embedding(entry.voice);
            KOKORO_LOGI("Kokoro LiteRT: selected voice %s for language %s",
                        entry.voice, entry.language);
        }
        return;
    }
}

size_t LiteRTKokoroTts::phoneme_count(const std::string& text) {
    // The normal chunks are tiny; this high cap prevents the counting pass
    // from silently accepting a value truncated to the graph's 128-token I/O.
    return phonemizer_.tokenize(text, 32768).size();
}

std::vector<std::string> LiteRTKokoroTts::split_in_half(
    const std::string& text) const {
    const std::string clean = trim_copy(text);
    if (clean.empty()) return {};

    std::vector<std::string> words;
    std::istringstream input(clean);
    for (std::string word; input >> word;) words.push_back(std::move(word));
    if (words.size() >= 2) {
        const size_t middle = words.size() / 2;
        std::string left;
        std::string right;
        for (size_t i = 0; i < words.size(); ++i) {
            std::string& side = i < middle ? left : right;
            if (!side.empty()) side += ' ';
            side += words[i];
        }
        return {std::move(left), std::move(right)};
    }

    // One unusually long word: split only at a UTF-8 code-point boundary.
    std::vector<size_t> boundaries{0};
    for (size_t i = 1; i < clean.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(clean[i]);
        if ((c & 0xC0) != 0x80) boundaries.push_back(i);
    }
    boundaries.push_back(clean.size());
    if (boundaries.size() <= 2) return {clean};
    const size_t split = boundaries[boundaries.size() / 2];
    return {clean.substr(0, split), clean.substr(split)};
}

std::vector<std::string> LiteRTKokoroTts::split_for_phoneme_limit(
    const std::string& text) {
    std::vector<std::string> words;
    std::istringstream input(text);
    for (std::string word; input >> word;) words.push_back(std::move(word));
    if (words.empty()) return {};

    // Fourteen active tokens at speed=1 is the proactive target established by
    // the accepted 60-frame profile (the parity fixture uses 14 tokens / 41
    // frames). Scale it for slower/faster speech, retain the hard 32-token
    // ALBERT ceiling, and keep the post-invoke 56-frame guard authoritative.
    const double scaled = static_cast<double>(kPreferredPhonemes) * speed_;
    const size_t preferred = static_cast<size_t>(
        scaled >= kActivePhonemes
            ? kActivePhonemes
            : std::max(4, static_cast<int>(std::floor(scaled))));

    std::vector<std::string> chunks;
    std::function<void(const std::string&, int)> append_oversized;
    append_oversized = [&](const std::string& value, int depth) {
        if (phoneme_count(value) <= preferred) {
            chunks.push_back(value);
            return;
        }
        if (depth >= 16) {
            throw std::runtime_error(
                "Kokoro LiteRT: could not split text below the preferred phoneme target");
        }
        const auto halves = split_in_half(value);
        if (halves.size() != 2 || halves[0] == value || halves[1] == value ||
            halves[0].empty() || halves[1].empty()) {
            throw std::runtime_error(
                "Kokoro LiteRT: one text unit exceeds the preferred phoneme target");
        }
        append_oversized(halves[0], depth + 1);
        append_oversized(halves[1], depth + 1);
    };

    std::string current;
    for (const auto& word : words) {
        const std::string candidate = current.empty() ? word : current + " " + word;
        if (phoneme_count(candidate) <= preferred) {
            current = candidate;
            continue;
        }
        if (!current.empty()) {
            chunks.push_back(current);
            current.clear();
        }
        if (phoneme_count(word) <= preferred) {
            current = word;
        } else {
            append_oversized(word, 0);
        }
    }
    if (!current.empty()) chunks.push_back(std::move(current));
    return chunks;
}

LiteRTKokoroTts::InferenceResult LiteRTKokoroTts::invoke_tokens(
    const std::vector<int64_t>& tokens,
    std::mt19937& rng) {
    if (tokens.size() > static_cast<size_t>(kActivePhonemes)) {
        throw std::runtime_error("Kokoro LiteRT: invocation exceeds 32 active phonemes");
    }
    if (staged_) return invoke_staged_tokens(tokens, rng);

    std::array<int64_t, kInputPhonemes> input_ids{};
    std::array<int64_t, kInputPhonemes> attention_mask{};
    for (size_t i = 0; i < tokens.size(); ++i) {
        input_ids[i] = tokens[i];
        attention_mask[i] = 1;
    }
    std::array<float, 9> phases{};
    std::uniform_real_distribution<float> phase_dist(0.0f, 1.0f);
    for (float& phase : phases) phase = phase_dist(rng);

    const std::array<const void*, 5> input_data = {
        input_ids.data(), attention_mask.data(), voice_embedding_.data(),
        &speed_, phases.data()};
    const std::array<size_t, 5> input_bytes = {
        sizeof(input_ids), sizeof(attention_mask),
        voice_embedding_.size() * sizeof(float), sizeof(speed_), sizeof(phases)};
    for (size_t i = 0; i < input_data.size(); ++i) {
        TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(
            interpreter_, static_cast<int32_t>(i));
        if (!tensor || TfLiteTensorByteSize(tensor) != input_bytes[i]) {
            throw std::runtime_error(
                "Kokoro LiteRT: input byte size changed at index " +
                std::to_string(i));
        }
        tflite_check(TfLiteTensorCopyFromBuffer(
                         tensor, input_data[i], input_bytes[i]),
                     "TensorCopyFromBuffer");
    }

    ++model_runs_last_;
    tflite_check(TfLiteInterpreterInvoke(interpreter_), "Invoke");

    InferenceResult result;
    result.audio.resize(kOutputSamples);
    const TfLiteTensor* out_audio =
        TfLiteInterpreterGetOutputTensor(interpreter_, audio_output_idx_);
    const TfLiteTensor* out_length =
        TfLiteInterpreterGetOutputTensor(interpreter_, length_output_idx_);
    tflite_check(TfLiteTensorCopyToBuffer(
                     out_audio, result.audio.data(), result.audio.size() * sizeof(float)),
                 "TensorCopyToBuffer(audio)");
    tflite_check(TfLiteTensorCopyToBuffer(
                     out_length, &result.valid_samples, sizeof(result.valid_samples)),
                 "TensorCopyToBuffer(length)");
    // The length tensor reports the unbounded duration prediction. It can be
    // larger than the fixed 36,000-sample waveform buffer; synthesize_piece()
    // uses that signal to discard the truncated waveform and retry smaller
    // text. Only a negative value is structurally invalid here.
    if (result.valid_samples < 0) {
        throw std::runtime_error(
            "Kokoro LiteRT: model returned an invalid audio length: " +
            std::to_string(result.valid_samples));
    }
    return result;
}

LiteRTKokoroTts::InferenceResult LiteRTKokoroTts::invoke_staged_tokens(
    const std::vector<int64_t>& tokens,
    std::mt19937& rng) {
    std::array<int64_t, kInputPhonemes> input_ids{};
    std::array<int64_t, kInputPhonemes> attention_mask{};
    std::array<float, kInputPhonemes> attention_float{};
    for (size_t i = 0; i < tokens.size(); ++i) {
        input_ids[i] = tokens[i];
        attention_mask[i] = 1;
        attention_float[i] = 1.0f;
    }
    std::array<float, 128> style{};
    if (voice_embedding_.size() != 256) {
        throw std::runtime_error("Kokoro LiteRT: voice embedding is not loaded");
    }
    std::copy_n(voice_embedding_.begin() + 128, style.size(), style.begin());
    std::array<float, 9> phases{};
    std::uniform_real_distribution<float> phase_dist(0.0f, 1.0f);
    for (float& phase : phases) phase = phase_dist(rng);

    const auto copy_input = [](
        TfLiteInterpreter* interpreter,
        std::initializer_list<int> shape,
        const void* data,
        size_t bytes,
        const char* label) {
        TfLiteTensor* tensor = find_input(interpreter, shape);
        if (TfLiteTensorByteSize(tensor) != bytes) {
            throw std::runtime_error(
                std::string("Kokoro LiteRT: byte size changed for ") + label);
        }
        tflite_check(TfLiteTensorCopyFromBuffer(tensor, data, bytes), label);
    };
    const auto copy_output = [](
        TfLiteInterpreter* interpreter,
        std::initializer_list<int> shape,
        void* data,
        size_t bytes,
        const char* label) {
        const TfLiteTensor* tensor = find_output(interpreter, shape);
        if (TfLiteTensorByteSize(tensor) != bytes) {
            throw std::runtime_error(
                std::string("Kokoro LiteRT: byte size changed for ") + label);
        }
        tflite_check(TfLiteTensorCopyToBuffer(tensor, data, bytes), label);
    };

    ++model_runs_last_;
    TfLiteInterpreter* encoder = staged_interpreters_[0];
    copy_input(
        encoder, {1, kInputPhonemes}, input_ids.data(), sizeof(input_ids),
        "encoder input_ids");
    copy_input(
        encoder, {1, 1, kInputPhonemes}, attention_mask.data(),
        sizeof(attention_mask), "encoder attention_mask");
    auto stage_started = std::chrono::steady_clock::now();
    tflite_check(TfLiteInterpreterInvoke(encoder), "encoder Invoke");
    last_stage_ms_[0] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stage_started).count();
    copy_output(
        encoder, {1, 512, kInputPhonemes}, duration_features_.data(),
        duration_features_.size() * sizeof(float), "encoder duration output");
    copy_output(
        encoder, {1, kInputPhonemes, 512}, text_features_.data(),
        text_features_.size() * sizeof(float), "encoder text output");

    TfLiteInterpreter* recurrent = staged_interpreters_[1];
    copy_input(
        recurrent, {1, 512, kInputPhonemes}, duration_features_.data(),
        duration_features_.size() * sizeof(float), "recurrent duration input");
    copy_input(
        recurrent, {1, kInputPhonemes, 512}, text_features_.data(),
        text_features_.size() * sizeof(float), "recurrent text input");
    copy_input(
        recurrent, {1, kInputPhonemes}, attention_float.data(),
        sizeof(attention_float), "recurrent attention_mask");
    copy_input(
        recurrent, {1, 1, 128}, style.data(), sizeof(style), "recurrent style");
    copy_input(recurrent, {1}, &speed_, sizeof(speed_), "recurrent speed");
    stage_started = std::chrono::steady_clock::now();
    tflite_check(TfLiteInterpreterInvoke(recurrent), "recurrent Invoke");
    last_stage_ms_[1] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stage_started).count();
    copy_output(
        recurrent, {1, 512, kMaxFrames}, aligned_text_.data(),
        aligned_text_.size() * sizeof(float), "recurrent aligned text output");
    copy_output(
        recurrent, {1, kMaxFrames, 512}, shared_features_.data(),
        shared_features_.size() * sizeof(float), "recurrent shared output");
    float total_frames = 0.0f;
    copy_output(
        recurrent, {1, 1}, &total_frames, sizeof(total_frames),
        "recurrent total frames output");

    TfLiteInterpreter* vocoder = staged_interpreters_[2];
    copy_input(
        vocoder, {1, 512, kMaxFrames}, aligned_text_.data(),
        aligned_text_.size() * sizeof(float), "vocoder aligned text input");
    copy_input(
        vocoder, {1, kMaxFrames, 512}, shared_features_.data(),
        shared_features_.size() * sizeof(float), "vocoder shared input");
    copy_input(
        vocoder, {1, 1}, &total_frames, sizeof(total_frames),
        "vocoder total frames input");
    copy_input(
        vocoder, {1, 256}, voice_embedding_.data(),
        voice_embedding_.size() * sizeof(float), "vocoder voice input");
    copy_input(
        vocoder, {1, 9}, phases.data(), sizeof(phases), "vocoder phases input");
    stage_started = std::chrono::steady_clock::now();
    tflite_check(TfLiteInterpreterInvoke(vocoder), "vocoder Invoke");
    last_stage_ms_[2] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - stage_started).count();

    InferenceResult result;
    result.audio.resize(kOutputSamples);
    copy_output(
        vocoder, {1, 1, kOutputSamples}, result.audio.data(),
        result.audio.size() * sizeof(float), "vocoder audio output");
    copy_output(
        vocoder, {1}, &result.valid_samples, sizeof(result.valid_samples),
        "vocoder length output");
    if (result.valid_samples < 0) {
        throw std::runtime_error(
            "Kokoro LiteRT: staged model returned an invalid audio length: " +
            std::to_string(result.valid_samples));
    }
    return result;
}

std::vector<float> LiteRTKokoroTts::finish_audio(
    std::vector<float> audio,
    size_t valid_samples,
    const std::string& text) {
    valid_samples = std::min(valid_samples, audio.size());
    audio.resize(valid_samples);
    if (audio.empty()) return audio;

    float peak = 0.0f;
    for (float value : audio) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Kokoro LiteRT: non-finite waveform for text '" + text + "'");
        }
        peak = std::max(peak, std::abs(value));
    }
    if (peak > 2.0f) {
        throw std::runtime_error(
            "Kokoro LiteRT: unstable waveform peak " + std::to_string(peak) +
            " for text '" + text + "'");
    }

    constexpr int sample_rate = 24000;
    constexpr float silence_rms = 0.030f;
    const size_t window = sample_rate / 20;  // 50 ms
    size_t speech_end = valid_samples;
    if (valid_samples > window) {
        for (size_t i = valid_samples - window; i > 0; i -= window / 2) {
            double sum_sq = 0.0;
            for (size_t j = 0; j < window; ++j) {
                const float value = audio[i + j];
                sum_sq += static_cast<double>(value) * value;
            }
            const float rms =
                std::sqrt(static_cast<float>(sum_sq / static_cast<double>(window)));
            if (rms > silence_rms) {
                speech_end = i + window;
                break;
            }
            if (i < window / 2) break;
        }
    }
    audio.resize(speech_end);

    const size_t fade_out = std::min<size_t>(speech_end, sample_rate / 100);
    if (fade_out >= 2) {
        const size_t start = speech_end - fade_out;
        const float denominator = static_cast<float>(fade_out - 1);
        for (size_t i = 0; i < fade_out; ++i) {
            audio[start + i] *=
                static_cast<float>(fade_out - 1 - i) / denominator;
        }
    }
    const size_t fade_in = std::min<size_t>(120, speech_end);
    if (fade_in >= 2) {
        const float denominator = static_cast<float>(fade_in - 1);
        for (size_t i = 0; i < fade_in; ++i) {
            audio[i] *= static_cast<float>(i) / denominator;
        }
    }
    return audio;
}

std::vector<float> LiteRTKokoroTts::synthesize_piece(
    const std::string& text,
    std::mt19937& rng,
    int depth) {
    if (cancelled_.load(std::memory_order_relaxed)) return {};
    if (depth >= 16) {
        throw std::runtime_error(
            "Kokoro LiteRT: exceeded the guarded re-chunking depth");
    }

    const auto tokens = phonemizer_.tokenize(text, kInputPhonemes);
    if (tokens.size() > static_cast<size_t>(kActivePhonemes)) {
        throw std::runtime_error(
            "Kokoro LiteRT: internal chunk exceeds 32 active phonemes");
    }
    InferenceResult result = invoke_tokens(tokens, rng);
    const int64_t safe_samples =
        static_cast<int64_t>(kSafeFrames) * kSamplesPerFrame;
    if (result.valid_samples <= safe_samples) {
        return finish_audio(
            std::move(result.audio), static_cast<size_t>(result.valid_samples), text);
    }

    // Outputs in frames 57-60 are inside the tensor but outside the validated
    // right-context guard band. Discard this waveform and retry smaller text.
    const auto halves = split_in_half(text);
    if (halves.size() != 2 || halves[0] == text || halves[1] == text ||
        halves[0].empty() || halves[1].empty()) {
        throw std::runtime_error(
            "Kokoro LiteRT: text exceeds the 56-frame safety ceiling and cannot be split");
    }
    KOKORO_LOGI("Kokoro LiteRT: retrying %lld-sample chunk as two guarded chunks",
                static_cast<long long>(result.valid_samples));
    std::vector<float> left = synthesize_piece(halves[0], rng, depth + 1);
    std::vector<float> right = synthesize_piece(halves[1], rng, depth + 1);
    left.insert(left.end(), right.begin(), right.end());
    return left;
}

void LiteRTKokoroTts::synthesize(const std::string& text,
                                 const std::string& language,
                                 TTSChunkCallback on_chunk) {
    if (!on_chunk) {
        throw std::invalid_argument("Kokoro LiteRT: on_chunk callback is empty");
    }
    cancelled_.store(false, std::memory_order_relaxed);
    model_runs_last_ = 0;

    const std::string clean = trim_copy(text);
    if (clean.empty()) return;

    const std::string lang = language.empty() ? "en" : language;
    phonemizer_.set_language(lang);
    auto_switch_voice(lang);

    seed_used_ = seed_;
    if (seed_used_ == 0) {
        std::random_device random_device;
        seed_used_ = random_device();
    }
    std::mt19937 rng(seed_used_);

    const auto pieces = split_for_phoneme_limit(clean);
    std::vector<float> audio;
    for (const auto& piece : pieces) {
        if (cancelled_.load(std::memory_order_relaxed)) return;
        std::vector<float> pcm = synthesize_piece(piece, rng, 0);
        audio.insert(audio.end(), pcm.begin(), pcm.end());
    }
    if (!cancelled_.load(std::memory_order_relaxed) && !audio.empty()) {
        KOKORO_LOGI("Kokoro LiteRT: chunks=%zu runs=%d samples=%zu seed=%u",
                    pieces.size(), model_runs_last_, audio.size(), seed_used_);
        on_chunk(audio.data(), audio.size(), true);
    }
}

}  // namespace speech_core

#undef KOKORO_LOGI
