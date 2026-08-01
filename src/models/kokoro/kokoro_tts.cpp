#include "speech_core/models/kokoro_tts.h"

#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/text_chunker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace speech_core {

static constexpr int MAX_PHONEMES = 128;

// Chunks below this are synthesized unreliably (the numerical-stability
// guard below drops ≤5-token outputs); merge such tails into the previous
// chunk when the selected graph profile's hard budget permits it.
static constexpr size_t MIN_TAIL_TOKENS = 12;
static constexpr size_t MIN_RETRY_TOKENS = 6;
static constexpr size_t MAX_RETRY_DEPTH = 4;
static constexpr size_t MAX_INFERENCE_ATTEMPTS = 15;
// Parity testing found a sharp quality cliff in the final two 25 ms decoder
// frames of the bounded export. Reserve a conservative eight-frame (200 ms)
// margin for the initial profile, regardless of profile-specific limits.
static constexpr size_t OUTPUT_TAIL_GUARD_SAMPLES = 4800;

namespace {

constexpr size_t guarded_output_capacity(
    size_t tensor_capacity, size_t configured_limit) {
    size_t safe = tensor_capacity > OUTPUT_TAIL_GUARD_SAMPLES
        ? tensor_capacity - OUTPUT_TAIL_GUARD_SAMPLES
        : 0;
    return configured_limit > 0 && configured_limit < safe
        ? configured_limit
        : safe;
}

static_assert(61200 <= guarded_output_capacity(72000, 67200));
static_assert(67800 > guarded_output_capacity(72000, 67200));
static_assert(71400 > guarded_output_capacity(72000, 67200));
static_assert(73800 <= guarded_output_capacity(84000, 79200));
static_assert(79800 > guarded_output_capacity(84000, 79200));
static_assert(83400 > guarded_output_capacity(84000, 79200));

class OrtValuesGuard {
public:
    OrtValuesGuard(const OrtApi* api, OrtValue** values, size_t count)
        : api_(api), values_(values), count_(count) {}

    ~OrtValuesGuard() {
        for (size_t i = count_; i > 0; --i) {
            if (values_[i - 1]) api_->ReleaseValue(values_[i - 1]);
        }
    }

    OrtValuesGuard(const OrtValuesGuard&) = delete;
    OrtValuesGuard& operator=(const OrtValuesGuard&) = delete;

private:
    const OrtApi* api_;
    OrtValue** values_;
    size_t count_;
};

int resolve_kokoro_threads() {
    if (const char* value = std::getenv("SPEECH_CORE_KOKORO_ORT_THREADS")) {
        const int threads = std::atoi(value);
        if (threads > 0) return threads;
    }
    if (const char* value = std::getenv("SPEECH_CORE_ORT_THREADS")) {
        const int threads = std::atoi(value);
        if (threads > 0) return threads;
    }
    // Kokoro is one large Conv-heavy Run, unlike Parakeet's many tiny decoder
    // calls that motivated OnnxEngine's conservative two-thread default.
    return 4;
}

}  // namespace

KokoroTts::Config KokoroTts::Config::short_turn_3s(bool hw_accel) {
    Config config;
    config.hw_accel = hw_accel;
    config.chunk_token_budget = 44;
    config.chunk_token_hard_cap = 49;
    config.max_safe_output_samples = 67200;
    return config;
}

KokoroTts::Config KokoroTts::Config::default_for_model_path(
    const std::string& model_path, bool hw_accel) {
    constexpr const char* kRealtimeGraph = "kokoro-e2e-realtime.onnx";
    const size_t slash = model_path.find_last_of("/\\");
    const std::string filename = slash == std::string::npos
        ? model_path
        : model_path.substr(slash + 1);
    if (filename == kRealtimeGraph) return short_turn_3s(hw_accel);
    return Config{hw_accel, 72, MAX_PHONEMES, 0};
}

KokoroTts::Config KokoroTts::Config::short_turn_3p5s(bool hw_accel) {
    Config config;
    config.hw_accel = hw_accel;
    config.chunk_token_budget = 44;
    config.chunk_token_hard_cap = 49;
    config.max_safe_output_samples = 79200;
    return config;
}

KokoroTts::KokoroTts(
    const std::string& model_path,
    const std::string& voices_dir,
    const std::string& data_dir,
    bool hw_accel)
    : KokoroTts(model_path, voices_dir, data_dir,
                Config::default_for_model_path(model_path, hw_accel))
{}

KokoroTts::KokoroTts(
    const std::string& model_path,
    const std::string& voices_dir,
    const std::string& data_dir,
    const Config& config)
    : voices_dir_(voices_dir), config_(config)
{
    if (config_.chunk_token_budget == 0 ||
        config_.chunk_token_budget > MAX_PHONEMES ||
        config_.chunk_token_hard_cap < config_.chunk_token_budget ||
        config_.chunk_token_hard_cap > MAX_PHONEMES) {
        throw std::invalid_argument("invalid Kokoro chunk token budgets");
    }

    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    session_ = engine.load(model_path, config_.hw_accel,
                           /*capture_hint=*/false,
                           resolve_kokoro_threads());

    // Load phonemizer vocabulary and dictionaries
    phonemizer_.load_vocab(data_dir + "/vocab_index.json");
    phonemizer_.load_dictionaries(data_dir);

    // Load optional non-English pronunciation dictionaries
    for (const char* lang : {"fr", "es", "it", "pt", "hi"}) {
        phonemizer_.load_language_dict(lang,
            data_dir + "/dict_" + lang + ".json");
    }

    // Load default voice
    set_voice("af_heart");
    voice_overridden_ = false;
    current_lang_ = "en";
}

KokoroTts::~KokoroTts() {
    if (session_) api_->ReleaseSession(session_);
}

void KokoroTts::set_voice(const std::string& name) {
    voice_embedding_ = load_voice_embedding(name);
    voice_overridden_ = true;
}

void KokoroTts::set_speed(float speed) {
    if (!std::isfinite(speed) || speed < 0.25f || speed > 4.0f) {
        throw std::invalid_argument("Kokoro speed must be in 0.25...4.0");
    }
    speed_ = speed;
}

std::vector<float> KokoroTts::load_voice_embedding(const std::string& name) {
    std::string path = voices_dir_ + "/" + name + ".bin";
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Voice file not found: %s", path.c_str());
        return std::vector<float>(256, 0.0f);
    }

    std::vector<float> embedding(256);
    file.read(reinterpret_cast<char*>(embedding.data()), 256 * sizeof(float));
    return embedding;
}

void KokoroTts::auto_switch_voice(const std::string& lang) {
    if (lang == current_lang_) return;
    current_lang_ = lang;
    if (voice_overridden_) return;

    // Map language to default voice
    struct LangVoice { const char* lang; const char* voice; };
    static const LangVoice map[] = {
        {"en", "af_heart"},
        {"fr", "ff_siwis"},
        {"es", "ef_dora"},
        {"it", "if_sara"},
        {"pt", "pf_dora"},
        {"hi", "hf_alpha"},
        {"ja", "jf_alpha"},
        {"zh", "zf_xiaobei"},
        {"ko", "kf_somi"},
    };

    for (auto& entry : map) {
        if (lang == entry.lang) {
            auto emb = load_voice_embedding(entry.voice);
            if (emb[0] != 0.0f || emb[1] != 0.0f) {  // check not zeroed (missing file)
                voice_embedding_ = std::move(emb);
                LOGI("TTS: auto-switched voice to %s for language %s", entry.voice, entry.lang);
            }
            return;
        }
    }
    // Unknown language — keep current voice
}

void KokoroTts::synthesize(
    const std::string& text, const std::string& language,
    TTSChunkCallback on_chunk)
{
    cancelled_.store(false, std::memory_order_relaxed);

    // Set language and auto-switch voice if language changed
    std::string lang = language.empty() ? "en" : language;
    phonemizer_.set_language(lang);
    auto_switch_voice(lang);

    // The E2E export bounds both its input and output tensors. Split long text
    // into sentence-preferring chunks using the selected graph profile's
    // validated token budget.
    auto count_tokens = [this](const std::string& t) {
        return phonemizer_.tokenize(t, 1 << 20).size();
    };
    auto chunks = chunk_text_for_synthesis(
        text, count_tokens, config_.chunk_token_budget,
        MAX_PHONEMES,
        MIN_TAIL_TOKENS);
    for (size_t i = 0; i < chunks.size(); i++) {
        if (cancelled_.load(std::memory_order_relaxed)) return;
        const bool is_final = i + 1 == chunks.size();
        size_t inference_attempts = 0;
        if (!synthesize_with_retry(
                chunks[i], on_chunk, is_final, 0, inference_attempts)) return;
    }
}

bool KokoroTts::synthesize_with_retry(
    const std::string& text, const TTSChunkCallback& on_chunk,
    bool is_final, size_t depth, size_t& inference_attempts)
{
    auto count_tokens = [this](const std::string& t) {
        return phonemizer_.tokenize(t, 1 << 20).size();
    };
    const size_t token_count = count_tokens(text);

    ChunkResult result = ChunkResult::RetrySmaller;
    if (token_count <= config_.chunk_token_hard_cap) {
        if (++inference_attempts > MAX_INFERENCE_ATTEMPTS) {
            throw std::runtime_error("Kokoro retry inference-attempt limit exceeded");
        }
        result = synthesize_chunk(text, on_chunk, is_final);
    } else {
        LOGI("TTS: preflight split for %zu tokens above model-run cap=%zu",
             token_count, config_.chunk_token_hard_cap);
    }
    if (result == ChunkResult::Emitted) return true;
    if (result == ChunkResult::Cancelled ||
        cancelled_.load(std::memory_order_relaxed)) return false;
    if (depth >= MAX_RETRY_DEPTH) {
        throw std::runtime_error("Kokoro output remained unsafe after retry limit");
    }

    if (token_count <= MIN_RETRY_TOKENS * 2) {
        throw std::runtime_error("Kokoro output is unsafe and too short to split");
    }

    auto pieces = split_text_for_synthesis_retry(
        text, count_tokens, MIN_RETRY_TOKENS, token_count - 1);
    if (pieces.size() < 2) {
        throw std::runtime_error("Kokoro retry splitter made no progress");
    }

    for (const auto& piece : pieces) {
        const size_t child_tokens = count_tokens(piece);
        if (piece.empty() || child_tokens < MIN_RETRY_TOKENS ||
            child_tokens >= token_count) {
            throw std::runtime_error("Kokoro retry splitter produced an unsafe chunk");
        }
    }

    LOGI("TTS: retrying %zu-token chunk as %zu smaller chunks",
         token_count, pieces.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (cancelled_.load(std::memory_order_relaxed)) return false;
        const bool child_final = is_final && i + 1 == pieces.size();
        if (!synthesize_with_retry(
                pieces[i], on_chunk, child_final, depth + 1,
                inference_attempts)) return false;
    }
    return true;
}

KokoroTts::ChunkResult KokoroTts::synthesize_chunk(
    const std::string& text, const TTSChunkCallback& on_chunk,
    bool is_final)
{
    auto* mem = OnnxEngine::get().cpu_memory();

    // Text → phoneme token IDs
    auto raw_tokens = phonemizer_.tokenize(text, MAX_PHONEMES + 1);
    if (cancelled_.load(std::memory_order_relaxed)) {
        return ChunkResult::Cancelled;
    }
    if (raw_tokens.empty()) return ChunkResult::RetrySmaller;
    if (raw_tokens.size() > MAX_PHONEMES) {
        LOGI("TTS: input exceeds %d phonemes; splitting and retrying",
             MAX_PHONEMES);
        return ChunkResult::RetrySmaller;
    }

    size_t token_count = raw_tokens.size();

    // Never log synthesis input: callers may send private or sensitive text.
    LOGI("TTS: tokens=%zu", token_count);

    // Pad to fixed MAX_PHONEMES with attention mask
    std::vector<int64_t> input_ids(MAX_PHONEMES, 0);
    std::vector<int64_t> attention_mask(MAX_PHONEMES, 0);
    for (size_t i = 0; i < token_count && i < MAX_PHONEMES; i++) {
        input_ids[i] = raw_tokens[i];
        attention_mask[i] = 1;
    }

    // --- input tensors ---

    const int64_t ids_shape[] = {1, MAX_PHONEMES};

    OrtValue* inputs[] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    OrtValuesGuard input_guard(api_, inputs, 5);

    // input_ids [1, 128]
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, input_ids.data(), input_ids.size() * sizeof(int64_t),
        ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]));

    // attention_mask [1, 128]
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, attention_mask.data(), attention_mask.size() * sizeof(int64_t),
        ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]));

    // ref_s / voice embedding [1, 256]
    const int64_t style_shape[] = {1, 256};
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, voice_embedding_.data(), voice_embedding_.size() * sizeof(float),
        style_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[2]));

    // speed [1]
    const int64_t speed_shape[] = {1};
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &speed_, sizeof(float),
        speed_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[3]));

    // random_phases [1, 9]
    float phases[9];
    for (int i = 0; i < 9; i++)
        phases[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    const int64_t phases_shape[] = {1, 9};
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, phases, sizeof(phases),
        phases_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[4]));

    // --- run ---

    const char* in_names[]  = {"input_ids", "attention_mask", "ref_s", "speed", "random_phases"};
    const char* out_names[] = {"audio", "audio_length_samples", "pred_dur"};
    OrtValue* outputs[] = {nullptr, nullptr, nullptr};
    OrtValuesGuard output_guard(api_, outputs, 3);

    ort_check(api_, api_->Run(
        session_, nullptr,
        in_names, inputs, 5,
        out_names, 3, outputs));

    if (!cancelled_.load(std::memory_order_relaxed)) {
        float* audio = nullptr;
        ort_check(api_, api_->GetTensorMutableData(outputs[0], (void**)&audio));

        // Get valid sample count from model
        int64_t* len_ptr = nullptr;
        ort_check(api_, api_->GetTensorMutableData(outputs[1], (void**)&len_ptr));
        const int64_t reported_samples = len_ptr[0];

        OrtTensorTypeAndShapeInfo* audio_info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(outputs[0], &audio_info));
        size_t audio_capacity = 0;
        // ort_check throws; release the info before checking the status so
        // the failure path cannot leak it.
        OrtStatus* count_status =
            api_->GetTensorShapeElementCount(audio_info, &audio_capacity);
        api_->ReleaseTensorTypeAndShapeInfo(audio_info);
        ort_check(api_, count_status);

        if (reported_samples <= 0 || audio_capacity == 0) {
            LOGI("TTS: invalid output length=%lld capacity=%zu; retrying",
                 static_cast<long long>(reported_samples), audio_capacity);
            return ChunkResult::RetrySmaller;
        }

        const size_t safe_capacity = guarded_output_capacity(
            audio_capacity, config_.max_safe_output_samples);
        if (static_cast<uint64_t>(reported_samples) >
            static_cast<uint64_t>(safe_capacity)) {
            LOGI("TTS: output length=%lld exceeds safe capacity=%zu "
                 "(tensor=%zu); splitting and retrying",
                 static_cast<long long>(reported_samples), safe_capacity,
                 audio_capacity);
            return ChunkResult::RetrySmaller;
        }
        const size_t valid_samples = static_cast<size_t>(reported_samples);

        // Inspect peak before any processing — short prompts (≤5 tokens) can
        // make the E2E ONNX export numerically explode (peak in the hundreds).
        // Non-finite or exploded outputs are split and retried once the input
        // is large enough; they are never emitted.
        float peak = 0.0f;
        bool finite = true;
        for (size_t i = 0; i < valid_samples; i++) {
            float a = std::abs(audio[i]);
            if (!std::isfinite(a)) {
                finite = false;
                break;
            }
            if (a > peak) peak = a;
        }
        if (!finite || peak > 2.0f) {
            LOGI("TTS: unstable output finite=%d peak=%.2f tokens=%zu; "
                 "splitting and retrying", finite ? 1 : 0, peak,
                 token_count);
            return ChunkResult::RetrySmaller;
        }

        // Trim trailing artifacts — Kokoro's E2E model often emits 100-300 ms
        // of low-energy noise + occasional loud spike past the real speech.
        constexpr int sample_rate = 24000;
        constexpr float silence_rms = 0.030f;
        const size_t win = std::max<size_t>(1, sample_rate / 20);  // 50 ms
        size_t speech_end = valid_samples;
        if (valid_samples > win) {
            for (size_t i = valid_samples - win; i > 0; i -= win / 2) {
                float sum_sq = 0.0f;
                for (size_t j = 0; j < win; j++) {
                    float v = audio[i + j];
                    sum_sq += v * v;
                }
                float rms = std::sqrt(sum_sq / static_cast<float>(win));
                if (rms > silence_rms) {
                    speech_end = i + win;
                    break;
                }
                if (i < win / 2) break;
            }
        }
        if (speech_end < valid_samples) {
            for (size_t k = speech_end; k < valid_samples; k++) audio[k] = 0.0f;
        }
        // ~10 ms linear fade-out at the new tail boundary so the seam is smooth.
        const size_t fade_out = std::min<size_t>(speech_end, sample_rate / 100);
        if (fade_out >= 2) {
            const size_t start = speech_end - fade_out;
            const float denom = static_cast<float>(fade_out - 1);
            for (size_t k = 0; k < fade_out; k++) {
                float gain = static_cast<float>(fade_out - 1 - k) / denom;
                audio[start + k] *= gain;
            }
        }
        // 5 ms fade-in to prevent click at start.
        const size_t fade_in = std::min<size_t>(120, speech_end);
        for (size_t i = 0; i < fade_in; i++) {
            audio[i] *= static_cast<float>(i) / static_cast<float>(fade_in);
        }

        LOGI("TTS: valid=%zu speech_end=%zu peak=%.4f final=%d",
             valid_samples, speech_end, peak, is_final ? 1 : 0);

        on_chunk(audio, speech_end, is_final);
        return ChunkResult::Emitted;
    }

    return ChunkResult::Cancelled;
}

void KokoroTts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

}  // namespace speech_core
