#include "speech_core/models/onnx_whisper_stt.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace speech_core {

namespace {

constexpr float kPi = 3.14159265358979323846f;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct OrtValueHandle {
    const OrtApi* api = nullptr;
    OrtValue* value = nullptr;

    OrtValueHandle() = default;
    OrtValueHandle(const OrtApi* a, OrtValue* v) : api(a), value(v) {}
    ~OrtValueHandle() { reset(); }

    OrtValueHandle(const OrtValueHandle&) = delete;
    OrtValueHandle& operator=(const OrtValueHandle&) = delete;

    OrtValueHandle(OrtValueHandle&& other) noexcept
        : api(other.api), value(other.value) {
        other.value = nullptr;
    }
    OrtValueHandle& operator=(OrtValueHandle&& other) noexcept {
        if (this != &other) {
            reset();
            api = other.api;
            value = other.value;
            other.value = nullptr;
        }
        return *this;
    }

    OrtValue* get() const { return value; }
    OrtValue* release() {
        OrtValue* v = value;
        value = nullptr;
        return v;
    }
    void reset(OrtValue* v = nullptr) {
        if (value && api) api->ReleaseValue(value);
        value = v;
    }
};

struct MetadataHandle {
    const OrtApi* api = nullptr;
    OrtModelMetadata* value = nullptr;
    ~MetadataHandle() { if (value && api) api->ReleaseModelMetadata(value); }
};

std::string trim(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto b = std::find_if(s.begin(), s.end(), not_space);
    auto e = std::find_if(s.rbegin(), s.rend(), not_space).base();
    if (b >= e) return {};
    return std::string(b, e);
}

std::string normalize_language(std::string s) {
    s = trim(std::move(s));
    auto dash = s.find('-');
    if (dash != std::string::npos) s.resize(dash);
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

int base64_ord(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::string base64_decode(const std::string& s) {
    if (s.empty()) return {};
    std::string out;
    out.reserve(s.size() / 4 * 3);

    for (size_t i = 0; i < s.size(); i += 4) {
        if (s[i] == '=') return " ";
        if (i + 1 >= s.size()) break;
        int a = base64_ord(s[i]);
        int b = base64_ord(s[i + 1]);
        if (a < 0 || b < 0) break;
        out.push_back(static_cast<char>((a << 2) | ((b & 0x30) >> 4)));

        if (i + 2 < s.size() && s[i + 2] != '=') {
            int c = base64_ord(s[i + 2]);
            if (c < 0) break;
            out.push_back(static_cast<char>(((b & 0x0f) << 4) | ((c & 0x3c) >> 2)));

            if (i + 3 < s.size() && s[i + 3] != '=') {
                int d = base64_ord(s[i + 3]);
                if (d < 0) break;
                out.push_back(static_cast<char>(((c & 0x03) << 6) | d));
            }
        }
    }
    return out;
}

std::vector<int64_t> parse_i64_list(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.push_back(std::stoll(item));
    }
    return out;
}

std::vector<int32_t> parse_i32_list(const std::string& s) {
    std::vector<int32_t> out;
    for (int64_t v : parse_i64_list(s)) {
        out.push_back(static_cast<int32_t>(v));
    }
    return out;
}

std::vector<std::string> parse_string_list(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::vector<int64_t> tensor_shape(const OrtApi* api, OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort_check(api, api->GetTensorTypeAndShape(value, &info));
    size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(info, &rank));
    std::vector<int64_t> shape(rank);
    ort_check(api, api->GetDimensions(info, shape.data(), rank));
    api->ReleaseTensorTypeAndShapeInfo(info);
    return shape;
}

OrtValueHandle make_f32_tensor(
    const OrtApi* api, OrtMemoryInfo* mem, float* data, size_t count,
    const int64_t* shape, size_t rank) {
    OrtValue* v = nullptr;
    ort_check(api, api->CreateTensorWithDataAsOrtValue(
        mem, data, count * sizeof(float), shape, rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v));
    return OrtValueHandle(api, v);
}

OrtValueHandle make_i64_tensor(
    const OrtApi* api, OrtMemoryInfo* mem, int64_t* data, size_t count,
    const int64_t* shape, size_t rank) {
    OrtValue* v = nullptr;
    ort_check(api, api->CreateTensorWithDataAsOrtValue(
        mem, data, count * sizeof(int64_t), shape, rank,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v));
    return OrtValueHandle(api, v);
}

int argmax(const float* values, int n) {
    int best = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        float v = values[i];
        if (std::isnan(v)) continue;
        if (v > best_value) {
            best_value = v;
            best = i;
        }
    }
    return best;
}

int resolve_whisper_threads(int config_threads) {
    if (config_threads > 0) return config_threads;
    if (const char* env = std::getenv("SPEECH_CORE_WHISPER_ORT_THREADS")) {
        int v = std::atoi(env);
        if (v > 0) return v;
    }
    return 0;
}

}  // namespace

OnnxWhisperStt::OnnxWhisperStt(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& tokens_path,
    bool hw_accel)
    : OnnxWhisperStt(encoder_path, decoder_path, tokens_path, Config{}, hw_accel) {}

OnnxWhisperStt::OnnxWhisperStt(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& tokens_path,
    const Config& config,
    bool hw_accel)
    : cfg_(config)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    const int intra_threads = resolve_whisper_threads(cfg_.intra_threads);
    encoder_ = engine.load(encoder_path, hw_accel, false, intra_threads);
    // Without IO binding, each decoder step would copy the full self-KV cache
    // between host and accelerator. Keep the AR decoder on CPU by default.
    decoder_ = engine.load(decoder_path, false, false, intra_threads);

    load_metadata();
    prepare_feature_tables();
    load_tokens(tokens_path);

    if (cfg_.max_audio_samples == 0) {
        int frames = std::max(1, cfg_.max_feature_frames - cfg_.reserve_tail_frames);
        cfg_.max_audio_samples = static_cast<size_t>(frames * cfg_.hop_length);
    }

    if (!cfg_.language.empty() && !set_language(cfg_.language)) {
        throw std::runtime_error("Unsupported Whisper language: " + cfg_.language);
    }

    LOGI("Whisper ONNX: mels=%d vocab=%d text_layers=%d text_ctx=%d tokens=%zu",
         meta_.n_mels, meta_.n_vocab, meta_.n_text_layer, meta_.n_text_ctx,
         token_table_.size());
}

OnnxWhisperStt::~OnnxWhisperStt() {
    if (decoder_) api_->ReleaseSession(decoder_);
    if (encoder_) api_->ReleaseSession(encoder_);
}

OnnxWhisperStt::Profile OnnxWhisperStt::last_profile() const {
    return last_profile_;
}

bool OnnxWhisperStt::set_language(const std::string& language) {
    std::string normalized = normalize_language(language);
    if (normalized.empty()) {
        cfg_.language.clear();
        return true;
    }
    if (!meta_.lang2id.empty() && !meta_.lang2id.count(normalized)) {
        return false;
    }
    cfg_.language = std::move(normalized);
    return true;
}

void OnnxWhisperStt::load_metadata() {
    OrtAllocator* allocator = nullptr;
    ort_check(api_, api_->GetAllocatorWithDefaultOptions(&allocator));

    MetadataHandle meta{api_, nullptr};
    ort_check(api_, api_->SessionGetModelMetadata(encoder_, &meta.value));

    auto get = [&](const char* key) -> std::string {
        char* raw = nullptr;
        ort_check(api_, api_->ModelMetadataLookupCustomMetadataMap(
            meta.value, allocator, key, &raw));
        if (!raw) return {};
        std::string value(raw);
        ort_check(api_, api_->AllocatorFree(allocator, raw));
        return value;
    };
    auto get_int = [&](const char* key, int fallback) -> int {
        std::string v = get(key);
        if (v.empty()) return fallback;
        return std::stoi(v);
    };

    meta_.n_mels = get_int("n_mels", meta_.n_mels);
    meta_.n_text_layer = get_int("n_text_layer", meta_.n_text_layer);
    meta_.n_text_ctx = get_int("n_text_ctx", meta_.n_text_ctx);
    meta_.n_text_state = get_int("n_text_state", meta_.n_text_state);
    meta_.n_vocab = get_int("n_vocab", meta_.n_vocab);
    meta_.sot = get_int("sot", meta_.sot);
    meta_.eot = get_int("eot", meta_.eot);
    meta_.transcribe = get_int("transcribe", meta_.transcribe);
    meta_.translate = get_int("translate", meta_.translate);
    meta_.no_timestamps = get_int("no_timestamps", meta_.no_timestamps);
    meta_.is_multilingual = get_int("is_multilingual", meta_.is_multilingual);

    meta_.sot_sequence = parse_i64_list(get("sot_sequence"));
    if (meta_.sot_sequence.empty()) {
        meta_.sot_sequence = {meta_.sot, 50259, meta_.transcribe};
    }

    meta_.all_language_tokens = parse_i32_list(get("all_language_tokens"));
    meta_.all_language_codes = parse_string_list(get("all_language_codes"));
    const size_t n = std::min(meta_.all_language_tokens.size(),
                              meta_.all_language_codes.size());
    for (size_t i = 0; i < n; ++i) {
        std::string code = normalize_language(meta_.all_language_codes[i]);
        int32_t id = meta_.all_language_tokens[i];
        meta_.lang2id[code] = id;
        meta_.id2lang[id] = code;
    }
}

void OnnxWhisperStt::load_tokens(const std::string& path) {
    std::string text = json::read_file(path);
    if (text.empty()) {
        throw std::runtime_error("Unable to read Whisper token file: " + path);
    }

    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string encoded;
        int id = -1;
        if (!(ls >> encoded >> id) || id < 0) continue;
        if (static_cast<size_t>(id) >= token_table_.size()) {
            token_table_.resize(static_cast<size_t>(id) + 1);
        }
        token_table_[static_cast<size_t>(id)] = base64_decode(encoded);
    }
    if (token_table_.empty()) {
        throw std::runtime_error("Whisper token file has no usable entries: " + path);
    }
}

std::vector<float> OnnxWhisperStt::make_mel_filterbank() const {
    const int n_bins = cfg_.n_fft / 2 + 1;
    const int n_mels = meta_.n_mels;

    auto hz_to_mel = [](float hz) {
        constexpr float f_sp = 200.0f / 3.0f;
        constexpr float min_log_hz = 1000.0f;
        constexpr float min_log_mel = min_log_hz / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;
        if (hz < min_log_hz) return hz / f_sp;
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    };
    auto mel_to_hz = [](float mel) {
        constexpr float f_sp = 200.0f / 3.0f;
        constexpr float min_log_hz = 1000.0f;
        constexpr float min_log_mel = min_log_hz / f_sp;
        const float logstep = std::log(6.4f) / 27.0f;
        if (mel < min_log_mel) return f_sp * mel;
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    };

    std::vector<float> fft_freqs(n_bins);
    for (int i = 0; i < n_bins; ++i) {
        fft_freqs[i] = static_cast<float>(i * cfg_.sample_rate) / cfg_.n_fft;
    }

    std::vector<float> hz_points(n_mels + 2);
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(static_cast<float>(cfg_.sample_rate) / 2.0f);
    for (int i = 0; i < n_mels + 2; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(n_mels + 1);
        hz_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * t);
    }

    std::vector<float> fb(n_mels * n_bins, 0.0f);
    for (int m = 0; m < n_mels; ++m) {
        float left = hz_points[m];
        float center = hz_points[m + 1];
        float right = hz_points[m + 2];
        float enorm = 2.0f / std::max(right - left, 1e-12f);
        for (int b = 0; b < n_bins; ++b) {
            float hz = fft_freqs[b];
            float v = 0.0f;
            if (hz >= left && hz <= center) {
                v = (hz - left) / std::max(center - left, 1e-12f);
            } else if (hz > center && hz <= right) {
                v = (right - hz) / std::max(right - center, 1e-12f);
            }
            fb[m * n_bins + b] = v * enorm;
        }
    }
    return fb;
}

void OnnxWhisperStt::prepare_feature_tables() {
    window_.resize(static_cast<size_t>(cfg_.win_length));
    for (int i = 0; i < cfg_.win_length; ++i) {
        window_[static_cast<size_t>(i)] =
            0.5f * (1.0f - std::cos(2.0f * kPi * i / cfg_.win_length));
    }

    const int n_bins = cfg_.n_fft / 2 + 1;
    cos_table_.resize(static_cast<size_t>(n_bins) * cfg_.n_fft);
    sin_table_.resize(cos_table_.size());
    for (int k = 0; k < n_bins; ++k) {
        for (int n = 0; n < cfg_.n_fft; ++n) {
            float angle = 2.0f * kPi * static_cast<float>(k * n) / cfg_.n_fft;
            cos_table_[static_cast<size_t>(k) * cfg_.n_fft + n] = std::cos(angle);
            sin_table_[static_cast<size_t>(k) * cfg_.n_fft + n] = std::sin(angle);
        }
    }

    mel_filterbank_ = make_mel_filterbank();
}

std::vector<float> OnnxWhisperStt::compute_features(
    const float* audio, size_t length, int* out_frames) const {
    *out_frames = 0;
    if (!audio || length == 0) return {};

    const int pad = cfg_.n_fft / 2;
    std::vector<float> padded(length + 2 * static_cast<size_t>(pad), 0.0f);
    for (int i = 0; i < pad; ++i) {
        size_t src = std::min(static_cast<size_t>(pad - i), length - 1);
        padded[static_cast<size_t>(i)] = audio[src];
    }
    std::copy(audio, audio + length, padded.begin() + pad);
    for (int i = 0; i < pad; ++i) {
        int src = static_cast<int>(length) - 2 - i;
        padded[static_cast<size_t>(pad) + length + static_cast<size_t>(i)] =
            audio[static_cast<size_t>(std::max(src, 0))];
    }

    int n_frames_all = static_cast<int>(
        (padded.size() - static_cast<size_t>(cfg_.n_fft)) / cfg_.hop_length) + 1;
    int n_frames = std::max(0, n_frames_all - 1);
    if (n_frames == 0) return {};

    const int n_bins = cfg_.n_fft / 2 + 1;
    const int n_mels = meta_.n_mels;
    std::vector<float> frame(cfg_.n_fft, 0.0f);
    std::vector<float> power(n_bins, 0.0f);
    std::vector<float> mel(n_frames * n_mels, 0.0f);

    for (int t = 0; t < n_frames; ++t) {
        std::fill(frame.begin(), frame.end(), 0.0f);
        size_t start = static_cast<size_t>(t * cfg_.hop_length);
        for (int i = 0; i < cfg_.win_length; ++i) {
            frame[i] = padded[start + static_cast<size_t>(i)] * window_[i];
        }

        for (int k = 0; k < n_bins; ++k) {
            const float* c = cos_table_.data() + static_cast<size_t>(k) * cfg_.n_fft;
            const float* s = sin_table_.data() + static_cast<size_t>(k) * cfg_.n_fft;
            float re = 0.0f;
            float im = 0.0f;
            for (int n = 0; n < cfg_.n_fft; ++n) {
                re += frame[n] * c[n];
                im -= frame[n] * s[n];
            }
            power[k] = re * re + im * im;
        }

        for (int m = 0; m < n_mels; ++m) {
            float sum = 0.0f;
            const float* filt = mel_filterbank_.data() + static_cast<size_t>(m) * n_bins;
            for (int k = 0; k < n_bins; ++k) sum += power[k] * filt[k];
            mel[t * n_mels + m] = std::max(sum, 1e-10f);
        }
    }

    float max_log = -std::numeric_limits<float>::infinity();
    for (float& v : mel) {
        v = std::log10(v);
        if (v > max_log) max_log = v;
    }
    float floor = max_log - 8.0f;
    for (float& v : mel) {
        v = (std::max(v, floor) + 4.0f) / 4.0f;
    }

    *out_frames = n_frames;
    return mel;
}

TranscriptionResult OnnxWhisperStt::transcribe(
    const float* audio, size_t length, int sample_rate) {
    TranscriptionResult out;
    last_profile_ = {};
    if (!audio || length == 0) return out;
    const auto t_total = Clock::now();
    Profile profile;

    std::vector<float> converted;
    const float* pcm = audio;
    size_t pcm_len = length;
    if (sample_rate <= 0) sample_rate = cfg_.sample_rate;
    if (sample_rate != cfg_.sample_rate) {
        converted = Resampler::resample(audio, length, sample_rate, cfg_.sample_rate);
        pcm = converted.data();
        pcm_len = converted.size();
    }

    std::vector<std::string> texts;
    std::string language;
    size_t offset = 0;
    const size_t chunk = std::max<size_t>(1, cfg_.max_audio_samples);
    while (offset < pcm_len) {
        size_t n = std::min(chunk, pcm_len - offset);
        double chunk_start_ms = ms_since(t_total);
        auto r = decode_chunk(pcm + offset, n);
        ++profile.chunks;
        profile.feature_ms += r.profile.feature_ms;
        profile.encoder_ms += r.profile.encoder_ms;
        profile.language_ms += r.profile.language_ms;
        profile.decoder_prompt_ms += r.profile.decoder_prompt_ms;
        profile.decoder_ms += r.profile.decoder_ms;
        profile.feature_frames += r.profile.feature_frames;
        profile.encoded_frames += r.profile.encoded_frames;
        profile.prompt_tokens += r.profile.prompt_tokens;
        profile.generated_tokens += r.profile.generated_tokens;
        if (profile.first_token_ms == 0.0 && r.profile.first_token_ms > 0.0) {
            profile.first_token_ms = chunk_start_ms + r.profile.first_token_ms;
        }
        if (!r.text.empty()) texts.push_back(std::move(r.text));
        if (language.empty()) language = std::move(r.language);
        offset += n;
    }

    std::string text;
    for (const auto& part : texts) {
        if (!text.empty()) text += ' ';
        text += part;
    }
    out.text = trim(std::move(text));
    out.language = std::move(language);
    out.confidence = 1.0f;
    out.end_time = static_cast<float>(length) / static_cast<float>(sample_rate);
    profile.total_ms = ms_since(t_total);
    last_profile_ = profile;
    return out;
}

OnnxWhisperStt::DecodeResult OnnxWhisperStt::decode_chunk(
    const float* audio, size_t length) {
    DecodeResult result;
    int num_frames = 0;
    const auto t_feature = Clock::now();
    std::vector<float> features = compute_features(audio, length, &num_frames);
    result.profile.feature_ms = ms_since(t_feature);
    result.profile.feature_frames = num_frames;
    if (features.empty() || num_frames <= 0) return result;

    const int max_real_frames =
        std::max(1, cfg_.max_feature_frames - cfg_.reserve_tail_frames);
    if (num_frames >= max_real_frames) num_frames = max_real_frames;

    const int actual_frames = std::min(
        cfg_.max_feature_frames, num_frames + std::max(0, cfg_.tail_padding_frames));
    result.profile.feature_frames = num_frames;
    result.profile.encoded_frames = actual_frames;

    std::vector<float> mel(static_cast<size_t>(meta_.n_mels) * actual_frames, 0.0f);
    for (int t = 0; t < num_frames; ++t) {
        for (int m = 0; m < meta_.n_mels; ++m) {
            mel[static_cast<size_t>(m) * actual_frames + t] =
                features[static_cast<size_t>(t) * meta_.n_mels + m];
        }
    }

    auto* mem = OnnxEngine::get().cpu_memory();
    int64_t mel_shape[] = {1, static_cast<int64_t>(meta_.n_mels),
                           static_cast<int64_t>(actual_frames)};
    auto t_mel = make_f32_tensor(api_, mem, mel.data(), mel.size(), mel_shape, 3);

    const char* enc_in[] = {"mel"};
    const char* enc_out[] = {"n_layer_cross_k", "n_layer_cross_v"};
    OrtValue* enc_inputs[] = {t_mel.get()};
    OrtValue* enc_outputs[] = {nullptr, nullptr};
    const auto t_encoder = Clock::now();
    ort_check(api_, api_->Run(encoder_, nullptr, enc_in, enc_inputs, 1,
                              enc_out, 2, enc_outputs));
    result.profile.encoder_ms = ms_since(t_encoder);

    OrtValueHandle cross_k(api_, enc_outputs[0]);
    OrtValueHandle cross_v(api_, enc_outputs[1]);
    auto decoded = decode_greedy(cross_k.get(), cross_v.get(), num_frames);
    decoded.profile.feature_ms = result.profile.feature_ms;
    decoded.profile.encoder_ms = result.profile.encoder_ms;
    decoded.profile.feature_frames = result.profile.feature_frames;
    decoded.profile.encoded_frames = result.profile.encoded_frames;
    decoded.profile.first_token_ms = result.profile.feature_ms
        + result.profile.encoder_ms
        + decoded.profile.language_ms
        + decoded.profile.decoder_prompt_ms;
    return decoded;
}

int32_t OnnxWhisperStt::detect_language(OrtValue* cross_k, OrtValue* cross_v) {
    if (meta_.all_language_tokens.empty()) {
        return meta_.sot_sequence.size() > 1
            ? static_cast<int32_t>(meta_.sot_sequence[1])
            : 50259;
    }

    auto* mem = OnnxEngine::get().cpu_memory();
    std::vector<float> self_k(static_cast<size_t>(meta_.n_text_layer)
                              * meta_.n_text_ctx * meta_.n_text_state, 0.0f);
    std::vector<float> self_v(self_k.size(), 0.0f);
    int64_t cache_shape[] = {meta_.n_text_layer, 1, meta_.n_text_ctx, meta_.n_text_state};
    auto t_self_k = make_f32_tensor(api_, mem, self_k.data(), self_k.size(), cache_shape, 4);
    auto t_self_v = make_f32_tensor(api_, mem, self_v.data(), self_v.size(), cache_shape, 4);

    int64_t token = meta_.sot;
    int64_t token_shape[] = {1, 1};
    auto t_tokens = make_i64_tensor(api_, mem, &token, 1, token_shape, 2);
    int64_t offset = 0;
    int64_t offset_shape[] = {1};
    auto t_offset = make_i64_tensor(api_, mem, &offset, 1, offset_shape, 1);

    const char* dec_in[] = {
        "tokens", "in_n_layer_self_k_cache", "in_n_layer_self_v_cache",
        "n_layer_cross_k", "n_layer_cross_v", "offset",
    };
    const char* dec_out[] = {
        "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache",
    };
    OrtValue* inputs[] = {
        t_tokens.get(), t_self_k.get(), t_self_v.get(),
        cross_k, cross_v, t_offset.get(),
    };
    OrtValue* outputs[] = {nullptr, nullptr, nullptr};
    ort_check(api_, api_->Run(decoder_, nullptr, dec_in, inputs, 6,
                              dec_out, 3, outputs));
    OrtValueHandle logits(api_, outputs[0]);
    OrtValueHandle out_k(api_, outputs[1]);
    OrtValueHandle out_v(api_, outputs[2]);

    float* p = nullptr;
    ort_check(api_, api_->GetTensorMutableData(logits.get(), reinterpret_cast<void**>(&p)));
    int32_t best = meta_.all_language_tokens[0];
    float best_logit = p[best];
    for (int32_t id : meta_.all_language_tokens) {
        if (id >= 0 && id < meta_.n_vocab && p[id] > best_logit) {
            best_logit = p[id];
            best = id;
        }
    }
    return best;
}

OnnxWhisperStt::DecodeResult OnnxWhisperStt::decode_greedy(
    OrtValue* cross_k, OrtValue* cross_v, int num_feature_frames) {
    DecodeResult result;
    const auto t_decoder_total = Clock::now();
    auto* mem = OnnxEngine::get().cpu_memory();

    std::vector<int64_t> initial = meta_.sot_sequence;
    int32_t language_token = 0;
    if (meta_.is_multilingual && initial.size() >= 2) {
        if (!cfg_.language.empty()) {
            auto it = meta_.lang2id.find(cfg_.language);
            if (it == meta_.lang2id.end()) {
                throw std::runtime_error("Unsupported Whisper language: " + cfg_.language);
            }
            language_token = it->second;
        } else {
            const auto t_language = Clock::now();
            language_token = detect_language(cross_k, cross_v);
            result.profile.language_ms = ms_since(t_language);
        }
        initial[1] = language_token;
        auto lang = meta_.id2lang.find(language_token);
        if (lang != meta_.id2lang.end()) result.language = lang->second;

        if (initial.size() >= 3 && cfg_.task == "translate") {
            initial[2] = meta_.translate;
        } else if (cfg_.task != "transcribe") {
            throw std::runtime_error("Unsupported Whisper task: " + cfg_.task);
        }
    }
    initial.push_back(meta_.no_timestamps);
    result.profile.prompt_tokens = static_cast<int>(initial.size());

    std::vector<float> self_k(static_cast<size_t>(meta_.n_text_layer)
                              * meta_.n_text_ctx * meta_.n_text_state, 0.0f);
    std::vector<float> self_v(self_k.size(), 0.0f);
    int64_t cache_shape[] = {meta_.n_text_layer, 1, meta_.n_text_ctx, meta_.n_text_state};
    OrtValueHandle t_self_k = make_f32_tensor(
        api_, mem, self_k.data(), self_k.size(), cache_shape, 4);
    OrtValueHandle t_self_v = make_f32_tensor(
        api_, mem, self_v.data(), self_v.size(), cache_shape, 4);

    const char* dec_in[] = {
        "tokens", "in_n_layer_self_k_cache", "in_n_layer_self_v_cache",
        "n_layer_cross_k", "n_layer_cross_v", "offset",
    };
    const char* dec_out[] = {
        "logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache",
    };

    auto run_decoder = [&](int64_t* tokens, size_t token_count, int64_t offset)
        -> std::pair<OrtValueHandle, int32_t> {
        int64_t token_shape[] = {1, static_cast<int64_t>(token_count)};
        auto t_tokens = make_i64_tensor(api_, mem, tokens, token_count, token_shape, 2);
        int64_t offset_shape[] = {1};
        auto t_offset = make_i64_tensor(api_, mem, &offset, 1, offset_shape, 1);

        OrtValue* inputs[] = {
            t_tokens.get(), t_self_k.get(), t_self_v.get(),
            cross_k, cross_v, t_offset.get(),
        };
        OrtValue* outputs[] = {nullptr, nullptr, nullptr};
        ort_check(api_, api_->Run(decoder_, nullptr, dec_in, inputs, 6,
                                  dec_out, 3, outputs));

        OrtValueHandle logits(api_, outputs[0]);
        OrtValueHandle next_k(api_, outputs[1]);
        OrtValueHandle next_v(api_, outputs[2]);

        auto shape = tensor_shape(api_, logits.get());
        int vocab = (shape.size() >= 3 && shape[2] > 0)
            ? static_cast<int>(shape[2])
            : meta_.n_vocab;
        int positions = (shape.size() >= 2 && shape[1] > 0)
            ? static_cast<int>(shape[1])
            : 1;
        float* p = nullptr;
        ort_check(api_, api_->GetTensorMutableData(
            logits.get(), reinterpret_cast<void**>(&p)));
        int32_t next = argmax(p + static_cast<size_t>(positions - 1) * vocab, vocab);

        t_self_k.reset(next_k.release());
        t_self_v.reset(next_v.release());
        return {std::move(logits), next};
    };

    const auto t_prompt = Clock::now();
    auto first = run_decoder(initial.data(), initial.size(), 0);
    result.profile.decoder_prompt_ms = ms_since(t_prompt);
    (void)first.first;
    int32_t next_token = first.second;
    std::vector<float>().swap(self_k);
    std::vector<float>().swap(self_v);

    int limit = static_cast<int>(num_feature_frames / 100.0f * 6.0f);
    if (meta_.n_text_ctx > 0) limit = std::min(limit, meta_.n_text_ctx / 2);
    if (cfg_.max_decode_tokens > 0) limit = std::min(limit, cfg_.max_decode_tokens);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(std::max(limit, 0)));
    int64_t offset = static_cast<int64_t>(initial.size());

    for (int i = 0; i < limit; ++i) {
        if (next_token == meta_.eot) break;
        generated.push_back(next_token);

        int64_t token = next_token;
        auto step = run_decoder(&token, 1, offset);
        (void)step.first;
        next_token = step.second;
        ++offset;
        if (offset >= meta_.n_text_ctx - 1) break;
    }

    result.text = decode_tokens(generated);
    result.profile.generated_tokens = static_cast<int>(generated.size());
    result.profile.decoder_ms = ms_since(t_decoder_total);
    return result;
}

std::string OnnxWhisperStt::decode_tokens(const std::vector<int32_t>& ids) const {
    std::string text;
    for (int32_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= token_table_.size()) continue;
        text += token_table_[static_cast<size_t>(id)];
    }
    return trim(std::move(text));
}

}  // namespace speech_core
