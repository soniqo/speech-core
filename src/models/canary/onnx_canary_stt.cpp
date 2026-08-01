#include "speech_core/models/onnx_canary_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace speech_core {

namespace {

constexpr const char* kSpMarker = "\xE2\x96\x81";  // U+2581

/// Control tokens are wrapped in <| |> and never belong in the transcript.
bool is_control_token(const std::string& token) {
    return token.size() >= 4 && token.rfind("<|", 0) == 0;
}

/// Language prompt tokens are exactly "<|xx|>" with a two-letter code.
bool is_language_token(const std::string& token, std::string* code) {
    if (token.size() != 6 || token.rfind("<|", 0) != 0 ||
        token.compare(4, 2, "|>") != 0) {
        return false;
    }
    for (size_t i = 2; i < 4; ++i) {
        const unsigned char c = static_cast<unsigned char>(token[i]);
        if (c < 'a' || c > 'z') return false;
    }
    *code = token.substr(2, 2);
    return true;
}

/// Releases its OrtValue on every exit path, including the throw out of
/// ort_check. The decode loop creates two tensors per token and holds the
/// encoder outputs across all of them, so a mid-decode error without this
/// leaks the entire encoder activation.
struct OrtValueHandle {
    const OrtApi* api = nullptr;
    OrtValue* value = nullptr;

    OrtValueHandle() = default;
    explicit OrtValueHandle(const OrtApi* a) : api(a) {}
    ~OrtValueHandle() { reset(); }

    OrtValueHandle(const OrtValueHandle&) = delete;
    OrtValueHandle& operator=(const OrtValueHandle&) = delete;

    OrtValue** slot() { reset(); return &value; }
    OrtValue* get() const { return value; }
    void reset() {
        if (value && api) api->ReleaseValue(value);
        value = nullptr;
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

OnnxCanaryStt::OnnxCanaryStt(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& vocab_path,
    bool hw_accel)
    : OnnxCanaryStt(encoder_path, decoder_path, vocab_path, Config{}, hw_accel) {}

OnnxCanaryStt::OnnxCanaryStt(
    const std::string& encoder_path,
    const std::string& decoder_path,
    const std::string& vocab_path,
    const Config& config,
    bool hw_accel)
    : cfg_(config)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    encoder_ = engine.load(encoder_path, hw_accel);
    decoder_ = engine.load(decoder_path, hw_accel);

    load_vocab(vocab_path);
    load_decode_contract();

    if (!language_tokens_.count(cfg_.language)) {
        throw std::runtime_error(
            "Canary bundle has no prompt token for source language: " + cfg_.language);
    }
    if (!language_tokens_.count(cfg_.target_language)) {
        throw std::runtime_error(
            "Canary bundle has no prompt token for target language: " +
            cfg_.target_language);
    }
}

OnnxCanaryStt::~OnnxCanaryStt() {
    if (decoder_) api_->ReleaseSession(decoder_);
    if (encoder_) api_->ReleaseSession(encoder_);
}

void OnnxCanaryStt::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Vocabulary — id -> piece, the same flat JSON object the Parakeet bundle ships
// ---------------------------------------------------------------------------

void OnnxCanaryStt::load_vocab(const std::string& path) {
    const std::string text = json::read_file(path);
    if (text.empty()) {
        throw std::runtime_error("Unable to read Canary vocab: " + path);
    }

    for (auto& [key, value] : json::parse_flat_object(text)) {
        int64_t id = 0;
        try {
            id = std::stoll(key);
        } catch (...) {
            continue;
        }
        if (id < 0) continue;
        vocab_[id] = value;

        // Language tokens are unique across the aggregated sub-tokenizers, so
        // indexing those by string is safe. Ordinary pieces are not — the same
        // piece appears once per sub-tokenizer — which is why nothing else
        // here resolves a token by its text.
        std::string code;
        if (is_language_token(value, &code)) language_tokens_[code] = id;
    }

    if (vocab_.empty()) {
        throw std::runtime_error("Canary vocab has no usable entries: " + path);
    }
}

// ---------------------------------------------------------------------------
// Decode contract — prompt, cache shape and end-of-text, from graph metadata
// ---------------------------------------------------------------------------

void OnnxCanaryStt::load_decode_contract() {
    OrtAllocator* allocator = nullptr;
    ort_check(api_, api_->GetAllocatorWithDefaultOptions(&allocator));

    struct MetadataHandle {
        const OrtApi* api = nullptr;
        OrtModelMetadata* value = nullptr;
        ~MetadataHandle() { if (value && api) api->ReleaseModelMetadata(value); }
    } meta{api_, nullptr};
    ort_check(api_, api_->SessionGetModelMetadata(decoder_, &meta.value));

    auto get = [&](const char* key) -> std::string {
        char* raw = nullptr;
        ort_check(api_, api_->ModelMetadataLookupCustomMetadataMap(
            meta.value, allocator, key, &raw));
        if (!raw) return {};
        std::string value(raw);
        ort_check(api_, api_->AllocatorFree(allocator, raw));
        return value;
    };

    const std::string prompt = get("prompt_ids");
    if (prompt.empty()) {
        throw std::runtime_error(
            "Canary decoder carries no prompt_ids metadata — this build expects "
            "the soniqo/Canary-*-ONNX bundle, which publishes its decode "
            "contract in the graph");
    }
    std::istringstream stream(prompt);
    std::string field;
    while (std::getline(stream, field, ',')) {
        if (field.empty()) continue;
        prompt_template_.push_back(std::stoll(field));
    }

    // The prompt carries one source/target language pair. Find it by token
    // rather than by position, so a checkpoint whose template differs still
    // switches language correctly.
    bool found = false;
    for (size_t i = 0; i + 1 < prompt_template_.size(); ++i) {
        auto a = vocab_.find(prompt_template_[i]);
        auto b = vocab_.find(prompt_template_[i + 1]);
        if (a == vocab_.end() || b == vocab_.end()) continue;
        std::string code;
        if (is_language_token(a->second, &code) &&
            is_language_token(b->second, &code)) {
            prompt_source_index_ = i;
            prompt_target_index_ = i + 1;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error(
            "Canary prompt metadata has no source/target language pair");
    }

    auto get_int = [&](const char* key, int64_t fallback) -> int64_t {
        const std::string v = get(key);
        return v.empty() ? fallback : std::stoll(v);
    };
    eos_id_ = get_int("eos_id", -1);
    mem_layers_ = get_int("decoder_mem_layers", 0);
    mem_width_ = get_int("decoder_hidden", 0);
    logits_are_log_probs_ = get_int("logits_are_log_probs", 0) != 0;

    if (eos_id_ < 0 || mem_layers_ <= 0 || mem_width_ <= 0) {
        throw std::runtime_error(
            "Canary decoder metadata is incomplete (eos_id / decoder_mem_layers "
            "/ decoder_hidden)");
    }
}

bool OnnxCanaryStt::set_language(const std::string& language) {
    if (!language_tokens_.count(language)) return false;
    cfg_.language = language;
    return true;
}

bool OnnxCanaryStt::set_target_language(const std::string& language) {
    if (!language_tokens_.count(language)) return false;
    cfg_.target_language = language;
    return true;
}

std::vector<int64_t> OnnxCanaryStt::build_prompt() const {
    std::vector<int64_t> prompt = prompt_template_;
    prompt[prompt_source_index_] = language_tokens_.at(cfg_.language);
    prompt[prompt_target_index_] = language_tokens_.at(cfg_.target_language);
    return prompt;
}

// ---------------------------------------------------------------------------
// Features — the NeMo contract Parakeet also uses
// ---------------------------------------------------------------------------

std::vector<float> OnnxCanaryStt::compute_features(
    const float* audio, size_t length) const
{
    std::vector<float> emphasized(length);
    if (length > 0) emphasized[0] = audio[0];
    for (size_t i = 1; i < length; i++) {
        emphasized[i] = audio[i] - cfg_.pre_emphasis * audio[i - 1];
    }

    // Canary's preprocessor config is the same AudioToMelSpectrogramPreprocessor
    // Parakeet trains with, so the flags are Parakeet's: torch.stft
    // center=True pad_mode="constant", Hann periodic=False, Slaney bank,
    // log(x + 2^-24). The unparameterised defaults are a different front end.
    auto mel = audio::mel_spectrogram(
        emphasized.data(), emphasized.size(),
        cfg_.sample_rate, cfg_.n_fft, cfg_.hop_length,
        cfg_.win_length, cfg_.num_mel_bins,
        /*slaney_norm=*/true, /*log_floor=*/5.960464478e-8f,
        /*center=*/true, /*torch_stft_layout=*/true,
        /*center_pad_zeros=*/true, /*symmetric_torch_window=*/true);

    // Per-feature normalisation, matching the reference extractor: sample
    // variance (Bessel, N-1 divisor) and a +1e-5 epsilon on the stddev.
    // mel layout: [num_mel_bins * num_frames], mel[m * num_frames + t]
    const int num_frames = static_cast<int>(mel.size() / cfg_.num_mel_bins);
    if (num_frames > 1) {
        for (int m = 0; m < cfg_.num_mel_bins; m++) {
            float sum = 0;
            for (int t = 0; t < num_frames; t++) {
                sum += mel[m * num_frames + t];
            }
            const float mean = sum / num_frames;
            float sq_dev = 0;
            for (int t = 0; t < num_frames; t++) {
                const float d = mel[m * num_frames + t] - mean;
                sq_dev += d * d;
            }
            const float var = sq_dev / (num_frames - 1);
            const float denom = ((var > 0) ? std::sqrt(var) : 0.0f) + 1e-5f;
            for (int t = 0; t < num_frames; t++) {
                mel[m * num_frames + t] = (mel[m * num_frames + t] - mean) / denom;
            }
        }
    }
    return mel;
}

std::string OnnxCanaryStt::detokenize(const std::vector<int64_t>& ids) const {
    std::string text;
    for (int64_t id : ids) {
        auto it = vocab_.find(id);
        if (it == vocab_.end()) continue;
        if (it->second.empty() || is_control_token(it->second)) continue;
        text += it->second;
    }

    size_t pos = 0;
    const std::string marker = kSpMarker;
    while ((pos = text.find(marker, pos)) != std::string::npos) {
        text.replace(pos, marker.size(), " ");
        pos += 1;
    }

    const size_t first = text.find_first_not_of(' ');
    if (first == std::string::npos) return "";
    const size_t last = text.find_last_not_of(' ');
    return text.substr(first, last - first + 1);
}

// ---------------------------------------------------------------------------
// Transcribe
// ---------------------------------------------------------------------------

TranscriptionResult OnnxCanaryStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    TranscriptionResult result;
    result.text = "";
    result.confidence = 0.0f;
    if (!audio || length == 0) return result;

    cancelled_.store(false, std::memory_order_relaxed);

    std::vector<float> converted;
    if (sample_rate <= 0) sample_rate = cfg_.sample_rate;
    if (sample_rate != cfg_.sample_rate) {
        converted = Resampler::resample(audio, length, sample_rate, cfg_.sample_rate);
        if (converted.empty()) return result;
        audio = converted.data();
        length = converted.size();
    }

    auto* mem = OnnxEngine::get().cpu_memory();

    // --- encoder ---

    auto features = compute_features(audio, length);
    const int64_t num_frames =
        static_cast<int64_t>(features.size() / cfg_.num_mel_bins);
    if (num_frames <= 0) return result;

    const int64_t feat_shape[] = {1, cfg_.num_mel_bins, num_frames};
    OrtValueHandle t_feat(api_);
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, features.data(), features.size() * sizeof(float),
        feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, t_feat.slot()));

    int64_t feat_len = num_frames;
    const int64_t len_shape[] = {1};
    OrtValueHandle t_len(api_);
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &feat_len, sizeof(int64_t),
        len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, t_len.slot()));

    const char* enc_in[] = {"audio_signal", "length"};
    const char* enc_out[] = {"encoder_embeddings", "encoder_mask"};
    OrtValue* enc_inputs[] = {t_feat.get(), t_len.get()};
    OrtValue* enc_raw[] = {nullptr, nullptr};

    ort_check(api_, api_->Run(
        encoder_, nullptr, enc_in, enc_inputs, 2, enc_out, 2, enc_raw));
    OrtValueHandle enc_emb(api_), enc_mask(api_);
    *enc_emb.slot() = enc_raw[0];
    *enc_mask.slot() = enc_raw[1];

    t_feat.reset();
    t_len.reset();

    // --- autoregressive decode ---

    std::vector<int64_t> tokens = build_prompt();
    std::vector<int64_t> generated;

    // Empty cache on the first step; the decoder then returns the state to
    // feed back, and only the newest token is passed from then on.
    std::vector<float> mems;
    int64_t mem_seq = 0;
    double score_sum = 0.0;

    for (int step = 0; step < cfg_.max_decode_tokens; ++step) {
        if (cancelled_.load(std::memory_order_relaxed)) break;

        const bool first = (mem_seq == 0);
        const int64_t* ids_data = first ? tokens.data() : &tokens.back();
        const int64_t ids_count = first ? static_cast<int64_t>(tokens.size()) : 1;
        const int64_t ids_shape[] = {1, ids_count};

        OrtValueHandle t_ids(api_);
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(ids_data), ids_count * sizeof(int64_t),
            ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, t_ids.slot()));

        const int64_t mems_shape[] = {mem_layers_, 1, mem_seq, mem_width_};
        OrtValueHandle t_mems(api_);
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, mems.empty() ? nullptr : mems.data(), mems.size() * sizeof(float),
            mems_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, t_mems.slot()));

        const char* dec_in[] = {
            "input_ids", "encoder_embeddings", "encoder_mask", "decoder_mems"};
        const char* dec_out[] = {"logits", "decoder_hidden_states"};
        OrtValue* dec_inputs[] = {
            t_ids.get(), enc_emb.get(), enc_mask.get(), t_mems.get()};
        OrtValue* dec_raw[] = {nullptr, nullptr};

        ort_check(api_, api_->Run(
            decoder_, nullptr, dec_in, dec_inputs, 4, dec_out, 2, dec_raw));
        OrtValueHandle t_logits(api_), t_states(api_);
        *t_logits.slot() = dec_raw[0];
        *t_states.slot() = dec_raw[1];

        // The bundle emits one position, but keep reading the last one so a
        // graph that returns every position still decodes correctly.
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(t_logits.get(), &info));
        size_t dims = 0;
        api_->GetDimensionsCount(info, &dims);
        std::vector<int64_t> logit_shape(dims);
        api_->GetDimensions(info, logit_shape.data(), dims);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        if (dims == 0) break;

        float* logits = nullptr;
        ort_check(api_, api_->GetTensorMutableData(t_logits.get(), (void**)&logits));

        const int64_t vocab = logit_shape[dims - 1];
        const int64_t positions = (dims >= 3) ? logit_shape[dims - 2] : 1;
        const float* last = logits + (positions - 1) * vocab;

        int64_t best = 0;
        float best_score = last[0];
        for (int64_t v = 1; v < vocab; ++v) {
            if (last[v] > best_score) {
                best_score = last[v];
                best = v;
            }
        }
        score_sum += best_score;

        // Carry the returned state forward as the next cache.
        OrtTensorTypeAndShapeInfo* mem_info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(t_states.get(), &mem_info));
        size_t mem_dims = 0;
        api_->GetDimensionsCount(mem_info, &mem_dims);
        std::vector<int64_t> mem_shape(mem_dims);
        api_->GetDimensions(mem_info, mem_shape.data(), mem_dims);
        api_->ReleaseTensorTypeAndShapeInfo(mem_info);

        float* mem_data = nullptr;
        ort_check(api_, api_->GetTensorMutableData(t_states.get(), (void**)&mem_data));

        size_t total = 1;
        for (int64_t d : mem_shape) total *= static_cast<size_t>(d);
        mems.assign(mem_data, mem_data + total);
        mem_seq = (mem_dims == 4) ? mem_shape[2] : mem_seq + 1;

        if (best == eos_id_) break;
        tokens.push_back(best);
        generated.push_back(best);
    }

    result.text = detokenize(generated);
    if (!generated.empty()) {
        const double mean = score_sum / static_cast<double>(generated.size());
        // The bundle's head is log_softmax, so the mean greedy score is a mean
        // log-probability and exp() puts it back on 0..1. Without that flag the
        // score is an uncalibrated logit, so squash it rather than pretend.
        result.confidence = logits_are_log_probs_
            ? static_cast<float>(std::exp(mean))
            : static_cast<float>(1.0 / (1.0 + std::exp(-mean * 0.1)));
    }
    return result;
}

}  // namespace speech_core
