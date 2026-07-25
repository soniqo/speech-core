#include "speech_core/models/onnx_canary_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace speech_core {

namespace {

constexpr const char* kSpMarker = "\xE2\x96\x81";  // U+2581

/// Control tokens are wrapped in <| |> and never belong in the transcript.
bool is_control_token(const std::string& token) {
    return token.size() >= 4 && token.rfind("<|", 0) == 0;
}

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

    eos_id_ = token_id("<|endoftext|>");
    if (eos_id_ < 0) {
        throw std::runtime_error(
            "Canary vocab has no <|endoftext|> token: " + vocab_path);
    }

    // decoder_mems is [layers, batch, seq, width]; the layer count and width
    // come from the export so the initial empty cache fits whatever was
    // loaded rather than a hard-coded model size.
    {
        OrtTypeInfo* type_info = nullptr;
        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        size_t input_count = 0;
        api_->SessionGetInputCount(decoder_, &input_count);
        OrtAllocator* allocator = nullptr;
        api_->GetAllocatorWithDefaultOptions(&allocator);

        for (size_t i = 0; i < input_count; ++i) {
            char* name = nullptr;
            api_->SessionGetInputName(decoder_, i, allocator, &name);
            const bool is_mems = name && std::string(name) == "decoder_mems";
            if (name) allocator->Free(allocator, name);
            if (!is_mems) continue;

            api_->SessionGetInputTypeInfo(decoder_, i, &type_info);
            api_->CastTypeInfoToTensorInfo(type_info, &shape_info);
            size_t dims = 0;
            api_->GetDimensionsCount(shape_info, &dims);
            std::vector<int64_t> shape(dims);
            api_->GetDimensions(shape_info, shape.data(), dims);
            if (dims == 4) {
                mem_layers_ = shape[0];
                mem_width_ = shape[3];
            }
            api_->ReleaseTypeInfo(type_info);
            break;
        }
    }
    if (mem_layers_ <= 0 || mem_width_ <= 0) {
        throw std::runtime_error(
            "Canary decoder does not expose a 4-D decoder_mems input");
    }

    // A missing control token silently becomes -1 in the prompt, and the
    // decoder answers garbage that still looks like speech — early stops and
    // repetition loops rather than an error. Fail here instead.
    const auto prompt = build_prompt();
    for (size_t i = 0; i < prompt.size(); ++i) {
        if (prompt[i] < 0) {
            throw std::runtime_error(
                "Canary prompt token " + std::to_string(i) +
                " missing from vocab: " + vocab_path);
        }
    }
}

OnnxCanaryStt::~OnnxCanaryStt() = default;

// ---------------------------------------------------------------------------
// Vocabulary — "<token> <id>" per line, as published for onnx-asr
// ---------------------------------------------------------------------------

void OnnxCanaryStt::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Unable to read Canary vocab: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        // Split on the LAST space: tokens may themselves be a space.
        const size_t split = line.rfind(' ');
        if (split == std::string::npos) continue;

        const std::string token = line.substr(0, split);
        int64_t id = 0;
        try {
            id = std::stoll(line.substr(split + 1));
        } catch (const std::exception&) {
            continue;
        }
        if (id < 0) continue;

        if (static_cast<size_t>(id) >= id_to_token_.size()) {
            id_to_token_.resize(static_cast<size_t>(id) + 1);
        }
        id_to_token_[static_cast<size_t>(id)] = token;
        token_to_id_[token] = id;
    }

    if (id_to_token_.empty()) {
        throw std::runtime_error("Canary vocab has no usable entries: " + path);
    }
}

int64_t OnnxCanaryStt::token_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    return it == token_to_id_.end() ? -1 : it->second;
}

bool OnnxCanaryStt::set_language(const std::string& language) {
    if (token_id("<|" + language + "|>") < 0) return false;
    cfg_.language = language;
    return true;
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

    auto mel = audio::mel_spectrogram(
        emphasized.data(), emphasized.size(),
        cfg_.sample_rate, cfg_.n_fft, cfg_.hop_length,
        cfg_.win_length, cfg_.num_mel_bins);

    // Per-feature normalisation (NeMo AudioToMelSpectrogramPreprocessor).
    const int num_frames = static_cast<int>(mel.size() / cfg_.num_mel_bins);
    if (num_frames > 1) {
        for (int m = 0; m < cfg_.num_mel_bins; m++) {
            float sum = 0, sq_sum = 0;
            for (int t = 0; t < num_frames; t++) {
                const float v = mel[m * num_frames + t];
                sum += v;
                sq_sum += v * v;
            }
            const float mean = sum / num_frames;
            const float var = sq_sum / num_frames - mean * mean;
            const float stddev = (var > 0) ? std::sqrt(var) : 1.0f;
            for (int t = 0; t < num_frames; t++) {
                mel[m * num_frames + t] = (mel[m * num_frames + t] - mean) / stddev;
            }
        }
    }
    return mel;
}

// ---------------------------------------------------------------------------
// Prompt — the ten control tokens Canary decodes from
// ---------------------------------------------------------------------------

std::vector<int64_t> OnnxCanaryStt::build_prompt() const {
    const std::string source = "<|" + cfg_.language + "|>";
    const std::string target = "<|" + cfg_.target_language + "|>";

    // Fall back to the source language rather than failing: an unknown target
    // means transcription, which is what a dictation caller wants anyway.
    const int64_t source_id = token_id(source) >= 0 ? token_id(source) : token_id("<|en|>");
    const int64_t target_id = token_id(target) >= 0 ? token_id(target) : source_id;

    return {
        // The reference implementation looks up " ", which resolves through a
        // loader that has already substituted the SentencePiece marker. The
        // published vocab has no bare-space token, so use the marker itself.
        token_id(kSpMarker),
        token_id("<|startofcontext|>"),
        token_id("<|startoftranscript|>"),
        token_id("<|emo:undefined|>"),
        source_id,
        target_id,
        token_id(cfg_.punctuation ? "<|pnc|>" : "<|nopnc|>"),
        token_id("<|noitn|>"),
        token_id("<|notimestamp|>"),
        token_id("<|nodiarize|>"),
    };
}

std::string OnnxCanaryStt::detokenize(const std::vector<int64_t>& ids) const {
    std::string text;
    for (int64_t id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) continue;
        const std::string& token = id_to_token_[static_cast<size_t>(id)];
        if (token.empty() || is_control_token(token)) continue;
        text += token;
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
    if (sample_rate != cfg_.sample_rate) return result;

    auto* mem = OnnxEngine::get().cpu_memory();

    // --- encoder ---

    auto features = compute_features(audio, length);
    const int64_t num_frames =
        static_cast<int64_t>(features.size() / cfg_.num_mel_bins);
    if (num_frames <= 0) return result;

    const int64_t feat_shape[] = {1, cfg_.num_mel_bins, num_frames};
    OrtValue* t_feat = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, features.data(), features.size() * sizeof(float),
        feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_feat));

    int64_t feat_len = num_frames;
    const int64_t len_shape[] = {1};
    OrtValue* t_len = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &feat_len, sizeof(int64_t),
        len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_len));

    const char* enc_in[] = {"audio_signal", "length"};
    const char* enc_out[] = {"encoder_embeddings", "encoder_mask"};
    OrtValue* enc_inputs[] = {t_feat, t_len};
    OrtValue* enc_outputs[] = {nullptr, nullptr};

    ort_check(api_, api_->Run(
        encoder_, nullptr, enc_in, enc_inputs, 2, enc_out, 2, enc_outputs));

    api_->ReleaseValue(t_feat);
    api_->ReleaseValue(t_len);

    // --- autoregressive decode ---

    std::vector<int64_t> tokens = build_prompt();
    const size_t prompt_size = tokens.size();
    std::vector<int64_t> generated;

    // Empty cache on the first step; the decoder then returns the state to
    // feed back, and only the newest token is passed from then on.
    std::vector<float> mems;
    int64_t mem_seq = 0;
    double logprob_sum = 0.0;

    for (int step = 0; step < cfg_.max_decode_tokens; ++step) {
        const bool first = (mem_seq == 0);
        const int64_t* ids_data = first ? tokens.data() : &tokens.back();
        const int64_t ids_count = first ? static_cast<int64_t>(tokens.size()) : 1;
        const int64_t ids_shape[] = {1, ids_count};

        OrtValue* t_ids = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(ids_data), ids_count * sizeof(int64_t),
            ids_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ids));

        const int64_t mems_shape[] = {mem_layers_, 1, mem_seq, mem_width_};
        OrtValue* t_mems = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, mems.empty() ? nullptr : mems.data(), mems.size() * sizeof(float),
            mems_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_mems));

        const char* dec_in[] = {
            "input_ids", "encoder_embeddings", "encoder_mask", "decoder_mems"};
        const char* dec_out[] = {"logits", "decoder_hidden_states"};
        OrtValue* dec_inputs[] = {t_ids, enc_outputs[0], enc_outputs[1], t_mems};
        OrtValue* dec_outputs[] = {nullptr, nullptr};

        ort_check(api_, api_->Run(
            decoder_, nullptr, dec_in, dec_inputs, 4, dec_out, 2, dec_outputs));

        // logits are [1, steps, vocab]; only the final position matters.
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(dec_outputs[0], &info));
        size_t dims = 0;
        api_->GetDimensionsCount(info, &dims);
        std::vector<int64_t> logit_shape(dims);
        api_->GetDimensions(info, logit_shape.data(), dims);
        api_->ReleaseTensorTypeAndShapeInfo(info);

        float* logits = nullptr;
        ort_check(api_, api_->GetTensorMutableData(dec_outputs[0], (void**)&logits));

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
        logprob_sum += best_score;

        // Carry the returned state forward as the next cache.
        {
            OrtTensorTypeAndShapeInfo* mem_info = nullptr;
            ort_check(api_, api_->GetTensorTypeAndShape(dec_outputs[1], &mem_info));
            size_t mem_dims = 0;
            api_->GetDimensionsCount(mem_info, &mem_dims);
            std::vector<int64_t> mem_shape(mem_dims);
            api_->GetDimensions(mem_info, mem_shape.data(), mem_dims);
            api_->ReleaseTensorTypeAndShapeInfo(mem_info);

            float* mem_data = nullptr;
            ort_check(api_, api_->GetTensorMutableData(dec_outputs[1], (void**)&mem_data));

            size_t total = 1;
            for (int64_t d : mem_shape) total *= static_cast<size_t>(d);
            mems.assign(mem_data, mem_data + total);
            mem_seq = (mem_dims == 4) ? mem_shape[2] : mem_seq + 1;
        }

        api_->ReleaseValue(dec_outputs[0]);
        api_->ReleaseValue(dec_outputs[1]);
        api_->ReleaseValue(t_ids);
        api_->ReleaseValue(t_mems);

        if (best == eos_id_) break;
        tokens.push_back(best);
        generated.push_back(best);
    }

    api_->ReleaseValue(enc_outputs[0]);
    api_->ReleaseValue(enc_outputs[1]);

    result.text = detokenize(generated);
    if (!generated.empty()) {
        // Mean greedy logit, squashed to 0..1 — comparable across utterances
        // but not a calibrated probability.
        const double mean = logprob_sum / static_cast<double>(generated.size());
        result.confidence = static_cast<float>(1.0 / (1.0 + std::exp(-mean)));
    }
    (void)prompt_size;
    return result;
}

}  // namespace speech_core
