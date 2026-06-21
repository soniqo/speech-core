#include "speech_core/models/litert_supertonic_tts.h"

#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace speech_core {
namespace {

// Exported fixed latent length (frames) of the vector_estimator / vocoder graphs. The published
// LiteRT bundle is fixed-shape (T=128, L=64); a dynamic-L export is the stated follow-up. Override
// at bring-up via SUPERTONIC_LATENT_FRAMES while validating against a re-exported graph.
int graph_latent_frames() {
    if (const char* e = std::getenv("SUPERTONIC_LATENT_FRAMES")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 64;
}

// Parse a flat JSON number array at s[i] (i must point at '[').
std::vector<float> parse_float_array(const std::string& s, size_t& i) {
    std::vector<float> out;
    json::skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') return out;
    ++i;
    while (i < s.size()) {
        json::skip_ws(s, i);
        if (s[i] == ']') { ++i; break; }
        if (s[i] == ',') { ++i; continue; }
        const std::string v = json::parse_value_raw(s, i);
        if (!v.empty()) out.push_back(std::strtof(v.c_str(), nullptr));
    }
    return out;
}

// Extract the "data" array from the top-level object value `key` (e.g. "style_ttl" → {dims,data}).
// util/json.h has no nested navigation, so walk by hand with its primitives.
std::vector<float> extract_style(const std::string& text, const std::string& key) {
    size_t i = 0;
    json::skip_ws(text, i);
    if (i >= text.size() || text[i] != '{') return {};
    ++i;
    while (i < text.size()) {
        json::skip_ws(text, i);
        if (text[i] == '}') break;
        if (text[i] == ',') { ++i; continue; }
        const std::string k = json::parse_string(text, i);
        json::skip_ws(text, i);
        if (i < text.size() && text[i] == ':') ++i;
        json::skip_ws(text, i);
        if (k == key && i < text.size() && text[i] == '{') {
            ++i;  // enter the {dims,data} object
            while (i < text.size()) {
                json::skip_ws(text, i);
                if (text[i] == '}') { ++i; break; }
                if (text[i] == ',') { ++i; continue; }
                const std::string kk = json::parse_string(text, i);
                json::skip_ws(text, i);
                if (i < text.size() && text[i] == ':') ++i;
                json::skip_ws(text, i);
                if (kk == "data" && i < text.size() && text[i] == '[')
                    return parse_float_array(text, i);
                json::skip_value(text, i);
            }
            return {};
        }
        json::skip_value(text, i);
    }
    return {};
}

// Load one voice style file (helper.py::load_voice_style format): {"style_ttl":{"dims","data"},
// "style_dp":{"dims","data"}} → flat row-major float vectors.
void load_style(const std::string& path, std::vector<float>& ttl, std::vector<float>& dp) {
    const std::string text = json::read_file(path);
    if (text.empty()) throw std::runtime_error("Supertonic: cannot read voice style " + path);
    ttl = extract_style(text, "style_ttl");
    dp  = extract_style(text, "style_dp");
}

}  // namespace

LiteRTSupertonicTts::LiteRTSupertonicTts(const std::string& duration_path,
                                         const std::string& text_encoder_path,
                                         const std::string& vector_estimator_path,
                                         const std::string& vocoder_path,
                                         const std::string& tokenizer_dir,
                                         const std::string& voice_styles_dir,
                                         bool hw_accel) {
    auto& engine = LiteRTEngine::get();
    engine.load(duration_path,         hw_accel, &duration_model_, &duration_compiled_);
    engine.load(text_encoder_path,     hw_accel, &encoder_model_,  &encoder_compiled_);
    engine.load(vector_estimator_path, hw_accel, &vector_model_,   &vector_compiled_);
    engine.load(vocoder_path,          hw_accel, &vocoder_model_,  &vocoder_compiled_);

    namespace fs = std::filesystem;
    tokenizer_ = std::make_unique<SupertonicTokenizer>(
        (fs::path(tokenizer_dir) / "unicode_indexer.json").string(),
        (fs::path(tokenizer_dir) / "tts.json").string());

    for (const auto& entry : fs::directory_iterator(voice_styles_dir)) {
        if (entry.path().extension() != ".json") continue;
        VoiceStyle v;
        load_style(entry.path().string(), v.style_ttl, v.style_dp);
        if (v.style_ttl.size() != kStyleTtlFloats || v.style_dp.size() != kStyleDpFloats) continue;
        voices_.emplace(entry.path().stem().string(), std::move(v));
    }
    if (voices_.empty())
        throw std::runtime_error("Supertonic: no voice styles in " + voice_styles_dir);
    if (!voices_.count(voice_id_)) voice_id_ = voices_.begin()->first;
}

LiteRTSupertonicTts::~LiteRTSupertonicTts() {
    // Free compiled before model (litert_engine.h contract).
    if (vocoder_compiled_)  LiteRtDestroyCompiledModel(vocoder_compiled_);
    if (vocoder_model_)     LiteRtDestroyModel(vocoder_model_);
    if (vector_compiled_)   LiteRtDestroyCompiledModel(vector_compiled_);
    if (vector_model_)      LiteRtDestroyModel(vector_model_);
    if (encoder_compiled_)  LiteRtDestroyCompiledModel(encoder_compiled_);
    if (encoder_model_)     LiteRtDestroyModel(encoder_model_);
    if (duration_compiled_) LiteRtDestroyCompiledModel(duration_compiled_);
    if (duration_model_)    LiteRtDestroyModel(duration_model_);
}

void LiteRTSupertonicTts::cancel() { cancelled_.store(true); }

void LiteRTSupertonicTts::set_voice(const std::string& voice_id) {
    if (!voices_.count(voice_id))
        throw std::invalid_argument("Supertonic: unknown voice '" + voice_id + "'");
    voice_id_ = voice_id;
}

std::vector<std::string> LiteRTSupertonicTts::voices() const {
    std::vector<std::string> ids;
    ids.reserve(voices_.size());
    for (const auto& kv : voices_) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    return ids;
}

const LiteRTSupertonicTts::VoiceStyle& LiteRTSupertonicTts::current_voice() const {
    auto it = voices_.find(voice_id_);
    if (it == voices_.end()) throw std::runtime_error("Supertonic: voice not loaded");
    return it->second;
}

void LiteRTSupertonicTts::synthesize(const std::string& text,
                                     const std::string& language,
                                     TTSChunkCallback on_chunk) {
    cancelled_.store(false);
    seed_used_ = seed_;
    if (seed_used_ == 0) {
        std::random_device rd;
        seed_used_ = rd();
    }

    const std::vector<std::string> chunks = tokenizer_->chunk(text, language);
    const int silence = static_cast<int>(chunk_silence_s_ * kSampleRate);

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (cancelled_.load()) return;
        std::vector<float> pcm = synth_chunk(chunks[ci], language);
        const bool is_final = (ci + 1 == chunks.size());

        if (ci > 0 && silence > 0) {
            std::vector<float> sil(static_cast<size_t>(silence), 0.0f);
            on_chunk(sil.data(), sil.size(), false);
        }
        on_chunk(pcm.data(), pcm.size(), is_final);
    }
}

std::vector<float> LiteRTSupertonicTts::synth_chunk(const std::string& chunk,
                                                    const std::string& language) {
    LiteRtEnvironment env = LiteRTEngine::get().env();
    const VoiceStyle& voice = current_voice();

    // --- tokenize → fixed-T ids + mask ---
    const SupertonicTokenizer::Tokens tok = tokenizer_->process(chunk, language, kTextT);

    const LiteRtRankedTensorType t_ids   = make_type(kLiteRtElementTypeInt32,   {1, kTextT});
    const LiteRtRankedTensorType t_mask  = make_type(kLiteRtElementTypeFloat32, {1, 1, kTextT});
    const LiteRtRankedTensorType t_ttl   = make_type(kLiteRtElementTypeFloat32, {1, 50, 256});
    const LiteRtRankedTensorType t_dp    = make_type(kLiteRtElementTypeFloat32, {1, 8, 16});
    const LiteRtRankedTensorType t_emb   = make_type(kLiteRtElementTypeFloat32, {1, 256, kTextT});
    const LiteRtRankedTensorType t_dur   = make_type(kLiteRtElementTypeFloat32, {1});

    LiteRtHostBuffer in_ids (env, t_ids,  tok.ids.size()  * sizeof(int32_t), tok.ids.data());
    LiteRtHostBuffer in_mask(env, t_mask, tok.mask.size() * sizeof(float),   tok.mask.data());
    LiteRtHostBuffer in_ttl (env, t_ttl,  voice.style_ttl.size() * sizeof(float), voice.style_ttl.data());
    LiteRtHostBuffer in_dp  (env, t_dp,   voice.style_dp.size()  * sizeof(float), voice.style_dp.data());

    // --- 1) duration_predictor (text_ids, style_dp, text_mask) → duration[1] ---
    float duration = 0.0f;
    {
        LiteRtHostBuffer out(env, t_dur, sizeof(float));
        LiteRtTensorBuffer ins[3]  = { in_ids.raw(), in_dp.raw(), in_mask.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(duration_compiled_, 0, 3, ins, 1, outs),
                     "duration_predictor Run");
        out.read(&duration, sizeof(float));
    }
    duration /= speed_;
    if (!(duration > 0.0f) || std::isnan(duration)) return {};

    // --- 2) text_encoder (text_ids, style_ttl, text_mask) → text_emb[1,256,T] ---
    std::vector<float> text_emb(static_cast<size_t>(256) * kTextT);
    {
        LiteRtHostBuffer out(env, t_emb, text_emb.size() * sizeof(float));
        LiteRtTensorBuffer ins[3]  = { in_ids.raw(), in_ttl.raw(), in_mask.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(encoder_compiled_, 0, 3, ins, 1, outs),
                     "text_encoder Run");
        out.read(text_emb.data(), text_emb.size() * sizeof(float));
    }

    // --- latent geometry: L_true = ceil(dur*SR / 3072); graph runs at fixed L ---
    const int chunk_size = kChunkSamples;  // 512 * 6 = 3072
    const int L_true = static_cast<int>(std::ceil(duration * kSampleRate / chunk_size));
    const int L      = graph_latent_frames();
    const int L_fill = std::min(std::max(L_true, 1), L);  // valid frames inside the fixed window

    // latent_mask[1,1,L]: 1.0 for the first L_fill frames, else 0.
    std::vector<float> latent_mask(static_cast<size_t>(L), 0.0f);
    for (int t = 0; t < L_fill; ++t) latent_mask[t] = 1.0f;

    // noisy[1,144,L] = randn * latent_mask (row-major c*L + t).
    std::mt19937 rng(seed_used_ + 0x9E3779B9u);  // decorrelate chunks via the golden ratio
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> xt(static_cast<size_t>(kLatentChannels) * L);
    for (int c = 0; c < kLatentChannels; ++c)
        for (int t = 0; t < L; ++t)
            xt[static_cast<size_t>(c) * L + t] = nd(rng) * latent_mask[t];

    const LiteRtRankedTensorType t_lat  = make_type(kLiteRtElementTypeFloat32, {1, kLatentChannels, L});
    const LiteRtRankedTensorType t_lmsk = make_type(kLiteRtElementTypeFloat32, {1, 1, L});
    const LiteRtRankedTensorType t_step = make_type(kLiteRtElementTypeFloat32, {1});

    LiteRtHostBuffer in_lmask(env, t_lmsk, latent_mask.size() * sizeof(float), latent_mask.data());

    // --- 3) vector_estimator × total_step (flow-matching ODE; xt fed forward) ---
    const float total_step_f = static_cast<float>(total_step_);
    for (int step = 0; step < total_step_; ++step) {
        if (cancelled_.load()) return {};
        const float cur_step_f = static_cast<float>(step);

        LiteRtHostBuffer in_noisy(env, t_lat,  xt.size() * sizeof(float), xt.data());
        LiteRtHostBuffer in_emb  (env, t_emb,  text_emb.size() * sizeof(float), text_emb.data());
        LiteRtHostBuffer in_cur  (env, t_step, sizeof(float), &cur_step_f);
        LiteRtHostBuffer in_tot  (env, t_step, sizeof(float), &total_step_f);
        LiteRtHostBuffer out     (env, t_lat,  xt.size() * sizeof(float));

        // export order: noisy, text_emb, style_ttl, latent_mask, text_mask, current_step, total_step
        LiteRtTensorBuffer ins[7]  = { in_noisy.raw(), in_emb.raw(), in_ttl.raw(),
                                       in_lmask.raw(), in_mask.raw(), in_cur.raw(), in_tot.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(vector_compiled_, 0, 7, ins, 1, outs),
                     "vector_estimator Run");
        out.read(xt.data(), xt.size() * sizeof(float));
    }

    // --- 4) vocoder (latent[1,144,L]) → wav[1, 3072*L] ---
    std::vector<float> wav(static_cast<size_t>(chunk_size) * L);
    {
        LiteRtHostBuffer in_latent(env, t_lat, xt.size() * sizeof(float), xt.data());
        const LiteRtRankedTensorType t_wav =
            make_type(kLiteRtElementTypeFloat32, {1, chunk_size * L});
        LiteRtHostBuffer out(env, t_wav, wav.size() * sizeof(float));
        LiteRtTensorBuffer ins[1]  = { in_latent.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(vocoder_compiled_, 0, 1, ins, 1, outs), "vocoder Run");
        out.read(wav.data(), wav.size() * sizeof(float));
    }

    // --- trim to floor(SR*dur), bounded by the valid latent window and the buffer ---
    size_t n = static_cast<size_t>(std::floor(kSampleRate * duration));
    n = std::min(n, static_cast<size_t>(chunk_size) * L_fill);
    n = std::min(n, wav.size());
    wav.resize(n);
    return wav;
}

}  // namespace speech_core
