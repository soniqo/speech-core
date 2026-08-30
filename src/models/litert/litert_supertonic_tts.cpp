#include "speech_core/models/litert_supertonic_tts.h"

#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>

namespace speech_core {
namespace {

// Exported fixed latent length (frames) of the vector_estimator / vocoder graphs. The published
// LiteRT bundle is fixed-shape (T=128, L=64). A dynamic-L export is currently BLOCKED upstream:
// litert_torch/odml_torch can't lower SupertonicTTS's relpos attention + ConvNeXt with a symbolic L
// (see speech-models/stmodels/export_litert.py). The host instead caps chunk length to this window
// (synthesize() below); longer chunks would need L-buckets or an upstream fix. SUPERTONIC_LATENT_FRAMES
// overrides this only for experiments against a re-exported graph.
int graph_latent_frames() {
    if (const char* e = std::getenv("SUPERTONIC_LATENT_FRAMES")) {
        int v = std::atoi(e);
        if (v > 0) return v;
    }
    return 64;
}

// Parse a JSON number array at s[i] (i must point at '['), FLATTENING any nesting — voice-style
// `data` is stored as a nested array matching `dims` (e.g. [1][50][256]), not a flat list.
std::vector<float> parse_float_array(const std::string& s, size_t& i) {
    std::vector<float> out;
    json::skip_ws(s, i);
    if (i >= s.size() || s[i] != '[') return out;
    int depth = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (c == '[') { ++depth; ++i; continue; }
        if (c == ']') { --depth; ++i; if (depth == 0) break; continue; }
        if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++i; continue; }
        const std::string v = json::parse_value_raw(s, i);  // one number; i advances to the delimiter
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

// Near-silence gate for the seam trims below (-46 dBFS; the vocoder's silence floor is well under it).
constexpr float kSeamSilenceGate = 0.005f;

size_t leading_silence(const std::vector<float>& pcm) {
    size_t i = 0;
    while (i < pcm.size() && std::fabs(pcm[i]) < kSeamSilenceGate) ++i;
    return i;
}

size_t trailing_silence(const std::vector<float>& pcm) {
    size_t n = 0;
    while (n < pcm.size() && std::fabs(pcm[pcm.size() - 1 - n]) < kSeamSilenceGate) ++n;
    return n;
}

// Cut leading / trailing near-silence beyond `keep` samples.
void trim_head(std::vector<float>& pcm, size_t keep) {
    const size_t s = leading_silence(pcm);
    if (s > keep) pcm.erase(pcm.begin(), pcm.begin() + static_cast<std::ptrdiff_t>(s - keep));
}

void trim_tail(std::vector<float>& pcm, size_t keep) {
    const size_t s = trailing_silence(pcm);
    if (s > keep) pcm.resize(pcm.size() - (s - keep));
}

}  // namespace

LiteRTSupertonicTts::LiteRTSupertonicTts(const std::string& duration_path,
                                         const std::string& text_encoder_path,
                                         const std::string& vector_estimator_path,
                                         const std::string& vocoder_path,
                                         const std::string& tokenizer_dir,
                                         const std::string& voice_styles_dir,
                                         bool hw_accel) {
    // The four LiteRt handles are raw (no per-member RAII), so a throw partway through loading would
    // leak the already-acquired graphs (a partially-constructed object never runs the dtor). Guard
    // the whole construction and release on failure.
    try {
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
    } catch (...) {
        destroy_graphs();  // release any LiteRt handles acquired before the failure, then rethrow
        throw;
    }
}

LiteRTSupertonicTts::~LiteRTSupertonicTts() { destroy_graphs(); }

// Free compiled-before-model (litert_engine.h contract); idempotent + nulls each handle so the ctor
// can call it on failure without risking a double-free.
void LiteRTSupertonicTts::destroy_graphs() noexcept {
    if (vocoder_compiled_)  { LiteRtDestroyCompiledModel(vocoder_compiled_);  vocoder_compiled_  = nullptr; }
    if (vocoder_model_)     { LiteRtDestroyModel(vocoder_model_);             vocoder_model_     = nullptr; }
    if (vector_compiled_)   { LiteRtDestroyCompiledModel(vector_compiled_);   vector_compiled_   = nullptr; }
    if (vector_model_)      { LiteRtDestroyModel(vector_model_);              vector_model_      = nullptr; }
    if (encoder_compiled_)  { LiteRtDestroyCompiledModel(encoder_compiled_);  encoder_compiled_  = nullptr; }
    if (encoder_model_)     { LiteRtDestroyModel(encoder_model_);             encoder_model_     = nullptr; }
    if (duration_compiled_) { LiteRtDestroyCompiledModel(duration_compiled_); duration_compiled_ = nullptr; }
    if (duration_model_)    { LiteRtDestroyModel(duration_model_);            duration_model_    = nullptr; }
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

    // The fixed latent window (L frames) bounds each piece's audio. A chars/sec heuristic only
    // decides which short sentences share a chunk; a longer sentence is kept whole by chunk(),
    // measured below with the duration predictor, and bisected at its best boundary only if it
    // really overflows — never word-packed at a character count (#140). A residual overflow (a
    // piece too short to split further) is logged + trimmed in synth_prepared().
    const int L = graph_latent_frames();
    const double window_s = static_cast<double>(L) * kChunkSamples / kSampleRate;
    const bool cjk = (language == "ko" || language == "ja");
    const int budget = std::max(8, static_cast<int>(window_s * (cjk ? 6 : 14) * 0.9));
    const std::vector<std::string> chunks = tokenizer_->chunk(text, language, budget);
    const int silence = static_cast<int>(chunk_silence_s_ * kSampleRate);

    // Preflight cache: a candidate that fits is synthesized from that prediction, not measured twice.
    std::map<std::string, Prepared> prepared;
    auto key = [](const std::string& t, bool continuation) {
        return std::string(continuation ? "," : ".") + t;
    };
    // A small overflow is absorbed by tempo (synth_prepared clamps the duration to the window);
    // only a larger one makes the planner split the text.
    const int max_frames = static_cast<int>(L * kWindowStretchMax);
    auto measure = [&](const std::string& t, bool continuation) -> int {
        if (cancelled_.load()) return 0;  // "fits"; the loop below exits before synthesizing
        auto it = prepared.find(key(t, continuation));
        if (it == prepared.end())
            it = prepared.emplace(key(t, continuation), prepare_chunk(t, language, continuation)).first;
        const int frames = it->second.latent_frames;
        if (frames > max_frames)
            LOGI("Supertonic: candidate needs L=%d > %d (window %d + stretch); splitting", frames, max_frames, L);
        return frames;
    };

    size_t piece_index = 0;
    for (size_t ci = 0; ci < chunks.size(); ++ci) {
        if (cancelled_.load()) return;
        prepared.clear();
        const std::vector<SupertonicPiece> pieces =
            SupertonicTokenizer::fit_to_window(chunks[ci], measure, max_frames);
        for (size_t pi = 0; pi < pieces.size(); ++pi) {
            if (cancelled_.load()) return;
            const SupertonicPiece& piece = pieces[pi];
            auto it = prepared.find(key(piece.text, piece.continuation));
            const Prepared p = it != prepared.end()
                ? std::move(it->second)
                : prepare_chunk(piece.text, language, piece.continuation);
            // Never log synthesis input: callers may send private or sensitive text.
            LOGI("Supertonic: chunk %zu/%zu piece %zu/%zu: L=%d/%d%s%s%s",
                 ci + 1, chunks.size(), pi + 1, pieces.size(), p.latent_frames, L,
                 p.latent_frames > L ? " (tempo-fit)" : "",
                 piece.continuation ? " (continues)" : "", piece.pause_before ? " (pause)" : "");
            std::vector<float> pcm = synth_prepared(p, piece_index++);
            if (cancelled_.load()) return;

            // A forced split lands mid-sentence, but every piece carries the model's own utterance
            // padding — ~400 ms of trailing and ~250–380 ms of leading silence — so the raw seam is
            // ~700 ms of dead air. Trim both sides down to a short comma-length gap (150 ms; the
            // model's own comma pause is ~250 ms, but the second piece restarts with sentence-
            // initial emphasis, and a tighter seam reads as more connected). Sentence boundaries
            // keep their padding.
            if (piece.continuation)
                trim_tail(pcm, static_cast<size_t>(kSeamTailMs * kSampleRate / 1000));
            if (pi > 0 && !piece.pause_before)
                trim_head(pcm, static_cast<size_t>(kSeamHeadMs * kSampleRate / 1000));

            const bool is_final = (ci + 1 == chunks.size()) && (pi + 1 == pieces.size());
            // Silence only where a sentence ends: between chunks, and between pieces whose cut
            // landed on a sentence boundary — never inside a sentence the window forced apart.
            const bool pause = (pi == 0) ? (ci > 0) : piece.pause_before;
            if (pause && silence > 0) {
                std::vector<float> sil(static_cast<size_t>(silence), 0.0f);
                on_chunk(sil.data(), sil.size(), false);
            }
            on_chunk(pcm.data(), pcm.size(), is_final);
        }
    }
}

LiteRTSupertonicTts::Prepared LiteRTSupertonicTts::prepare_chunk(const std::string& text,
                                                                 const std::string& language,
                                                                 bool continuation) {
    LiteRtEnvironment env = LiteRTEngine::get().env();
    const VoiceStyle& voice = current_voice();

    Prepared p;
    p.tok = tokenizer_->process(text, language, kTextT, continuation);

    // The exported LiteRT graphs take text_ids as INT64 (ai_edge_torch traces ids with torch.long;
    // the CoreML export is int32 via a wrapper, but LiteRT stays int64 — confirmed against the
    // published .tflite signatures). Widen the i32 tokenizer ids into an i64 input buffer.
    const std::vector<int64_t> ids64(p.tok.ids.begin(), p.tok.ids.end());
    const LiteRtRankedTensorType t_ids  = make_type(kLiteRtElementTypeInt64,   {1, kTextT});
    const LiteRtRankedTensorType t_mask = make_type(kLiteRtElementTypeFloat32, {1, 1, kTextT});
    const LiteRtRankedTensorType t_dp   = make_type(kLiteRtElementTypeFloat32, {1, 8, 16});
    const LiteRtRankedTensorType t_dur  = make_type(kLiteRtElementTypeFloat32, {1});

    LiteRtHostBuffer in_ids (env, t_ids,  ids64.size() * sizeof(int64_t), ids64.data());
    LiteRtHostBuffer in_mask(env, t_mask, p.tok.mask.size() * sizeof(float), p.tok.mask.data());
    LiteRtHostBuffer in_dp  (env, t_dp,   voice.style_dp.size() * sizeof(float), voice.style_dp.data());

    // --- 1) duration_predictor → duration[1] ---
    // LiteRtRunCompiledModel binds ins[] by the graph's tensor-INDEX order, which the ai_edge_torch
    // export permutes away from the (args_0=ids, args_1=style_dp, args_2=text_mask) declaration.
    // Introspected order of the published duration_predictor.tflite: [text_mask, text_ids, style_dp].
    float duration = 0.0f;
    {
        LiteRtHostBuffer out(env, t_dur, sizeof(float));
        LiteRtTensorBuffer ins[3]  = { in_mask.raw(), in_ids.raw(), in_dp.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(duration_compiled_, 0, 3, ins, 1, outs),
                     "duration_predictor Run");
        out.read(&duration, sizeof(float));
    }
    duration /= speed_;
    if (!(duration > 0.0f) || std::isnan(duration)) return p;  // nothing to synthesize

    // --- latent geometry: L_true = ceil(int(dur*SR) / 3072) ---
    // Match the reference (infer.py): truncate dur*SR to integer samples BEFORE the ceil.
    const long long wav_len = static_cast<long long>(duration * kSampleRate);
    p.duration      = duration;
    p.latent_frames = static_cast<int>((wav_len + kChunkSamples - 1) / kChunkSamples);
    return p;
}

std::vector<float> LiteRTSupertonicTts::synth_prepared(const Prepared& p, size_t piece_index) {
    if (!(p.duration > 0.0f)) return {};

    LiteRtEnvironment env = LiteRTEngine::get().env();
    const VoiceStyle& voice = current_voice();

    const std::vector<int64_t> ids64(p.tok.ids.begin(), p.tok.ids.end());
    const LiteRtRankedTensorType t_ids  = make_type(kLiteRtElementTypeInt64,   {1, kTextT});
    const LiteRtRankedTensorType t_mask = make_type(kLiteRtElementTypeFloat32, {1, 1, kTextT});
    const LiteRtRankedTensorType t_ttl  = make_type(kLiteRtElementTypeFloat32, {1, 50, 256});
    const LiteRtRankedTensorType t_emb  = make_type(kLiteRtElementTypeFloat32, {1, 256, kTextT});

    LiteRtHostBuffer in_ids (env, t_ids,  ids64.size() * sizeof(int64_t), ids64.data());
    LiteRtHostBuffer in_mask(env, t_mask, p.tok.mask.size() * sizeof(float), p.tok.mask.data());
    LiteRtHostBuffer in_ttl (env, t_ttl,  voice.style_ttl.size() * sizeof(float), voice.style_ttl.data());

    // --- 2) text_encoder → text_emb[1,256,T] ---
    // Introspected tensor-index order of text_encoder.tflite: [text_mask, text_ids, style_ttl].
    std::vector<float> text_emb(static_cast<size_t>(256) * kTextT);
    {
        LiteRtHostBuffer out(env, t_emb, text_emb.size() * sizeof(float));
        LiteRtTensorBuffer ins[3]  = { in_mask.raw(), in_ids.raw(), in_ttl.raw() };
        LiteRtTensorBuffer outs[1] = { out.raw() };
        litert_check(LiteRtRunCompiledModel(encoder_compiled_, 0, 3, ins, 1, outs),
                     "text_encoder Run");
        out.read(text_emb.data(), text_emb.size() * sizeof(float));
    }

    // --- latent window: the graph runs at fixed L; L_true valid frames inside it ---
    const int chunk_size = kChunkSamples;  // 512 * 6 = 3072
    const int L          = graph_latent_frames();
    const double window_s = static_cast<double>(chunk_size) * L / kSampleRate;
    int   L_true   = p.latent_frames;
    float duration = p.duration;
    if (L_true > L && L_true <= static_cast<int>(L * kWindowStretchMax)) {
        // Tempo-fit: clamp the duration to the window so the piece is spoken slightly faster (the
        // model fills exactly L frames) instead of cutting the sentence.
        duration = static_cast<float>(window_s);
        L_true   = L;
    } else if (L_true > L) {
        LOGE("Supertonic: piece needs L=%d frames > fixed graph L=%d and cannot be split further; "
             "audio truncated to %.2fs (dynamic L is blocked upstream).",
             L_true, L, window_s);
    }
    const int L_fill = std::min(std::max(L_true, 1), L);  // valid frames inside the fixed window

    // latent_mask[1,1,L]: 1.0 for the first L_fill frames, else 0.
    std::vector<float> latent_mask(static_cast<size_t>(L), 0.0f);
    for (int t = 0; t < L_fill; ++t) latent_mask[t] = 1.0f;

    // noisy[1,144,L] = randn * latent_mask (row-major c*L + t). Mix the piece index into the seed so
    // multi-piece utterances draw distinct noise per piece (the reference advances one shared RNG).
    std::mt19937 rng(seed_used_ + 0x9E3779B9u * static_cast<uint32_t>(piece_index + 1));
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

        // Introspected tensor-index order of vector_estimator.tflite:
        // [current_step, style_ttl, latent_mask, noisy, total_step, text_mask, text_emb].
        LiteRtTensorBuffer ins[7]  = { in_cur.raw(), in_ttl.raw(), in_lmask.raw(),
                                       in_noisy.raw(), in_tot.raw(), in_mask.raw(), in_emb.raw() };
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
