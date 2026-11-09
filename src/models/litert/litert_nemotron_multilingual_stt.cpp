#include "speech_core/models/litert_nemotron_multilingual_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/transcription/timed_words.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

namespace speech_core {

// The directory a bundle file sits in, which holds the bundle's config.json.
static std::string bundle_directory(const std::string& file) {
    const auto slash = file.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : file.substr(0, slash);
}

LiteRTNemotronMultilingualStt::LiteRTNemotronMultilingualStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const std::string& languages_path, bool hw_accel)
    : LiteRTNemotronMultilingualStt(encoder_path, decoder_path, joint_path,
                                    vocab_path, languages_path, Config{}, hw_accel) {}

LiteRTNemotronMultilingualStt::LiteRTNemotronMultilingualStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const std::string& languages_path, const Config& config, bool hw_accel)
    : cfg_(config)
{
    auto& engine = LiteRTEngine::get();
    engine.load(encoder_path, hw_accel, &enc_model_, &enc_compiled_);
    engine.load(decoder_path, false,    &dec_model_, &dec_compiled_);
    engine.load(joint_path,   false,    &jnt_model_, &jnt_compiled_);

    load_vocab(vocab_path);
    load_languages(languages_path);
    load_bundle_config(bundle_directory(vocab_path) + "/config.json");
    query_litert_io_order();

    LOGI("Nemotron multilingual (LiteRT): vocab=%zu prompts=%d enc_hidden=%d "
         "dec_hidden=%d T_out=%d window=%d samples lang_slot=%d",
         vocab_.size(), cfg_.num_prompts, cfg_.encoder_hidden, cfg_.decoder_hidden,
         enc_t_out_, chunk_samples(), lang_slot_);
}

LiteRTNemotronMultilingualStt::~LiteRTNemotronMultilingualStt() {
    if (jnt_compiled_) LiteRtDestroyCompiledModel(jnt_compiled_);
    if (jnt_model_)    LiteRtDestroyModel(jnt_model_);
    if (dec_compiled_) LiteRtDestroyCompiledModel(dec_compiled_);
    if (dec_model_)    LiteRtDestroyModel(dec_model_);
    if (enc_compiled_) LiteRtDestroyCompiledModel(enc_compiled_);
    if (enc_model_)    LiteRtDestroyModel(enc_model_);
}

// ---------------------------------------------------------------------------
// Vocabulary / languages (runtime-agnostic; mirrors NemotronMultilingualStt)
// ---------------------------------------------------------------------------

bool LiteRTNemotronMultilingualStt::load_vocab(const std::string& path) {
    auto text = json::read_file(path);
    if (text.empty()) return false;
    auto flat = json::parse_flat_object(text);

    int max_id = -1;
    for (auto& [key, val] : flat) {
        (void)val;
        try { max_id = std::max(max_id, std::stoi(key)); } catch (...) {}
    }
    if (max_id < 0) return false;

    vocab_.assign(static_cast<size_t>(max_id) + 1, std::string{});
    for (auto& [key, val] : flat) {
        try {
            int id = std::stoi(key);
            if (id >= 0 && id <= max_id) vocab_[id] = val;
        } catch (...) {}
    }
    cfg_.vocab_size = static_cast<int>(vocab_.size());
    cfg_.blank_id   = cfg_.vocab_size;
    return true;
}

bool LiteRTNemotronMultilingualStt::load_languages(const std::string& path) {
    auto text = json::read_file(path);
    if (text.empty()) return false;

    auto top = json::parse_flat_object(text);
    if (auto it = top.find("numPrompts"); it != top.end()) {
        try { cfg_.num_prompts = std::stoi(it->second); } catch (...) {}
    }
    if (auto it = top.find("autoSlot"); it != top.end()) {
        try { auto_slot_ = std::stoi(it->second); } catch (...) {}
    }

    auto key_pos = text.find("\"promptDictionary\"");
    if (key_pos != std::string::npos) {
        size_t brace = text.find('{', key_pos);
        if (brace != std::string::npos) {
            int depth = 0; size_t end = brace;
            for (size_t i = brace; i < text.size(); ++i) {
                if (text[i] == '{') depth++;
                else if (text[i] == '}') { if (--depth == 0) { end = i; break; } }
            }
            std::string block = text.substr(brace, end - brace + 1);
            auto dict = json::parse_flat_object(block);
            for (auto& [locale, slot] : dict) {
                try { lang2slot_[locale] = std::stoi(slot); } catch (...) {}
            }
        }
    }
    if (!set_language("en-US")) lang_slot_ = 0;
    return !lang2slot_.empty();
}

// The bundle's config.json says how many mel frames one encoder output frame
// covers, which turns the frame a token was emitted on into a time. Absent or
// unreadable leaves the published export's value.
void LiteRTNemotronMultilingualStt::load_bundle_config(const std::string& path) {
    auto text = json::read_file(path);
    if (text.empty()) return;
    auto top = json::parse_flat_object(text);
    if (auto it = top.find("subsamplingFactor"); it != top.end()) {
        try {
            const int value = std::stoi(it->second);
            if (value > 0) cfg_.subsampling = value;
        } catch (...) {}
    }
}

bool LiteRTNemotronMultilingualStt::set_language(const std::string& locale) {
    auto it = lang2slot_.find(locale);
    if (it != lang2slot_.end()) { lang_slot_ = it->second; return true; }
    if (auto_slot_ >= 0) lang_slot_ = auto_slot_;
    return false;
}

std::string LiteRTNemotronMultilingualStt::token_to_text(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size())) return {};
    const std::string& piece = vocab_[id];
    if (piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 &&
        static_cast<unsigned char>(piece[1]) == 0x96 &&
        static_cast<unsigned char>(piece[2]) == 0x81) {
        return " " + piece.substr(3);
    }
    return piece;
}

audio::StreamingMelSpectrogram::Config LiteRTNemotronMultilingualStt::mel_config() const {
    audio::StreamingMelSpectrogram::Config mel;
    mel.sample_rate = cfg_.sample_rate;
    mel.n_fft = cfg_.n_fft;
    mel.hop_length = cfg_.hop_length;
    mel.win_length = cfg_.win_length;
    mel.num_mel_bins = cfg_.mel_bins;
    mel.slaney_norm = true;
    mel.log_floor = 1.0f / static_cast<float>(1 << 24);  // 2^-24
    mel.pre_emphasis = cfg_.pre_emphasis;
    mel.torch_stft_layout = true;
    mel.center_pad_zeros = true;
    return mel;
}

float LiteRTNemotronMultilingualStt::frame_seconds() const {
    return static_cast<float>(cfg_.hop_length * cfg_.subsampling)
         / static_cast<float>(cfg_.sample_rate);
}

std::string LiteRTNemotronMultilingualStt::run_pending_window() {
    const size_t bins = static_cast<size_t>(cfg_.mel_bins);
    const size_t frames = static_cast<size_t>(cfg_.mel_frames);
    // Pending frames are [frame, bin]; the encoder takes [bin, frame]. A frame
    // the stream has not produced stays zero, as the whole-buffer front-end
    // left it.
    std::vector<float> window(bins * frames, 0.0f);
    const size_t available = std::min(frames, pending_frames_.size() / bins);
    for (size_t f = 0; f < available; ++f) {
        for (size_t b = 0; b < bins; ++b) {
            window[b * frames + f] = pending_frames_[f * bins + b];
        }
    }
    pending_frames_.erase(
        pending_frames_.begin(),
        pending_frames_.begin() + static_cast<std::ptrdiff_t>(available * bins));
    std::string text = run_window(window.data());
    ++decoded_windows_;
    return text;
}

// Parse the trailing integer after `token` in a tflite tensor name, e.g.
// "serving_default_args_5:0" + "args_" -> 5; "StatefulPartitionedCall:4" +
// "StatefulPartitionedCall:" -> 4. Returns -1 if not found.
static int parse_index_after(const std::string& name, const char* token) {
    auto p = name.find(token);
    if (p == std::string::npos) return -1;
    p += std::strlen(token);
    int v = 0; bool any = false;
    while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
        v = v * 10 + (name[p] - '0'); ++p; any = true;
    }
    return any ? v : -1;
}

// Canonical role index per physical subgraph input/output slot, parsed from
// the tensor name (the converter reorders slots vs the signature).
static std::vector<int> litert_roles(LiteRtModel m, bool inputs) {
    std::vector<int> roles;
    LiteRtParamIndex main = 0;
    if (LiteRtGetMainModelSubgraphIndex(m, &main) != kLiteRtStatusOk) return roles;
    LiteRtSubgraph sg = nullptr;
    if (LiteRtGetModelSubgraph(m, main, &sg) != kLiteRtStatusOk) return roles;
    LiteRtParamIndex n = 0;
    if (inputs) LiteRtGetNumSubgraphInputs(sg, &n);
    else        LiteRtGetNumSubgraphOutputs(sg, &n);
    roles.assign(static_cast<size_t>(n), -1);
    for (LiteRtParamIndex i = 0; i < n; ++i) {
        LiteRtTensor t = nullptr;
        LiteRtStatus s = inputs ? LiteRtGetSubgraphInput(sg, i, &t)
                                : LiteRtGetSubgraphOutput(sg, i, &t);
        if (s != kLiteRtStatusOk || !t) continue;
        const char* nm = nullptr;
        if (LiteRtGetTensorName(t, &nm) != kLiteRtStatusOk || !nm) continue;
        std::string name(nm);
        roles[i] = inputs ? parse_index_after(name, "args_")
                          : parse_index_after(name, "StatefulPartitionedCall:");
    }
    return roles;
}

void LiteRTNemotronMultilingualStt::query_litert_io_order() {
    enc_in_role_  = litert_roles(enc_model_, true);
    enc_out_role_ = litert_roles(enc_model_, false);
    dec_in_role_  = litert_roles(dec_model_, true);
    dec_out_role_ = litert_roles(dec_model_, false);
    jnt_in_role_  = litert_roles(jnt_model_, true);
    jnt_out_role_ = litert_roles(jnt_model_, false);

    // enc_t_out_ from the encoded_output (role 0) compiled layout.
    LiteRtLayout outs[8]{};
    if (LiteRtGetCompiledModelOutputTensorLayouts(enc_compiled_, 0, 8, outs, false) == kLiteRtStatusOk) {
        for (size_t pos = 0; pos < enc_out_role_.size() && pos < 8; ++pos) {
            if (enc_out_role_[pos] == 0 && outs[pos].rank >= 3 && outs[pos].dimensions[1] > 0) {
                enc_t_out_ = static_cast<int>(outs[pos].dimensions[1]);
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One 320 ms window: encoder (cache-aware) -> greedy RNN-T over all frames
// ---------------------------------------------------------------------------

std::string LiteRTNemotronMultilingualStt::run_window(const float* mel_window) {
    auto env = LiteRTEngine::get().env();
    const int H  = cfg_.encoder_hidden;
    const int Hd = cfg_.decoder_hidden;

    auto t_mel  = make_type(kLiteRtElementTypeFloat32, {1, cfg_.mel_bins, cfg_.mel_frames});
    auto t_i64  = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_lang = make_type(kLiteRtElementTypeFloat32, {1, cfg_.num_prompts});
    auto t_pre  = make_type(kLiteRtElementTypeFloat32, {1, cfg_.mel_bins, cfg_.pre_cache_size});
    auto t_clc  = make_type(kLiteRtElementTypeFloat32, {cfg_.encoder_layers, 1, cfg_.att_left_context, H});
    auto t_clt  = make_type(kLiteRtElementTypeFloat32, {cfg_.encoder_layers, 1, H, cfg_.conv_cache_size});
    auto t_enc  = make_type(kLiteRtElementTypeFloat32, {1, enc_t_out_, H});
    auto t_i32  = make_type(kLiteRtElementTypeInt32,   {1});

    std::vector<float> lang_mask(static_cast<size_t>(cfg_.num_prompts), 0.0f);
    if (lang_slot_ >= 0 && lang_slot_ < cfg_.num_prompts) lang_mask[lang_slot_] = 1.0f;
    int64_t audio_len = cfg_.mel_frames;
    int64_t ch_len    = cache_last_channel_len_;

    // Inputs/outputs by canonical role (physical slot order resolved at load).
    LiteRtHostBuffer in_mel (env, t_mel,  static_cast<size_t>(cfg_.mel_bins) * cfg_.mel_frames * sizeof(float), mel_window);
    LiteRtHostBuffer in_alen(env, t_i64,  sizeof(int64_t),                                 &audio_len);
    LiteRtHostBuffer in_lang(env, t_lang, lang_mask.size() * sizeof(float),                lang_mask.data());
    LiteRtHostBuffer in_pre (env, t_pre,  pre_cache_.size() * sizeof(float),               pre_cache_.data());
    LiteRtHostBuffer in_clc (env, t_clc,  cache_last_channel_.size() * sizeof(float),      cache_last_channel_.data());
    LiteRtHostBuffer in_clt (env, t_clt,  cache_last_time_.size() * sizeof(float),         cache_last_time_.data());
    LiteRtHostBuffer in_chl (env, t_i64,  sizeof(int64_t),                                 &ch_len);
    LiteRtTensorBuffer in_by_role[7] = { in_mel.raw(), in_alen.raw(), in_lang.raw(), in_pre.raw(),
                                         in_clc.raw(), in_clt.raw(), in_chl.raw() };

    LiteRtHostBuffer out_enc (env, t_enc, static_cast<size_t>(enc_t_out_) * H * sizeof(float));
    LiteRtHostBuffer out_elen(env, t_i32, sizeof(int32_t));
    LiteRtHostBuffer out_pre (env, t_pre, pre_cache_.size() * sizeof(float));
    LiteRtHostBuffer out_clc (env, t_clc, cache_last_channel_.size() * sizeof(float));
    LiteRtHostBuffer out_clt (env, t_clt, cache_last_time_.size() * sizeof(float));
    LiteRtHostBuffer out_chl (env, t_i32, sizeof(int32_t));
    LiteRtTensorBuffer out_by_role[6] = { out_enc.raw(), out_elen.raw(), out_pre.raw(),
                                          out_clc.raw(), out_clt.raw(), out_chl.raw() };

    std::vector<LiteRtTensorBuffer> eins(enc_in_role_.size());
    for (size_t pos = 0; pos < enc_in_role_.size(); ++pos) {
        int r = enc_in_role_[pos];
        eins[pos] = (r >= 0 && r < 7) ? in_by_role[r] : in_by_role[pos < 7 ? pos : 0];
    }
    std::vector<LiteRtTensorBuffer> eouts(enc_out_role_.size());
    for (size_t pos = 0; pos < enc_out_role_.size(); ++pos) {
        int r = enc_out_role_[pos];
        eouts[pos] = (r >= 0 && r < 6) ? out_by_role[r] : out_by_role[pos < 6 ? pos : 0];
    }
    litert_check(LiteRtRunCompiledModel(enc_compiled_, 0, eins.size(), eins.data(),
                                        eouts.size(), eouts.data()),
                 "Nemotron multilingual encoder Run");

    // Roll caches forward.
    out_pre.read(pre_cache_.data(),          pre_cache_.size() * sizeof(float));
    out_clc.read(cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float));
    out_clt.read(cache_last_time_.data(),    cache_last_time_.size() * sizeof(float));
    int32_t chl_new = 0; out_chl.read(&chl_new, sizeof(int32_t));
    cache_last_channel_len_ = static_cast<int64_t>(chl_new);
    int32_t enc_len = 0; out_elen.read(&enc_len, sizeof(int32_t));
    if (enc_len > enc_t_out_) enc_len = enc_t_out_;

    std::vector<float> encoded(static_cast<size_t>(enc_t_out_) * H);
    out_enc.read(encoded.data(), encoded.size() * sizeof(float));

    // Greedy RNN-T over every emitted encoder frame.
    const size_t n_logits = static_cast<size_t>(cfg_.vocab_size) + 1;
    auto t_tok   = make_type(kLiteRtElementTypeInt64,   {1, 1});
    auto t_state = make_type(kLiteRtElementTypeFloat32, {cfg_.decoder_layers, 1, Hd});
    auto t_dhid  = make_type(kLiteRtElementTypeFloat32, {1, 1, Hd});
    auto t_encf  = make_type(kLiteRtElementTypeFloat32, {1, 1, H});
    auto t_log   = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.vocab_size + 1});

    std::string emitted;
    for (int t = 0; t < enc_len; ++t) {
        const float* frame = encoded.data() + static_cast<size_t>(t) * H;
        for (int expand = 0; expand < cfg_.max_symbols; ++expand) {
            // --- decoder ---
            LiteRtHostBuffer in_tok (env, t_tok,   sizeof(int64_t),                 &last_token_);
            LiteRtHostBuffer in_h   (env, t_state, dec_h_.size() * sizeof(float),   dec_h_.data());
            LiteRtHostBuffer in_c   (env, t_state, dec_c_.size() * sizeof(float),   dec_c_.data());
            LiteRtHostBuffer out_dh (env, t_dhid,  static_cast<size_t>(Hd) * sizeof(float));
            LiteRtHostBuffer out_h  (env, t_state, dec_h_.size() * sizeof(float));
            LiteRtHostBuffer out_c  (env, t_state, dec_c_.size() * sizeof(float));

            LiteRtTensorBuffer dec_in_by_role [3] = { in_tok.raw(), in_h.raw(), in_c.raw() };
            LiteRtTensorBuffer dec_out_by_role[3] = { out_dh.raw(), out_h.raw(), out_c.raw() };
            std::vector<LiteRtTensorBuffer> dins(dec_in_role_.size());
            for (size_t p = 0; p < dec_in_role_.size(); ++p) {
                int r = dec_in_role_[p]; dins[p] = (r >= 0 && r < 3) ? dec_in_by_role[r] : dec_in_by_role[p < 3 ? p : 0];
            }
            std::vector<LiteRtTensorBuffer> douts(dec_out_role_.size());
            for (size_t p = 0; p < dec_out_role_.size(); ++p) {
                int r = dec_out_role_[p]; douts[p] = (r >= 0 && r < 3) ? dec_out_by_role[r] : dec_out_by_role[p < 3 ? p : 0];
            }
            litert_check(LiteRtRunCompiledModel(dec_compiled_, 0, dins.size(), dins.data(),
                                                douts.size(), douts.data()),
                         "Nemotron multilingual decoder Run");

            std::vector<float> dhid(static_cast<size_t>(Hd));
            out_dh.read(dhid.data(), dhid.size() * sizeof(float));

            // --- joint ---
            LiteRtHostBuffer in_encf(env, t_encf, static_cast<size_t>(H) * sizeof(float), frame);
            LiteRtHostBuffer in_dhid(env, t_dhid, dhid.size() * sizeof(float),            dhid.data());
            LiteRtHostBuffer out_log(env, t_log,  n_logits * sizeof(float));

            LiteRtTensorBuffer jnt_in_by_role[2] = { in_encf.raw(), in_dhid.raw() };
            std::vector<LiteRtTensorBuffer> jins(jnt_in_role_.size());
            for (size_t p = 0; p < jnt_in_role_.size(); ++p) {
                int r = jnt_in_role_[p]; jins[p] = (r >= 0 && r < 2) ? jnt_in_by_role[r] : jnt_in_by_role[p < 2 ? p : 0];
            }
            LiteRtTensorBuffer jouts[1] = { out_log.raw() };
            litert_check(LiteRtRunCompiledModel(jnt_compiled_, 0, jins.size(), jins.data(), 1, jouts),
                         "Nemotron multilingual joint Run");

            std::vector<float> logits(n_logits);
            out_log.read(logits.data(), n_logits * sizeof(float));

            int   best   = 0;
            float best_v = logits[0];
            for (size_t i = 1; i < n_logits; ++i) {
                if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int>(i); }
            }
            if (best == cfg_.blank_id) break;

            const std::string piece = token_to_text(best);
            emitted += piece;
            transcription::append_sentencepiece_token(
                stream_words_, piece, encoder_frames_ + t, frame_seconds());
            last_token_ = best;
            out_h.read(dec_h_.data(), dec_h_.size() * sizeof(float));
            out_c.read(dec_c_.data(), dec_c_.size() * sizeof(float));
        }
    }

    encoder_frames_ += enc_len;
    accumulated_text_ += emitted;
    return emitted;
}

// ---------------------------------------------------------------------------
// Streaming (identical structure to the ONNX wrapper)
// ---------------------------------------------------------------------------

void LiteRTNemotronMultilingualStt::reset_stream_state() {
    mel_stream_ = std::make_unique<audio::StreamingMelSpectrogram>(mel_config());
    pending_frames_.clear();
    samples_pushed_ = 0;
    decoded_windows_ = 0;
    pre_cache_.assign(static_cast<size_t>(cfg_.mel_bins) * cfg_.pre_cache_size, 0.0f);
    cache_last_channel_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.att_left_context * cfg_.encoder_hidden, 0.0f);
    cache_last_time_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.encoder_hidden * cfg_.conv_cache_size, 0.0f);
    cache_last_channel_len_ = 0;
    dec_h_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    dec_c_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    last_token_ = cfg_.blank_id;
    accumulated_text_.clear();
    stream_words_.clear();
    encoder_frames_ = 0;
}

void LiteRTNemotronMultilingualStt::begin_stream(int sample_rate) {
    cfg_.sample_rate = sample_rate;
    reset_stream_state();
    stream_init_ = true;
}

// Features are computed as audio arrives, and a window is decoded once its
// 320 ms and n_fft/2 samples of right context are in — the moment the
// whole-buffer front-end decoded it, so partial text appears exactly when it
// used to. Every frame of that window lies inside the audio by then, so it
// equals the frame the whole-utterance mel produced.
PartialResult LiteRTNemotronMultilingualStt::push_chunk(const float* audio, size_t length) {
    if (!stream_init_) begin_stream(cfg_.sample_rate);
    if (audio != nullptr && length > 0) {
        mel_stream_->push(audio, length, pending_frames_);
        samples_pushed_ += length;
    }

    const size_t win_samples   = static_cast<size_t>(chunk_samples());
    const size_t right_ctx     = static_cast<size_t>(cfg_.n_fft) / 2;
    const size_t window_values = static_cast<size_t>(cfg_.mel_bins) * cfg_.mel_frames;

    std::string text;
    while (samples_pushed_ >= (decoded_windows_ + 1) * win_samples + right_ctx
           && pending_frames_.size() >= window_values) {
        text += run_pending_window();
    }

    PartialResult out;
    out.text = std::move(text);
    out.words = stream_words_;
    return out;
}

void LiteRTNemotronMultilingualStt::flush_stream() {
    // No-op: trailing audio is flushed at end_stream().
}

TranscriptionResult LiteRTNemotronMultilingualStt::end_stream() {
    const float duration = static_cast<float>(samples_pushed_)
                         / static_cast<float>(cfg_.sample_rate);
    // Complete every remaining feature frame with zeros after pre-emphasis.
    // Centred final windows need right context even at an exact chunk boundary.
    if (stream_init_ && samples_pushed_ > 0) {
        const size_t win_samples = static_cast<size_t>(chunk_samples());
        const size_t total_windows = (samples_pushed_ + win_samples - 1) / win_samples;
        mel_stream_->finish(total_windows * static_cast<size_t>(cfg_.mel_frames), pending_frames_);
        while (decoded_windows_ < total_windows) {
            run_pending_window();
        }
    }
    TranscriptionResult out;
    out.text = accumulated_text_;
    out.words = transcription::clamp_word_times(stream_words_, duration);
    stream_init_ = false;
    return out;
}

void LiteRTNemotronMultilingualStt::cancel_stream() {
    if (mel_stream_) mel_stream_->reset();
    pending_frames_.clear();
    samples_pushed_ = 0;
    accumulated_text_.clear();
    stream_words_.clear();
    encoder_frames_ = 0;
    decoded_windows_ = 0;
    stream_init_ = false;
}

TranscriptionResult LiteRTNemotronMultilingualStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    begin_stream(sample_rate);
    push_chunk(audio, length);
    return end_stream();
}

}  // namespace speech_core
