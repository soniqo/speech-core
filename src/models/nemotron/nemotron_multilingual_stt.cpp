#include "speech_core/models/nemotron_multilingual_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace speech_core {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

NemotronMultilingualStt::NemotronMultilingualStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const std::string& languages_path, bool hw_accel)
    : NemotronMultilingualStt(encoder_path, decoder_path, joint_path,
                              vocab_path, languages_path, Config{}, hw_accel) {}

NemotronMultilingualStt::NemotronMultilingualStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const std::string& languages_path, const Config& config, bool hw_accel)
    : cfg_(config)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    encoder_ = engine.load(encoder_path, hw_accel);
    // Decoder + joint are tiny FP32 graphs invoked many times per utterance;
    // like Parakeet's decoder-joint they stay on CPU (GPU/NNAPI dispatch cost
    // dominates their per-call compute).
    decoder_ = engine.load(decoder_path, false);
    joint_   = engine.load(joint_path,   false);

    load_vocab(vocab_path);
    load_languages(languages_path);
    query_io_names();

    LOGI("Nemotron multilingual: vocab=%zu prompts=%d enc_hidden=%d dec_hidden=%d "
         "mel_frames=%d window=%d samples lang_slot=%d",
         vocab_.size(), cfg_.num_prompts, cfg_.encoder_hidden, cfg_.decoder_hidden,
         cfg_.mel_frames, chunk_samples(), lang_slot_);
}

NemotronMultilingualStt::~NemotronMultilingualStt() {
    if (joint_)   api_->ReleaseSession(joint_);
    if (decoder_) api_->ReleaseSession(decoder_);
    if (encoder_) api_->ReleaseSession(encoder_);
}

// ---------------------------------------------------------------------------
// Vocabulary / languages
// ---------------------------------------------------------------------------

bool NemotronMultilingualStt::load_vocab(const std::string& path) {
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

bool NemotronMultilingualStt::load_languages(const std::string& path) {
    auto text = json::read_file(path);
    if (text.empty()) return false;

    // Top-level scalars (numPrompts, autoSlot) — parse_flat_object skips the
    // nested promptDictionary, so these come back clean.
    auto top = json::parse_flat_object(text);
    if (auto it = top.find("numPrompts"); it != top.end()) {
        try { cfg_.num_prompts = std::stoi(it->second); } catch (...) {}
    }
    if (auto it = top.find("autoSlot"); it != top.end()) {
        try { auto_slot_ = std::stoi(it->second); } catch (...) {}
    }

    // Extract the nested promptDictionary {...} block and flat-parse it.
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
    // Default to English; fall back to slot 0 if the bundle lacks "en-US".
    if (!set_language("en-US")) lang_slot_ = 0;
    return !lang2slot_.empty();
}

bool NemotronMultilingualStt::set_language(const std::string& locale) {
    auto it = lang2slot_.find(locale);
    if (it != lang2slot_.end()) { lang_slot_ = it->second; return true; }
    if (auto_slot_ >= 0) { lang_slot_ = auto_slot_; }
    return false;
}

void NemotronMultilingualStt::query_io_names() {
    OrtAllocator* alloc = nullptr;
    api_->GetAllocatorWithDefaultOptions(&alloc);
    auto names = [&](OrtSession* s, bool inputs) {
        std::vector<std::string> out;
        size_t n = 0;
        if (inputs) api_->SessionGetInputCount(s, &n);
        else        api_->SessionGetOutputCount(s, &n);
        for (size_t i = 0; i < n; ++i) {
            char* nm = nullptr;
            OrtStatus* st = inputs ? api_->SessionGetInputName(s, i, alloc, &nm)
                                   : api_->SessionGetOutputName(s, i, alloc, &nm);
            if (st != nullptr) { api_->ReleaseStatus(st); break; }
            out.emplace_back(nm);
            alloc->Free(alloc, nm);
        }
        return out;
    };
    enc_in_ = names(encoder_, true);  enc_out_ = names(encoder_, false);
    dec_in_ = names(decoder_, true);  dec_out_ = names(decoder_, false);
    jnt_in_ = names(joint_,   true);  jnt_out_ = names(joint_,   false);
}

std::string NemotronMultilingualStt::token_to_text(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size())) return {};
    const std::string& piece = vocab_[id];
    // SentencePiece ▁ (U+2581, UTF-8 E2 96 81) → leading space.
    if (piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 &&
        static_cast<unsigned char>(piece[1]) == 0x96 &&
        static_cast<unsigned char>(piece[2]) == 0x81) {
        return " " + piece.substr(3);
    }
    return piece;
}

// ---------------------------------------------------------------------------
// Mel — pre-emphasis + Slaney log-mel, log floor 2^-24, no per-feature norm
// (NeMo normalize=NA). Matches export/onnx_inference.py compute_mel_chunk.
// ---------------------------------------------------------------------------

std::vector<float> NemotronMultilingualStt::compute_mel(
    const float* audio, size_t length) const
{
    if (length == 0) return {};
    std::vector<float> emph(length);
    emph[0] = audio[0];
    for (size_t i = 1; i < length; ++i) {
        emph[i] = audio[i] - cfg_.pre_emphasis * audio[i - 1];
    }
    constexpr float kLogFloor = 1.0f / static_cast<float>(1 << 24);  // 2^-24
    return audio::mel_spectrogram(
        emph.data(), emph.size(), cfg_.sample_rate,
        cfg_.n_fft, cfg_.hop_length, cfg_.win_length, cfg_.mel_bins,
        /*slaney=*/true, kLogFloor, /*center=*/true);
}

// ---------------------------------------------------------------------------
// One 320 ms window: encoder (cache-aware) -> greedy RNN-T over all frames
// ---------------------------------------------------------------------------

std::string NemotronMultilingualStt::run_window(const float* mel_window) {
    auto* mem = OnnxEngine::get().cpu_memory();
    const int H  = cfg_.encoder_hidden;
    const int Hd = cfg_.decoder_hidden;

    auto mk_f32 = [&](void* data, size_t bytes, const int64_t* shape, size_t rank) {
        OrtValue* v = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, data, bytes, shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v));
        return v;
    };
    auto mk_i32 = [&](void* data, const int64_t* shape, size_t rank) {
        OrtValue* v = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, data, sizeof(int32_t), shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &v));
        return v;
    };
    auto out_index = [](const std::vector<std::string>& v, const char* name, int fb) {
        for (size_t k = 0; k < v.size(); ++k) if (v[k] == name) return static_cast<int>(k);
        return fb;
    };

    // --- encoder inputs ---
    std::vector<float> lang_mask(static_cast<size_t>(cfg_.num_prompts), 0.0f);
    if (lang_slot_ >= 0 && lang_slot_ < cfg_.num_prompts) lang_mask[lang_slot_] = 1.0f;
    int32_t audio_len = cfg_.mel_frames;
    int32_t ch_len    = cache_last_channel_len_;

    const int64_t s_mel[3]  = {1, cfg_.mel_bins, cfg_.mel_frames};
    const int64_t s_len[1]  = {1};
    const int64_t s_lang[2] = {1, cfg_.num_prompts};
    const int64_t s_pre[3]  = {1, cfg_.mel_bins, cfg_.pre_cache_size};
    const int64_t s_clc[4]  = {cfg_.encoder_layers, 1, cfg_.att_left_context, H};
    const int64_t s_clt[4]  = {cfg_.encoder_layers, 1, H, cfg_.conv_cache_size};

    std::unordered_map<std::string, OrtValue*> in;
    in["audio_signal"]           = mk_f32(const_cast<float*>(mel_window),
                                          static_cast<size_t>(cfg_.mel_bins) * cfg_.mel_frames * sizeof(float), s_mel, 3);
    in["audio_length"]           = mk_i32(&audio_len, s_len, 1);
    in["language_mask"]          = mk_f32(lang_mask.data(), lang_mask.size() * sizeof(float), s_lang, 2);
    in["pre_cache"]              = mk_f32(pre_cache_.data(), pre_cache_.size() * sizeof(float), s_pre, 3);
    in["cache_last_channel"]     = mk_f32(cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float), s_clc, 4);
    in["cache_last_time"]        = mk_f32(cache_last_time_.data(), cache_last_time_.size() * sizeof(float), s_clt, 4);
    in["cache_last_channel_len"] = mk_i32(&ch_len, s_len, 1);

    std::vector<const char*> enc_in_names;
    std::vector<OrtValue*>   enc_in_vals;
    for (auto& nm : enc_in_) {
        auto it = in.find(nm);
        if (it != in.end()) { enc_in_names.push_back(nm.c_str()); enc_in_vals.push_back(it->second); }
    }

    std::vector<const char*> enc_out_names;
    for (auto& nm : enc_out_) enc_out_names.push_back(nm.c_str());
    std::vector<OrtValue*> enc_out_vals(enc_out_names.size(), nullptr);

    ort_check(api_, api_->Run(
        encoder_, nullptr, enc_in_names.data(), enc_in_vals.data(), enc_in_names.size(),
        enc_out_names.data(), enc_out_names.size(), enc_out_vals.data()));

    const int i_enc  = out_index(enc_out_, "encoded_output", 0);
    const int i_elen = out_index(enc_out_, "encoded_length", 1);
    const int i_pre  = out_index(enc_out_, "new_pre_cache", 2);
    const int i_clc  = out_index(enc_out_, "new_cache_last_channel", 3);
    const int i_clt  = out_index(enc_out_, "new_cache_last_time", 4);
    const int i_chl  = out_index(enc_out_, "new_cache_last_channel_len", 5);

    float*   encoded = nullptr;  api_->GetTensorMutableData(enc_out_vals[i_enc], (void**)&encoded);
    int32_t* elen_p  = nullptr;  api_->GetTensorMutableData(enc_out_vals[i_elen], (void**)&elen_p);
    int32_t  enc_len = elen_p[0];

    // Roll caches forward for the next window.
    auto copy_cache = [&](int idx, std::vector<float>& dst) {
        float* src = nullptr; api_->GetTensorMutableData(enc_out_vals[idx], (void**)&src);
        std::memcpy(dst.data(), src, dst.size() * sizeof(float));
    };
    copy_cache(i_pre, pre_cache_);
    copy_cache(i_clc, cache_last_channel_);
    copy_cache(i_clt, cache_last_time_);
    int32_t* chl_p = nullptr; api_->GetTensorMutableData(enc_out_vals[i_chl], (void**)&chl_p);
    cache_last_channel_len_ = chl_p[0];

    // Greedy RNN-T over every emitted encoder frame.
    std::string emitted;
    const size_t n_logits = static_cast<size_t>(cfg_.vocab_size) + 1;
    const int64_t s_tok[2]   = {1, 1};
    const int64_t s_state[3] = {cfg_.decoder_layers, 1, Hd};
    const int64_t s_frame[3] = {1, 1, H};
    const int64_t s_dhid[3]  = {1, 1, Hd};

    for (int t = 0; t < enc_len; ++t) {
        float* frame = encoded + static_cast<size_t>(t) * H;
        for (int expand = 0; expand < cfg_.max_symbols; ++expand) {
            // --- decoder ---
            OrtValue* t_tok = nullptr;
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, &last_token_, sizeof(int64_t), s_tok, 2,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_tok));
            OrtValue* t_h = mk_f32(dec_h_.data(), dec_h_.size() * sizeof(float), s_state, 3);
            OrtValue* t_c = mk_f32(dec_c_.data(), dec_c_.size() * sizeof(float), s_state, 3);

            std::array<const char*, 3> dnames = {dec_in_[0].c_str(), dec_in_[1].c_str(), dec_in_[2].c_str()};
            std::array<OrtValue*, 3>   dvals  = {t_tok, t_h, t_c};
            std::vector<const char*> donames;
            for (auto& nm : dec_out_) donames.push_back(nm.c_str());
            std::vector<OrtValue*> douts(donames.size(), nullptr);
            ort_check(api_, api_->Run(decoder_, nullptr, dnames.data(), dvals.data(), 3,
                                      donames.data(), donames.size(), douts.data()));

            const int di_out = out_index(dec_out_, "decoder_output", 0);
            const int di_h   = out_index(dec_out_, "h", 1);
            const int di_c   = out_index(dec_out_, "c", 2);
            float* dec_out = nullptr; api_->GetTensorMutableData(douts[di_out], (void**)&dec_out);

            // --- joint ---
            OrtValue* t_frame = mk_f32(frame, static_cast<size_t>(H) * sizeof(float), s_frame, 3);
            OrtValue* t_dhid  = mk_f32(dec_out, static_cast<size_t>(Hd) * sizeof(float), s_dhid, 3);
            std::array<const char*, 2> jnames = {jnt_in_[0].c_str(), jnt_in_[1].c_str()};
            std::array<OrtValue*, 2>   jvals  = {t_frame, t_dhid};
            const char* jo = jnt_out_[0].c_str();
            OrtValue* j_out = nullptr;
            ort_check(api_, api_->Run(joint_, nullptr, jnames.data(), jvals.data(), 2,
                                      &jo, 1, &j_out));
            float* logits = nullptr; api_->GetTensorMutableData(j_out, (void**)&logits);

            int   best = 0;
            float best_v = logits[0];
            for (size_t i = 1; i < n_logits; ++i) {
                if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int>(i); }
            }

            const bool is_blank = (best == cfg_.blank_id);
            if (!is_blank) {
                emitted += token_to_text(best);
                last_token_ = best;
                float* nh = nullptr; api_->GetTensorMutableData(douts[di_h], (void**)&nh);
                float* nc = nullptr; api_->GetTensorMutableData(douts[di_c], (void**)&nc);
                std::memcpy(dec_h_.data(), nh, dec_h_.size() * sizeof(float));
                std::memcpy(dec_c_.data(), nc, dec_c_.size() * sizeof(float));
            }

            api_->ReleaseValue(j_out);
            api_->ReleaseValue(t_dhid);
            api_->ReleaseValue(t_frame);
            for (auto* v : douts) api_->ReleaseValue(v);
            api_->ReleaseValue(t_c);
            api_->ReleaseValue(t_h);
            api_->ReleaseValue(t_tok);

            if (is_blank) break;
        }
    }

    for (auto* v : enc_out_vals) api_->ReleaseValue(v);
    for (auto& kv : in) api_->ReleaseValue(kv.second);

    accumulated_text_ += emitted;
    return emitted;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

void NemotronMultilingualStt::reset_stream_state() {
    stream_audio_.clear();
    decoded_windows_ = 0;
    pre_cache_.assign(static_cast<size_t>(cfg_.mel_bins) * cfg_.pre_cache_size, 0.0f);
    cache_last_channel_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.att_left_context * cfg_.encoder_hidden, 0.0f);
    cache_last_time_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.encoder_hidden * cfg_.conv_cache_size, 0.0f);
    cache_last_channel_len_ = 0;
    dec_h_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    dec_c_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    last_token_ = cfg_.blank_id;  // RNN-T blank primes the predictor
    accumulated_text_.clear();
}

void NemotronMultilingualStt::begin_stream(int sample_rate) {
    cfg_.sample_rate = sample_rate;
    reset_stream_state();
    stream_init_ = true;
}

// Decode any windows that are now fully covered by the accumulated audio.
// The encoder is cache-aware, so each fixed window is computed from a
// continuous whole-buffer mel (matching the reference's whole_mel) sliced at
// [w*mel_frames, (w+1)*mel_frames). A window needs its 320 ms plus n_fft/2
// right-context samples before its trailing mel frames are final.
PartialResult NemotronMultilingualStt::push_chunk(const float* audio, size_t length) {
    if (!stream_init_) begin_stream(cfg_.sample_rate);
    stream_audio_.insert(stream_audio_.end(), audio, audio + length);

    const size_t win_samples = static_cast<size_t>(chunk_samples());
    const size_t right_ctx   = static_cast<size_t>(cfg_.n_fft) / 2;

    std::string text;
    while (stream_audio_.size() >= (decoded_windows_ + 1) * win_samples + right_ctx) {
        auto mel = compute_mel(stream_audio_.data(), stream_audio_.size());
        const int produced = static_cast<int>(mel.size()) / cfg_.mel_bins;
        const int f0 = static_cast<int>(decoded_windows_) * cfg_.mel_frames;
        if (f0 + cfg_.mel_frames > produced) break;
        // Slice channels-first window [mel_bins, mel_frames] from the
        // [mel_bins, produced] continuous mel.
        std::vector<float> window(static_cast<size_t>(cfg_.mel_bins) * cfg_.mel_frames);
        for (int b = 0; b < cfg_.mel_bins; ++b) {
            std::copy_n(&mel[static_cast<size_t>(b) * produced + f0], cfg_.mel_frames,
                        &window[static_cast<size_t>(b) * cfg_.mel_frames]);
        }
        text += run_window(window.data());
        ++decoded_windows_;
    }

    PartialResult out;
    out.text = std::move(text);
    return out;
}

void NemotronMultilingualStt::flush_stream() {
    // No-op: trailing audio is flushed at end_stream().
}

TranscriptionResult NemotronMultilingualStt::end_stream() {
    // Decode the remaining tail: pad to a whole number of windows (reflecting
    // the reference, which zero-pads the utterance to a chunk multiple).
    if (stream_init_ && !stream_audio_.empty()) {
        const size_t win_samples = static_cast<size_t>(chunk_samples());
        size_t total_windows = (stream_audio_.size() + win_samples - 1) / win_samples;
        if (stream_audio_.size() % win_samples != 0) {
            stream_audio_.resize(total_windows * win_samples, 0.0f);
        }
        auto mel = compute_mel(stream_audio_.data(), stream_audio_.size());
        const int produced = static_cast<int>(mel.size()) / cfg_.mel_bins;
        while (decoded_windows_ < total_windows) {
            const int f0 = static_cast<int>(decoded_windows_) * cfg_.mel_frames;
            std::vector<float> window(static_cast<size_t>(cfg_.mel_bins) * cfg_.mel_frames, 0.0f);
            const int avail = std::min(cfg_.mel_frames, std::max(0, produced - f0));
            for (int b = 0; b < cfg_.mel_bins; ++b) {
                if (avail > 0)
                    std::copy_n(&mel[static_cast<size_t>(b) * produced + f0], avail,
                                &window[static_cast<size_t>(b) * cfg_.mel_frames]);
            }
            run_window(window.data());
            ++decoded_windows_;
        }
    }
    TranscriptionResult out;
    out.text = accumulated_text_;
    stream_init_ = false;
    return out;
}

void NemotronMultilingualStt::cancel_stream() {
    stream_audio_.clear();
    accumulated_text_.clear();
    decoded_windows_ = 0;
    stream_init_ = false;
}

// ---------------------------------------------------------------------------
// Batch convenience: begin -> push everything -> end. Matches the reference
// validator (whole-utterance mel, fixed windows, carried caches).
// ---------------------------------------------------------------------------

TranscriptionResult NemotronMultilingualStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    begin_stream(sample_rate);
    push_chunk(audio, length);
    return end_stream();
}

}  // namespace speech_core
