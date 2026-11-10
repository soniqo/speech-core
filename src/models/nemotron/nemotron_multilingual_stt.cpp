#include "speech_core/models/nemotron_multilingual_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/nemotron_language_guide.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/transcription/timed_words.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>

namespace speech_core {

// The directory a bundle file sits in, which holds the bundle's config.json.
static std::string bundle_directory(const std::string& file) {
    const auto slash = file.find_last_of("/\\");
    return slash == std::string::npos ? std::string(".") : file.substr(0, slash);
}

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
    graphs_ = std::make_shared<Graphs>();
    api_ = graphs_->api = engine.api();
    encoder_ = graphs_->encoder = engine.load(encoder_path, hw_accel);
    // Decoder + joint are tiny FP32 graphs invoked many times per utterance;
    // like Parakeet's decoder-joint they stay on CPU (GPU/NNAPI dispatch cost
    // dominates their per-call compute).
    decoder_ = graphs_->decoder = engine.load(decoder_path, false);
    joint_   = graphs_->joint   = engine.load(joint_path,   false);

    load_vocab(vocab_path);
    load_languages(languages_path);
    load_bundle_config(bundle_directory(vocab_path) + "/config.json");
    query_io_names();

    LOGI("Nemotron multilingual: vocab=%zu prompts=%d enc_hidden=%d dec_hidden=%d "
         "mel_frames=%d window=%d samples lang_slot=%d",
         vocab_.size(), cfg_.num_prompts, cfg_.encoder_hidden, cfg_.decoder_hidden,
         cfg_.mel_frames, chunk_samples(), lang_slot_);
}

NemotronMultilingualStt::NemotronMultilingualStt(
    const NemotronMultilingualStt& other)
    : graphs_(other.graphs_),
      api_(other.api_),
      encoder_(other.encoder_),
      decoder_(other.decoder_),
      joint_(other.joint_),
      cfg_(other.cfg_),
      vocab_(other.vocab_),
      lang2slot_(other.lang2slot_),
      lang_slot_(other.lang_slot_),
      auto_slot_(other.auto_slot_),
      language_guide_mask_(other.language_guide_mask_),
      enc_in_(other.enc_in_), enc_out_(other.enc_out_),
      dec_in_(other.dec_in_), dec_out_(other.dec_out_),
      jnt_in_(other.jnt_in_), jnt_out_(other.jnt_out_)
{
    // Everything above is what loading produced and is immutable afterwards.
    // The caches below belong to a stream, and this is a new one, so they
    // start empty rather than carrying the other stream's history.
    reset_stream_state();
}

NemotronMultilingualStt::Graphs::~Graphs() {
    if (joint)   api->ReleaseSession(joint);
    if (decoder) api->ReleaseSession(decoder);
    if (encoder) api->ReleaseSession(encoder);
}

NemotronMultilingualStt::~NemotronMultilingualStt() = default;

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

// The bundle's config.json says how many mel frames one encoder output frame
// covers, which turns the frame a token was emitted on into a time. Absent or
// unreadable leaves the published export's value.
void NemotronMultilingualStt::load_bundle_config(const std::string& path) {
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

bool NemotronMultilingualStt::set_language(const std::string& locale) {
    language_guide_mask_.clear();
    auto it = lang2slot_.find(locale);
    if (it != lang2slot_.end()) { lang_slot_ = it->second; return true; }
    if (auto_slot_ >= 0) { lang_slot_ = auto_slot_; }
    return false;
}

void NemotronMultilingualStt::set_language_guide(const std::vector<std::string>& expected, float weight) {
    if (stream_init_) throw std::invalid_argument("Set the language guide before begin_stream");
    auto mask = nemotron_language_guide_mask(expected, weight, cfg_.num_prompts,
        auto_slot_, lang_slot_, lang2slot_, vocab_);
    language_guide_mask_ = std::move(mask);
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

audio::StreamingMelSpectrogram::Config NemotronMultilingualStt::mel_config() const {
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

float NemotronMultilingualStt::frame_seconds() const {
    return static_cast<float>(cfg_.hop_length * cfg_.subsampling)
         / static_cast<float>(cfg_.sample_rate);
}

std::string NemotronMultilingualStt::run_pending_window() {
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
    if (!language_guide_mask_.empty()) lang_mask = language_guide_mask_;
    else if (lang_slot_ >= 0 && lang_slot_ < cfg_.num_prompts) lang_mask[lang_slot_] = 1.0f;
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
                const std::string piece = token_to_text(best);
                emitted += piece;
                transcription::append_sentencepiece_token(
                    stream_words_, piece, encoder_frames_ + t, frame_seconds());
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

    encoder_frames_ += enc_len;
    accumulated_text_ += emitted;
    return emitted;
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

void NemotronMultilingualStt::reset_stream_state() {
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
    last_token_ = cfg_.blank_id;  // RNN-T blank primes the predictor
    accumulated_text_.clear();
    stream_words_.clear();
    encoder_frames_ = 0;
}

void NemotronMultilingualStt::begin_stream(int sample_rate) {
    cfg_.sample_rate = sample_rate;
    reset_stream_state();
    stream_init_ = true;
}

// Features are computed as audio arrives, and a window is decoded once its
// 320 ms and n_fft/2 samples of right context are in — the moment the
// whole-buffer front-end decoded it, so partial text appears exactly when it
// used to. Every frame of that window lies inside the audio by then, so it
// equals the frame the whole-utterance mel produced.
PartialResult NemotronMultilingualStt::push_chunk(const float* audio, size_t length) {
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

void NemotronMultilingualStt::flush_stream() {
    // No-op: trailing audio is flushed at end_stream().
}

TranscriptionResult NemotronMultilingualStt::end_stream() {
    const float duration = static_cast<float>(samples_pushed_)
                         / static_cast<float>(cfg_.sample_rate);
    // Decode the remaining tail: pad to a whole number of windows (reflecting
    // the reference, which zero-pads the utterance to a chunk multiple). The
    // padding is applied after pre-emphasis, including the right context of
    // a centred window at an exact chunk boundary.
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

void NemotronMultilingualStt::cancel_stream() {
    if (mel_stream_) mel_stream_->reset();
    pending_frames_.clear();
    samples_pushed_ = 0;
    accumulated_text_.clear();
    stream_words_.clear();
    encoder_frames_ = 0;
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
