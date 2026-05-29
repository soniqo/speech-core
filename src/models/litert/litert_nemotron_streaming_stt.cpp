#include "speech_core/models/litert_nemotron_streaming_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace speech_core {

LiteRTNemotronStreamingStt::LiteRTNemotronStreamingStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path, bool hw_accel)
    : LiteRTNemotronStreamingStt(encoder_path, decoder_path, joint_path,
                                 vocab_path, Config{}, hw_accel) {}

LiteRTNemotronStreamingStt::LiteRTNemotronStreamingStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const Config& config, bool hw_accel)
    : cfg_(config)
{
    auto& engine = LiteRTEngine::get();
    engine.load(encoder_path, hw_accel, &enc_model_, &enc_compiled_);
    engine.load(decoder_path, false,    &dec_model_, &dec_compiled_);
    engine.load(joint_path,   false,    &jnt_model_, &jnt_compiled_);

    load_vocab(vocab_path);

    // Encoder output[0] = encoded [1, T_out, H]; read T_out for buffer sizing.
    // The export is fixed-shape so the static layout is available without a
    // resize. Tolerate failure — fall back to the default (2 @ 80 ms).
    LiteRtLayout outs[6]{};
    if (LiteRtGetCompiledModelOutputTensorLayouts(enc_compiled_, 0, 6, outs, false) == kLiteRtStatusOk
        && outs[0].rank >= 3 && outs[0].dimensions[1] > 0) {
        enc_t_out_ = static_cast<int>(outs[0].dimensions[1]);
    }

    LOGI("Nemotron streaming: vocab=%zu enc_hidden=%d dec_hidden=%d T_out=%d window=%d samples",
         vocab_.size(), cfg_.encoder_hidden, cfg_.decoder_hidden, enc_t_out_, chunk_samples());
}

LiteRTNemotronStreamingStt::~LiteRTNemotronStreamingStt() {
    if (jnt_compiled_) LiteRtDestroyCompiledModel(jnt_compiled_);
    if (jnt_model_)    LiteRtDestroyModel(jnt_model_);
    if (dec_compiled_) LiteRtDestroyCompiledModel(dec_compiled_);
    if (dec_model_)    LiteRtDestroyModel(dec_model_);
    if (enc_compiled_) LiteRtDestroyCompiledModel(enc_compiled_);
    if (enc_model_)    LiteRtDestroyModel(enc_model_);
}

bool LiteRTNemotronStreamingStt::load_vocab(const std::string& path) {
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

std::string LiteRTNemotronStreamingStt::token_to_text(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size())) return {};
    std::string piece = vocab_[id];
    // SentencePiece ▁ (U+2581, UTF-8 E2 96 81) → leading space.
    if (piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 &&
        static_cast<unsigned char>(piece[1]) == 0x96 &&
        static_cast<unsigned char>(piece[2]) == 0x81) {
        return " " + piece.substr(3);
    }
    return piece;
}

void LiteRTNemotronStreamingStt::reset_stream_state() {
    pending_.clear();
    pre_cache_.assign(static_cast<size_t>(cfg_.mel_bins) * cfg_.pre_cache_size, 0.0f);
    cache_last_channel_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.attn_left_context * cfg_.encoder_hidden, 0.0f);
    cache_last_time_.assign(
        static_cast<size_t>(cfg_.encoder_layers) * cfg_.encoder_hidden * cfg_.conv_cache_size, 0.0f);
    cache_last_channel_len_ = 0;
    dec_h_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    dec_c_.assign(static_cast<size_t>(cfg_.decoder_layers) * cfg_.decoder_hidden, 0.0f);
    dec_hidden_.assign(static_cast<size_t>(cfg_.decoder_hidden), 0.0f);
    accumulated_text_.clear();
    // Prime the decoder LSTM with the blank token so the first joint() call
    // sees a meaningful hidden state (matches the reference validator).
    run_decoder_step(cfg_.blank_id);
}

void LiteRTNemotronStreamingStt::run_decoder_step(int64_t token_id) {
    auto env      = LiteRTEngine::get().env();
    auto t_tok    = make_type(kLiteRtElementTypeInt64,   {1, 1});
    auto t_state  = make_type(kLiteRtElementTypeFloat32, {cfg_.decoder_layers, 1, cfg_.decoder_hidden});
    auto t_hidden = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.decoder_hidden});

    int64_t tok = token_id;
    LiteRtHostBuffer in_tok   (env, t_tok,    sizeof(int64_t),                 &tok);
    LiteRtHostBuffer in_h     (env, t_state,  dec_h_.size() * sizeof(float),   dec_h_.data());
    LiteRtHostBuffer in_c     (env, t_state,  dec_c_.size() * sizeof(float),   dec_c_.data());
    LiteRtHostBuffer out_hid  (env, t_hidden, dec_hidden_.size() * sizeof(float));
    LiteRtHostBuffer out_h    (env, t_state,  dec_h_.size() * sizeof(float));
    LiteRtHostBuffer out_c    (env, t_state,  dec_c_.size() * sizeof(float));

    // decoder I/O: in [token, h, c] -> out [hidden, h_new, c_new]
    LiteRtTensorBuffer ins [3] = { in_tok.raw(), in_h.raw(), in_c.raw() };
    LiteRtTensorBuffer outs[3] = { out_hid.raw(), out_h.raw(), out_c.raw() };
    litert_check(LiteRtRunCompiledModel(dec_compiled_, 0, 3, ins, 3, outs), "Nemotron decoder Run");

    out_hid.read(dec_hidden_.data(), dec_hidden_.size() * sizeof(float));
    out_h  .read(dec_h_.data(),      dec_h_.size() * sizeof(float));
    out_c  .read(dec_c_.data(),      dec_c_.size() * sizeof(float));
}

// Drain one window from pending_, run mel -> encoder (cache-aware) -> greedy
// RNN-T over the first encoder frame, return the emitted text.
std::string LiteRTNemotronStreamingStt::run_window() {
    const int n_frames = cfg_.actual_mel_frames;
    const int win      = chunk_samples();
    if (static_cast<int>(pending_.size()) < win) return {};

    std::vector<float> pcm(pending_.begin(), pending_.begin() + win);
    pending_.erase(pending_.begin(), pending_.begin() + win);

    constexpr float kLogFloor = 1.0f / static_cast<float>(1 << 24);  // 2^-24
    auto mel = audio::mel_spectrogram(
        pcm.data(), pcm.size(), cfg_.sample_rate,
        cfg_.n_fft, cfg_.hop_length, cfg_.win_length, cfg_.mel_bins,
        /*slaney=*/true, kLogFloor, /*center=*/true);

    const int produced = static_cast<int>(mel.size()) / cfg_.mel_bins;
    if (produced < n_frames) return {};
    if (produced > n_frames) {  // trim trailing frames (centred padding overshoot)
        std::vector<float> trimmed(static_cast<size_t>(cfg_.mel_bins) * n_frames);
        for (int b = 0; b < cfg_.mel_bins; ++b) {
            std::copy_n(&mel[static_cast<size_t>(b) * produced], n_frames,
                        &trimmed[static_cast<size_t>(b) * n_frames]);
        }
        mel.swap(trimmed);
    }

    auto env = LiteRTEngine::get().env();
    auto t_mel  = make_type(kLiteRtElementTypeFloat32, {1, cfg_.mel_bins, n_frames});
    auto t_len  = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_pre  = make_type(kLiteRtElementTypeFloat32, {1, cfg_.mel_bins, cfg_.pre_cache_size});
    auto t_clc  = make_type(kLiteRtElementTypeFloat32,
                            {cfg_.encoder_layers, 1, cfg_.attn_left_context, cfg_.encoder_hidden});
    auto t_clt  = make_type(kLiteRtElementTypeFloat32,
                            {cfg_.encoder_layers, 1, cfg_.encoder_hidden, cfg_.conv_cache_size});
    auto t_chl  = make_type(kLiteRtElementTypeInt64,   {1});
    auto t_enc  = make_type(kLiteRtElementTypeFloat32, {1, enc_t_out_, cfg_.encoder_hidden});
    auto t_i32  = make_type(kLiteRtElementTypeInt32,   {1});

    int64_t mel_len = n_frames;
    int64_t ch_len  = cache_last_channel_len_;

    LiteRtHostBuffer in_mel (env, t_mel, mel.size() * sizeof(float),                mel.data());
    LiteRtHostBuffer in_mlen(env, t_len, sizeof(int64_t),                           &mel_len);
    LiteRtHostBuffer in_pre (env, t_pre, pre_cache_.size() * sizeof(float),         pre_cache_.data());
    LiteRtHostBuffer in_clc (env, t_clc, cache_last_channel_.size() * sizeof(float), cache_last_channel_.data());
    LiteRtHostBuffer in_clt (env, t_clt, cache_last_time_.size() * sizeof(float),   cache_last_time_.data());
    LiteRtHostBuffer in_chl (env, t_chl, sizeof(int64_t),                           &ch_len);

    LiteRtHostBuffer out_enc (env, t_enc, static_cast<size_t>(enc_t_out_) * cfg_.encoder_hidden * sizeof(float));
    LiteRtHostBuffer out_elen(env, t_i32, sizeof(int32_t));
    LiteRtHostBuffer out_pre (env, t_pre, pre_cache_.size() * sizeof(float));
    LiteRtHostBuffer out_clc (env, t_clc, cache_last_channel_.size() * sizeof(float));
    LiteRtHostBuffer out_clt (env, t_clt, cache_last_time_.size() * sizeof(float));
    LiteRtHostBuffer out_chl (env, t_i32, sizeof(int32_t));

    // encoder I/O order (traced export):
    //   in:  mel, mel_len, pre_cache, cache_last_channel, cache_last_time, ch_len
    //   out: encoded, encoded_len, pre_cache', cache_last_channel', cache_last_time', ch_len'
    LiteRtTensorBuffer eins [6] = { in_mel.raw(), in_mlen.raw(), in_pre.raw(),
                                    in_clc.raw(), in_clt.raw(), in_chl.raw() };
    LiteRtTensorBuffer eouts[6] = { out_enc.raw(), out_elen.raw(), out_pre.raw(),
                                    out_clc.raw(), out_clt.raw(), out_chl.raw() };
    litert_check(LiteRtRunCompiledModel(enc_compiled_, 0, 6, eins, 6, eouts), "Nemotron encoder Run");

    // Roll caches forward for the next window.
    out_pre.read(pre_cache_.data(),          pre_cache_.size() * sizeof(float));
    out_clc.read(cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float));
    out_clt.read(cache_last_time_.data(),    cache_last_time_.size() * sizeof(float));
    int32_t chl_new = 0;
    out_chl.read(&chl_new, sizeof(int32_t));
    cache_last_channel_len_ = static_cast<int64_t>(chl_new);

    std::vector<float> encoded(static_cast<size_t>(enc_t_out_) * cfg_.encoder_hidden);
    out_enc.read(encoded.data(), encoded.size() * sizeof(float));

    // Greedy RNN-T over the FIRST encoder frame only (frame 0 = first
    // encoder_hidden floats). The second frame is right-context lookahead,
    // re-settled on the next window — feeding it duplicates output.
    auto t_encf   = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.encoder_hidden});
    auto t_dechid = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.decoder_hidden});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.vocab_size + 1});
    const size_t n_logits = static_cast<size_t>(cfg_.vocab_size) + 1;

    std::string emitted;
    for (int expand = 0; expand < cfg_.max_symbols; ++expand) {
        LiteRtHostBuffer in_encf  (env, t_encf,   static_cast<size_t>(cfg_.encoder_hidden) * sizeof(float), encoded.data());
        LiteRtHostBuffer in_dechid(env, t_dechid, dec_hidden_.size() * sizeof(float),                       dec_hidden_.data());
        LiteRtHostBuffer out_log  (env, t_logits, n_logits * sizeof(float));

        LiteRtTensorBuffer jins [2] = { in_encf.raw(), in_dechid.raw() };
        LiteRtTensorBuffer jouts[1] = { out_log.raw() };
        litert_check(LiteRtRunCompiledModel(jnt_compiled_, 0, 2, jins, 1, jouts), "Nemotron joint Run");

        std::vector<float> logits(n_logits);
        out_log.read(logits.data(), n_logits * sizeof(float));

        int   best   = 0;
        float best_v = logits[0];
        for (size_t i = 1; i < n_logits; ++i) {
            if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int>(i); }
        }
        if (best == cfg_.blank_id) break;  // advance to the next encoder frame

        emitted += token_to_text(best);
        run_decoder_step(best);  // advance the predictor for the next expansion
    }

    return emitted;
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

void LiteRTNemotronStreamingStt::begin_stream(int sample_rate) {
    cfg_.sample_rate = sample_rate;
    reset_stream_state();
    stream_init_ = true;
}

PartialResult LiteRTNemotronStreamingStt::push_chunk(const float* audio, size_t length) {
    if (!stream_init_) begin_stream(cfg_.sample_rate);
    pending_.insert(pending_.end(), audio, audio + length);

    std::string text;
    while (static_cast<int>(pending_.size()) >= chunk_samples()) {
        text += run_window();
    }
    accumulated_text_ += text;

    PartialResult out;
    out.text = std::move(text);
    return out;
}

void LiteRTNemotronStreamingStt::flush_stream() {
    // No-op: only full windows are decoded; a trailing partial window is held.
}

TranscriptionResult LiteRTNemotronStreamingStt::end_stream() {
    TranscriptionResult out;
    out.text = accumulated_text_;
    stream_init_ = false;
    return out;
}

void LiteRTNemotronStreamingStt::cancel_stream() {
    pending_.clear();
    accumulated_text_.clear();
    stream_init_ = false;
}

// ---------------------------------------------------------------------------
// Batch convenience: begin -> push everything -> end.
// ---------------------------------------------------------------------------

TranscriptionResult LiteRTNemotronStreamingStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    begin_stream(sample_rate);
    push_chunk(audio, length);
    return end_stream();
}

}  // namespace speech_core
