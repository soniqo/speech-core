#include "speech_core/models/litert_nemotron_streaming_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>

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

    // Auto-detect chunk-mode shape pins from the encoder input layouts. The
    // English Nemotron streaming model ships in 4 chunk-mode exports (80,
    // 160, 560, 1120 ms) whose mel_frames / pre_cache_size differ. Without
    // this, the wrapper assumes 80 ms defaults and silently feeds misshaped
    // tensors to larger-chunk exports.
    //   input[0] = audio_signal       [1, mel_bins, actual_mel_frames]
    //   input[2] = pre_cache          [1, mel_bins, pre_cache_size]
    //   input[3] = cache_last_channel [L, 1, attn_left, encoder_hidden]
    //   input[4] = cache_last_time    [L, 1, encoder_hidden, conv_cache_size]
    LiteRtLayout in_layout{};
    if (LiteRtGetCompiledModelInputTensorLayout(enc_compiled_, 0, 0, &in_layout) == kLiteRtStatusOk
        && in_layout.rank >= 3 && in_layout.dimensions[2] > 0) {
        cfg_.actual_mel_frames = static_cast<int>(in_layout.dimensions[2]);
    }
    if (LiteRtGetCompiledModelInputTensorLayout(enc_compiled_, 0, 2, &in_layout) == kLiteRtStatusOk
        && in_layout.rank >= 3 && in_layout.dimensions[2] > 0) {
        cfg_.pre_cache_size = static_cast<int>(in_layout.dimensions[2]);
    }
    if (LiteRtGetCompiledModelInputTensorLayout(enc_compiled_, 0, 3, &in_layout) == kLiteRtStatusOk
        && in_layout.rank >= 4) {
        if (in_layout.dimensions[0] > 0) cfg_.encoder_layers    = static_cast<int>(in_layout.dimensions[0]);
        if (in_layout.dimensions[2] > 0) cfg_.attn_left_context = static_cast<int>(in_layout.dimensions[2]);
        if (in_layout.dimensions[3] > 0) cfg_.encoder_hidden    = static_cast<int>(in_layout.dimensions[3]);
    }
    if (LiteRtGetCompiledModelInputTensorLayout(enc_compiled_, 0, 4, &in_layout) == kLiteRtStatusOk
        && in_layout.rank >= 4 && in_layout.dimensions[3] > 0) {
        cfg_.conv_cache_size = static_cast<int>(in_layout.dimensions[3]);
    }

    // Committed encoder frames per window. The cache-aware streaming encoder
    // emits T_out frames per Run; the FIRST output_frames_ are committed
    // (their text belongs in the transcript), the remainder are right-context
    // lookahead re-computed (with more left context) in the next window.
    // Without this, only frame 0 was decoded — fine for 80 ms but catastrophic
    // at larger chunk modes. Source: config.json streaming.outputFrames;
    // fall back to T_out - 1.
    output_frames_ = std::max(1, enc_t_out_ - 1);
    // LiteRT's introspection of input tensor layouts is unreliable for our
    // exports: the SignatureRunner's input-index order does not match the
    // semantic forward() argument order, and the layouts come back in an
    // arbitrary tensor-id order that may not put audio_signal at index 0.
    // Worse, on the freshly-exported 560 ms bundle the introspection
    // returned 80 ms-default dims for every input — i.e. it failed silently.
    // Read everything we can from config.json (the canonical source) and
    // only fall back to introspection / defaults when the file is missing.
    {
        std::string cfg_path = encoder_path;
        auto slash = cfg_path.find_last_of("/\\");
        if (slash != std::string::npos) cfg_path = cfg_path.substr(0, slash);
        cfg_path += "/config.json";
        std::string text = json::read_file(cfg_path);
        auto read_int = [&](const std::string& key) -> int {
            const std::string k = "\"" + key + "\"";
            auto pos = text.find(k);
            if (pos == std::string::npos) return -1;
            pos = text.find(':', pos);
            if (pos == std::string::npos) return -1;
            ++pos;
            while (pos < text.size()
                   && (text[pos] == ' ' || text[pos] == '\t'
                       || text[pos] == '\n' || text[pos] == '\r')) ++pos;
            int v = 0;
            bool any = false;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
                v = v * 10 + (text[pos] - '0'); ++pos; any = true;
            }
            return any ? v : -1;
        };
        if (!text.empty()) {
            int v;
            if ((v = read_int("outputFrames"))    > 0) output_frames_         = v;
            if ((v = read_int("melFrames"))       > 0) cfg_.actual_mel_frames = v;
            if ((v = read_int("preCacheSize"))    > 0) cfg_.pre_cache_size    = v;
            if ((v = read_int("attentionContext"))> 0) cfg_.attn_left_context = v;
            if ((v = read_int("convCacheSize"))   > 0) cfg_.conv_cache_size   = v;
            if ((v = read_int("encoderHidden"))   > 0) cfg_.encoder_hidden    = v;
            if ((v = read_int("encoderLayers"))   > 0) cfg_.encoder_layers    = v;
            if ((v = read_int("decoderHidden"))   > 0) cfg_.decoder_hidden    = v;
            if ((v = read_int("decoderLayers"))   > 0) cfg_.decoder_layers    = v;
            if ((v = read_int("vocabSize"))       > 0) { cfg_.vocab_size = v; cfg_.blank_id = v; }
            // Authoritative T_out: chunkSize + rightContext. The cache-aware
            // streaming Conformer emits committed frames (chunkSize) plus
            // right-context lookahead frames in one Run; only chunkSize gets
            // appended to the transcript, the lookahead is re-settled (with
            // more left context) on the next window.
            //
            // The earlier override `enc_t_out_ = output_frames_` was wrong:
            // it sized the out_enc host buffer to `output_frames_ * H * 4`
            // bytes while LiteRT writes `(output_frames_ + right_context) *
            // H * 4` bytes. The trailing right_context * H * 4 bytes
            // overflowed onto adjacent LiteRtHostBuffers (out_elen / out_pre
            // / out_clc), corrupted their heap headers, and the process died
            // silently mid-Run with no LiteRT error message — matching the
            // "exit 127, no error" symptom on 560 ms bundles.
            //
            // For 80 ms: chunkSize=1, rightContext=1 → T_out=2. ✓
            // For 320 ms: chunkSize=4, rightContext=3 → T_out=7.
            // For 560 ms: chunkSize=?, rightContext=? → T_out=8 (per bundle dump).
            int chunk_size  = read_int("chunkSize");
            int right_ctx   = read_int("rightContext");
            if (chunk_size > 0 && right_ctx >= 0) {
                enc_t_out_     = chunk_size + right_ctx;
                output_frames_ = chunk_size;
            }
        }
    }

    // Build the semantic-role -> signature-position mapping by reading the
    // encoder's input/output names from its signature. Each tensor is named
    // "serving_default_args_N"; the N is the original positional index from
    // the Python wrapper's forward() args (so 0=mel, 1=mlen, 2=pre_cache,
    // 3=cache_last_channel, 4=cache_last_time, 5=cache_last_channel_len).
    // The published 80 ms bundle has identity mapping (pos N has args_N),
    // but bundles produced by the WSL-side patched-NeMo + litert-torch 0.8
    // path scramble the tensor-id order, leaving the semantic ID encoded
    // only in the name. Without this remap, run_window() feeds buffers into
    // the wrong slots and LiteRT rejects them with "Cannot auto-resize
    // tensor args_2: no dims_signature exists" — same shape, wrong position.
    {
        LiteRtSignature sig = nullptr;
        if (LiteRtGetModelSignature(enc_model_, 0, &sig) == kLiteRtStatusOk && sig) {
            auto parse_args_id = [](const char* name) -> int {
                if (!name) return -1;
                // Try "args_N" (input naming) and "output_N" (output naming).
                for (const char* token : {"args_", "output_"}) {
                    const char* m = std::strstr(name, token);
                    if (m) {
                        int n = std::atoi(m + std::strlen(token));
                        if (n >= 0 && n < 6) return n;
                    }
                }
                // "StatefulPartitionedCall:N" fallback.
                const char* colon = std::strrchr(name, ':');
                if (colon) {
                    int n = std::atoi(colon + 1);
                    if (n >= 0 && n < 6) return n;
                }
                return -1;
            };
            // Inputs
            int sem_to_pos_in[6] = {-1,-1,-1,-1,-1,-1};
            for (int p = 0; p < 6; ++p) {
                const char* name = nullptr;
                if (LiteRtGetSignatureInputName(sig, p, &name) != kLiteRtStatusOk) continue;
                int sem = parse_args_id(name);
                if (sem >= 0) sem_to_pos_in[sem] = p;
            }
            bool full_in = true;
            for (int s = 0; s < 6; ++s) if (sem_to_pos_in[s] < 0) full_in = false;
            if (full_in) {
                for (int s = 0; s < 6; ++s) enc_in_sig_pos_[s] = sem_to_pos_in[s];
            }
            // Outputs
            int sem_to_pos_out[6] = {-1,-1,-1,-1,-1,-1};
            for (int p = 0; p < 6; ++p) {
                const char* name = nullptr;
                if (LiteRtGetSignatureOutputName(sig, p, &name) != kLiteRtStatusOk) continue;
                int sem = parse_args_id(name);
                if (sem < 0) {
                    const char* colon = name ? std::strrchr(name, ':') : nullptr;
                    if (colon) sem = std::atoi(colon + 1);
                }
                if (sem >= 0 && sem < 6) sem_to_pos_out[sem] = p;
            }
            bool full_out = true;
            for (int s = 0; s < 6; ++s) if (sem_to_pos_out[s] < 0) full_out = false;
            if (full_out) {
                for (int s = 0; s < 6; ++s) enc_out_sig_pos_[s] = sem_to_pos_out[s];
            }
        }
        // Same scrambling can hit the decoder (3 in / 3 out). Joint has been
        // identity in observed bundles, but we apply the same approach for
        // safety if it ever scrambles.
        LiteRtSignature dsig = nullptr;
        if (LiteRtGetModelSignature(dec_model_, 0, &dsig) == kLiteRtStatusOk && dsig) {
            auto parse3 = [](const char* name) -> int {
                if (!name) return -1;
                for (const char* token : {"args_", "output_"}) {
                    const char* m = std::strstr(name, token);
                    if (m) {
                        int n = std::atoi(m + std::strlen(token));
                        if (n >= 0 && n < 3) return n;
                    }
                }
                const char* colon = std::strrchr(name, ':');
                if (colon) {
                    int n = std::atoi(colon + 1);
                    if (n >= 0 && n < 3) return n;
                }
                return -1;
            };
            int dec_in_map[3] = {-1,-1,-1};
            for (int p = 0; p < 3; ++p) {
                const char* name = nullptr;
                if (LiteRtGetSignatureInputName(dsig, p, &name) != kLiteRtStatusOk) continue;
                int sem = parse3(name);
                if (sem >= 0) dec_in_map[sem] = p;
            }
            bool fi = true;
            for (int s = 0; s < 3; ++s) if (dec_in_map[s] < 0) fi = false;
            if (fi) for (int s = 0; s < 3; ++s) dec_in_sig_pos_[s] = dec_in_map[s];

            int dec_out_map[3] = {-1,-1,-1};
            for (int p = 0; p < 3; ++p) {
                const char* name = nullptr;
                if (LiteRtGetSignatureOutputName(dsig, p, &name) != kLiteRtStatusOk) continue;
                int sem = parse3(name);
                if (sem >= 0) dec_out_map[sem] = p;
            }
            bool fo = true;
            for (int s = 0; s < 3; ++s) if (dec_out_map[s] < 0) fo = false;
            if (fo) for (int s = 0; s < 3; ++s) dec_out_sig_pos_[s] = dec_out_map[s];
        }
    }

    LOGI("Nemotron streaming: vocab=%zu enc_hidden=%d dec_hidden=%d T_out=%d output_frames=%d mel_frames=%d pre_cache=%d attn=%d conv=%d window=%d samples",
         vocab_.size(), cfg_.encoder_hidden, cfg_.decoder_hidden, enc_t_out_,
         output_frames_, cfg_.actual_mel_frames, cfg_.pre_cache_size,
         cfg_.attn_left_context, cfg_.conv_cache_size, chunk_samples());
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
    // decoder I/O semantic order: in [token, h, c] -> out [hidden, h_new, c_new]
    // Some 560 ms-style bundles scramble tensor IDs; remap via dec_in_sig_pos_.
    LiteRtTensorBuffer sem_ins [3] = { in_tok.raw(), in_h.raw(), in_c.raw() };
    LiteRtTensorBuffer sem_outs[3] = { out_hid.raw(), out_h.raw(), out_c.raw() };
    LiteRtTensorBuffer ins[3], outs[3];
    for (int s = 0; s < 3; ++s) {
        ins [dec_in_sig_pos_[s]]  = sem_ins[s];
        outs[dec_out_sig_pos_[s]] = sem_outs[s];
    }
    litert_check(LiteRtRunCompiledModel(dec_compiled_, 0, 3, ins, 3, outs), "Nemotron decoder Run");

    out_hid.read(dec_hidden_.data(), dec_hidden_.size() * sizeof(float));
    out_h  .read(dec_h_.data(),      dec_h_.size() * sizeof(float));
    out_c  .read(dec_c_.data(),      dec_c_.size() * sizeof(float));
}

// Drain one window from pending_, run mel -> encoder (cache-aware) -> greedy
// RNN-T over the first encoder frame, return the emitted text.
//
// Per-Run timing is opt-in via SPEECH_LITERT_PROFILE=1. Cumulative totals are
// printed to stderr at end_stream(). Off by default — the gettime calls add
// ~100 ns per Run which is noise but the prints aren't.
namespace { struct ProfAcc {
    double mel_ms = 0, enc_ms = 0, jnt_ms = 0, dec_ms = 0;
    int    n_windows = 0, n_jnt = 0, n_dec = 0;
    bool   on = std::getenv("SPEECH_LITERT_PROFILE") != nullptr;
    ~ProfAcc() {
        if (!on || n_windows == 0) return;
        std::fprintf(stderr,
            "\n[profile] windows=%d mel=%.1f enc=%.1f jnt=%.1f(n=%d) dec=%.1f(n=%d) total=%.1f ms\n",
            n_windows, mel_ms, enc_ms, jnt_ms, n_jnt, dec_ms, n_dec,
            mel_ms + enc_ms + jnt_ms + dec_ms);
    }
}; }
static thread_local ProfAcc g_prof;

std::string LiteRTNemotronStreamingStt::run_window() {
    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
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

    if (g_prof.on) g_prof.mel_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();

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

    // encoder I/O semantic order:
    //   in:  mel, mel_len, pre_cache, cache_last_channel, cache_last_time, ch_len
    //   out: encoded, encoded_len, pre_cache', cache_last_channel', cache_last_time', ch_len'
    // The actual signature position of each tensor depends on the bundle:
    // identity for the published 80 ms bundle, scrambled for our newly-
    // exported bundles. enc_in_sig_pos_/enc_out_sig_pos_ remap.
    LiteRtTensorBuffer sem_eins[6] = {
        in_mel.raw(), in_mlen.raw(), in_pre.raw(),
        in_clc.raw(), in_clt.raw(), in_chl.raw()
    };
    LiteRtTensorBuffer sem_eouts[6] = {
        out_enc.raw(), out_elen.raw(), out_pre.raw(),
        out_clc.raw(), out_clt.raw(), out_chl.raw()
    };
    LiteRtTensorBuffer eins[6], eouts[6];
    for (int s = 0; s < 6; ++s) {
        eins [enc_in_sig_pos_[s]]  = sem_eins[s];
        eouts[enc_out_sig_pos_[s]] = sem_eouts[s];
    }
    auto t_enc_start = clk::now();
    litert_check(LiteRtRunCompiledModel(enc_compiled_, 0, 6, eins, 6, eouts), "Nemotron encoder Run");
    if (g_prof.on) g_prof.enc_ms += std::chrono::duration<double, std::milli>(clk::now() - t_enc_start).count();

    // Roll caches forward for the next window.
    out_pre.read(pre_cache_.data(),          pre_cache_.size() * sizeof(float));
    out_clc.read(cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float));
    out_clt.read(cache_last_time_.data(),    cache_last_time_.size() * sizeof(float));
    int32_t chl_new = 0;
    out_chl.read(&chl_new, sizeof(int32_t));
    cache_last_channel_len_ = static_cast<int64_t>(chl_new);

    std::vector<float> encoded(static_cast<size_t>(enc_t_out_) * cfg_.encoder_hidden);
    out_enc.read(encoded.data(), encoded.size() * sizeof(float));

    // Greedy RNN-T over the COMMITTED encoder frames (frames 0..output_frames_-1).
    // The remaining T_out - output_frames_ frames are right-context lookahead,
    // re-settled on the next window — feeding them duplicates output. For
    // 80 ms chunks (output_frames_=1) we decode only frame 0, same as before.
    // For 160/560/1120 ms (output_frames_=2/7/14) we now decode multiple
    // frames per window — this closes the streaming-quality gap.
    auto t_encf   = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.encoder_hidden});
    auto t_dechid = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.decoder_hidden});
    auto t_logits = make_type(kLiteRtElementTypeFloat32, {1, 1, cfg_.vocab_size + 1});
    const size_t n_logits = static_cast<size_t>(cfg_.vocab_size) + 1;

    std::string emitted;
    for (int frame = 0; frame < output_frames_; ++frame) {
        const size_t frame_off = static_cast<size_t>(frame) * cfg_.encoder_hidden;
        for (int expand = 0; expand < cfg_.max_symbols; ++expand) {
            LiteRtHostBuffer in_encf  (env, t_encf,
                static_cast<size_t>(cfg_.encoder_hidden) * sizeof(float),
                encoded.data() + frame_off);
            LiteRtHostBuffer in_dechid(env, t_dechid, dec_hidden_.size() * sizeof(float), dec_hidden_.data());
            LiteRtHostBuffer out_log  (env, t_logits, n_logits * sizeof(float));

            LiteRtTensorBuffer jins [2] = { in_encf.raw(), in_dechid.raw() };
            LiteRtTensorBuffer jouts[1] = { out_log.raw() };
            auto t_jnt = clk::now();
            litert_check(LiteRtRunCompiledModel(jnt_compiled_, 0, 2, jins, 1, jouts), "Nemotron joint Run");
            if (g_prof.on) { g_prof.jnt_ms += std::chrono::duration<double, std::milli>(clk::now() - t_jnt).count(); ++g_prof.n_jnt; }

            std::vector<float> logits(n_logits);
            out_log.read(logits.data(), n_logits * sizeof(float));

            int   best   = 0;
            float best_v = logits[0];
            for (size_t i = 1; i < n_logits; ++i) {
                if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int>(i); }
            }
            if (best == cfg_.blank_id) break;  // advance to the next encoder frame

            emitted += token_to_text(best);
            auto t_dec = clk::now();
            run_decoder_step(best);  // advance the predictor for the next expansion
            if (g_prof.on) { g_prof.dec_ms += std::chrono::duration<double, std::milli>(clk::now() - t_dec).count(); ++g_prof.n_dec; }
        }
    }

    if (g_prof.on) ++g_prof.n_windows;
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
    // Pad any leftover partial window with silence so the encoder gets
    // a final pass on the trailing audio. Otherwise the last 0..chunk_ms
    // of every utterance is dropped — bucketing the LibriSpeech-100
    // corpus on the ORT path (same architecture, same export contract)
    // showed this trailing-loss concentrated 7.23 absolute WER points
    // on utterances <5s. The same bug bites here.
    if (!stream_init_) return;
    if (pending_.empty()) return;
    const int win = chunk_samples();
    if (static_cast<int>(pending_.size()) >= win) return;
    pending_.resize(static_cast<size_t>(win), 0.0f);
    accumulated_text_ += run_window();
}

TranscriptionResult LiteRTNemotronStreamingStt::end_stream() {
    flush_stream();
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
