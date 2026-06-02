#include "speech_core/models/onnx_nemotron_streaming_stt.h"

#include "speech_core/audio/mel.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/util/json.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace speech_core {

namespace {

// RAII for an OrtAllocator-allocated string (input/output name).
struct OrtStringHandle {
    const OrtApi* api = nullptr;
    OrtAllocator* alloc = nullptr;
    char* p = nullptr;
    OrtStringHandle(const OrtApi* a, OrtAllocator* al) : api(a), alloc(al) {}
    ~OrtStringHandle() { if (p && api && alloc) api->AllocatorFree(alloc, p); }
    OrtStringHandle(const OrtStringHandle&) = delete;
    OrtStringHandle& operator=(const OrtStringHandle&) = delete;
};

}  // namespace

void OnnxNemotronStreamingStt::query_io_names(OrtSession* session, IoNames& names) {
    OrtAllocator* alloc = nullptr;
    ort_check(api_, api_->GetAllocatorWithDefaultOptions(&alloc));

    size_t n_in = 0;
    ort_check(api_, api_->SessionGetInputCount(session, &n_in));
    names.in_names_str.reserve(n_in);
    for (size_t i = 0; i < n_in; ++i) {
        OrtStringHandle h(api_, alloc);
        ort_check(api_, api_->SessionGetInputName(session, i, alloc, &h.p));
        names.in_names_str.emplace_back(h.p);
    }
    size_t n_out = 0;
    ort_check(api_, api_->SessionGetOutputCount(session, &n_out));
    names.out_names_str.reserve(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        OrtStringHandle h(api_, alloc);
        ort_check(api_, api_->SessionGetOutputName(session, i, alloc, &h.p));
        names.out_names_str.emplace_back(h.p);
    }

    names.in_names.clear();
    names.in_names.reserve(names.in_names_str.size());
    for (auto& s : names.in_names_str) names.in_names.push_back(s.c_str());
    names.out_names.clear();
    names.out_names.reserve(names.out_names_str.size());
    for (auto& s : names.out_names_str) names.out_names.push_back(s.c_str());
}

OnnxNemotronStreamingStt::OnnxNemotronStreamingStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path, bool hw_accel)
    : OnnxNemotronStreamingStt(encoder_path, decoder_path, joint_path,
                               vocab_path, Config{}, hw_accel) {}

OnnxNemotronStreamingStt::OnnxNemotronStreamingStt(
    const std::string& encoder_path, const std::string& decoder_path,
    const std::string& joint_path, const std::string& vocab_path,
    const Config& config, bool hw_accel)
    : cfg_(config)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    // Encoder is the heavy graph (24-layer Conformer ~580M params) — runs on
    // the GPU when available. Decoder LSTM + joint are tiny (~50M decoder,
    // ~1M joint) and ship small tensors per expansion; CUDA dispatch beats
    // their compute time, same anti-pattern as Parakeet's decoder_joint.
    enc_ = engine.load(encoder_path, hw_accel);
    dec_ = engine.load(decoder_path, false);
    jnt_ = engine.load(joint_path,   false);

    query_io_names(enc_, enc_io_);
    query_io_names(dec_, dec_io_);
    query_io_names(jnt_, jnt_io_);

    load_vocab(vocab_path);

    // Encoder output[0] = encoded_output [1, T_out, H]; read T_out for buffer
    // sizing. Static shape, available without a resize. Tolerate failure —
    // fall back to the default (2 @ 80 ms).
    OrtTypeInfo* ti = nullptr;
    if (api_->SessionGetOutputTypeInfo(enc_, 0, &ti) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (api_->CastTypeInfoToTensorInfo(ti, &shape) == nullptr && shape) {
            size_t rank = 0;
            api_->GetDimensionsCount(shape, &rank);
            if (rank >= 3) {
                std::vector<int64_t> dims(rank);
                api_->GetDimensions(shape, dims.data(), rank);
                if (dims[1] > 0) enc_t_out_ = static_cast<int>(dims[1]);
            }
        }
        api_->ReleaseTypeInfo(ti);
    }

    // Encoder input[0] = audio_signal [1, mel_bins, frames]; read frames to
    // auto-detect chunk size. Different exports pin different chunk_ms
    // (80/160/560/1120 -> 9/17/64/121 mel frames). Without this, the wrapper
    // assumes 80ms and feeds misaligned windows to a larger-chunk export.
    OrtTypeInfo* in_ti = nullptr;
    if (api_->SessionGetInputTypeInfo(enc_, 0, &in_ti) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (api_->CastTypeInfoToTensorInfo(in_ti, &shape) == nullptr && shape) {
            size_t rank = 0;
            api_->GetDimensionsCount(shape, &rank);
            if (rank >= 3) {
                std::vector<int64_t> dims(rank);
                api_->GetDimensions(shape, dims.data(), rank);
                if (dims[2] > 0) cfg_.actual_mel_frames = static_cast<int>(dims[2]);
            }
        }
        api_->ReleaseTypeInfo(in_ti);
    }

    // Encoder input[2] = pre_cache [1, mel_bins, pre_cache_size]. The
    // pre_cache_size pin differs across chunk-mode exports (16 at 80/160 ms,
    // 9 at 560/1120 ms) and the wrong size shape-mismatches the Run call
    // (which on the CUDA EP fails silently — exit code 0, no transcript).
    OrtTypeInfo* in_ti2 = nullptr;
    if (api_->SessionGetInputTypeInfo(enc_, 2, &in_ti2) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (api_->CastTypeInfoToTensorInfo(in_ti2, &shape) == nullptr && shape) {
            size_t rank = 0;
            api_->GetDimensionsCount(shape, &rank);
            if (rank >= 3) {
                std::vector<int64_t> dims(rank);
                api_->GetDimensions(shape, dims.data(), rank);
                if (dims[2] > 0) cfg_.pre_cache_size = static_cast<int>(dims[2]);
            }
        }
        api_->ReleaseTypeInfo(in_ti2);
    }

    // Encoder input[3] = cache_last_channel [L, 1, attn_left, H]. Auto-detect
    // attn_left_context too, in case any chunk-mode export deviates from 70.
    OrtTypeInfo* in_ti3 = nullptr;
    if (api_->SessionGetInputTypeInfo(enc_, 3, &in_ti3) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (api_->CastTypeInfoToTensorInfo(in_ti3, &shape) == nullptr && shape) {
            size_t rank = 0;
            api_->GetDimensionsCount(shape, &rank);
            if (rank >= 4) {
                std::vector<int64_t> dims(rank);
                api_->GetDimensions(shape, dims.data(), rank);
                if (dims[0] > 0) cfg_.encoder_layers    = static_cast<int>(dims[0]);
                if (dims[2] > 0) cfg_.attn_left_context = static_cast<int>(dims[2]);
                if (dims[3] > 0) cfg_.encoder_hidden    = static_cast<int>(dims[3]);
            }
        }
        api_->ReleaseTypeInfo(in_ti3);
    }

    // Encoder input[4] = cache_last_time [L, 1, H, conv_cache_size]. Auto-
    // detect conv_cache_size (kernel - 1; typically 8 but worth confirming).
    OrtTypeInfo* in_ti4 = nullptr;
    if (api_->SessionGetInputTypeInfo(enc_, 4, &in_ti4) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape = nullptr;
        if (api_->CastTypeInfoToTensorInfo(in_ti4, &shape) == nullptr && shape) {
            size_t rank = 0;
            api_->GetDimensionsCount(shape, &rank);
            if (rank >= 4) {
                std::vector<int64_t> dims(rank);
                api_->GetDimensions(shape, dims.data(), rank);
                if (dims[3] > 0) cfg_.conv_cache_size = static_cast<int>(dims[3]);
            }
        }
        api_->ReleaseTypeInfo(in_ti4);
    }

    // Committed encoder frames per window. The cache-aware streaming encoder
    // emits T_out frames per Run, of which the FIRST `output_frames_` are
    // committed (their text gets emitted into the transcript) and the
    // remaining T_out - output_frames_ are right-context lookahead that gets
    // re-computed (with more left context) in the next window. Without this,
    // the wrapper only decoded frame 0 regardless of chunk size — fine for
    // 80 ms (output_frames=1) but catastrophic for 160/560/1120 ms exports
    // where output_frames is 2/7/14.
    //
    // Source of truth: config.json's streaming.outputFrames field. Fall back
    // to T_out - 1 (assume one lookahead frame) if config.json isn't on disk.
    output_frames_ = std::max(1, enc_t_out_ - 1);
    {
        // Look for config.json next to the encoder. Strip the filename.
        std::string cfg_path = encoder_path;
        auto slash = cfg_path.find_last_of("/\\");
        if (slash != std::string::npos) cfg_path = cfg_path.substr(0, slash);
        cfg_path += "/config.json";
        std::string text = json::read_file(cfg_path);
        if (!text.empty()) {
            // Cheap scan for "outputFrames": <int>. Tolerates nesting.
            const std::string key = "\"outputFrames\"";
            auto pos = text.find(key);
            if (pos != std::string::npos) {
                pos = text.find(':', pos);
                if (pos != std::string::npos) {
                    ++pos;
                    while (pos < text.size()
                           && (text[pos] == ' ' || text[pos] == '\t'
                               || text[pos] == '\n' || text[pos] == '\r')) ++pos;
                    int v = 0;
                    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
                        v = v * 10 + (text[pos] - '0');
                        ++pos;
                    }
                    if (v > 0 && v <= enc_t_out_) output_frames_ = v;
                }
            }
        }
    }

    LOGI("Nemotron streaming (ORT): vocab=%zu enc_hidden=%d dec_hidden=%d T_out=%d output_frames=%d mel_frames=%d window=%d samples",
         vocab_.size(), cfg_.encoder_hidden, cfg_.decoder_hidden, enc_t_out_,
         output_frames_, cfg_.actual_mel_frames, chunk_samples());
}

OnnxNemotronStreamingStt::~OnnxNemotronStreamingStt() {
    if (jnt_) api_->ReleaseSession(jnt_);
    if (dec_) api_->ReleaseSession(dec_);
    if (enc_) api_->ReleaseSession(enc_);
}

bool OnnxNemotronStreamingStt::load_vocab(const std::string& path) {
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

std::string OnnxNemotronStreamingStt::token_to_text(int id) const {
    if (id < 0 || id >= static_cast<int>(vocab_.size())) return {};
    std::string piece = vocab_[id];
    // SentencePiece \xE2\x96\x81 (U+2581) → leading space.
    if (piece.size() >= 3 &&
        static_cast<unsigned char>(piece[0]) == 0xE2 &&
        static_cast<unsigned char>(piece[1]) == 0x96 &&
        static_cast<unsigned char>(piece[2]) == 0x81) {
        return " " + piece.substr(3);
    }
    return piece;
}

void OnnxNemotronStreamingStt::reset_stream_state() {
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
    // sees a meaningful hidden state (matches the LiteRT impl).
    run_decoder_step(cfg_.blank_id);
}

void OnnxNemotronStreamingStt::run_decoder_step(int64_t token_id) {
    auto* mem = OnnxEngine::get().cpu_memory();

    const int64_t s_tok[2]   = {1, 1};
    const int64_t s_state[3] = {cfg_.decoder_layers, 1, cfg_.decoder_hidden};

    int64_t tok = token_id;
    OrtValue* t_tok = nullptr;
    OrtValue* t_h   = nullptr;
    OrtValue* t_c   = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &tok, sizeof(int64_t), s_tok, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_tok));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, dec_h_.data(), dec_h_.size() * sizeof(float), s_state, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_h));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, dec_c_.data(), dec_c_.size() * sizeof(float), s_state, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_c));

    OrtValue* in[3]  = {t_tok, t_h, t_c};
    OrtValue* out[3] = {nullptr, nullptr, nullptr};
    ort_check(api_, api_->Run(
        dec_, nullptr,
        dec_io_.in_names.data(),  in,  dec_io_.in_names.size(),
        dec_io_.out_names.data(), dec_io_.out_names.size(), out));

    auto copy_out = [&](OrtValue* v, void* dst, size_t bytes) {
        void* p = nullptr;
        ort_check(api_, api_->GetTensorMutableData(v, &p));
        std::memcpy(dst, p, bytes);
    };
    copy_out(out[0], dec_hidden_.data(), dec_hidden_.size() * sizeof(float));
    copy_out(out[1], dec_h_.data(),      dec_h_.size()      * sizeof(float));
    copy_out(out[2], dec_c_.data(),      dec_c_.size()      * sizeof(float));

    for (int i = 2; i >= 0; --i) api_->ReleaseValue(out[i]);
    api_->ReleaseValue(t_c);
    api_->ReleaseValue(t_h);
    api_->ReleaseValue(t_tok);
}

// Drain one window from pending_, run mel -> encoder (cache-aware) -> greedy
// RNN-T over the first encoder frame, return the emitted text.
std::string OnnxNemotronStreamingStt::run_window() {
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

    auto* mem = OnnxEngine::get().cpu_memory();

    const int64_t s_mel  [3] = {1, cfg_.mel_bins, n_frames};
    const int64_t s_mlen [1] = {1};
    const int64_t s_pre  [3] = {1, cfg_.mel_bins, cfg_.pre_cache_size};
    const int64_t s_clc  [4] = {cfg_.encoder_layers, 1, cfg_.attn_left_context, cfg_.encoder_hidden};
    const int64_t s_clt  [4] = {cfg_.encoder_layers, 1, cfg_.encoder_hidden, cfg_.conv_cache_size};
    const int64_t s_chl  [1] = {1};

    int64_t mel_len = n_frames;
    int64_t ch_len  = cache_last_channel_len_;

    OrtValue* t_mel  = nullptr;
    OrtValue* t_mlen = nullptr;
    OrtValue* t_pre  = nullptr;
    OrtValue* t_clc  = nullptr;
    OrtValue* t_clt  = nullptr;
    OrtValue* t_chl  = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, mel.data(), mel.size() * sizeof(float), s_mel, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_mel));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &mel_len, sizeof(int64_t), s_mlen, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_mlen));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, pre_cache_.data(), pre_cache_.size() * sizeof(float), s_pre, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_pre));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float),
        s_clc, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_clc));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, cache_last_time_.data(), cache_last_time_.size() * sizeof(float),
        s_clt, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_clt));
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, &ch_len, sizeof(int64_t), s_chl, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_chl));

    OrtValue* enc_in[6]  = {t_mel, t_mlen, t_pre, t_clc, t_clt, t_chl};
    OrtValue* enc_out[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    if (std::getenv("NEMOTRON_DEBUG")) {
        std::fprintf(stderr,
            "[nem] run_window: mel=%dx%d pcm=%zu pre_cache=%zu clc=%zu clt=%zu ch_len=%lld\n",
            cfg_.mel_bins, n_frames, pcm.size(), pre_cache_.size(),
            cache_last_channel_.size(), cache_last_time_.size(),
            static_cast<long long>(ch_len));
    }
    ort_check(api_, api_->Run(
        enc_, nullptr,
        enc_io_.in_names.data(),  enc_in,  enc_io_.in_names.size(),
        enc_io_.out_names.data(), enc_io_.out_names.size(), enc_out));
    if (std::getenv("NEMOTRON_DEBUG")) {
        std::fprintf(stderr, "[nem] encoder Run ok\n");
    }

    auto copy_out = [&](OrtValue* v, void* dst, size_t bytes) {
        void* p = nullptr;
        ort_check(api_, api_->GetTensorMutableData(v, &p));
        std::memcpy(dst, p, bytes);
    };

    // Roll caches forward for the next window. Encoder I/O order matches the
    // LiteRT export: encoded_output, encoded_length, new_pre_cache,
    // new_cache_last_channel, new_cache_last_time, new_cache_last_channel_len.
    std::vector<float> encoded(static_cast<size_t>(enc_t_out_) * cfg_.encoder_hidden);
    copy_out(enc_out[0], encoded.data(),             encoded.size()             * sizeof(float));
    copy_out(enc_out[2], pre_cache_.data(),          pre_cache_.size()          * sizeof(float));
    copy_out(enc_out[3], cache_last_channel_.data(), cache_last_channel_.size() * sizeof(float));
    copy_out(enc_out[4], cache_last_time_.data(),    cache_last_time_.size()    * sizeof(float));
    {
        // encoded_length / cache_last_channel_len come back as scalars (int32
        // on the LiteRT export; ONNX preserves the same dtype). Read the
        // type info and copy accordingly.
        void* p = nullptr;
        ort_check(api_, api_->GetTensorMutableData(enc_out[5], &p));
        int64_t v = 0;
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check(api_, api_->GetTensorTypeAndShape(enc_out[5], &info));
        ONNXTensorElementDataType etype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        api_->GetTensorElementType(info, &etype);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            v = static_cast<int64_t>(*static_cast<int32_t*>(p));
        } else if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            v = *static_cast<int64_t*>(p);
        }
        cache_last_channel_len_ = v;
    }

    for (int i = 5; i >= 0; --i) api_->ReleaseValue(enc_out[i]);
    api_->ReleaseValue(t_chl);
    api_->ReleaseValue(t_clt);
    api_->ReleaseValue(t_clc);
    api_->ReleaseValue(t_pre);
    api_->ReleaseValue(t_mlen);
    api_->ReleaseValue(t_mel);

    // Greedy RNN-T over the COMMITTED encoder frames (frames 0..output_frames_-1).
    // The remaining T_out - output_frames_ frames are right-context lookahead,
    // re-settled on the next window — feeding them duplicates output. For
    // 80 ms chunks (output_frames_=1) we decode only frame 0, same as before.
    // For 160/560/1120 ms (output_frames_=2/7/14) we now decode multiple
    // frames per window — this is what closes the streaming-quality gap.
    const int64_t s_encf  [3] = {1, 1, cfg_.encoder_hidden};
    const int64_t s_dechid[3] = {1, 1, cfg_.decoder_hidden};
    const size_t  n_logits     = static_cast<size_t>(cfg_.vocab_size) + 1;

    std::string emitted;
    for (int frame = 0; frame < output_frames_; ++frame) {
        // Offset into encoded[]: frame 0 starts at index 0, frame 1 at
        // encoder_hidden, etc. The joint sees one encoder_hidden vector.
        const size_t frame_off = static_cast<size_t>(frame) * cfg_.encoder_hidden;
        for (int expand = 0; expand < cfg_.max_symbols; ++expand) {
            OrtValue* t_encf   = nullptr;
            OrtValue* t_dechid = nullptr;
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, encoded.data() + frame_off,
                static_cast<size_t>(cfg_.encoder_hidden) * sizeof(float),
                s_encf, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_encf));
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, dec_hidden_.data(), dec_hidden_.size() * sizeof(float),
                s_dechid, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_dechid));

            OrtValue* jin[2]  = {t_encf, t_dechid};
            OrtValue* jout[1] = {nullptr};
            ort_check(api_, api_->Run(
                jnt_, nullptr,
                jnt_io_.in_names.data(),  jin,  jnt_io_.in_names.size(),
                jnt_io_.out_names.data(), jnt_io_.out_names.size(), jout));

            std::vector<float> logits(n_logits);
            copy_out(jout[0], logits.data(), n_logits * sizeof(float));

            api_->ReleaseValue(jout[0]);
            api_->ReleaseValue(t_dechid);
            api_->ReleaseValue(t_encf);

            int   best   = 0;
            float best_v = logits[0];
            for (size_t i = 1; i < n_logits; ++i) {
                if (logits[i] > best_v) { best_v = logits[i]; best = static_cast<int>(i); }
            }
            if (best == cfg_.blank_id) break;  // advance to the next encoder frame

            emitted += token_to_text(best);
            run_decoder_step(best);  // advance the predictor for the next expansion
        }
    }
    return emitted;
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

void OnnxNemotronStreamingStt::begin_stream(int sample_rate) {
    cfg_.sample_rate = sample_rate;
    reset_stream_state();
    stream_init_ = true;
}

PartialResult OnnxNemotronStreamingStt::push_chunk(const float* audio, size_t length) {
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

void OnnxNemotronStreamingStt::flush_stream() {
    // No-op: only full windows are decoded; a trailing partial window is held.
}

TranscriptionResult OnnxNemotronStreamingStt::end_stream() {
    TranscriptionResult out;
    out.text = accumulated_text_;
    stream_init_ = false;
    return out;
}

void OnnxNemotronStreamingStt::cancel_stream() {
    pending_.clear();
    accumulated_text_.clear();
    stream_init_ = false;
}

TranscriptionResult OnnxNemotronStreamingStt::transcribe(
    const float* audio, size_t length, int sample_rate)
{
    begin_stream(sample_rate);
    push_chunk(audio, length);
    return end_stream();
}

}  // namespace speech_core
