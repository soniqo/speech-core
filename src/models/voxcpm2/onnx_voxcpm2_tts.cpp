#include "speech_core/models/onnx_voxcpm2_tts.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace speech_core {

namespace {

// RAII for an OrtAllocator-allocated string (input/output name).
struct OrtStringHandle {
    const OrtApi* api = nullptr;
    OrtAllocator* alloc = nullptr;
    char* p = nullptr;
    OrtStringHandle(const OrtApi* a, OrtAllocator* al) : api(a), alloc(al) {}
    ~OrtStringHandle() {
        if (p && api && alloc) api->AllocatorFree(alloc, p);
    }
    OrtStringHandle(const OrtStringHandle&) = delete;
    OrtStringHandle& operator=(const OrtStringHandle&) = delete;
};

}  // namespace

// ---------------------------------------------------------------------------
// IO name introspection
// ---------------------------------------------------------------------------

void OnnxVoxCPM2Tts::query_io_names(OrtSession* session, IoNames& names) {
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

    // Build parallel const char*[] for api_->Run().
    names.in_names.clear();
    names.in_names.reserve(names.in_names_str.size());
    for (auto& s : names.in_names_str) names.in_names.push_back(s.c_str());
    names.out_names.clear();
    names.out_names.reserve(names.out_names_str.size());
    for (auto& s : names.out_names_str) names.out_names.push_back(s.c_str());
}

int OnnxVoxCPM2Tts::query_prefill_context(OrtSession* session) {
    constexpr int kDefault = 512;
    if (!session) return kDefault;

    size_t n_in = 0;
    if (api_->SessionGetInputCount(session, &n_in) != nullptr) return kDefault;

    for (size_t i = 0; i < n_in; ++i) {
        OrtTypeInfo* type_info = nullptr;
        if (api_->SessionGetInputTypeInfo(session, i, &type_info) != nullptr) continue;
        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        if (api_->CastTypeInfoToTensorInfo(type_info, &shape_info) != nullptr || !shape_info) {
            api_->ReleaseTypeInfo(type_info);
            continue;
        }
        size_t rank = 0;
        api_->GetDimensionsCount(shape_info, &rank);
        if (rank == 4) {  // audio_feats: [1, max_text, patch, feat]
            std::vector<int64_t> dims(rank);
            api_->GetDimensions(shape_info, dims.data(), rank);
            api_->ReleaseTypeInfo(type_info);
            const int v = static_cast<int>(dims[1]);
            if (v > 0) return v;
            return kDefault;
        }
        api_->ReleaseTypeInfo(type_info);
    }
    return kDefault;
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

OnnxVoxCPM2Tts::OnnxVoxCPM2Tts(const std::string& decoder_path,
                                const std::string& audio_encoder_path,
                                const std::string& audio_decoder_path,
                                const std::string& tokenizer_path,
                                bool hw_accel)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    // The unified graph runs both the one-shot prefill and the 25-60 per-step
    // token decodes. Token-step shapes are fixed, so this is still the CUDA
    // Graph capture target (SPEECH_CORE_CUDA_GRAPH=1).
    decoder_session_       = engine.load(decoder_path, hw_accel,
                                         engine.cuda_graph_enabled());
    audio_encoder_session_ = engine.load(audio_encoder_path, hw_accel);
    audio_decoder_session_ = engine.load(audio_decoder_path, hw_accel);

    // Split the unified graph's I/O into prefill (no prefix) and token (ts_).
    IoNames full;
    query_io_names(decoder_session_, full);
    auto is_ts = [](const std::string& s) { return s.rfind("ts_", 0) == 0; };
    for (auto& s : full.in_names_str)
        (is_ts(s) ? step_io_ : prefill_io_).in_names_str.push_back(s);
    for (auto& s : full.out_names_str)
        (is_ts(s) ? step_io_ : prefill_io_).out_names_str.push_back(s);
    auto rebuild = [](IoNames& io) {
        io.in_names.clear();
        for (auto& s : io.in_names_str) io.in_names.push_back(s.c_str());
        io.out_names.clear();
        for (auto& s : io.out_names_str) io.out_names.push_back(s.c_str());
    };
    rebuild(prefill_io_);
    rebuild(step_io_);

    query_io_names(audio_encoder_session_, encoder_io_);
    query_io_names(audio_decoder_session_, decoder_io_);

    // Context window comes from the prefill graph's rank-4 audio_feats input.
    max_text_                = query_prefill_context(decoder_session_);
    full_seq_                = max_text_ + kMaxGenerated;
    base_cache_floats_       = 2L * kBaseLayers     * kKvHeads * full_seq_ * kHeadDim;
    residual_cache_floats_   = 2L * kResidualLayers * kKvHeads * full_seq_ * kHeadDim;
    base_prefill_floats_     = 2L * kBaseLayers     * kKvHeads * max_text_ * kHeadDim;
    residual_prefill_floats_ = 2L * kResidualLayers * kKvHeads * max_text_ * kHeadDim;

    tokenizer_         = std::make_unique<VoxCPM2Tokenizer>(tokenizer_path);
    audio_start_token_ = tokenizer_->token_id("<|audio_start|>");
    if (audio_start_token_ < 0) {
        throw std::runtime_error(
            "ONNX VoxCPM2: tokenizer is missing <|audio_start|> — bundle is malformed");
    }
    ref_audio_start_token_ = tokenizer_->token_id("<|ref_audio_start|>");
    ref_audio_end_token_   = tokenizer_->token_id("<|ref_audio_end|>");
}

OnnxVoxCPM2Tts::~OnnxVoxCPM2Tts() {
    if (audio_decoder_session_)  api_->ReleaseSession(audio_decoder_session_);
    if (audio_encoder_session_)  api_->ReleaseSession(audio_encoder_session_);
    if (decoder_session_)        api_->ReleaseSession(decoder_session_);
}

void OnnxVoxCPM2Tts::cancel() {
    cancelled_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Reference conditioning
// ---------------------------------------------------------------------------

void OnnxVoxCPM2Tts::clear_reference() {
    ref_feats_.clear();
    ref_feats_.shrink_to_fit();
    ref_frames_ = 0;
}

void OnnxVoxCPM2Tts::set_reference(const float* pcm, size_t length, int sample_rate) {
    clear_reference();
    if (!pcm || length == 0) return;
    if (ref_audio_start_token_ < 0 || ref_audio_end_token_ < 0) {
        throw std::runtime_error(
            "ONNX VoxCPM2: tokenizer lacks <|ref_audio_start|>/<|ref_audio_end|> — "
            "this bundle does not support voice cloning");
    }

    // Resample to the encoder's conditioning rate (16 kHz).
    std::vector<float> mono;
    if (sample_rate != kCondSampleRate) {
        mono = Resampler::resample(pcm, length, sample_rate, kCondSampleRate);
    } else {
        mono.assign(pcm, pcm + length);
    }
    if (mono.empty()) return;

    // Trim leading silence (same gate as the LiteRT variant).
    {
        float peak = 0.0f;
        for (float v : mono) peak = std::max(peak, std::abs(v));
        const float gate = std::max(0.01f, 0.1f * peak);
        constexpr int kWin = 320;  // 20 ms @ 16 kHz
        size_t start = 0;
        for (; start + kWin <= mono.size(); start += kWin) {
            double ss = 0.0;
            for (int i = 0; i < kWin; ++i) ss += double(mono[start + i]) * mono[start + i];
            if (std::sqrt(ss / kWin) > gate) break;
        }
        if (start + kWin <= mono.size() && start > 0) {
            const size_t preroll = std::min<size_t>(start, kCondSampleRate / 10);
            mono.erase(mono.begin(), mono.begin() + (start - preroll));
        }
    }

    // Rescue quiet reference clips (same recipe as the LiteRT variant). See
    // litert_voxcpm2_tts.cpp set_reference() for the full rationale.
    {
        double ss = 0.0;
        float peak = 0.0f;
        for (float v : mono) {
            ss += double(v) * v;
            peak = std::max(peak, std::abs(v));
        }
        if (!mono.empty() && peak > 1e-6f) {
            const double rms = std::sqrt(ss / mono.size());
            const float kQuietThreshold = 0.04f;
            const float kTargetRms      = 0.08f;
            const float kPeakCap        = 0.95f;
            if (rms < kQuietThreshold) {
                float scale = static_cast<float>(kTargetRms / rms);
                if (peak * scale > kPeakCap) scale = kPeakCap / peak;
                for (float& v : mono) v *= scale;
            }
        }
    }

    const size_t real_samples = std::min<size_t>(mono.size(), kRefAudioSamples);
    int real_frames = static_cast<int>((real_samples + kRefFrameStride - 1) / kRefFrameStride);
    real_frames = std::clamp(real_frames, 1, kMaxRefFrames);

    // Pad/truncate to the encoder's fixed-length input.
    mono.resize(kRefAudioSamples, 0.0f);

    auto* mem = OnnxEngine::get().cpu_memory();

    const int64_t in_shape[2]  = {1, kRefAudioSamples};
    OrtValue* t_in = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, mono.data(), mono.size() * sizeof(float),
        in_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_in));

    OrtValue* enc_inputs[1]  = {t_in};
    OrtValue* enc_outputs[1] = {nullptr};
    ort_check(api_, api_->Run(
        audio_encoder_session_, nullptr,
        encoder_io_.in_names.data(),  enc_inputs,  encoder_io_.in_names.size(),
        encoder_io_.out_names.data(), encoder_io_.out_names.size(), enc_outputs));

    float* out_ptr = nullptr;
    ort_check(api_, api_->GetTensorMutableData(enc_outputs[0], (void**)&out_ptr));

    std::vector<float> enc_out(static_cast<size_t>(kMaxRefFrames) * kPredFeatFloats);
    std::memcpy(enc_out.data(), out_ptr, enc_out.size() * sizeof(float));

    api_->ReleaseValue(enc_outputs[0]);
    api_->ReleaseValue(t_in);

    ref_feats_.assign(enc_out.begin(),
                      enc_out.begin() + static_cast<size_t>(real_frames) * kPredFeatFloats);
    ref_frames_ = real_frames;

    if (std::getenv("VOXCPM2_DEBUG")) {
        double ss = 0.0;
        for (float v : ref_feats_) ss += static_cast<double>(v) * v;
        const double rms = ref_feats_.empty() ? 0.0 : std::sqrt(ss / ref_feats_.size());
        LOGI("VoxCPM2 (ORT) reference: %d frames (%zu real samples), feats rms=%.4f",
             ref_frames_, real_samples, rms);
    }
}

// ---------------------------------------------------------------------------
// synthesize()
// ---------------------------------------------------------------------------

void OnnxVoxCPM2Tts::synthesize(const std::string& text,
                                 const std::string& /*language*/,
                                 TTSChunkCallback on_chunk)
{
    if (!on_chunk) return;
    cancelled_.store(false, std::memory_order_relaxed);

    tokens_generated_      = 0;
    stopped_on_stop_token_ = false;
    seed_used_             = 0;

    // --- 1. Build the prefill sequence (same layout as the LiteRT wrapper).
    std::string prompt = "(" + instruction_ + ")" + text;
    std::vector<int> target_ids = tokenizer_->encode(prompt);
    target_ids.push_back(audio_start_token_);

    if (ref_frames_ > 0 && !target_ids.empty()
        && target_ids.front() == tokenizer_->bos_id()) {
        target_ids.erase(target_ids.begin());
    }

    std::vector<int64_t> text_tokens(max_text_, 0);
    std::vector<float>   text_mask  (max_text_, 0.0f);
    std::vector<float>   audio_feats(static_cast<size_t>(max_text_) * kPredFeatFloats, 0.0f);
    std::vector<float>   audio_mask (max_text_, 0.0f);

    int pos = 0;
    if (ref_frames_ > 0) {
        if (2 + ref_frames_ + 1 > max_text_) {
            throw std::runtime_error(
                "ONNX VoxCPM2: reference block does not fit the context window");
        }
        text_tokens[pos] = ref_audio_start_token_;
        text_mask[pos]   = 1.0f;
        ++pos;
        for (int f = 0; f < ref_frames_; ++f) {
            audio_mask[pos] = 1.0f;
            std::memcpy(audio_feats.data() + static_cast<size_t>(pos) * kPredFeatFloats,
                        ref_feats_.data()  + static_cast<size_t>(f)   * kPredFeatFloats,
                        kPredFeatFloats * sizeof(float));
            ++pos;
        }
        text_tokens[pos] = ref_audio_end_token_;
        text_mask[pos]   = 1.0f;
        ++pos;
    }

    const int avail = max_text_ - pos;
    if (static_cast<int>(target_ids.size()) > avail) {
        std::vector<int> trimmed;
        trimmed.reserve(static_cast<size_t>(avail));
        for (int i = 0; i < avail - 1; ++i) trimmed.push_back(target_ids[i]);
        trimmed.push_back(audio_start_token_);
        target_ids.swap(trimmed);
    }
    for (int id : target_ids) {
        text_tokens[pos] = id;
        text_mask[pos]   = 1.0f;
        ++pos;
    }
    const int     context_length        = pos;
    const int64_t context_length_scalar = context_length;

    auto* mem = OnnxEngine::get().cpu_memory();

    // The unified graph exposes both modes' inputs; ORT requires every graph
    // input in each Run even though we request only one mode's outputs (it
    // executes just that subgraph). make_dummy mints a zero tensor for an idle
    // input — minimal shape, since the nodes that read it are never run.
    auto make_dummy = [&](const int64_t* shape, int rank,
                          ONNXTensorElementDataType dt,
                          std::vector<std::vector<uint8_t>>& bufs,
                          std::vector<OrtValue*>& vals) {
        size_t elems = 1;
        for (int i = 0; i < rank; ++i) elems *= static_cast<size_t>(shape[i] > 0 ? shape[i] : 1);
        const size_t esz = (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ? 8u : 4u;
        bufs.emplace_back(elems * esz, 0);
        OrtValue* v = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, bufs.back().data(), bufs.back().size(), shape, rank, dt, &v));
        vals.push_back(v);
    };

    // --- 2. prefill mode → initial hiddens + caches.
    //
    // TODO: text_prefill ONNX graph does not yet exist with real weights.
    // The speech-models export at facca69 only emits a tiny-random-init
    // version (a parallel sub-agent is implementing the full export). This
    // wrapper will still drive it correctly — the AR loop shape matches the
    // LiteRT contract — but until the real-weights export lands, the audio
    // produced from random-init prefill is meaningless and serves only as a
    // link/load smoke test.
    std::vector<float> lm_hidden       (kHidden);
    std::vector<float> residual_hidden (kHidden);
    std::vector<float> prefix_feat_cond(kPredFeatFloats);
    std::vector<float> base_prefill    (static_cast<size_t>(base_prefill_floats_));
    std::vector<float> residual_prefill(static_cast<size_t>(residual_prefill_floats_));
    {
        const int64_t s_text_tokens[2] = {1, max_text_};
        const int64_t s_text_mask  [2] = {1, max_text_};
        const int64_t s_audio_feats[4] = {1, max_text_, kPatchSize, kFeatDim};
        const int64_t s_audio_mask [2] = {1, max_text_};
        const int64_t* s_ctxlen        = nullptr;  // scalar (rank 0)

        OrtValue* t_tokens = nullptr;
        OrtValue* t_tmask  = nullptr;
        OrtValue* t_afeats = nullptr;
        OrtValue* t_amask  = nullptr;
        OrtValue* t_ctxlen = nullptr;

        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, text_tokens.data(), text_tokens.size() * sizeof(int64_t),
            s_text_tokens, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_tokens));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, text_mask.data(), text_mask.size() * sizeof(float),
            s_text_mask, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_tmask));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, audio_feats.data(), audio_feats.size() * sizeof(float),
            s_audio_feats, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_afeats));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, audio_mask.data(), audio_mask.size() * sizeof(float),
            s_audio_mask, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_amask));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(&context_length_scalar), sizeof(int64_t),
            s_ctxlen, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_ctxlen));

        // Idle (token-step) inputs: minimal-shape zero dummies, in step_io_
        // input order (ts_lm, ts_resid, ts_pfc, ts_base, ts_rcache, ts_pos, ts_noise).
        const int64_t d_lm[2]     = {1, kHidden};
        const int64_t d_pfc[3]    = {1, kPatchSize, kFeatDim};
        const int64_t d_bcache[6] = {2, kBaseLayers,     1, kKvHeads, 1, kHeadDim};
        const int64_t d_rcache[6] = {2, kResidualLayers, 1, kKvHeads, 1, kHeadDim};
        const int64_t* d_scalar   = nullptr;
        const int64_t d_noise[3]  = {1, kFeatDim, kPatchSize};
        std::vector<std::vector<uint8_t>> ts_dummy_bufs;
        std::vector<OrtValue*>            ts_dummy_vals;
        make_dummy(d_lm,     2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_lm,     2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_pfc,    3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_bcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_rcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_scalar, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, ts_dummy_bufs, ts_dummy_vals);
        make_dummy(d_noise,  3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ts_dummy_bufs, ts_dummy_vals);

        // Full input set = real prefill (prefill_io_ order) + ts_ dummies.
        std::vector<const char*> in_names = prefill_io_.in_names;
        in_names.insert(in_names.end(), step_io_.in_names.begin(), step_io_.in_names.end());
        std::vector<OrtValue*> in_vals = {t_tokens, t_tmask, t_afeats, t_amask, t_ctxlen};
        in_vals.insert(in_vals.end(), ts_dummy_vals.begin(), ts_dummy_vals.end());

        OrtValue* prefill_out[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        ort_check(api_, api_->Run(
            decoder_session_, nullptr,
            in_names.data(),  in_vals.data(), in_names.size(),
            prefill_io_.out_names.data(), prefill_io_.out_names.size(), prefill_out));

        for (OrtValue* v : ts_dummy_vals) api_->ReleaseValue(v);

        auto copy_out = [&](OrtValue* v, void* dst, size_t bytes) {
            void* p = nullptr;
            ort_check(api_, api_->GetTensorMutableData(v, &p));
            std::memcpy(dst, p, bytes);
        };
        // Output order mirrors the LiteRT graph: lm_hidden, residual_hidden,
        // prefix_feat_cond, base_cache, residual_cache.
        copy_out(prefill_out[0], lm_hidden.data(),        lm_hidden.size()        * sizeof(float));
        copy_out(prefill_out[1], residual_hidden.data(),  residual_hidden.size()  * sizeof(float));
        copy_out(prefill_out[2], prefix_feat_cond.data(), prefix_feat_cond.size() * sizeof(float));
        copy_out(prefill_out[3], base_prefill.data(),     base_prefill.size()     * sizeof(float));
        copy_out(prefill_out[4], residual_prefill.data(), residual_prefill.size() * sizeof(float));

        for (int i = 4; i >= 0; --i) api_->ReleaseValue(prefill_out[i]);
        api_->ReleaseValue(t_ctxlen);
        api_->ReleaseValue(t_amask);
        api_->ReleaseValue(t_afeats);
        api_->ReleaseValue(t_tmask);
        api_->ReleaseValue(t_tokens);
    }

    if (std::getenv("VOXCPM2_DEBUG")) {
        auto rms = [](const std::vector<float>& v) {
            double s = 0.0; for (float x : v) s += double(x) * x;
            return v.empty() ? 0.0 : std::sqrt(s / v.size());
        };
        LOGI("VoxCPM2 (ORT) prefill: ctx_len=%d ref_frames=%d lm_rms=%.4f resid_rms=%.4f pfc_rms=%.4f",
             context_length, ref_frames_, rms(lm_hidden), rms(residual_hidden),
             rms(prefix_feat_cond));
    }

    // --- 3. Grow caches from prefill window to full_seq_ by zero-padding axis 4.
    auto pad_cache = [pref_seq = max_text_, full_seq = full_seq_](
                         const std::vector<float>& src, std::vector<float>& dst,
                         int layers, int kv_heads, int head_dim) {
        std::fill(dst.begin(), dst.end(), 0.0f);
        const size_t per_kv  = static_cast<size_t>(layers) * kv_heads;
        const size_t src_row = static_cast<size_t>(pref_seq) * head_dim;
        const size_t dst_row = static_cast<size_t>(full_seq) * head_dim;
        for (int kv = 0; kv < 2; ++kv) {
            for (size_t i = 0; i < per_kv; ++i) {
                const float* src_ptr = src.data() + (kv * per_kv + i) * src_row;
                float*       dst_ptr = dst.data() + (kv * per_kv + i) * dst_row;
                std::memcpy(dst_ptr, src_ptr, src_row * sizeof(float));
            }
        }
    };
    std::vector<float> base_cache    (static_cast<size_t>(base_cache_floats_));
    std::vector<float> residual_cache(static_cast<size_t>(residual_cache_floats_));
    pad_cache(base_prefill,     base_cache,     kBaseLayers,     kKvHeads, kHeadDim);
    pad_cache(residual_prefill, residual_cache, kResidualLayers, kKvHeads, kHeadDim);
    base_prefill    .clear(); base_prefill    .shrink_to_fit();
    residual_prefill.clear(); residual_prefill.shrink_to_fit();

    // --- 4. AR loop.
    std::vector<float> pred_feat   (kPredFeatFloats);
    std::vector<float> stop_logits (2);
    std::vector<float> next_lm     (kHidden);
    std::vector<float> next_resid  (kHidden);
    std::vector<float> noise       (kPredFeatFloats);
    std::vector<float> feature_buffer; feature_buffer.reserve(kDecoderInputFloats);
    std::vector<float> decoder_input(kDecoderInputFloats, 0.0f);
    std::vector<float> pcm         (kDecoderOutputFloats);

    uint32_t rng_seed = seed_;
    if (rng_seed == 0) {
        std::random_device rd;
        rng_seed = rd();
    }
    seed_used_ = rng_seed;
    std::mt19937 rng(rng_seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Static tensor shapes used inside the step loop.
    const int64_t s_lm     [2] = {1, kHidden};
    const int64_t s_resid  [2] = {1, kHidden};
    const int64_t s_pfc    [3] = {1, kPatchSize, kFeatDim};
    const int64_t s_bcache [6] = {2, kBaseLayers,     1, kKvHeads, full_seq_, kHeadDim};
    const int64_t s_rcache [6] = {2, kResidualLayers, 1, kKvHeads, full_seq_, kHeadDim};
    const int64_t* s_pos       = nullptr;  // scalar
    const int64_t s_noise  [3] = {1, kFeatDim, kPatchSize};
    const int64_t s_dec_in [3] = {1, kFramesPerChunk, kPredFeatFloats};
    (void)s_dec_in;  // referenced by the decoder Run below

    // Idle (prefill) inputs for every token-step Run: zero dummies in
    // prefill_io_ input order, created once and reused each step. Prefill input
    // shapes are fixed (not dynamic), so the dummies use the real dimensions.
    const int64_t pd_tokens[2] = {1, max_text_};
    const int64_t pd_mask[2]   = {1, max_text_};
    const int64_t pd_afeats[4] = {1, max_text_, kPatchSize, kFeatDim};
    const int64_t pd_amask[2]  = {1, max_text_};
    const int64_t* pd_ctx      = nullptr;
    std::vector<std::vector<uint8_t>> prefill_dummy_bufs;
    std::vector<OrtValue*>            prefill_dummy_vals;
    make_dummy(pd_tokens, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, prefill_dummy_bufs, prefill_dummy_vals);
    make_dummy(pd_mask,   2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, prefill_dummy_bufs, prefill_dummy_vals);
    make_dummy(pd_afeats, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, prefill_dummy_bufs, prefill_dummy_vals);
    make_dummy(pd_amask,  2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, prefill_dummy_bufs, prefill_dummy_vals);
    make_dummy(pd_ctx,    0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, prefill_dummy_bufs, prefill_dummy_vals);

    auto flush_decoder = [&](size_t valid_steps, bool is_final) {
        std::fill(decoder_input.begin(), decoder_input.end(), 0.0f);
        const size_t cap_steps = std::min<size_t>(
            valid_steps, static_cast<size_t>(kFramesPerChunk));
        for (size_t step_idx = 0; step_idx < cap_steps; ++step_idx) {
            const float* pred = feature_buffer.data() + step_idx * kPredFeatFloats;
            for (int p = 0; p < kPatchSize; ++p) {
                const int t = static_cast<int>(step_idx) * kPatchSize + p;
                if (t >= kDecoderInputFloats / kFeatDim) break;
                for (int c = 0; c < kFeatDim; ++c) {
                    decoder_input[c * (kDecoderInputFloats / kFeatDim) + t]
                        = pred[p * kFeatDim + c];
                }
            }
        }

        const int64_t s_in[3] = {1, kFramesPerChunk, kPredFeatFloats};
        OrtValue* t_in = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, decoder_input.data(), decoder_input.size() * sizeof(float),
            s_in, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_in));

        OrtValue* dec_in[1]  = {t_in};
        OrtValue* dec_out[1] = {nullptr};
        ort_check(api_, api_->Run(
            audio_decoder_session_, nullptr,
            decoder_io_.in_names.data(),  dec_in,  decoder_io_.in_names.size(),
            decoder_io_.out_names.data(), decoder_io_.out_names.size(), dec_out));

        float* out_pcm_ptr = nullptr;
        ort_check(api_, api_->GetTensorMutableData(dec_out[0], (void**)&out_pcm_ptr));
        std::memcpy(pcm.data(), out_pcm_ptr, pcm.size() * sizeof(float));

        api_->ReleaseValue(dec_out[0]);
        api_->ReleaseValue(t_in);

        const size_t valid_samples = std::min<size_t>(
            static_cast<size_t>(valid_steps) * kSamplesPerStep, pcm.size());
        on_chunk(pcm.data(), valid_samples, is_final);

        feature_buffer.clear();
    };

    int  steps_done       = 0;
    int  steps_in_chunk   = 0;
    bool stopped_by_model = false;
    const bool debug_steps = std::getenv("VOXCPM2_DEBUG") != nullptr;

    // CUDA IoBinding path — keep heavy state (lm_hidden, residual_hidden,
    // base_cache, residual_cache) GPU-resident between AR steps. Each step
    // would otherwise re-upload ~200 MB of state to the device, dominating
    // wall time at this scale. Off-by-default for CPU sessions so the
    // existing well-tested codepath keeps running unchanged.
    //
    // Names use the graph's actual input/output names (introspected at load
    // via query_io_names) so we don't hard-code "lm_hidden" / "next_lm_hidden"
    // here — index into step_io_ to stay model-agnostic.
    // Try to set up GPU IoBinding when the engine resolved a CUDA provider.
    // Per-session might still be on CPU (when hw_accel=false at construction,
    // e.g. the load smoke test) — in that case CreateAllocator returns a
    // status with "No requested allocator available" and we fall through to
    // the host path.
    bool use_gpu_bind =
        OnnxEngine::get().gpu_provider() != OrtGpuProvider::None
        && std::getenv("SPEECH_CORE_VOXCPM2_NO_IOBINDING") == nullptr;

    OrtMemoryInfo* cuda_mem = nullptr;
    OrtAllocator*  cuda_alloc = nullptr;
    OrtIoBinding*  binding = nullptr;
    // Ping-pong sets of GPU state tensors (lm, resid, base, rcache).
    OrtValue* gpu_lm[2]      = {nullptr, nullptr};
    OrtValue* gpu_resid[2]   = {nullptr, nullptr};
    OrtValue* gpu_base[2]    = {nullptr, nullptr};
    OrtValue* gpu_rcache[2]  = {nullptr, nullptr};
    int       slot_in   = 0;  // index of the "current" (input) slot
    int       slot_out  = 1;  // index of the "next" (output) slot

    if (use_gpu_bind) {
        // CUDA EP registers its device allocator under name "Cuda" with type
        // OrtDeviceAllocator (NOT OrtArenaAllocator).
        OrtStatus* s = api_->CreateMemoryInfo(
            "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cuda_mem);
        if (s == nullptr) {
            s = api_->CreateAllocator(decoder_session_, cuda_mem, &cuda_alloc);
        }
        if (s != nullptr) {
            // Session was created on CPU (e.g. hw_accel=false) — drop back
            // to the host path silently. Free the partial state.
            api_->ReleaseStatus(s);
            if (cuda_mem) { api_->ReleaseMemoryInfo(cuda_mem); cuda_mem = nullptr; }
            use_gpu_bind = false;
        }
    }

    if (use_gpu_bind) {
        ort_check(api_, api_->CreateIoBinding(decoder_session_, &binding));

        auto alloc_gpu = [&](const int64_t* shape, int rank, OrtValue** out) {
            ort_check(api_, api_->CreateTensorAsOrtValue(
                cuda_alloc, shape, rank,
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, out));
        };
        for (int i = 0; i < 2; ++i) {
            alloc_gpu(s_lm,     2, &gpu_lm[i]);
            alloc_gpu(s_resid,  2, &gpu_resid[i]);
            alloc_gpu(s_bcache, 6, &gpu_base[i]);
            alloc_gpu(s_rcache, 6, &gpu_rcache[i]);
        }
    }

    for (int step = 0; step < max_steps_; ++step) {
        if (cancelled_.load(std::memory_order_relaxed)) break;

        for (float& v : noise) v = normal(rng);
        const int64_t position_id = static_cast<int64_t>(context_length + step);

      if (!use_gpu_bind) {
        OrtValue* t_lm    = nullptr;
        OrtValue* t_resid = nullptr;
        OrtValue* t_pfc   = nullptr;
        OrtValue* t_bcache= nullptr;
        OrtValue* t_rcache= nullptr;
        OrtValue* t_pos   = nullptr;
        OrtValue* t_noise = nullptr;

        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, lm_hidden.data(), lm_hidden.size() * sizeof(float),
            s_lm, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_lm));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, residual_hidden.data(), residual_hidden.size() * sizeof(float),
            s_resid, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_resid));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, prefix_feat_cond.data(), prefix_feat_cond.size() * sizeof(float),
            s_pfc, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_pfc));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, base_cache.data(), base_cache.size() * sizeof(float),
            s_bcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_bcache));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, residual_cache.data(), residual_cache.size() * sizeof(float),
            s_rcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_rcache));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(&position_id), sizeof(int64_t),
            s_pos, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_pos));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, noise.data(), noise.size() * sizeof(float),
            s_noise, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_noise));

        // Input order mirrors the graph: ts_lm_hidden, ts_residual_hidden,
        // ts_prefix_feat_cond, ts_base_cache, ts_residual_cache, ts_position_id,
        // ts_noise — plus the idle prefill dummies (ORT needs every input).
        std::vector<const char*> step_in_names = step_io_.in_names;
        step_in_names.insert(step_in_names.end(),
                             prefill_io_.in_names.begin(), prefill_io_.in_names.end());
        std::vector<OrtValue*> step_in_vals = {t_lm, t_resid, t_pfc, t_bcache, t_rcache, t_pos, t_noise};
        step_in_vals.insert(step_in_vals.end(),
                            prefill_dummy_vals.begin(), prefill_dummy_vals.end());

        OrtValue* step_out[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        ort_check(api_, api_->Run(
            decoder_session_, nullptr,
            step_in_names.data(),  step_in_vals.data(), step_in_names.size(),
            step_io_.out_names.data(), step_io_.out_names.size(), step_out));

        // Output order mirrors LiteRT: pred_feat, stop_logits, next_lm,
        // next_residual, base_cache, residual_cache.
        auto copy_out = [&](OrtValue* v, void* dst, size_t bytes) {
            void* p = nullptr;
            ort_check(api_, api_->GetTensorMutableData(v, &p));
            std::memcpy(dst, p, bytes);
        };
        copy_out(step_out[0], pred_feat.data(),      pred_feat.size()      * sizeof(float));
        copy_out(step_out[1], stop_logits.data(),    stop_logits.size()    * sizeof(float));
        copy_out(step_out[2], next_lm.data(),        next_lm.size()        * sizeof(float));
        copy_out(step_out[3], next_resid.data(),     next_resid.size()     * sizeof(float));
        copy_out(step_out[4], base_cache.data(),     base_cache.size()     * sizeof(float));
        copy_out(step_out[5], residual_cache.data(), residual_cache.size() * sizeof(float));

        for (int i = 5; i >= 0; --i) api_->ReleaseValue(step_out[i]);
        api_->ReleaseValue(t_noise);
        api_->ReleaseValue(t_pos);
        api_->ReleaseValue(t_rcache);
        api_->ReleaseValue(t_bcache);
        api_->ReleaseValue(t_pfc);
        api_->ReleaseValue(t_resid);
        api_->ReleaseValue(t_lm);

        lm_hidden        = next_lm;
        residual_hidden  = next_resid;
        prefix_feat_cond = pred_feat;
      } else {
        // ------ GPU IoBinding path ------
        // pred_feat (256 floats), stop_logits (2 floats), pfc (256 floats),
        // position_id (1 i64), noise (256 floats): all small enough to keep
        // bound to host memory; ORT handles GPU<->CPU sync via
        // SynchronizeBoundOutputs / SynchronizeBoundInputs.
        OrtValue* t_pos = nullptr;
        OrtValue* t_noise = nullptr;
        OrtValue* t_pfc_in = nullptr;
        OrtValue* t_pred_feat_out = nullptr;
        OrtValue* t_stop_logits_out = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, const_cast<int64_t*>(&position_id), sizeof(int64_t),
            s_pos, 0, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &t_pos));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, noise.data(), noise.size() * sizeof(float),
            s_noise, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_noise));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, prefix_feat_cond.data(), prefix_feat_cond.size() * sizeof(float),
            s_pfc, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_pfc_in));
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, pred_feat.data(), pred_feat.size() * sizeof(float),
            s_pfc, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_pred_feat_out));
        // Graph outputs stop_logits as rank-2 [1, 2], not rank-1.
        const int64_t s_stop[2] = {1, 2};
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, stop_logits.data(), stop_logits.size() * sizeof(float),
            s_stop, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_stop_logits_out));

        // On step 0, the state inputs (lm/resid/base/rcache) come from the
        // host (post-prefill); ORT will upload them to the device implicitly
        // when we BindInput a CPU OrtValue. On step 1+, the previous step's
        // GPU output slot is the current input.
        OrtValue* in_lm     = nullptr;
        OrtValue* in_resid  = nullptr;
        OrtValue* in_base   = nullptr;
        OrtValue* in_rcache = nullptr;
        if (step == 0) {
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, lm_hidden.data(), lm_hidden.size() * sizeof(float),
                s_lm, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_lm));
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, residual_hidden.data(), residual_hidden.size() * sizeof(float),
                s_resid, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_resid));
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, base_cache.data(), base_cache.size() * sizeof(float),
                s_bcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_base));
            ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
                mem, residual_cache.data(), residual_cache.size() * sizeof(float),
                s_rcache, 6, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_rcache));
        }

        api_->ClearBoundInputs(binding);
        api_->ClearBoundOutputs(binding);

        // Bind inputs — names come from the loaded graph's introspected list.
        // Index layout from query_io_names: 0=lm_hidden, 1=residual_hidden,
        // 2=prefix_feat_cond, 3=base_cache, 4=residual_cache, 5=position_id,
        // 6=noise. (Same ordering as the existing host path's step_in array.)
        const char* const* in_n = step_io_.in_names.data();
        ort_check(api_, api_->BindInput(binding, in_n[0],
            step == 0 ? in_lm     : gpu_lm[slot_in]));
        ort_check(api_, api_->BindInput(binding, in_n[1],
            step == 0 ? in_resid  : gpu_resid[slot_in]));
        ort_check(api_, api_->BindInput(binding, in_n[2], t_pfc_in));
        ort_check(api_, api_->BindInput(binding, in_n[3],
            step == 0 ? in_base   : gpu_base[slot_in]));
        ort_check(api_, api_->BindInput(binding, in_n[4],
            step == 0 ? in_rcache : gpu_rcache[slot_in]));
        ort_check(api_, api_->BindInput(binding, in_n[5], t_pos));
        ort_check(api_, api_->BindInput(binding, in_n[6], t_noise));
        // Idle prefill inputs — bound to the persistent zero dummies so the
        // unified graph has every input present (the prefill subgraph isn't run).
        for (size_t i = 0; i < prefill_io_.in_names.size(); ++i)
            ort_check(api_, api_->BindInput(
                binding, prefill_io_.in_names[i], prefill_dummy_vals[i]));

        // Output layout: 0=pred_feat, 1=stop_logits, 2=next_lm_hidden,
        // 3=next_residual_hidden, 4=base_cache, 5=residual_cache.
        const char* const* out_n = step_io_.out_names.data();
        ort_check(api_, api_->BindOutput(binding, out_n[0], t_pred_feat_out));
        ort_check(api_, api_->BindOutput(binding, out_n[1], t_stop_logits_out));
        ort_check(api_, api_->BindOutput(binding, out_n[2], gpu_lm[slot_out]));
        ort_check(api_, api_->BindOutput(binding, out_n[3], gpu_resid[slot_out]));
        ort_check(api_, api_->BindOutput(binding, out_n[4], gpu_base[slot_out]));
        ort_check(api_, api_->BindOutput(binding, out_n[5], gpu_rcache[slot_out]));

        ort_check(api_, api_->RunWithBinding(decoder_session_, nullptr, binding));
        // CPU-bound outputs (pred_feat, stop_logits) need an explicit
        // device->host sync; GPU-bound outputs stay on device.
        ort_check(api_, api_->SynchronizeBoundOutputs(binding));

        if (step == 0) {
            api_->ReleaseValue(in_rcache);
            api_->ReleaseValue(in_base);
            api_->ReleaseValue(in_resid);
            api_->ReleaseValue(in_lm);
        }
        api_->ReleaseValue(t_stop_logits_out);
        api_->ReleaseValue(t_pred_feat_out);
        api_->ReleaseValue(t_pfc_in);
        api_->ReleaseValue(t_noise);
        api_->ReleaseValue(t_pos);

        // pred_feat host buffer now holds the new value (ORT synced it).
        // Feed it back as next iter's prefix_feat_cond input.
        prefix_feat_cond = pred_feat;

        // Swap which GPU slot is "current" for next iter's state inputs.
        std::swap(slot_in, slot_out);
      }

        feature_buffer.insert(feature_buffer.end(), pred_feat.begin(), pred_feat.end());
        ++steps_done;
        ++steps_in_chunk;

        const bool stop_signal = stop_on_stop_token_
                              && stop_logits[1] > stop_logits[0]
                              && steps_done > min_stop_steps_;
        if (stop_signal) stopped_by_model = true;

        if (debug_steps && (step < 6 || stop_signal)) {
            float pmax = 0.0f; double sum = 0.0, sq = 0.0;
            for (float v : pred_feat) { pmax = std::max(pmax, std::abs(v)); sum += v; sq += double(v) * v; }
            const double mean = sum / pred_feat.size();
            const double sd = std::sqrt(std::max(0.0, sq / pred_feat.size() - mean * mean));
            LOGI("VoxCPM2 (ORT) step %d: pred|max|=%.4f mean=%.4f std=%.4f stop=[%.2f,%.2f]%s",
                 step, pmax, mean, sd, stop_logits[0], stop_logits[1],
                 stop_signal ? " STOP" : "");
        }

        if (steps_in_chunk == kFramesPerChunk) {
            flush_decoder(kFramesPerChunk, /*is_final=*/false);
            steps_in_chunk = 0;
        }
        if (stop_signal) break;
    }

    if (use_gpu_bind) {
        for (int i = 0; i < 2; ++i) {
            if (gpu_rcache[i])  api_->ReleaseValue(gpu_rcache[i]);
            if (gpu_base[i])    api_->ReleaseValue(gpu_base[i]);
            if (gpu_resid[i])   api_->ReleaseValue(gpu_resid[i]);
            if (gpu_lm[i])      api_->ReleaseValue(gpu_lm[i]);
        }
        if (binding)    api_->ReleaseIoBinding(binding);
        if (cuda_alloc) api_->ReleaseAllocator(cuda_alloc);
        if (cuda_mem)   api_->ReleaseMemoryInfo(cuda_mem);
    }

    for (OrtValue* v : prefill_dummy_vals) api_->ReleaseValue(v);

    if (steps_in_chunk > 0) {
        flush_decoder(static_cast<size_t>(steps_in_chunk), /*is_final=*/true);
    } else {
        on_chunk(nullptr, 0, /*is_final=*/true);
    }

    tokens_generated_      = steps_done;
    stopped_on_stop_token_ = stopped_by_model;
}

}  // namespace speech_core
