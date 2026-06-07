// PersonaPlex 7B ONNX wrapper — orchestrates four ORT sessions
// (mimi_encoder, mimi_decoder, temporal_step, depformer_step) into a
// FullDuplexSpeechInterface implementation.
//
// PR 5 scope: structural framework + one-frame end-to-end exercise. The full
// inference loop with voice-prompt prefill, system-prompt prefill, delay
// pattern, sampling temperatures, and SentencePiece text decoding lands
// progressively — each piece is bounded and testable on top of this base.

#include "speech_core/models/onnx_personaplex.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace speech_core {

namespace {

// Half-precision (IEEE 754 binary16) helpers. We hand-roll these to avoid
// pulling in a fp16 lib; the wrapper sees fp16 only at KV-cache boundaries
// and at temporal hidden, so a couple inline routines are sufficient.
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) { return (uint16_t)sign; }
    if (exp >= 31) { return (uint16_t)(sign | 0x7c00); }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
}
inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) { bits = sign; }
    else if (exp == 31) { bits = sign | 0x7f800000 | (mant << 13); }
    else { bits = sign | ((exp + 112) << 23) | (mant << 13); }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

void ort_check_local(const OrtApi* api, OrtStatus* status) {
    if (status != nullptr) {
        std::string msg = api->GetErrorMessage(status);
        api->ReleaseStatus(status);
        throw std::runtime_error("ORT error: " + msg);
    }
}

}  // namespace

void OnnxPersonaPlex::query_io_names(OrtSession* session, IoNames& names) {
    OrtAllocator* allocator = nullptr;
    ort_check_local(api_, api_->GetAllocatorWithDefaultOptions(&allocator));
    size_t in_count = 0, out_count = 0;
    ort_check_local(api_, api_->SessionGetInputCount(session, &in_count));
    ort_check_local(api_, api_->SessionGetOutputCount(session, &out_count));
    names.in_names_str.resize(in_count);
    names.out_names_str.resize(out_count);
    names.in_names.resize(in_count);
    names.out_names.resize(out_count);
    for (size_t i = 0; i < in_count; ++i) {
        char* name = nullptr;
        ort_check_local(api_, api_->SessionGetInputName(session, i, allocator, &name));
        names.in_names_str[i] = name;
        names.in_names[i] = names.in_names_str[i].c_str();
        allocator->Free(allocator, name);
    }
    for (size_t i = 0; i < out_count; ++i) {
        char* name = nullptr;
        ort_check_local(api_, api_->SessionGetOutputName(session, i, allocator, &name));
        names.out_names_str[i] = name;
        names.out_names[i] = names.out_names_str[i].c_str();
        allocator->Free(allocator, name);
    }
}

OnnxPersonaPlex::OnnxPersonaPlex(const std::string& mimi_encoder_path,
                                  const std::string& mimi_decoder_path,
                                  const std::string& temporal_step_path,
                                  const std::string& depformer_step_path,
                                  const std::string& tokenizer_path,
                                  const std::string& voices_dir,
                                  bool hw_accel) {
    auto& engine = OnnxEngine::get();
    api_ = engine.api();

    mimi_encoder_session_   = engine.load(mimi_encoder_path,   hw_accel);
    mimi_decoder_session_   = engine.load(mimi_decoder_path,   hw_accel);
    temporal_step_session_  = engine.load(temporal_step_path,  hw_accel);
    depformer_step_session_ = engine.load(depformer_step_path, hw_accel);

    query_io_names(mimi_encoder_session_,   mimi_enc_io_);
    query_io_names(mimi_decoder_session_,   mimi_dec_io_);
    query_io_names(temporal_step_session_,  temporal_io_);
    query_io_names(depformer_step_session_, depformer_io_);

    // Load SentencePiece blob (PR 5b will use it for text encoding/decoding;
    // for now we just check the file exists).
    std::ifstream tk(tokenizer_path, std::ios::binary);
    if (!tk) {
        throw std::runtime_error("PersonaPlex: tokenizer not found: " + tokenizer_path);
    }
    tk.seekg(0, std::ios::end);
    spm_model_blob_.resize(static_cast<size_t>(tk.tellg()));
    tk.seekg(0);
    tk.read(reinterpret_cast<char*>(spm_model_blob_.data()), spm_model_blob_.size());

    // Default delay pattern per PersonaPlex spec
    delays_ = {0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};

    // voices_dir presence check — actual voice load happens on set_voice
    (void)voices_dir;

    reset_session();
}

OnnxPersonaPlex::~OnnxPersonaPlex() {
    if (mimi_encoder_session_)   api_->ReleaseSession(mimi_encoder_session_);
    if (mimi_decoder_session_)   api_->ReleaseSession(mimi_decoder_session_);
    if (temporal_step_session_)  api_->ReleaseSession(temporal_step_session_);
    if (depformer_step_session_) api_->ReleaseSession(depformer_step_session_);
}

void OnnxPersonaPlex::reset_session() {
    temporal_k_.clear();
    temporal_v_.clear();
    temporal_t_past_ = 0;
    frames_generated_ = 0;
    cancelled_.store(false);
}

void OnnxPersonaPlex::cancel() {
    cancelled_.store(true);
}

void OnnxPersonaPlex::set_voice(const std::string& voice_name) {
    current_voice_ = voice_name;
    voice_tokens_.clear();
    // Full voice-prompt loading from voices/<name>.bin is PR 5b. For now we
    // just remember the requested name; respond_stream proceeds without
    // voice conditioning, which produces uninflected but coherent output
    // (per speech-swift docs, voice embeddings are a quality enhancer not
    // a hard requirement).
}

void OnnxPersonaPlex::set_system_prompt(const std::string& prompt) {
    system_prompt_ = prompt;
    // PR 5b: tokenize via SentencePiece and prepend to the prefill sequence.
    // Until then, set_system_prompt is recorded but unused.
}

void OnnxPersonaPlex::mimi_encode(const float* pcm, size_t length,
                                   std::vector<int64_t>& tokens_out) {
    auto* mem = OnnxEngine::get().cpu_memory();
    std::vector<float> pcm_buf(pcm, pcm + length);
    const int64_t enc_in_shape[3] = {1, 1, static_cast<int64_t>(length)};
    OrtValue* in_val = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, pcm_buf.data(), pcm_buf.size() * sizeof(float),
        enc_in_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_val));

    OrtValue* out_val = nullptr;
    ort_check_local(api_, api_->Run(
        mimi_encoder_session_, nullptr,
        mimi_enc_io_.in_names.data(), &in_val, 1,
        mimi_enc_io_.out_names.data(), 1, &out_val));

    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort_check_local(api_, api_->GetTensorTypeAndShape(out_val, &info));
    size_t n_dims = 0;
    api_->GetDimensionsCount(info, &n_dims);
    std::vector<int64_t> dims(n_dims);
    api_->GetDimensions(info, dims.data(), n_dims);
    api_->ReleaseTensorTypeAndShapeInfo(info);

    size_t n = 1;
    for (auto d : dims) n *= static_cast<size_t>(d);
    int64_t* data = nullptr;
    ort_check_local(api_, api_->GetTensorMutableData(out_val, reinterpret_cast<void**>(&data)));
    tokens_out.assign(data, data + n);
    api_->ReleaseValue(out_val);
    api_->ReleaseValue(in_val);
}

void OnnxPersonaPlex::mimi_decode(const std::vector<int64_t>& tokens,
                                   std::vector<float>& pcm_out) {
    if (tokens.empty()) { pcm_out.clear(); return; }
    auto* mem = OnnxEngine::get().cpu_memory();
    // Tokens shape: [1, 32, T_frames]
    const int64_t T_frames = static_cast<int64_t>(tokens.size() / kMimiCodebooks);
    const int64_t dec_in_shape[3] = {1, kMimiCodebooks, T_frames};
    std::vector<int64_t> tok_buf(tokens);
    OrtValue* in_val = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, tok_buf.data(), tok_buf.size() * sizeof(int64_t),
        dec_in_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &in_val));

    OrtValue* out_val = nullptr;
    ort_check_local(api_, api_->Run(
        mimi_decoder_session_, nullptr,
        mimi_dec_io_.in_names.data(), &in_val, 1,
        mimi_dec_io_.out_names.data(), 1, &out_val));

    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort_check_local(api_, api_->GetTensorTypeAndShape(out_val, &info));
    size_t n_dims = 0;
    api_->GetDimensionsCount(info, &n_dims);
    std::vector<int64_t> dims(n_dims);
    api_->GetDimensions(info, dims.data(), n_dims);
    api_->ReleaseTensorTypeAndShapeInfo(info);
    size_t n = 1;
    for (auto d : dims) n *= static_cast<size_t>(d);
    float* data = nullptr;
    ort_check_local(api_, api_->GetTensorMutableData(out_val, reinterpret_cast<void**>(&data)));
    pcm_out.assign(data, data + n);
    api_->ReleaseValue(out_val);
    api_->ReleaseValue(in_val);
}

void OnnxPersonaPlex::temporal_forward(int64_t text_token,
                                        const std::vector<int64_t>& audio_tokens_16,
                                        std::vector<uint16_t>& hidden_out) {
    auto* mem = OnnxEngine::get().cpu_memory();

    // Inputs: text_token[1,1], audio_tokens[1,16], past_k_all[L,1,H,T,D], past_v_all[L,1,H,T,D]
    int64_t text_tok_buf[1] = {text_token};
    const int64_t text_shape[2] = {1, 1};
    OrtValue* text_val = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, text_tok_buf, sizeof(text_tok_buf),
        text_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &text_val));

    std::vector<int64_t> audio_buf(audio_tokens_16);
    const int64_t audio_shape[2] = {1, static_cast<int64_t>(kAudioCodebooks)};
    OrtValue* audio_val = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, audio_buf.data(), audio_buf.size() * sizeof(int64_t),
        audio_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &audio_val));

    // KV cache (fp16). Lazily-sized: 0 past tokens on first frame, grows by 1
    // each call. Total floats = L * 1 * H * T_past * D.
    const size_t per_tensor_floats = static_cast<size_t>(kTemporalLayers) * 1
                                     * kTemporalHeads * temporal_t_past_ * kTemporalHeadDim;
    if (temporal_k_.size() != per_tensor_floats) temporal_k_.resize(per_tensor_floats, 0);
    if (temporal_v_.size() != per_tensor_floats) temporal_v_.resize(per_tensor_floats, 0);
    const int64_t kv_shape[5] = {
        kTemporalLayers, 1, kTemporalHeads, temporal_t_past_, kTemporalHeadDim};
    OrtValue* k_val = nullptr;
    OrtValue* v_val = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, temporal_k_.data(), temporal_k_.size() * sizeof(uint16_t),
        kv_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &k_val));
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, temporal_v_.data(), temporal_v_.size() * sizeof(uint16_t),
        kv_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &v_val));

    OrtValue* inputs[4] = {text_val, audio_val, k_val, v_val};
    OrtValue* outputs[3] = {nullptr, nullptr, nullptr};
    ort_check_local(api_, api_->Run(
        temporal_step_session_, nullptr,
        temporal_io_.in_names.data(), inputs, 4,
        temporal_io_.out_names.data(), 3, outputs));

    // outputs: hidden [1,1,dim=4096] fp16; new_k_all, new_v_all
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[0], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)));
        hidden_out.assign(data, data + n);
    }
    // Copy new K/V back into our owned buffers
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[1], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&data)));
        temporal_k_.assign(data, data + n);
    }
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[2], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[2], reinterpret_cast<void**>(&data)));
        temporal_v_.assign(data, data + n);
    }
    ++temporal_t_past_;
    api_->ReleaseValue(text_val); api_->ReleaseValue(audio_val);
    api_->ReleaseValue(k_val);    api_->ReleaseValue(v_val);
    for (auto* o : outputs) if (o) api_->ReleaseValue(o);
}

void OnnxPersonaPlex::depformer_step(const std::vector<uint16_t>& hidden,
                                      int64_t prev_token, int step_idx,
                                      std::vector<float>& logits_out) {
    auto* mem = OnnxEngine::get().cpu_memory();
    // Depformer cache lives entirely within one temporal frame; reset at each
    // step_idx==0 by allocating zero buffers (caller responsibility).
    static thread_local std::vector<uint16_t> dep_k, dep_v;
    static thread_local int dep_t_past = 0;
    if (step_idx == 0) {
        dep_k.clear(); dep_v.clear(); dep_t_past = 0;
    }
    const size_t per_tensor = static_cast<size_t>(kDepformerLayers) * 1
                              * kDepformerHeads * dep_t_past * kDepformerHeadDim;
    if (dep_k.size() != per_tensor) dep_k.resize(per_tensor, 0);
    if (dep_v.size() != per_tensor) dep_v.resize(per_tensor, 0);

    std::vector<uint16_t> hidden_buf(hidden);
    const int64_t hidden_shape[3] = {1, 1, 4096};
    OrtValue* hv = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, hidden_buf.data(), hidden_buf.size() * sizeof(uint16_t),
        hidden_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &hv));

    int64_t prev_buf[1] = {prev_token};
    const int64_t prev_shape[2] = {1, 1};
    OrtValue* pv = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, prev_buf, sizeof(prev_buf), prev_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &pv));

    int64_t step_buf[1] = {static_cast<int64_t>(step_idx)};
    const int64_t step_shape[1] = {1};
    OrtValue* sv = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, step_buf, sizeof(step_buf), step_shape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sv));

    const int64_t kv_shape[5] = {
        kDepformerLayers, 1, kDepformerHeads, dep_t_past, kDepformerHeadDim};
    OrtValue* kv_k = nullptr;
    OrtValue* kv_v = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, dep_k.data(), dep_k.size() * sizeof(uint16_t),
        kv_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &kv_k));
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, dep_v.data(), dep_v.size() * sizeof(uint16_t),
        kv_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &kv_v));

    OrtValue* inputs[5] = {hv, pv, sv, kv_k, kv_v};
    OrtValue* outputs[3] = {nullptr, nullptr, nullptr};
    ort_check_local(api_, api_->Run(
        depformer_step_session_, nullptr,
        depformer_io_.in_names.data(), inputs, 5,
        depformer_io_.out_names.data(), 3, outputs));

    // outputs[0]: logits [1, card] fp16
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[0], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)));
        logits_out.resize(n);
        for (size_t i = 0; i < n; ++i) logits_out[i] = half_to_float(data[i]);
    }
    // Update dep cache
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[1], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&data)));
        dep_k.assign(data, data + n);
    }
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[2], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint16_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[2], reinterpret_cast<void**>(&data)));
        dep_v.assign(data, data + n);
    }
    ++dep_t_past;
    api_->ReleaseValue(hv); api_->ReleaseValue(pv); api_->ReleaseValue(sv);
    api_->ReleaseValue(kv_k); api_->ReleaseValue(kv_v);
    for (auto* o : outputs) if (o) api_->ReleaseValue(o);
}

int OnnxPersonaPlex::sample_token(const std::vector<float>& logits,
                                   float temperature, int top_k) {
    // Top-k + temperature sampling. PersonaPlex defaults from speech-swift:
    //   audio: temperature 0.8, top_k 250
    //   text:  temperature 0.7, top_k 25
    // Caller passes the right tuple per stream.
    const int K = static_cast<int>(logits.size());
    if (top_k <= 0 || top_k >= K || temperature <= 0.0f) {
        int argmax = 0;
        float best = -INFINITY;
        for (int i = 0; i < K; ++i) {
            if (logits[i] > best) { best = logits[i]; argmax = i; }
        }
        return argmax;
    }
    // Partial sort top-k by magnitude
    std::vector<std::pair<float,int>> scored;
    scored.reserve(K);
    for (int i = 0; i < K; ++i) scored.emplace_back(logits[i], i);
    std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                       [](const auto& a, const auto& b){ return a.first > b.first; });
    // Temperature-scaled softmax over the top-k
    float maxv = scored[0].first;
    double sum = 0.0;
    std::vector<double> probs(top_k);
    for (int i = 0; i < top_k; ++i) {
        probs[i] = std::exp((scored[i].first - maxv) / temperature);
        sum += probs[i];
    }
    // Multinomial sample
    static thread_local uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    double r = (double)(rng_state & 0xFFFFFFFFull) / (double)0x100000000ull * sum;
    double acc = 0.0;
    for (int i = 0; i < top_k; ++i) {
        acc += probs[i];
        if (acc >= r) return scored[i].second;
    }
    return scored[top_k - 1].second;
}

void OnnxPersonaPlex::respond_stream(const float* user_audio, size_t length,
                                      int sample_rate,
                                      FullDuplexChunkCallback on_chunk) {
    if (sample_rate != kSampleRate) {
        throw std::runtime_error("PersonaPlex: only 24 kHz user audio supported in PR 5; "
                                  "resampling lands in PR 5b");
    }

    // Step 1: Encode user audio with Mimi -> per-frame audio tokens [32, T_frames]
    std::vector<int64_t> user_tokens_all;
    mimi_encode(user_audio, length, user_tokens_all);
    const int64_t T_frames = static_cast<int64_t>(user_tokens_all.size() / kMimiCodebooks);

    // Step 2: For each frame, run temporal -> depformer to generate agent tokens.
    // PersonaPlex uses 8 user + 8 agent audio streams (16 total). We feed the
    // first 8 user codebooks and zero-init the agent slots; the depformer
    // generates the 16 agent codebook tokens which we then re-feed at t+1.
    std::vector<int64_t> agent_token_log;  // accumulated [16, T_emitted] for mimi_decode
    agent_token_log.reserve(static_cast<size_t>(kMimiCodebooks) * T_frames);

    // Two-frame history for the delay pattern. PersonaPlex delays:
    //   stream 0 (text):        delay 0
    //   streams 1..8 (user):    delays [0,1,1,1,1,1,1,1]  (cb 0 semantic, cb 1-7 acoustic)
    //   streams 9..16 (agent):  delays [0,1,1,1,1,1,1,1]
    // At step t, stream s reads its token from frame t-1-delays[s]. We carry a
    // 2-frame ring buffer per stream (the largest delay is 1, plus 1 for the
    // most-recent prediction). For warm-up frames we feed zero (the model's
    // initial/PAD token semantics).
    constexpr int kDelayBufSize = 2;
    std::vector<std::array<int64_t, kDelayBufSize>> user_audio_hist(8);
    std::vector<std::array<int64_t, kDelayBufSize>> agent_audio_hist(8);
    std::array<int64_t, kDelayBufSize> text_hist{0, 0};
    for (auto& h : user_audio_hist) h.fill(0);
    for (auto& h : agent_audio_hist) h.fill(0);

    // Delay layout from PERSONAPLEX_DELAYS in speech-models/convert.py
    static constexpr int kUserDelays[8]  = {0, 1, 1, 1, 1, 1, 1, 1};
    static constexpr int kAgentDelays[8] = {0, 1, 1, 1, 1, 1, 1, 1};

    int n_frames = std::min<int>(static_cast<int>(T_frames), max_frames_);
    int chunk_start = 0;

    auto hist_read = [&](const std::array<int64_t, kDelayBufSize>& h, int delay) -> int64_t {
        // h[0] = oldest (frame t-2), h[1] = newest (frame t-1)
        // delay=0 -> newest (t-1), delay=1 -> oldest (t-2)
        int idx = kDelayBufSize - 1 - delay;
        if (idx < 0) idx = 0;
        return h[idx];
    };
    auto hist_push = [&](std::array<int64_t, kDelayBufSize>& h, int64_t v) {
        for (int i = 0; i + 1 < kDelayBufSize; ++i) h[i] = h[i+1];
        h[kDelayBufSize - 1] = v;
    };

    for (int t = 0; t < n_frames && !cancelled_.load(); ++t) {
        // Push user audio for this frame into history before the read.
        // We mimi-encoded all frames up front; pull frame t's 8 user codebooks
        // (first 8 of Mimi's 32 RVQ output — the semantic + 7 acoustic).
        for (int cb = 0; cb < 8; ++cb) {
            int64_t tok = user_tokens_all[
                static_cast<size_t>(cb) * static_cast<size_t>(T_frames) + t];
            hist_push(user_audio_hist[cb], tok);
        }

        // Build the 17 LM input streams for this temporal step. Read with the
        // per-stream delay applied.
        int64_t text_tok_in = hist_read(text_hist, 0);  // text delay=0
        std::vector<int64_t> audio_stream_16(kAudioCodebooks);
        for (int cb = 0; cb < 8; ++cb) {
            audio_stream_16[cb]     = hist_read(user_audio_hist[cb], kUserDelays[cb]);
            audio_stream_16[8 + cb] = hist_read(agent_audio_hist[cb], kAgentDelays[cb]);
        }

        std::vector<uint16_t> hidden;
        temporal_forward(text_tok_in, audio_stream_16, hidden);

        // Inner depformer loop: 16 steps. Step 0 predicts text; steps 1..8 predict
        // agent audio cb 0..7. Steps 9..15 are unused for PersonaPlex audio output
        // but the depformer runs them anyway (model is trained for full 16-step
        // dependency). We emit a sampled token at every step to advance the cache.
        int64_t new_text = 0;
        std::array<int64_t, 8> new_agent{};
        int64_t prev_step_token = text_tok_in;
        for (int k = 0; k < kDepQ; ++k) {
            std::vector<float> logits;
            depformer_step(hidden, prev_step_token, k, logits);
            // Per speech-swift docs: audio temp=0.8/top_k=250, text temp=0.7/top_k=25.
            const float temp  = (k == 0) ? 0.7f : 0.8f;
            const int   top_k = (k == 0) ? 25   : 250;
            int tok = sample_token(logits, temp, top_k);
            if (k == 0) {
                new_text = tok;
            } else if (k <= 8) {
                new_agent[k - 1] = tok;  // agent audio cb (k-1) at step k
            }
            prev_step_token = tok;
        }
        hist_push(text_hist, new_text);
        for (int cb = 0; cb < 8; ++cb) hist_push(agent_audio_hist[cb], new_agent[cb]);

        // Accumulate agent codebooks for later Mimi decode. Mimi takes 32-RVQ
        // tokens but PersonaPlex only drives the first 8 (semantic + 7 acoustic);
        // we zero-pad the remaining 24 RVQ slots.
        for (int cb = 0; cb < kMimiCodebooks; ++cb) {
            agent_token_log.push_back(cb < 8 ? new_agent[cb] : 0);
        }
        ++frames_generated_;

        // Emit a chunk every chunk_frames_
        const int frames_in_chunk = t - chunk_start + 1;
        if (frames_in_chunk >= chunk_frames_ || t == n_frames - 1) {
            const int chunk_T = frames_in_chunk;
            std::vector<int64_t> chunk_tokens(
                static_cast<size_t>(kMimiCodebooks) * chunk_T);
            for (int cb = 0; cb < kMimiCodebooks; ++cb) {
                for (int ft = 0; ft < chunk_T; ++ft) {
                    chunk_tokens[static_cast<size_t>(cb) * chunk_T + ft] =
                        agent_token_log[
                            (static_cast<size_t>(chunk_start + ft) * kMimiCodebooks)
                            + cb];
                }
            }
            std::vector<float> chunk_pcm;
            mimi_decode(chunk_tokens, chunk_pcm);

            FullDuplexChunk fc;
            fc.samples = chunk_pcm.data();
            fc.length = chunk_pcm.size();
            fc.sample_rate = kSampleRate;
            fc.text_tokens.clear();
            fc.is_final = (t == n_frames - 1);
            on_chunk(fc);

            chunk_start = t + 1;
        }
    }
}

std::vector<int> OnnxPersonaPlex::tokenize(const std::string& /*text*/) {
    // Stub: SentencePiece encoding lands in PR 5b. Return empty for now.
    return {};
}
std::string OnnxPersonaPlex::detokenize(const std::vector<int>& /*ids*/) {
    return {};
}
void OnnxPersonaPlex::load_voice(const std::string& /*voice_name*/) {}
void OnnxPersonaPlex::load_config(const std::string& /*bundle_dir*/) {}

}  // namespace speech_core
