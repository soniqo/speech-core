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

void OnnxPersonaPlex::detect_temporal_kv_dtype() {
    // KV: input "past_k_all" — index 2 (after text_token + audio_tokens).
    OrtTypeInfo* type_info = nullptr;
    if (api_->SessionGetInputTypeInfo(temporal_step_session_, 2, &type_info) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        api_->CastTypeInfoToTensorInfo(type_info, &shape_info);
        if (shape_info) {
            ONNXTensorElementDataType elem_type;
            api_->GetTensorElementType(shape_info, &elem_type);
            temporal_kv_onnx_type_ = static_cast<int>(elem_type);
            temporal_kv_elem_size_ = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? 4 : 2;
        }
        api_->ReleaseTypeInfo(type_info);
    }
    // Hidden: output 0 ("hidden"). Can differ from KV dtype on bundles where
    // KV is wrapped in Cast(FP16) nodes but the internal compute (and hidden
    // output) is still FP32.
    type_info = nullptr;
    if (api_->SessionGetOutputTypeInfo(temporal_step_session_, 0, &type_info) == nullptr) {
        const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        api_->CastTypeInfoToTensorInfo(type_info, &shape_info);
        if (shape_info) {
            ONNXTensorElementDataType elem_type;
            api_->GetTensorElementType(shape_info, &elem_type);
            temporal_hidden_onnx_type_ = static_cast<int>(elem_type);
            temporal_hidden_elem_size_ = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? 4 : 2;
        }
        api_->ReleaseTypeInfo(type_info);
    }
}

void OnnxPersonaPlex::detect_depformer_kv_dtype() {
    // Depformer input layout: transformer_out, prev_token, step_idx,
    // past_k_all (index 3), past_v_all. Falls back to FP16 on failure.
    OrtTypeInfo* type_info = nullptr;
    if (api_->SessionGetInputTypeInfo(depformer_step_session_, 3, &type_info) != nullptr) {
        return;
    }
    const OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    api_->CastTypeInfoToTensorInfo(type_info, &shape_info);
    if (shape_info) {
        ONNXTensorElementDataType elem_type;
        api_->GetTensorElementType(shape_info, &elem_type);
        depformer_kv_onnx_type_ = static_cast<int>(elem_type);
        depformer_kv_elem_size_ = (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) ? 4 : 2;
    }
    api_->ReleaseTypeInfo(type_info);
}

std::vector<uint8_t> OnnxPersonaPlex::hidden_to_depformer_dtype(
        const std::vector<uint8_t>& src) {
    if (temporal_hidden_elem_size_ == depformer_kv_elem_size_) {
        return src;
    }
    const size_t n_elems = src.size() / temporal_hidden_elem_size_;
    std::vector<uint8_t> dst(n_elems * depformer_kv_elem_size_);
    if (temporal_hidden_elem_size_ == 4 && depformer_kv_elem_size_ == 2) {
        // FP32 -> FP16
        const float* sp = reinterpret_cast<const float*>(src.data());
        uint16_t* dp = reinterpret_cast<uint16_t*>(dst.data());
        for (size_t i = 0; i < n_elems; ++i) dp[i] = float_to_half(sp[i]);
    } else if (temporal_hidden_elem_size_ == 2 && depformer_kv_elem_size_ == 4) {
        // FP16 -> FP32
        const uint16_t* sp = reinterpret_cast<const uint16_t*>(src.data());
        float* dp = reinterpret_cast<float*>(dst.data());
        for (size_t i = 0; i < n_elems; ++i) dp[i] = half_to_float(sp[i]);
    }
    return dst;
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

    // Detect KV cache precision from the loaded temporal_step model. FP16
    // standard bundle uses ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10;
    // INT8-quantized bundle (from --stage quantize on the FP32 export) uses
    // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1. Hidden + Mimi tokens follow
    // the same precision in both cases.
    detect_temporal_kv_dtype();
    detect_depformer_kv_dtype();

    // GPU-resident KV path: when we're on a CUDA EP, keep the temporal_step
    // output K/V OrtValues alive across calls instead of copying through
    // host vectors. Eliminates ~1.5 GB of host RAM + 16 Memcpy nodes/step.
    // Opt-out via SPEECH_CORE_PP_GPU_KV=0 to fall back to the host path
    // (useful for debugging or CPU-only sessions).
    if (OnnxEngine::get().has_gpu_provider()) {
        const char* off = std::getenv("SPEECH_CORE_PP_GPU_KV");
        gpu_kv_enabled_ = (!off || std::strcmp(off, "0") != 0);

        // Try to set up a CUDA memory info + IoBinding. With BindOutputToDevice
        // ORT allocates the dynamic-shape outputs directly on the device, no
        // host staging at Run boundaries. OPT-IN via SPEECH_CORE_PP_IOBIND=1
        // (default off — the IOBinding path currently segfaults on first
        // RunWithBinding with our growing-shape KV cache; under investigation).
        if (gpu_kv_enabled_) {
            const char* iob_on = std::getenv("SPEECH_CORE_PP_IOBIND");
            const bool want_iob = iob_on && std::strcmp(iob_on, "0") != 0 &&
                                  std::strcmp(iob_on, "") != 0;
            if (want_iob) {
                OrtStatus* s = api_->CreateMemoryInfo(
                    "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cuda_mem_info_);
                if (s != nullptr) {
                    api_->ReleaseStatus(s);
                    cuda_mem_info_ = nullptr;
                } else if (api_->CreateIoBinding(
                                temporal_step_session_, &temporal_binding_) == nullptr) {
                    iobind_enabled_ = true;
                }
            }
        }
    }

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

    voices_dir_ = voices_dir;

    // Load pre-tokenized system prompts from <bundle>/system_prompts.bin
    // (produced by tokenize_system_prompts.py). Optional — wrapper still runs
    // without it, just without system-prompt conditioning.
    {
        const std::string sp_path = tokenizer_path.substr(
            0, tokenizer_path.find_last_of("/\\"));
        std::ifstream sf(sp_path + "/system_prompts.bin", std::ios::binary);
        if (sf) {
            auto rd = [&](void* p, size_t n) { sf.read(reinterpret_cast<char*>(p), n); };
            uint32_t magic = 0, version = 0, num_prompts = 0;
            rd(&magic, 4); rd(&version, 4); rd(&num_prompts, 4);
            if (magic == 0x50506573u && version == 1u) {
                for (uint32_t i = 0; i < num_prompts; ++i) {
                    uint32_t name_len = 0; rd(&name_len, 4);
                    std::string name(name_len, '\0'); rd(name.data(), name_len);
                    uint32_t num_tokens = 0; rd(&num_tokens, 4);
                    std::vector<int32_t> ids(num_tokens);
                    rd(ids.data(), num_tokens * sizeof(int32_t));
                    system_prompts_[name] = std::move(ids);
                }
            }
        }
    }

    reset_session();
}

OnnxPersonaPlex::~OnnxPersonaPlex() {
    if (past_k_value_) api_->ReleaseValue(past_k_value_);
    if (past_v_value_) api_->ReleaseValue(past_v_value_);
    if (temporal_binding_) api_->ReleaseIoBinding(temporal_binding_);
    if (cuda_mem_info_)    api_->ReleaseMemoryInfo(cuda_mem_info_);
    if (mimi_encoder_session_)   api_->ReleaseSession(mimi_encoder_session_);
    if (mimi_decoder_session_)   api_->ReleaseSession(mimi_decoder_session_);
    if (temporal_step_session_)  api_->ReleaseSession(temporal_step_session_);
    if (depformer_step_session_) api_->ReleaseSession(depformer_step_session_);
}

void OnnxPersonaPlex::reset_session() {
    // .clear() leaves capacity allocated. Swap with an empty vector to actually
    // release the KV buffer — important for memory-conscious long-running
    // services that cycle through many sessions.
    std::vector<uint8_t>().swap(temporal_k_);
    std::vector<uint8_t>().swap(temporal_v_);
    if (past_k_value_) { api_->ReleaseValue(past_k_value_); past_k_value_ = nullptr; }
    if (past_v_value_) { api_->ReleaseValue(past_v_value_); past_v_value_ = nullptr; }
    temporal_t_past_ = 0;
    frames_generated_ = 0;
    cancelled_.store(false);

    // PersonaPlex prefill sequence (matches speech-swift's MLX layout):
    //   0. Voice embedding prefix (50 frames of pre-recorded hidden states)
    //   1. Voice prompt replay (4-token cache tail)
    //   2. Silence spacer ~0.5s = 6 frames
    //   3. Text system prompt (one token per frame)
    //   4. Silence spacer ~0.5s = 6 frames
    //   5. respond_stream begins -> user audio frames
    // Each section pushes KV state but discards the hidden output. Silence
    // spacers feed PAD text + zero audio tokens through temporal_step so the
    // model gets clean transition boundaries between conditioning sections.
    std::vector<uint8_t> warmup_hidden;
    constexpr int kSilenceSpacerFrames = 6;
    constexpr int64_t kPadTextToken    = 3;  // matches Moshi PAD id

    // 0) Voice embedding prefix — ENABLED by default with scale factor 10
    // (measured-best across NATM0/NATM1/VARM0/VARF2/VARF4 on the
    // 'Can you guarantee shipping?' fixture; produces 'We're concerned
    // about it.' on VARF2, plus coherent responses on 4 other voices).
    // Override scale via env SPEECH_CORE_PP_EMB_SCALE (set to 0 to disable
    // the prefix entirely). The .bin embeddings are stored at ~0.03 std;
    // the depformer was trained on temporal output at ~1.5 std; ~10x is
    // empirically right.
    float emb_scale = 10.0f;
    if (const char* s = std::getenv("SPEECH_CORE_PP_EMB_SCALE")) {
        emb_scale = static_cast<float>(std::atof(s));
    }
    if (emb_scale > 0.0f &&
        !voice_embeddings_.empty() && voice_embedding_frames_ > 0) {
        const int F = voice_embedding_frames_;
        for (int f = 0; f < F && !cancelled_.load(); ++f) {
            std::vector<uint8_t> emb_bytes;
            const float* emb_src = voice_embeddings_.data() + static_cast<size_t>(f) * 4096;
            // Embedding feeds into depformer_step as the hidden state, so it
            // must match the temporal-hidden dtype (NOT the KV dtype — they
            // may differ on KV16-wrapped bundles).
            if (temporal_hidden_elem_size_ == 4) {
                emb_bytes.resize(4096 * 4);
                float* dst = reinterpret_cast<float*>(emb_bytes.data());
                for (int i = 0; i < 4096; ++i) dst[i] = emb_src[i] * emb_scale;
            } else {
                emb_bytes.resize(4096 * 2);
                uint16_t* dst = reinterpret_cast<uint16_t*>(emb_bytes.data());
                for (int i = 0; i < 4096; ++i) dst[i] = float_to_half(emb_src[i] * emb_scale);
            }
            int64_t prev_step_token = 0;
            int64_t new_text = 0;
            std::array<int64_t, 8> new_agent{};
            for (int k = 0; k < kDepQ; ++k) {
                std::vector<float> logits;
                depformer_step(emb_bytes, prev_step_token, k, logits);
                int tok = sample_token(logits, k == 0 ? 0.7f : 0.8f,
                                        k == 0 ? 25 : 250);
                if (k == 0) new_text = tok;
                else if (k <= 8) new_agent[k - 1] = tok;
                prev_step_token = tok;
            }
            std::vector<int64_t> audio_in(kAudioCodebooks, 0);
            for (int cb = 0; cb < 8; ++cb) audio_in[8 + cb] = new_agent[cb];
            temporal_forward(new_text, audio_in, warmup_hidden);
        }
    }

    // 1) Voice prompt replay (4-token cache tail)
    if (!voice_cache_.empty() && voice_history_size_ > 0) {
        const int H = voice_history_size_;
        for (int t = 0; t < H && !cancelled_.load(); ++t) {
            int64_t text_tok = voice_cache_[0 * H + t];
            std::vector<int64_t> audio_tok(kAudioCodebooks);
            for (int cb = 0; cb < 8; ++cb) {
                audio_tok[cb]     = voice_cache_[(1 + cb) * H + t];
                audio_tok[8 + cb] = voice_cache_[(9 + cb) * H + t];
            }
            temporal_forward(text_tok, audio_tok, warmup_hidden);
        }
    }

    // 2) Silence spacer after voice
    {
        std::vector<int64_t> silence_audio(kAudioCodebooks, 0);
        for (int i = 0; i < kSilenceSpacerFrames && !cancelled_.load(); ++i) {
            temporal_forward(kPadTextToken, silence_audio, warmup_hidden);
        }
    }

    // 3) System-prompt prefill: one text token per frame
    if (!system_prompts_.empty()) {
        auto it = system_prompts_.find(system_prompt_);
        if (it == system_prompts_.end()) {
            it = system_prompts_.find("helpful");
        }
        if (it != system_prompts_.end()) {
            const std::vector<int32_t>& ids = it->second;
            std::vector<int64_t> silence_audio(kAudioCodebooks, 0);
            for (int32_t id : ids) {
                if (cancelled_.load()) break;
                temporal_forward(static_cast<int64_t>(id), silence_audio, warmup_hidden);
            }
        }
    }

    // 4) Silence spacer after system prompt — boundary before user audio
    {
        std::vector<int64_t> silence_audio(kAudioCodebooks, 0);
        for (int i = 0; i < kSilenceSpacerFrames && !cancelled_.load(); ++i) {
            temporal_forward(kPadTextToken, silence_audio, warmup_hidden);
        }
    }
}

void OnnxPersonaPlex::cancel() {
    cancelled_.store(true);
}

void OnnxPersonaPlex::set_voice(const std::string& voice_name) {
    current_voice_ = voice_name;
    voice_cache_.clear();
    voice_embeddings_.clear();
    voice_history_size_ = 0;
    voice_embedding_frames_ = 0;
    if (voices_dir_.empty()) return;
    const std::string path = voices_dir_ + "/" + voice_name + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) return;  // silently ignore missing voice file
    auto read = [&](void* p, size_t n) { f.read(reinterpret_cast<char*>(p), n); };
    uint32_t magic = 0, version = 0, num_streams = 0, history = 0;
    read(&magic, 4); read(&version, 4); read(&num_streams, 4); read(&history, 4);
    if (magic != 0x50506558u || version != 1u) return;
    voice_cache_.resize(static_cast<size_t>(num_streams) * history);
    read(voice_cache_.data(), voice_cache_.size() * sizeof(int64_t));
    voice_history_size_ = static_cast<int>(history);
    uint32_t F = 0, D = 0;
    read(&F, 4); read(&D, 4);
    voice_embeddings_.resize(static_cast<size_t>(F) * D);
    read(voice_embeddings_.data(), voice_embeddings_.size() * sizeof(float));
    voice_embedding_frames_ = static_cast<int>(F);
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
                                        std::vector<uint8_t>& hidden_out) {
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

    // KV cache. Precision was detected at construction time:
    //   FP16 (elem_size=2) for the standard fp16 bundle
    //   FP32 (elem_size=4) for the INT8-quantized bundle
    // Buffer is raw bytes; ONNX dtype routed via temporal_kv_onnx_type_.
    const ONNXTensorElementDataType kv_type =
        static_cast<ONNXTensorElementDataType>(temporal_kv_onnx_type_);
    OrtValue* k_val = nullptr;
    OrtValue* v_val = nullptr;
    bool kv_owned_here = false;  // whether we own k_val/v_val (vs. ORT-owned)

    if (gpu_kv_enabled_ && past_k_value_ != nullptr) {
        // Reuse the OrtValue from the previous step — it's still on GPU
        k_val = past_k_value_;
        v_val = past_v_value_;
    } else {
        const size_t per_tensor_elems = static_cast<size_t>(kTemporalLayers) * 1
                                         * kTemporalHeads * temporal_t_past_ * kTemporalHeadDim;
        const size_t per_tensor_bytes = per_tensor_elems * temporal_kv_elem_size_;
        if (temporal_k_.size() != per_tensor_bytes) temporal_k_.assign(per_tensor_bytes, 0);
        if (temporal_v_.size() != per_tensor_bytes) temporal_v_.assign(per_tensor_bytes, 0);
        const int64_t kv_shape[5] = {
            kTemporalLayers, 1, kTemporalHeads, temporal_t_past_, kTemporalHeadDim};
        ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, temporal_k_.data(), temporal_k_.size(),
            kv_shape, 5, kv_type, &k_val));
        ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, temporal_v_.data(), temporal_v_.size(),
            kv_shape, 5, kv_type, &v_val));
        kv_owned_here = true;
    }

    OrtValue* inputs[4] = {text_val, audio_val, k_val, v_val};
    OrtValue* outputs[3] = {nullptr, nullptr, nullptr};

    if (iobind_enabled_) {
        // BindOutputToDevice path: ORT allocates the dynamic-shape outputs
        // directly on the CUDA device. ClearBoundInputs at the end so the
        // input OrtValues we own can be released without affecting next step.
        ort_check_local(api_, api_->BindInput(temporal_binding_, temporal_io_.in_names[0], text_val));
        ort_check_local(api_, api_->BindInput(temporal_binding_, temporal_io_.in_names[1], audio_val));
        ort_check_local(api_, api_->BindInput(temporal_binding_, temporal_io_.in_names[2], k_val));
        ort_check_local(api_, api_->BindInput(temporal_binding_, temporal_io_.in_names[3], v_val));
        ort_check_local(api_, api_->BindOutputToDevice(temporal_binding_, temporal_io_.out_names[0], cuda_mem_info_));
        ort_check_local(api_, api_->BindOutputToDevice(temporal_binding_, temporal_io_.out_names[1], cuda_mem_info_));
        ort_check_local(api_, api_->BindOutputToDevice(temporal_binding_, temporal_io_.out_names[2], cuda_mem_info_));
        ort_check_local(api_, api_->RunWithBinding(
            temporal_step_session_, nullptr, temporal_binding_));
        // Retrieve the bound outputs. The allocator is just for the OrtValue
        // array; the underlying device tensors stay device-resident.
        OrtAllocator* host_alloc = nullptr;
        ort_check_local(api_, api_->GetAllocatorWithDefaultOptions(&host_alloc));
        OrtValue** bound = nullptr;
        size_t bound_count = 0;
        ort_check_local(api_, api_->GetBoundOutputValues(
            temporal_binding_, host_alloc, &bound, &bound_count));
        if (bound_count == 3) {
            outputs[0] = bound[0];
            outputs[1] = bound[1];
            outputs[2] = bound[2];
        }
        if (bound) host_alloc->Free(host_alloc, bound);
        api_->ClearBoundInputs(temporal_binding_);
        api_->ClearBoundOutputs(temporal_binding_);
    } else {
        ort_check_local(api_, api_->Run(
            temporal_step_session_, nullptr,
            temporal_io_.in_names.data(), inputs, 4,
            temporal_io_.out_names.data(), 3, outputs));
    }

    // outputs: hidden [1,1,dim=4096] fp16; new_k_all, new_v_all
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[0], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint8_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)));
        hidden_out.assign(data, data + n * temporal_hidden_elem_size_);
    }
    // K/V output handling: GPU-resident path keeps the OrtValue alive
    // across calls (no host mirror, no per-frame device<->host copy).
    // Host-mirror path copies the data through std::vector<uint8_t> as before.
    if (gpu_kv_enabled_) {
        if (past_k_value_) api_->ReleaseValue(past_k_value_);
        if (past_v_value_) api_->ReleaseValue(past_v_value_);
        past_k_value_ = outputs[1];
        past_v_value_ = outputs[2];
        outputs[1] = nullptr;  // don't double-release at function bottom
        outputs[2] = nullptr;
    } else {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[1], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint8_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&data)));
        temporal_k_.assign(data, data + n * temporal_kv_elem_size_);

        info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[2], &info));
        nd = 0; api_->GetDimensionsCount(info, &nd);
        dims.resize(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[2], reinterpret_cast<void**>(&data)));
        temporal_v_.assign(data, data + n * temporal_kv_elem_size_);
    }
    ++temporal_t_past_;
    // Soft cap at kMaxContext (model's training context). Once we hit it,
    // drop the oldest column from each [L, B, H, T_past, D] block to keep
    // memory bounded and quality stable (positions beyond train ctx are
    // attention-poison anyway). Layout in temporal_k_/v_:
    //   for each layer L: for each B,H: T_past columns of D elems each
    // Ring-shift cap operates on the host buffer. In GPU-resident mode the
    // KV cache lives in an ORT-managed CUDA buffer; trimming it there would
    // require IOBinding with a max-shape staging buffer. We skip the cap in
    // GPU-mode for now — practical sessions don't approach kMaxContext=3000.
    if (!gpu_kv_enabled_ && temporal_t_past_ > kMaxContext) {
        const size_t blocks   = static_cast<size_t>(kTemporalLayers) * kTemporalHeads;
        const size_t old_T    = static_cast<size_t>(temporal_t_past_);
        const size_t new_T    = static_cast<size_t>(kMaxContext);
        const size_t D_bytes  = static_cast<size_t>(kTemporalHeadDim) * temporal_kv_elem_size_;
        const size_t old_row  = old_T * D_bytes;
        const size_t new_row  = new_T * D_bytes;
        std::vector<uint8_t> trimmed_k(blocks * new_row);
        std::vector<uint8_t> trimmed_v(blocks * new_row);
        for (size_t b = 0; b < blocks; ++b) {
            // Skip the oldest column (D_bytes) of each block
            std::memcpy(trimmed_k.data() + b * new_row,
                        temporal_k_.data() + b * old_row + D_bytes, new_row);
            std::memcpy(trimmed_v.data() + b * new_row,
                        temporal_v_.data() + b * old_row + D_bytes, new_row);
        }
        temporal_k_ = std::move(trimmed_k);
        temporal_v_ = std::move(trimmed_v);
        temporal_t_past_ = kMaxContext;
    }
    api_->ReleaseValue(text_val); api_->ReleaseValue(audio_val);
    // Only release k/v if we own them. In GPU-resident mode, k_val/v_val
    // came from the previous step's past_k_value_/past_v_value_ — these were
    // already released above when we adopted the new outputs[1]/outputs[2].
    if (kv_owned_here) {
        api_->ReleaseValue(k_val);
        api_->ReleaseValue(v_val);
    }
    for (auto* o : outputs) if (o) api_->ReleaseValue(o);
}

void OnnxPersonaPlex::depformer_step(const std::vector<uint8_t>& hidden,
                                      int64_t prev_token, int step_idx,
                                      std::vector<float>& logits_out) {
    auto* mem = OnnxEngine::get().cpu_memory();
    // Depformer cache lives entirely within one temporal frame; reset at each
    // step_idx==0 by allocating zero buffers (caller responsibility). Byte
    // storage matches temporal precision (FP16 or FP32 depending on bundle).
    // Host-mirror state (used when GPU-resident path is off).
    static thread_local std::vector<uint8_t> dep_k, dep_v;
    static thread_local int dep_t_past = 0;
    // GPU-resident state — held across the 16 inner steps to skip per-call
    // GetTensorMutableData copies. Reset (released) at step_idx==0 each frame.
    static thread_local OrtValue* dep_k_value = nullptr;
    static thread_local OrtValue* dep_v_value = nullptr;
    if (step_idx == 0) {
        dep_k.clear(); dep_v.clear(); dep_t_past = 0;
        if (dep_k_value) { api_->ReleaseValue(dep_k_value); dep_k_value = nullptr; }
        if (dep_v_value) { api_->ReleaseValue(dep_v_value); dep_v_value = nullptr; }
    }
    const ONNXTensorElementDataType kv_type =
        static_cast<ONNXTensorElementDataType>(depformer_kv_onnx_type_);
    OrtValue* kv_k = nullptr;
    OrtValue* kv_v = nullptr;
    bool dep_kv_owned_here = false;

    if (gpu_kv_enabled_ && dep_k_value != nullptr) {
        kv_k = dep_k_value;
        kv_v = dep_v_value;
    } else {
        const size_t per_tensor_elems = static_cast<size_t>(kDepformerLayers) * 1
                                  * kDepformerHeads * dep_t_past * kDepformerHeadDim;
        const size_t per_tensor_bytes = per_tensor_elems * depformer_kv_elem_size_;
        if (dep_k.size() != per_tensor_bytes) dep_k.assign(per_tensor_bytes, 0);
        if (dep_v.size() != per_tensor_bytes) dep_v.assign(per_tensor_bytes, 0);
        const int64_t kv_shape[5] = {
            kDepformerLayers, 1, kDepformerHeads, dep_t_past, kDepformerHeadDim};
        ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, dep_k.data(), dep_k.size(),
            kv_shape, 5, kv_type, &kv_k));
        ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, dep_v.data(), dep_v.size(),
            kv_shape, 5, kv_type, &kv_v));
        dep_kv_owned_here = true;
    }

    // Hidden may need a Cast: temporal might be FP32 (INT8 bundle) while
    // depformer is FP16. hidden_to_depformer_dtype is a no-op if they match.
    std::vector<uint8_t> hidden_buf = hidden_to_depformer_dtype(hidden);
    const int64_t hidden_shape[3] = {1, 1, 4096};
    OrtValue* hv = nullptr;
    ort_check_local(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, hidden_buf.data(), hidden_buf.size(),
        hidden_shape, 3, kv_type, &hv));

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

    // kv_k / kv_v were set up at the top of the function (either GPU-resident
    // reuse from the previous inner step, or freshly-created host tensors).
    OrtValue* inputs[5] = {hv, pv, sv, kv_k, kv_v};
    OrtValue* outputs[3] = {nullptr, nullptr, nullptr};
    ort_check_local(api_, api_->Run(
        depformer_step_session_, nullptr,
        depformer_io_.in_names.data(), inputs, 5,
        depformer_io_.out_names.data(), 3, outputs));

    // outputs[0]: logits [1, card] in same dtype as hidden (fp16 or fp32)
    {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[0], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint8_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)));
        logits_out.resize(n);
        if (depformer_kv_elem_size_ == 2) {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(data);
            for (size_t i = 0; i < n; ++i) logits_out[i] = half_to_float(p[i]);
        } else {
            const float* p = reinterpret_cast<const float*>(data);
            for (size_t i = 0; i < n; ++i) logits_out[i] = p[i];
        }
    }
    // Update dep cache. GPU-resident path adopts the output OrtValues;
    // host path copies the data back via GetTensorMutableData.
    if (gpu_kv_enabled_) {
        if (dep_k_value) api_->ReleaseValue(dep_k_value);
        if (dep_v_value) api_->ReleaseValue(dep_v_value);
        dep_k_value = outputs[1];
        dep_v_value = outputs[2];
        outputs[1] = nullptr;
        outputs[2] = nullptr;
    } else {
        OrtTensorTypeAndShapeInfo* info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[1], &info));
        size_t nd = 0; api_->GetDimensionsCount(info, &nd);
        std::vector<int64_t> dims(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        size_t n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        uint8_t* data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[1], reinterpret_cast<void**>(&data)));
        dep_k.assign(data, data + n * depformer_kv_elem_size_);

        info = nullptr;
        ort_check_local(api_, api_->GetTensorTypeAndShape(outputs[2], &info));
        nd = 0; api_->GetDimensionsCount(info, &nd);
        dims.resize(nd); api_->GetDimensions(info, dims.data(), nd);
        api_->ReleaseTensorTypeAndShapeInfo(info);
        n = 1; for (auto d : dims) n *= static_cast<size_t>(d);
        data = nullptr;
        ort_check_local(api_, api_->GetTensorMutableData(outputs[2], reinterpret_cast<void**>(&data)));
        dep_v.assign(data, data + n * depformer_kv_elem_size_);
    }
    ++dep_t_past;
    api_->ReleaseValue(hv); api_->ReleaseValue(pv); api_->ReleaseValue(sv);
    if (dep_kv_owned_here) {
        api_->ReleaseValue(kv_k);
        api_->ReleaseValue(kv_v);
    }
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

    // NOTE: voice prefill is done in reset_session() by replaying the cached
    // 4-token tail through temporal_step (populates KV cache with speaker-
    // conditioned state). The delay history is INTENTIONALLY left zeroed here
    // — re-seeding it would double-condition the model on the voice (KV +
    // input both voice-flavoured), producing over-energetic / clipped output.

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

        std::vector<uint8_t> hidden;
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
