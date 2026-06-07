#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// PersonaPlex 7B — NVIDIA's full-duplex speech-to-speech model on Moshi
/// architecture, via ONNX Runtime. 24 kHz I/O at 12.5 Hz frame rate.
/// Model files: soniqo/PersonaPlex-7B-ONNX (planned) — four .onnx graphs:
///   `mimi_encoder.onnx`    — 24 kHz PCM → 16 audio codebooks @ 12.5 Hz
///   `mimi_decoder.onnx`    — 16 audio codebooks @ 12.5 Hz → 24 kHz PCM
///   `temporal_step.onnx`   — one frame of 32-layer 7B temporal transformer,
///                            explicit KV cache I/O (concat-style, grows by 1
///                            position per call up to context=3000)
///   `depformer_step.onnx`  — one inner step of the 6-layer depformer,
///                            per-step weight gathering by step_idx in [0,16),
///                            explicit KV cache I/O
/// Plus:
///   `tokenizer_spm_32k_3.model` — SentencePiece text tokenizer
///   `voices/<name>.bin`         — per-voice audio embeddings (~6 KB each)
///   `config.json`               — bundle config (delays, vocab sizes, etc.)
///
/// Hardware accel is automatic via OnnxEngine: CUDA EP on Linux/Windows when
/// `SPEECH_CORE_WITH_CUDA=ON`, NNAPI on Android, CPU fallback otherwise.
/// The temporal KV cache is ~1.5 GB fp16 at full ctx; GPU-resident via
/// IOBinding when CUDA EP is active (PR 6).
///
/// Implements FullDuplexSpeechInterface. Mirrors speech-swift's
/// PersonaPlexModel.respondStream contract: caller pushes user PCM, the
/// on_chunk callback receives ~2 s windows of agent audio + interleaved
/// text tokens.
class OnnxPersonaPlex : public FullDuplexSpeechInterface {
public:
    /// Constructor. Each path is the ONNX graph from the bundle plus a
    /// SentencePiece model path and a voices directory.
    OnnxPersonaPlex(const std::string& mimi_encoder_path,
                    const std::string& mimi_decoder_path,
                    const std::string& temporal_step_path,
                    const std::string& depformer_step_path,
                    const std::string& tokenizer_path,
                    const std::string& voices_dir,
                    bool hw_accel = true);
    ~OnnxPersonaPlex() override;

    // --- FullDuplexSpeechInterface ---
    void respond_stream(const float* user_audio, size_t length, int sample_rate,
                        FullDuplexChunkCallback on_chunk) override;
    void set_voice(const std::string& voice_name) override;
    void set_system_prompt(const std::string& prompt) override;
    void set_max_frames(int max_frames) override { max_frames_ = max_frames; }
    void reset_session() override;
    void cancel() override;
    int output_sample_rate() const override { return kSampleRate; }

    /// Set the streaming chunk size (frames per emitted audio chunk).
    /// Default 25 frames = 2 s of 24 kHz audio. Smaller values reduce
    /// latency at the cost of more callback overhead.
    void set_chunk_frames(int frames) { chunk_frames_ = frames; }

    /// Number of frames the most recent respond_stream emitted.
    int frames_generated() const { return frames_generated_; }

private:
    // Bundle-invariant constants (from PersonaPlex spec; see speech-swift
    // docs/models/personaplex.md and speech-models ONNX_EXPORT_PLAN.md).
    static constexpr int kSampleRate      = 24000;
    static constexpr int kFrameRate       = 12;            // 12.5 Hz, kept as int (samples_per_frame is canonical)
    static constexpr int kSamplesPerFrame = 1920;          // 24000 / 12.5
    static constexpr int kTemporalLayers  = 32;
    static constexpr int kTemporalHeads   = 32;
    static constexpr int kTemporalHeadDim = 128;
    static constexpr int kDepformerLayers = 6;
    static constexpr int kDepformerHeads  = 16;
    static constexpr int kDepformerHeadDim = 64;
    static constexpr int kDepQ            = 16;            // depformer codebook steps
    static constexpr int kAudioCodebooks  = 16;            // LM audio stream count
    static constexpr int kMimiCodebooks   = 32;            // Mimi internal RVQ count
    static constexpr int kTextVocab       = 32001;
    static constexpr int kAudioVocab      = 2049;
    static constexpr int kMaxContext      = 3000;          // temporal context

    struct IoNames {
        std::vector<std::string> in_names_str;
        std::vector<std::string> out_names_str;
        std::vector<const char*> in_names;
        std::vector<const char*> out_names;
    };
    void query_io_names(OrtSession* session, IoNames& names);

    // Frame loop helpers (PR 5 implementation; some may stub until PR 6/7).
    void mimi_encode(const float* pcm, size_t length, std::vector<int64_t>& tokens_out);
    void mimi_decode(const std::vector<int64_t>& tokens, std::vector<float>& pcm_out);
    void temporal_forward(int64_t text_token,
                          const std::vector<int64_t>& audio_tokens_16,
                          std::vector<uint8_t>& hidden_out);
    void depformer_step(const std::vector<uint8_t>& hidden,
                        int64_t prev_token, int step_idx,
                        std::vector<float>& logits_out);
    void detect_temporal_kv_dtype();
    void detect_depformer_kv_dtype();
    // Cast hidden bytes from temporal's dtype to depformer's dtype if they
    // differ (e.g. INT8 temporal -> FP16 depformer). Returns a buffer in
    // depformer dtype. If the dtypes match, returns the input unchanged.
    std::vector<uint8_t> hidden_to_depformer_dtype(const std::vector<uint8_t>& src);
    int  sample_token(const std::vector<float>& logits, float temperature, int top_k);

    // SentencePiece text decoding/encoding (minimal vendored impl in PR 5).
    std::vector<int> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int>& ids);

    void load_voice(const std::string& voice_name);
    void load_config(const std::string& bundle_dir);

    const OrtApi* api_ = nullptr;
    OrtSession*   mimi_encoder_session_   = nullptr;
    OrtSession*   mimi_decoder_session_   = nullptr;
    OrtSession*   temporal_step_session_  = nullptr;
    OrtSession*   depformer_step_session_ = nullptr;

    IoNames mimi_enc_io_;
    IoNames mimi_dec_io_;
    IoNames temporal_io_;
    IoNames depformer_io_;

    // KV cache buffers — concat-style. Two storage modes:
    //   - HOST mirror (vectors): the original path. Each frame copies the
    //     output K/V from device back to host, then re-uploads next frame.
    //     Used when running on CPU EP.
    //   - GPU-resident (OrtValue*): the output OrtValue from the prior
    //     step is kept alive and passed directly as the next step's input.
    //     No host mirror, no per-frame memcpy. Used when running on CUDA EP
    //     and SPEECH_CORE_PP_GPU_KV != "0". Detected at construction.
    // Stored as raw bytes when host; OrtValue* when GPU-resident.
    std::vector<uint8_t> temporal_k_;
    std::vector<uint8_t> temporal_v_;
    int                  temporal_t_past_ = 0;
    OrtValue*            past_k_value_ = nullptr;  // GPU-resident path only
    OrtValue*            past_v_value_ = nullptr;
    bool                 gpu_kv_enabled_ = false;
    // IoBinding state for temporal_step. With BindOutputToDevice we let ORT
    // allocate the output buffers on the CUDA device on its own; no host
    // staging at Run boundaries. Falls back to the simpler GPU-resident
    // OrtValue handoff when this isn't available.
    OrtMemoryInfo*       cuda_mem_info_ = nullptr;
    OrtIoBinding*        temporal_binding_ = nullptr;
    bool                 iobind_enabled_ = false;
    int                  temporal_kv_onnx_type_ = 10;  // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 default
    size_t               temporal_kv_elem_size_ = 2;   // bytes per element
    // Depformer dtype tracked independently so a mixed-precision bundle
    // (INT8 temporal with FP32 KV/hidden + FP16 depformer) is supported —
    // saves ~3 GB by shipping the FP16 depformer instead of FP32.
    int                  depformer_kv_onnx_type_ = 10;
    size_t               depformer_kv_elem_size_ = 2;

    // Voice prompt (loaded from voices/<name>.bin via voices_to_bin.py).
    // Format: magic + version + num_streams=17 + history=4 + cache[17*4] int64
    //         + num_emb_frames + emb_dim=4096 + embeddings[F * 4096] fp32
    // We currently consume cache[] to pre-populate the delay history. The
    // embeddings would prefill temporal hidden but require an alternate
    // graph signature — deferred follow-up.
    std::vector<int64_t> voice_cache_;          // 17*4 cache tokens
    int                  voice_history_size_ = 0;  // typically 4
    std::vector<float>   voice_embeddings_;     // [F*4096] fp32
    int                  voice_embedding_frames_ = 0;
    std::string          voices_dir_;
    std::string          current_voice_;

    // System prompts pre-tokenized by tokenize_system_prompts.py and shipped
    // as system_prompts.bin next to the SPM tokenizer. Maps prompt name to
    // token IDs. The wrapper feeds these as the text stream during
    // reset_session voice-prefill warmup, so the model is conditioned on the
    // chosen role before user audio arrives. Avoids a SentencePiece C++ dep.
    std::unordered_map<std::string, std::vector<int32_t>> system_prompts_;

    // SentencePiece model (raw bytes; we vendor a minimal decoder; the full
    // SP runtime arrives in PR 5b if needed).
    std::vector<uint8_t> spm_model_blob_;
    std::vector<std::string> spm_vocab_;  // optional, populated when we read pieces

    // Delay pattern: text + 16 audio streams. PersonaPlex uses
    // [0, 0,1,1,1,1,1,1,1, 0,1,1,1,1,1,1,1] (1 text + 8 user + 8 agent).
    std::vector<int> delays_;

    std::string system_prompt_;
    int         max_frames_     = 2000;   // ~2.7 min at 12.5 Hz
    int         chunk_frames_   = 25;     // 2 s emit cadence
    int         frames_generated_ = 0;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
