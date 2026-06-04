#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Nemotron-3.5 ASR Streaming Multilingual 0.6B — cache-aware streaming RNN-T
/// via ONNX Runtime. Prompt-conditioned FastConformer encoder + RNN-T
/// decoder/joint, exported as three ONNX graphs
/// (soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-*):
///
///   encoder  in:  audio_signal[1,bins,frames] f32, audio_length[1] i32,
///                 language_mask[1,num_prompts] f32 (one-hot language slot),
///                 pre_cache[1,bins,pre] f32,
///                 cache_last_channel[L,1,attn,H] f32,
///                 cache_last_time[L,1,H,conv] f32,
///                 cache_last_channel_len[1] i32
///            out: encoded_output[1,T_out,H] f32, encoded_length[1] i32,
///                 + the four rolled caches (new_*)
///   decoder  in:  token[1,1] i64, h[L,1,Hd] f32, c[L,1,Hd] f32
///            out: decoder_output[1,1,Hd] f32, h_new, c_new
///   joint    in:  encoder_output[1,1,H] f32, decoder_output[1,1,Hd] f32
///            out: logits[1,1,vocab+1] f32
///
/// Unlike Parakeet, language is NOT auto-detected — the caller selects a
/// prompt slot via set_language() (locale -> slot from languages.json). The
/// default slot is "en-US". One instance == one stream. Not thread-safe.
///
/// Mirrors the reference validator export/onnx_inference.py: whole-utterance
/// mel sliced into fixed 320 ms windows (pre-emphasis 0.97, Slaney mel,
/// log-floor 2^-24, NeMo normalize=NA — i.e. no per-feature normalization),
/// greedy RNN-T over every emitted encoder frame, encoder caches carried
/// across windows.
class NemotronMultilingualStt : public STTInterface {
public:
    struct Config {
        int   sample_rate       = 16000;
        int   mel_bins          = 128;
        int   n_fft             = 512;
        int   hop_length        = 160;
        int   win_length        = 400;
        float pre_emphasis      = 0.97f;
        int   mel_frames        = 32;    // 320 ms window
        int   pre_cache_size    = 9;
        int   encoder_layers    = 24;
        int   encoder_hidden    = 1024;
        int   att_left_context  = 56;    // cache_last_channel time dim
        int   conv_cache_size   = 8;     // cache_last_time conv dim
        int   decoder_layers    = 2;
        int   decoder_hidden    = 640;
        int   num_prompts       = 128;
        int   vocab_size        = 13087; // refined from vocab.json
        int   blank_id          = 13087; // = vocab_size
        int   max_symbols       = 10;    // expansions per encoder frame
    };

    NemotronMultilingualStt(const std::string& encoder_path,
                            const std::string& decoder_path,
                            const std::string& joint_path,
                            const std::string& vocab_path,
                            const std::string& languages_path,
                            bool hw_accel = true);
    NemotronMultilingualStt(const std::string& encoder_path,
                            const std::string& decoder_path,
                            const std::string& joint_path,
                            const std::string& vocab_path,
                            const std::string& languages_path,
                            const Config& config,
                            bool hw_accel = true);
    ~NemotronMultilingualStt() override;

    /// Select the language prompt slot for subsequent transcription.
    /// Accepts a locale key from languages.json ("en-US", "fr", "ja-JP", ...).
    /// Unknown keys fall back to the "auto" slot if present, else slot 0.
    /// Returns true if the key resolved to a known slot.
    bool set_language(const std::string& locale);
    int  language_slot() const { return lang_slot_; }

    // --- STTInterface (batch convenience = begin/push/end) ---
    TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }

    // --- STTInterface (streaming — the real path) ---
    bool supports_streaming() const override { return true; }
    void begin_stream(int sample_rate) override;
    PartialResult push_chunk(const float* audio, size_t length) override;
    void flush_stream() override;
    TranscriptionResult end_stream() override;
    void cancel_stream() override;

private:
    bool load_vocab(const std::string& path);
    bool load_languages(const std::string& path);
    void reset_stream_state();
    void query_io_names();

    /// Compute the continuous log-mel of `audio` (pre-emphasis + Slaney + log
    /// floor, channels-first [bins, frames]). Matches compute_mel_chunk() in
    /// the reference validator.
    std::vector<float> compute_mel(const float* audio, size_t length) const;

    /// Run one 320 ms window: encoder (cache-aware) -> greedy RNN-T over every
    /// emitted encoder frame. `mel_window` is [mel_bins * mel_frames]
    /// channels-first. Appends emitted text to accumulated_text_.
    std::string run_window(const float* mel_window);
    std::string token_to_text(int id) const;
    int chunk_samples() const { return cfg_.mel_frames * cfg_.hop_length; }

    const OrtApi* api_ = nullptr;
    OrtSession* encoder_ = nullptr;
    OrtSession* decoder_ = nullptr;
    OrtSession* joint_   = nullptr;

    Config cfg_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> lang2slot_;
    int lang_slot_ = 0;
    int auto_slot_ = -1;

    // Cached I/O names (queried once at load; positional order preserved).
    std::vector<std::string> enc_in_, enc_out_;
    std::vector<std::string> dec_in_, dec_out_;
    std::vector<std::string> jnt_in_, jnt_out_;

    // ---- per-stream state ----
    std::vector<float> stream_audio_;        // all PCM for the current utterance
    size_t             decoded_windows_ = 0; // windows already run from stream_audio_
    std::vector<float> pre_cache_;           // [1, bins, pre_cache_size]
    std::vector<float> cache_last_channel_;  // [L, 1, attn, H]
    std::vector<float> cache_last_time_;     // [L, 1, H, conv]
    int32_t            cache_last_channel_len_ = 0;
    std::vector<float> dec_h_, dec_c_;       // [L, 1, Hd]
    int64_t            last_token_ = 0;      // primed to blank in reset
    std::string        accumulated_text_;
    bool               stream_init_ = false;
};

}  // namespace speech_core
