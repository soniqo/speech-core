#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Nemotron-3.5 ASR Streaming Multilingual 0.6B — cache-aware streaming RNN-T
/// via LiteRT (TFLite). Prompt-conditioned FastConformer encoder + RNN-T
/// decoder/joint, three graphs
/// (soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-LiteRT-*):
///
///   encoder  in (serving order):
///                 audio_signal[1,bins,frames] f32, audio_length[1] i64,
///                 language_mask[1,num_prompts] f32, pre_cache[1,bins,pre] f32,
///                 cache_last_channel[L,1,attn,H] f32,
///                 cache_last_time[L,1,H,conv] f32,
///                 cache_last_channel_len[1] i64
///            out: encoded_output[1,T_out,H] f32, encoded_length[1] i32,
///                 new_pre_cache, new_cache_last_channel, new_cache_last_time,
///                 new_cache_last_channel_len[1] i32
///   decoder  in:  token[1,1] i64, h[L,1,Hd] f32, c[L,1,Hd] f32
///            out: decoder_output[1,1,Hd] f32, h_out, c_out
///   joint    in:  encoder_output[1,1,H] f32, decoder_output[1,1,Hd] f32
///            out: logits[1,1,vocab+1] f32
///
/// Differs from the older LiteRTNemotronStreamingStt: 320 ms windows (32 mel
/// frames, 4 emitted encoder frames decoded per chunk), a one-hot
/// language_mask input (caller selects the prompt slot via set_language), and
/// a 13 087-token vocab. The continuous whole-utterance mel (pre-emphasis 0.97,
/// Slaney, log-floor 2^-24) is sliced into fixed windows. One instance == one
/// stream. Not thread-safe.
///
/// Runtime note: the channelwise-INT8 encoder needs an Android NNAPI/XNNPACK
/// delegate; the FP16 encoder runs on the plain CPU interpreter.
class LiteRTNemotronMultilingualStt : public STTInterface {
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
        int   att_left_context  = 56;
        int   conv_cache_size   = 8;
        int   decoder_layers    = 2;
        int   decoder_hidden    = 640;
        int   num_prompts       = 128;
        int   vocab_size        = 13087; // refined from vocab.json
        int   blank_id          = 13087; // = vocab_size
        int   max_symbols       = 10;
    };

    LiteRTNemotronMultilingualStt(const std::string& encoder_path,
                                  const std::string& decoder_path,
                                  const std::string& joint_path,
                                  const std::string& vocab_path,
                                  const std::string& languages_path,
                                  bool hw_accel = true);
    LiteRTNemotronMultilingualStt(const std::string& encoder_path,
                                  const std::string& decoder_path,
                                  const std::string& joint_path,
                                  const std::string& vocab_path,
                                  const std::string& languages_path,
                                  const Config& config,
                                  bool hw_accel = true);
    ~LiteRTNemotronMultilingualStt() override;

    /// Select the language prompt slot (locale key from languages.json).
    /// Returns true if resolved; unknown keys fall back to "auto"/slot 0.
    bool set_language(const std::string& locale);
    int  language_slot() const { return lang_slot_; }

    // --- STTInterface (batch convenience = begin/push/end) ---
    TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }

    // --- STTInterface (streaming) ---
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
    std::vector<float> compute_mel(const float* audio, size_t length) const;
    std::string run_window(const float* mel_window);  // encoder + greedy RNN-T
    std::string token_to_text(int id) const;
    int chunk_samples() const { return cfg_.mel_frames * cfg_.hop_length; }

    /// LiteRT binds Run buffers in subgraph-input/output order, which the
    /// converter scrambles relative to the canonical signature. At load we read
    /// each graph's tensor names and parse the canonical index back out
    /// ("serving_default_args_N:0" -> N for inputs; "StatefulPartitionedCall:M"
    /// -> M for outputs), so run_window can place buffers by role regardless of
    /// the physical order.
    void query_litert_io_order();

    LiteRtModel         enc_model_    = nullptr;
    LiteRtModel         dec_model_    = nullptr;
    LiteRtModel         jnt_model_    = nullptr;
    LiteRtCompiledModel enc_compiled_ = nullptr;
    LiteRtCompiledModel dec_compiled_ = nullptr;
    LiteRtCompiledModel jnt_compiled_ = nullptr;

    Config cfg_;
    int    enc_t_out_ = 4;  // encoder output frames per window (read at load)
    // Per-graph role order: vec[physical_slot] = canonical index parsed from
    // the tensor name. enc inputs 0..6 = audio_signal, audio_length,
    // language_mask, pre_cache, cache_last_channel, cache_last_time,
    // cache_last_channel_len; enc outputs 0..5 = encoded_output,
    // encoded_length, new_pre_cache, new_cache_last_channel,
    // new_cache_last_time, new_cache_last_channel_len. decoder in 0..2 =
    // token, h, c; out 0..2 = decoder_output, h, c. joint in 0..1 =
    // encoder_output, decoder_output; out 0 = logits.
    std::vector<int> enc_in_role_, enc_out_role_;
    std::vector<int> dec_in_role_, dec_out_role_;
    std::vector<int> jnt_in_role_, jnt_out_role_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> lang2slot_;
    int lang_slot_ = 0;
    int auto_slot_ = -1;

    // ---- per-stream state ----
    std::vector<float> stream_audio_;
    size_t             decoded_windows_ = 0;
    std::vector<float> pre_cache_;
    std::vector<float> cache_last_channel_;
    std::vector<float> cache_last_time_;
    int64_t            cache_last_channel_len_ = 0;
    std::vector<float> dec_h_, dec_c_;
    int64_t            last_token_ = 0;
    std::string        accumulated_text_;
    bool               stream_init_ = false;
};

}  // namespace speech_core
