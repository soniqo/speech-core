#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// Nemotron Speech Streaming 0.6B — cache-aware streaming RNN-T via LiteRT.
///
/// Three graphs (soniqo/Nemotron-Speech-Streaming-LiteRT):
///   encoder  in:  mel[1,bins,frames], mel_len[1], pre_cache[1,bins,pre],
///                 cache_last_channel[L,1,attn,H], cache_last_time[L,1,H,conv],
///                 cache_last_channel_len[1]
///            out: encoded[1,T_out,H], encoded_len[1], + the three rolled
///                 caches + new cache_last_channel_len
///   decoder  in:  token[1,1] i64, h[L,1,Hd], c[L,1,Hd]
///            out: hidden[1,1,Hd], h_new, c_new
///   joint    in:  enc_frame[1,1,H], dec_hidden[1,1,Hd]
///            out: logits[1,1,vocab+1]
///
/// True chunkwise streaming: `push_chunk` drains fixed windows
/// ((actual_mel_frames-1)*hop samples ≈ 80 ms), advancing the encoder cache +
/// decoder LSTM state across calls and greedily decoding the first encoder
/// frame (the second frame is right-context lookahead, re-settled next chunk).
/// One instance == one stream. Not thread-safe.
class LiteRTNemotronStreamingStt : public STTInterface {
public:
    struct Config {
        int sample_rate       = 16000;
        int mel_bins          = 128;
        int n_fft             = 512;
        int hop_length        = 160;
        int win_length        = 400;
        int encoder_layers    = 24;
        int encoder_hidden    = 1024;
        int decoder_layers    = 2;
        int decoder_hidden    = 640;
        int vocab_size        = 1024;  // refined from vocab.json
        int blank_id          = 1024;  // = vocab_size
        int pre_cache_size    = 16;
        int actual_mel_frames = 9;     // 80 ms window
        int attn_left_context = 70;    // cache_last_channel time dim
        int conv_cache_size   = 8;     // cache_last_time conv dim (kernel-1)
        int max_symbols       = 8;     // expansions per encoder frame
    };

    LiteRTNemotronStreamingStt(const std::string& encoder_path,
                               const std::string& decoder_path,
                               const std::string& joint_path,
                               const std::string& vocab_path,
                               bool hw_accel = true);
    LiteRTNemotronStreamingStt(const std::string& encoder_path,
                               const std::string& decoder_path,
                               const std::string& joint_path,
                               const std::string& vocab_path,
                               const Config& config,
                               bool hw_accel = true);
    ~LiteRTNemotronStreamingStt() override;

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
    void reset_stream_state();
    void run_decoder_step(int64_t token_id);
    std::string run_window();          // drain one window from pending_, return emitted text
    std::string token_to_text(int id) const;
    int chunk_samples() const { return (cfg_.actual_mel_frames - 1) * cfg_.hop_length; }

    LiteRtModel         enc_model_    = nullptr;
    LiteRtModel         dec_model_    = nullptr;
    LiteRtModel         jnt_model_    = nullptr;
    LiteRtCompiledModel enc_compiled_ = nullptr;
    LiteRtCompiledModel dec_compiled_ = nullptr;
    LiteRtCompiledModel jnt_compiled_ = nullptr;

    Config cfg_;
    int    enc_t_out_ = 2;  // encoder output frames per window (read at load)
    std::vector<std::string> vocab_;

    // ---- per-stream state ----
    std::vector<float> pending_;             // accumulated PCM float32
    std::vector<float> pre_cache_;           // [1, bins, pre_cache_size]
    std::vector<float> cache_last_channel_;  // [L, 1, attn, H]
    std::vector<float> cache_last_time_;     // [L, 1, H, conv]
    int64_t            cache_last_channel_len_ = 0;
    std::vector<float> dec_h_, dec_c_;       // [L, 1, Hd]
    std::vector<float> dec_hidden_;          // [Hd] — joint input
    std::string        accumulated_text_;
    bool               stream_init_ = false;
};

}  // namespace speech_core
