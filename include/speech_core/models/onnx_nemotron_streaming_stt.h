#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/context_graph.h"

#include <onnxruntime_c_api.h>

#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// Nemotron Speech Streaming 0.6B — cache-aware streaming RNN-T via ORT.
///
/// Wire-level identical to LiteRTNemotronStreamingStt: same three graphs,
/// same I/O contract, same streaming state machine — only the runtime
/// differs (ONNX Runtime instead of LiteRT). On a desktop/server build the
/// CUDA EP takes over for the encoder (heaviest of the three) while the
/// decoder LSTM + joint stay CPU (small tensors, dispatch overhead beats
/// kernel time — same anti-pattern as Parakeet's decoder_joint).
///
/// Three graphs (soniqo/Nemotron-Speech-Streaming-ONNX):
///   encoder  in:  audio_signal[1,bins,frames], audio_length[1],
///                 pre_cache[1,bins,pre], cache_last_channel[L,1,attn,H],
///                 cache_last_time[L,1,H,conv], cache_last_channel_len[1]
///            out: encoded_output[1,T_out,H], encoded_length[1],
///                 new_pre_cache, new_cache_last_channel,
///                 new_cache_last_time, new_cache_last_channel_len
///   decoder  in:  token[1,1] i64, h[L,1,Hd], c[L,1,Hd]
///            out: decoder_output[1,1,Hd], h_out, c_out
///   joint    in:  encoder_output[1,1,H], decoder_output[1,1,Hd]
///            out: logits[1,1,vocab+1]
///
/// One instance == one stream. Not thread-safe.
class OnnxNemotronStreamingStt : public STTInterface {
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
        int subsampling_factor = 8;    // encoder frame : mel frame ratio (EOU streaming shift)
        // End-of-utterance / end-of-boundary tokens (Parakeet-EOU). Read from
        // config.json's eouTokenId / eobTokenId. -1 = disabled (plain Nemotron
        // streaming), in which case these ids are never special-cased and the
        // decode path is byte-identical to before. When set, <EOU> ends the
        // current utterance (surfaced via end_of_utterance()) and <EOB> is a
        // soft boundary that is not emitted into the transcript.
        int eou_token_id      = -1;
        int eob_token_id      = -1;
        // Pre-emphasis coefficient applied to the waveform before the mel
        // frontend (NeMo `preemph`, read from config.json's preEmphasis).
        // 0 = disabled, keeping the plain-Nemotron path unchanged.
        float preemph         = 0.0f;
        // RNN-T decoding: <=1 keeps the original greedy path (byte-identical);
        // >1 enables modified beam search, which is what contextual biasing
        // (set_context_phrases) rides on. Greedy stays the default.
        int   beam_size       = 0;
    };

    OnnxNemotronStreamingStt(const std::string& encoder_path,
                             const std::string& decoder_path,
                             const std::string& joint_path,
                             const std::string& vocab_path,
                             bool hw_accel = true);
    OnnxNemotronStreamingStt(const std::string& encoder_path,
                             const std::string& decoder_path,
                             const std::string& joint_path,
                             const std::string& vocab_path,
                             const Config& config,
                             bool hw_accel = true);
    ~OnnxNemotronStreamingStt() override;

    // --- STTInterface (batch convenience = begin/push/end) ---
    TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }

    // True once an <EOU> token has been decoded in the current stream
    // (Parakeet-EOU only). Lets a voice agent detect turn end from the ASR
    // stream itself. Reset by begin_stream() / cancel_stream().
    bool end_of_utterance() const { return eou_detected_; }

    // Contextual biasing (shallow fusion). Nudges beam search toward the given
    // surface phrases — command words, a brand name, live contact/track names.
    // No effect unless Config.beam_size > 1. Rebuild per utterance to inject the
    // entities currently on the device. Empty list clears biasing.
    void set_context_phrases(const std::vector<std::string>& phrases,
                             float per_char = 1.5f, float completion = 3.0f);

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
    // Pure predictor (decoder LSTM) step: reads (token, h_in, c_in), writes the
    // fresh joint-input hidden + next LSTM state. No shared-state side effects,
    // so beam hypotheses each carry their own predictor state. run_decoder_step
    // is this applied to the greedy path's shared members.
    void decoder_step(int64_t token_id,
                      const std::vector<float>& h_in, const std::vector<float>& c_in,
                      std::vector<float>& hidden_out,
                      std::vector<float>& h_out, std::vector<float>& c_out);
    // Run the joint network for one encoder frame + predictor hidden -> logits.
    void joint_logits(const float* enc_frame, const std::vector<float>& dec_hidden,
                      std::vector<float>& logits);
    std::string run_window();
    std::string decode_greedy(const std::vector<float>& encoded);
    std::string decode_beam(const std::vector<float>& encoded);
    std::string token_to_text(int id) const;

    // One beam-search hypothesis: emitted text, running log-prob score, its own
    // predictor state, and its position in the context-biasing automaton.
    struct BeamHyp {
        std::string        text;
        double             score = 0.0;
        std::vector<float> dec_h, dec_c, dec_hidden;
        ContextGraph::State ctx = 0;
        bool               eou = false;
    };
    const BeamHyp* best_beam() const;
    // Mel window fed to the encoder, in samples. EOU streams melFrames*hop
    // windows advanced by a smaller shift (see run_window); Nemotron uses the
    // original (melFrames-1)*hop stride where window == shift.
    int chunk_samples() const {
        const int frames = (cfg_.eou_token_id >= 0)
            ? cfg_.actual_mel_frames
            : cfg_.actual_mel_frames - 1;
        return frames * cfg_.hop_length;
    }
    // Samples to advance per window: for EOU only outputFrames*subsampling mel
    // frames are fresh audio (the rest is right-context re-settled next window);
    // Nemotron advances the whole window.
    int shift_samples() const {
        if (cfg_.eou_token_id < 0) return chunk_samples();
        return output_frames_ * cfg_.subsampling_factor * cfg_.hop_length;
    }

    const OrtApi* api_ = nullptr;
    OrtSession*   enc_ = nullptr;
    OrtSession*   dec_ = nullptr;
    OrtSession*   jnt_ = nullptr;

    // Cached I/O name vectors (introspected at load) — same idiom as
    // OnnxVoxCPM2Tts::IoNames.
    struct IoNames {
        std::vector<std::string>  in_names_str;
        std::vector<std::string>  out_names_str;
        std::vector<const char*>  in_names;
        std::vector<const char*>  out_names;
    };
    void query_io_names(OrtSession* session, IoNames& names);
    IoNames enc_io_;
    IoNames dec_io_;
    IoNames jnt_io_;

    Config cfg_;
    int    enc_t_out_     = 2;  // encoder output frames per window (read at load)
    int    output_frames_ = 1;  // committed frames per window (T_out minus lookahead;
                                // overridden from config.json's streaming.outputFrames).
    std::vector<std::string> vocab_;

    // ---- per-stream state ----
    std::vector<float> pending_;             // accumulated PCM float32
    std::vector<float> pre_cache_;           // [1, bins, pre_cache_size]
    std::vector<float> cache_last_channel_;  // [L, 1, attn, H]
    std::vector<float> cache_last_time_;     // [L, 1, H, conv]
    int64_t            cache_last_channel_len_ = 0;
    std::vector<float> dec_h_, dec_c_;       // [L, 1, Hd]  (greedy path)
    std::vector<float> dec_hidden_;          // [Hd] — joint input (greedy path)
    std::string        accumulated_text_;
    bool               stream_init_ = false;
    bool               eou_detected_ = false;  // Parakeet-EOU: <EOU> seen this stream
    float              preemph_prev_ = 0.0f;   // last raw sample, for cross-window pre-emphasis

    // ---- beam-search state (Config.beam_size > 1) ----
    std::vector<BeamHyp> beams_;   // carried across windows, like the greedy predictor state
    ContextGraph         ctx_;     // contextual-biasing automaton (empty = no biasing)
};

}  // namespace speech_core
