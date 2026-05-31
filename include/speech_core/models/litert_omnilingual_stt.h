#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <string>
#include <vector>

namespace speech_core {

/// Meta Omnilingual ASR CTC-300M via LiteRT.
///
/// Single-model CTC:
///   Input:  raw waveform [1, S] f32, z-score normalised
///   Output: logits [1, T, vocab] CTC logits at 50 Hz (320× downsample)
/// Uses a SentencePiece (.model) tokenizer for decode. The fixed chunk length
/// and vocab size are read from the model's output layout at construction.
/// Not thread-safe — one per worker.
class LiteRTOmnilingualStt : public STTInterface {
public:
    struct Config {
        int sample_rate       = 16000;
        int max_audio_samples = 160000;  // refined from the model at load
        int frame_rate        = 50;      // 320× downsample
        int vocab_size        = 10288;   // refined from the model at load
        int blank_id          = 0;       // CTC blank is typically 0
    };

    LiteRTOmnilingualStt(const std::string& model_path,
                         const std::string& tokenizer_path,
                         bool hw_accel = true);
    ~LiteRTOmnilingualStt() override;

    TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }

private:
    /// CTC greedy decode: per-frame argmax, collapse repeats, drop blanks.
    std::string ctc_decode(const float* logits, int num_frames);

    /// Minimal SentencePiece .model parse → id → piece table.
    bool load_tokenizer(const std::string& path);

    LiteRtModel         model_    = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
    Config cfg_;
    int frames_per_chunk_ = 0;  // T from the model output layout

    // Input tensor shape pinned at construction via the non-strict resize API,
    // so the runtime doesn't reject our buffer for missing dims_signature.
    static constexpr int kMaxRank = 4;
    int input_rank_ = 0;
    int input_dims_[kMaxRank] = {0};

    std::vector<std::string> id_to_piece_;
};

}  // namespace speech_core
