#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/litert_engine.h"

#include <string>
#include <vector>

namespace speech_core {

/// Pyannote speaker segmentation 3.0 via LiteRT — streaming mode.
///
/// Matches soniqo/Pyannote-Segmentation-LiteRT:
///   Input 0:  audio      [1, 1, 16000]   — one 1-second chunk at 16 kHz
///   Input 1:  lstm_state [2, 8, 1, 128]  — carried across chunks
///   Output 0: lstm_state [2, 8, 1, 128]
///   Output 1: posteriors [1, 56, 7]
///
/// A 10-second window is 10 consecutive 1-s chunks with state carried between
/// calls; state is reset at each window start, then the window slides by
/// window_step. Not thread-safe — one per worker.
class LiteRTPyannoteSegmentation : public SegmentationInterface {
public:
    struct Config {
        int   sample_rate          = 16000;
        int   chunk_samples        = 16000;  // 1 s @ 16 kHz
        int   frames_per_chunk     = 56;     // per config.json
        int   num_powerset_classes = 7;
        int   max_local_speakers   = 3;
        int   chunks_per_window    = 10;     // 10-s window
        float window_duration      = 10.0f;
        float window_step          = 5.0f;
    };

    static constexpr int kStateDim0 = 2;
    static constexpr int kStateDim1 = 8;
    static constexpr int kStateDim2 = 1;
    static constexpr int kStateDim3 = 128;
    static constexpr int kStateSize = kStateDim0 * kStateDim1 * kStateDim2 * kStateDim3;

    explicit LiteRTPyannoteSegmentation(const std::string& model_path, bool hw_accel = true);
    ~LiteRTPyannoteSegmentation() override;

    std::vector<SegmentationWindow> segment(
        const float* audio, size_t length, int sample_rate) override;
    int input_sample_rate() const override { return cfg_.sample_rate; }
    int max_local_speakers() const override { return cfg_.max_local_speakers; }

    const Config& config() const { return cfg_; }

private:
    /// Run one 1-s chunk: append its 56 frames of posteriors to
    /// `window_posteriors` and advance state_. Caller resets state_ per window.
    void process_chunk(const float* audio, size_t length,
                       std::vector<float>& window_posteriors);

    void decode_powerset(const float* posteriors, int num_frames,
                         std::vector<float>& speaker_activity);

    LiteRtModel         model_    = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
    Config cfg_;
    std::vector<float> state_;
    std::vector<float> chunk_buffer_;
};

}  // namespace speech_core
