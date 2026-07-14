#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <string>
#include <vector>

namespace speech_core {

/// Pyannote speaker segmentation 3.0 via ONNX Runtime.
///
/// Matches soniqo/Pyannote-Segmentation-ONNX:
///   Input:  audio      [1, 1, 160000]  — one 10-second window at 16 kHz
///   Output: posteriors [1, F, 7]       — per-frame powerset LOG-probabilities
///
/// Differs from LiteRTPyannoteSegmentation in more than the runtime. The LiteRT
/// bundle decomposes this model into 1-second chunks with the LSTM state carried
/// across them by hand (56 frames per chunk, 560 per window). This graph runs the
/// LSTM internally over the whole window and emits F = 589 frames (~17.0 ms each).
/// F is therefore READ FROM THE GRAPH at construction, never assumed: a hardcoded
/// frame count would silently shift every segment's timestamps by a few percent —
/// transcripts would still look right while speaker boundaries drifted.
///
/// The single-call-per-window shape is also why this is ~8.5x faster than the
/// LiteRT path on CPU (18.8 ms vs 160.3 ms per window): the chunked bundle paid
/// ten invocations plus state marshalling per window.
///
/// Not thread-safe — one per worker.
class OnnxPyannoteSegmentation : public SegmentationInterface {
public:
    struct Config {
        int   sample_rate          = 16000;
        int   num_powerset_classes = 7;
        int   max_local_speakers   = 3;
        float window_duration      = 10.0f;  // seconds per inference window
        float window_step          = 5.0f;   // hop between windows
    };

    explicit OnnxPyannoteSegmentation(const std::string& model_path,
                                      bool hw_accel = true);
    ~OnnxPyannoteSegmentation() override;

    OnnxPyannoteSegmentation(const OnnxPyannoteSegmentation&) = delete;
    OnnxPyannoteSegmentation& operator=(const OnnxPyannoteSegmentation&) = delete;

    std::vector<SegmentationWindow> segment(
        const float* audio, size_t length, int sample_rate) override;

    int input_sample_rate() const override { return cfg_.sample_rate; }
    int max_local_speakers() const override { return cfg_.max_local_speakers; }

    /// Frames the graph emits per window — resolved from the model, not assumed.
    int frames_per_window() const { return frames_per_window_; }

private:
    /// Powerset {∅,{1},{2},{3},{1,2},{1,3},{2,3}} -> per-speaker activity.
    /// The graph emits log-probabilities; the max-subtract/exp/normalise below is
    /// idempotent on those (it recovers the same probabilities), which is why this
    /// is identical to the LiteRT decoder even though the input differs.
    void decode_powerset(const float* posteriors, int num_frames,
                         std::vector<float>& speaker_activity) const;

    const OrtApi* api_     = nullptr;
    OrtSession*   session_ = nullptr;

    Config cfg_;
    int    window_samples_    = 0;
    int    frames_per_window_ = 0;
};

}  // namespace speech_core
