#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <string>
#include <vector>

namespace speech_core {

/// DeepFilterNet3 — real-time speech enhancement / noise cancellation.
/// Processes audio at 48 kHz using STFT + ERB filterbank + neural network.
/// Model size: ~2.1M parameters (~8 MB FP16).
class DeepFilterEnhancer : public EnhancerInterface {
public:
    struct Config {
        int fft_size    = 960;
        int hop_size    = 480;
        int erb_bands   = 32;
        int df_bins     = 96;   // deep-filtered frequency bins
        int df_order    = 5;    // filter taps
        int df_lookahead = 2;
        int freq_bins   = 481;  // fft_size / 2 + 1
        int sample_rate = 48000;
    };

    /// Canonical DSP tables are generated in-process. An optional legacy
    /// auxiliary file is checked for parity, but its matrices are never
    /// trusted for inference.
    explicit DeepFilterEnhancer(const std::string& model_path,
                                const std::string& auxiliary_path = {},
                                bool hw_accel = true);
    ~DeepFilterEnhancer() override;

    /// Enhance audio by removing noise.
    /// @param audio       Input PCM Float32 at 48 kHz
    /// @param length      Number of samples
    /// @param sample_rate Input sample rate (must be 48000)
    /// @param output      Pre-allocated output buffer (same length)
    void enhance(const float* audio, size_t length, int sample_rate,
                 float* output) override;

    int input_sample_rate() const override { return cfg_.sample_rate; }

private:
    void load_auxiliary(const std::string& path);

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
    Config cfg_;

    // Canonical libdf frontend data. The historical auxiliary path remains in
    // the constructor for bundle compatibility, but these values are derived
    // in-process so a transposed/normalized artifact cannot corrupt audio.
    std::vector<int> erb_widths_;
    std::vector<float> window_;
};

}  // namespace speech_core
