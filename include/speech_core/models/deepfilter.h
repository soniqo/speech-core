#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// DeepFilterNet3 — offline and streaming speech enhancement.
/// Processes audio at 48 kHz using STFT + ERB filterbank + neural network.
/// Model size: ~2.1M parameters (~8.2 MiB FP32).
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

    /// Enhance one complete, independent recording with delay compensation.
    /// Does not change the state used by enhance_stream().
    /// @param audio       Input PCM Float32 at 48 kHz
    /// @param length      Number of samples
    /// @param sample_rate Input sample rate (must be 48000)
    /// @param output      Pre-allocated output buffer (same length)
    void enhance(const float* audio, size_t length, int sample_rate,
                 float* output) override;

    /// Enhance consecutive capture packets with persistent neural/DSP state.
    /// Accepts arbitrary packet lengths, including in-place audio/output.
    /// The first stream_latency_samples() output samples are zero; subsequent
    /// samples match complete-buffer enhancement after this fixed delay.
    /// Uses the published v0.5.6 FP32 model and ORT >= 1.18 on CPU. Other batch
    /// models remain usable with enhance() but fail explicitly here.
    void enhance_stream(const float* audio, size_t length, int sample_rate,
                        float* output) override;

    /// Emit the delayed tail once, preserving the complete input recording.
    /// Append this to enhance_stream() output, then remove the initial delay
    /// for an aligned recording. Call reset() before starting another stream.
    std::vector<float> flush_stream();

    /// Discard buffered audio and reset normalization, GRUs, and convolutions.
    void reset() override;
    size_t stream_latency_samples() const { return 4 * static_cast<size_t>(cfg_.hop_size); }

    int input_sample_rate() const override { return cfg_.sample_rate; }

private:
    void load_auxiliary(const std::string& path);

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
    std::string model_path_;
    struct StreamState;
    std::unique_ptr<StreamState> stream_;
    Config cfg_;

    // Canonical libdf frontend data. The historical auxiliary path remains in
    // the constructor for bundle compatibility, but these values are derived
    // in-process so a transposed/normalized artifact cannot corrupt audio.
    std::vector<int> erb_widths_;
    std::vector<float> window_;
};

}  // namespace speech_core
