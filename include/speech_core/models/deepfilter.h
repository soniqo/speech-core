#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// Mono RuntimeParams::default() from DeepFilterNet's Rust DfTract runtime.
/// The upstream CLI uses different thresholds (-15 / 35 / 35 dB).
struct DeepFilterRustStreamingOptions {
    float min_snr_db = -10.0f;
    float max_erb_snr_db = 30.0f;
    float max_df_snr_db = 20.0f;
    float post_filter_beta = 0.0f;  // zero disables the post-filter
    float attenuation_limit_db = 100.0f;  // abs(db) >= 100 means unlimited
};

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
    /// In the default compatibility mode, the first stream_latency_samples()
    /// output samples are zero; subsequent samples match complete-buffer
    /// enhancement after this fixed delay. enable_rust_streaming() selects the
    /// upstream runtime's stage selection and startup behavior instead.
    /// Default mode uses the published v0.5.6 FP32 model and ORT >= 1.18 on CPU.
    /// Other batch models remain usable with enhance() but fail explicitly in
    /// default streaming mode; Rust mode uses its separate official bundle.
    void enhance_stream(const float* audio, size_t length, int sample_rate,
                        float* output) override;

    /// Opt into the official Rust streaming behavior. model_directory contains
    /// the pinned official enc.onnx / erb_dec.onnx / df_dec.onnx bundle, which
    /// includes the local-SNR head missing from the batch model. Initializes
    /// CPU sessions now and starts a fresh stream; rejected configurations
    /// preserve an existing stream. Offline enhance() remains unchanged.
    void enable_rust_streaming(const std::string& model_directory,
                              const DeepFilterRustStreamingOptions& options = {});

    /// Return to offline-compatible streaming and discard capture history.
    void disable_rust_streaming();

    /// Emit the delayed tail once, preserving the complete input recording.
    /// Append this to enhance_stream() output, then remove the initial delay
    /// for an aligned recording. Call reset() before starting another stream.
    std::vector<float> flush_stream();

    /// Discard buffered audio and reset normalization, GRUs, and convolutions.
    void reset() override;
    /// Alignment delay: 40 ms normally, 10 ms for Rust's zero-dB bypass.
    size_t stream_latency_samples() const;

    int input_sample_rate() const override { return cfg_.sample_rate; }

private:
    void load_auxiliary(const std::string& path);

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
    std::string model_path_;
    struct StreamState;
    std::unique_ptr<StreamState> stream_;
    struct RustStreamState;
    std::unique_ptr<RustStreamState> rust_stream_;
    Config cfg_;

    // Canonical libdf frontend data. The historical auxiliary path remains in
    // the constructor for bundle compatibility, but these values are derived
    // in-process so a transposed/normalized artifact cannot corrupt audio.
    std::vector<int> erb_widths_;
    std::vector<float> window_;
};

}  // namespace speech_core
