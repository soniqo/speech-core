#pragma once

#include <cstddef>
#include <vector>

namespace speech_core::deepfilter_dsp {

// Pure C++ signal-processing contract around the DeepFilterNet3 neural graph.
// Kept independent of ONNX Runtime so it can be regression-tested directly.
struct Config {
    int fft_size = 960;
    int hop_size = 480;
    int erb_bands = 32;
    int df_bins = 96;
    int df_order = 5;
    int df_lookahead = 2;
    int freq_bins = 481;
    int sample_rate = 48000;
    int min_erb_freqs = 2;
    float norm_tau = 1.0f;
};

std::vector<int> make_erb_widths(const Config& cfg);
std::vector<float> make_vorbis_window(int fft_size);
std::vector<float> make_mean_norm_state(int erb_bands);
std::vector<float> make_unit_norm_state(int df_bins);
float normalization_alpha(const Config& cfg);

int frame_count(size_t input_length, const Config& cfg);

// Streaming-compatible libdf analysis, evaluated as one batch. A zeroed
// analysis history is prepended and a full fft_size tail is appended.
void analyze(const float* audio, size_t length,
             const Config& cfg, const std::vector<float>& window,
             std::vector<float>& spec_real,
             std::vector<float>& spec_imag);

// Produces feat_erb [1,1,T,E] and channel-major feat_spec [1,2,T,F].
// Normalization state starts from the upstream libdf initial values for each
// batch invocation.
void compute_features(const std::vector<float>& spec_real,
                      const std::vector<float>& spec_imag,
                      int num_frames,
                      const Config& cfg,
                      const std::vector<int>& erb_widths,
                      std::vector<float>& feat_erb,
                      std::vector<float>& feat_spec);

// Fuses the graph outputs with the immutable noisy spectrum. erb_mask is
// [1,1,T,E]; df_coefs is [1,O,T,F,2].
void apply_network_output(const std::vector<float>& spec_real,
                          const std::vector<float>& spec_imag,
                          const float* erb_mask,
                          const float* df_coefs,
                          int num_frames,
                          const Config& cfg,
                          const std::vector<int>& erb_widths,
                          std::vector<float>& enhanced_real,
                          std::vector<float>& enhanced_imag);

// Streaming overlap-add synthesis followed by removal of the fft_size-hop
// algorithmic delay. Writes exactly output_length samples.
void synthesize(const std::vector<float>& spec_real,
                const std::vector<float>& spec_imag,
                int num_frames,
                const Config& cfg,
                const std::vector<float>& window,
                float* output,
                size_t output_length);

}  // namespace speech_core::deepfilter_dsp
