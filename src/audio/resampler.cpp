#include "speech_core/audio/resampler.h"

#include <algorithm>
#include <cmath>

namespace speech_core {

std::mutex Resampler::cache_mutex_;
std::unordered_map<uint64_t, Resampler::KernelTable> Resampler::cache_;

// Blackman window on [-half, +half]
static inline double blackman(double x, double half) {
    if (x <= -half || x >= half) return 0.0;
    double n = (x / half + 1.0) * 0.5;  // normalize to [0, 1]
    return 0.42 - 0.5 * std::cos(2.0 * M_PI * n)
                + 0.08 * std::cos(4.0 * M_PI * n);
}

static inline double sinc(double x) {
    if (std::abs(x) < 1e-9) return 1.0;
    double px = M_PI * x;
    return std::sin(px) / px;
}

Resampler::KernelTable Resampler::build_kernel(int from_rate, int to_rate) {
    double cutoff = std::min(1.0, static_cast<double>(to_rate) / from_rate);
    int hw = static_cast<int>(std::ceil(kHalfWidth / cutoff));
    int taps = 2 * hw + 1;

    KernelTable kt;
    kt.half_width = hw;
    kt.taps = taps;
    kt.table.resize(kTablePhases * taps);

    for (int phase = 0; phase < kTablePhases; phase++) {
        double frac = static_cast<double>(phase) / kTablePhases;
        double wsum = 0.0;
        int base = phase * taps;

        for (int j = -hw; j <= hw; j++) {
            double d = frac - j;
            double w = sinc(d * cutoff) * blackman(d, hw + 0.5) * cutoff;
            kt.table[base + j + hw] = static_cast<float>(w);
            wsum += w;
        }

        // Normalize so filter taps sum to 1.0
        if (wsum != 0.0) {
            float inv = static_cast<float>(1.0 / wsum);
            for (int j = 0; j < taps; j++) {
                kt.table[base + j] *= inv;
            }
        }
    }

    return kt;
}

const Resampler::KernelTable& Resampler::get_kernel(int from_rate, int to_rate) {
    uint64_t key = (static_cast<uint64_t>(from_rate) << 32)
                 | static_cast<uint64_t>(static_cast<uint32_t>(to_rate));

    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    auto [inserted, _] = cache_.emplace(key, build_kernel(from_rate, to_rate));
    return inserted->second;
}

std::vector<float> Resampler::resample(
    const float* input, size_t input_length,
    int from_rate, int to_rate) {

    if (from_rate == to_rate || input_length == 0) {
        return {input, input + input_length};
    }

    const auto& kt = get_kernel(from_rate, to_rate);

    double ratio = static_cast<double>(from_rate) / static_cast<double>(to_rate);
    size_t output_length = static_cast<size_t>(
        static_cast<double>(input_length) / ratio);

    std::vector<float> output(output_length);

    for (size_t i = 0; i < output_length; i++) {
        double src_pos = static_cast<double>(i) * ratio;
        int center = static_cast<int>(src_pos);
        double frac = src_pos - center;

        // Look up the nearest precomputed phase
        int phase = static_cast<int>(frac * kTablePhases);
        if (phase >= kTablePhases) phase = kTablePhases - 1;

        const float* kernel = &kt.table[phase * kt.taps];
        float sum = 0.0f;

        for (int j = -kt.half_width; j <= kt.half_width; j++) {
            int idx = center + j;
            if (idx < 0) idx = 0;
            else if (idx >= static_cast<int>(input_length)) idx = static_cast<int>(input_length) - 1;
            sum += input[idx] * kernel[j + kt.half_width];
        }

        output[i] = sum;
    }

    return output;
}

void Resampler::clear_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

}  // namespace speech_core
