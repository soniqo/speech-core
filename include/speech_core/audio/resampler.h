#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Windowed-sinc resampler with precomputed filter kernels.
///
/// Uses a Blackman-windowed sinc filter for anti-aliased sample rate conversion.
/// Kernel tables are cached per (from_rate, to_rate) pair for repeated use.
///
/// Suitable for speech. For music or ultra-low-latency paths,
/// platform-specific implementations (vDSP, Oboe) may be preferred.
class Resampler {
public:
    /// Resample audio from one sample rate to another.
    /// @param input        Source samples
    /// @param input_length Number of source samples
    /// @param from_rate    Source sample rate in Hz
    /// @param to_rate      Target sample rate in Hz
    /// @return Resampled audio
    static std::vector<float> resample(
        const float* input, size_t input_length,
        int from_rate, int to_rate);

    /// Clear all cached filter kernels.
    static void clear_cache();

private:
    /// Filter half-width in output-domain taps (before scaling by ratio).
    static constexpr int kHalfWidth = 8;

    /// Number of sub-sample phases for polyphase interpolation.
    static constexpr int kTablePhases = 256;

    struct KernelTable {
        int half_width;           // actual half-width after ratio scaling
        int taps;                 // 2 * half_width + 1
        std::vector<float> table; // kTablePhases * taps
    };

    static const KernelTable& get_kernel(int from_rate, int to_rate);
    static KernelTable build_kernel(int from_rate, int to_rate);

    static std::mutex cache_mutex_;
    static std::unordered_map<uint64_t, KernelTable> cache_;
};

}  // namespace speech_core
