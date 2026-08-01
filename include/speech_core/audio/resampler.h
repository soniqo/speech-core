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
    friend class StreamingResampler;

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

/// Stateful windowed-sinc resampler for packetized live capture.
///
/// Unlike repeatedly calling Resampler::resample() for each device packet,
/// this preserves the fractional source position and filter history across
/// packet boundaries. That guarantees long-run sample conservation and avoids
/// a small clock drift that otherwise breaks timestamp-aligned AEC.
class StreamingResampler {
public:
    StreamingResampler(int from_rate, int to_rate);

    /// Append one source packet and return every output sample whose filter
    /// has enough future context. A small filter-delay tail remains buffered.
    std::vector<float> process(const float* input, size_t input_length);

    /// Emit the buffered tail using edge extension. Call once when capture
    /// ends; subsequent process() calls require reset().
    std::vector<float> flush();

    /// Discard filter and fractional-rate state after a capture discontinuity.
    void reset();

    int input_sample_rate() const { return from_rate_; }
    int output_sample_rate() const { return to_rate_; }
    uint64_t total_input_samples() const { return total_input_samples_; }
    uint64_t total_output_samples() const { return total_output_samples_; }

private:
    std::vector<float> produce(bool finishing);
    uint64_t target_output_count() const;
    void advance_source_position();
    void prune_input();

    int from_rate_ = 0;
    int to_rate_ = 0;
    Resampler::KernelTable kernel_{};
    std::vector<float> input_;
    uint64_t input_base_sample_ = 0;
    uint64_t total_input_samples_ = 0;
    uint64_t total_output_samples_ = 0;
    uint64_t next_source_sample_ = 0;
    uint64_t next_source_fraction_ = 0;
    bool flushed_ = false;
};

}  // namespace speech_core
