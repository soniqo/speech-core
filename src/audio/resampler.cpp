#include "speech_core/audio/resampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace speech_core {

// MSVC's <cmath> doesn't define M_PI without _USE_MATH_DEFINES.
static constexpr double kPi = 3.14159265358979323846;

std::mutex Resampler::cache_mutex_;
std::unordered_map<uint64_t, Resampler::KernelTable> Resampler::cache_;

// Blackman window on [-half, +half]
static inline double blackman(double x, double half) {
    if (x <= -half || x >= half) return 0.0;
    double n = (x / half + 1.0) * 0.5;  // normalize to [0, 1]
    return 0.42 - 0.5 * std::cos(2.0 * kPi * n)
                + 0.08 * std::cos(4.0 * kPi * n);
}

static inline double sinc(double x) {
    if (std::abs(x) < 1e-9) return 1.0;
    double px = kPi * x;
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

StreamingResampler::StreamingResampler(int from_rate, int to_rate)
    : from_rate_(from_rate), to_rate_(to_rate) {
    if (from_rate <= 0 || to_rate <= 0) {
        throw std::invalid_argument(
            "Streaming resampler rates must be positive");
    }
    if (from_rate != to_rate) {
        // Keep an owned copy so Resampler::clear_cache() cannot invalidate a
        // live packetized stream.
        kernel_ = Resampler::get_kernel(from_rate, to_rate);
    }
}

std::vector<float> StreamingResampler::process(
    const float* input, size_t input_length) {
    if (!input && input_length != 0) {
        throw std::invalid_argument(
            "Streaming resampler input must not be null");
    }
    if (flushed_) {
        throw std::logic_error(
            "Streaming resampler must be reset after flush");
    }
    if (input_length == 0) return {};
    for (size_t index = 0; index < input_length; ++index) {
        if (!std::isfinite(input[index])) {
            throw std::invalid_argument(
                "Streaming resampler input contains NaN or infinity");
        }
    }
    if (total_input_samples_
        > std::numeric_limits<uint64_t>::max() - input_length) {
        throw std::overflow_error(
            "Streaming resampler input sample counter overflowed");
    }
    input_.insert(input_.end(), input, input + input_length);
    total_input_samples_ += static_cast<uint64_t>(input_length);
    return produce(false);
}

std::vector<float> StreamingResampler::flush() {
    if (flushed_) return {};
    flushed_ = true;
    return produce(true);
}

void StreamingResampler::reset() {
    input_.clear();
    input_base_sample_ = 0;
    total_input_samples_ = 0;
    total_output_samples_ = 0;
    next_source_sample_ = 0;
    next_source_fraction_ = 0;
    flushed_ = false;
}

uint64_t StreamingResampler::target_output_count() const {
    // floor(total_input_samples * to_rate / from_rate), written without a
    // potentially overflowing multiplication so multi-day captures remain
    // exact on MSVC as well as compilers with 128-bit integers.
    const uint64_t from = static_cast<uint64_t>(from_rate_);
    const uint64_t to = static_cast<uint64_t>(to_rate_);
    const uint64_t whole = total_input_samples_ / from;
    const uint64_t remainder = total_input_samples_ % from;
    if (whole > std::numeric_limits<uint64_t>::max() / to) {
        throw std::overflow_error(
            "Streaming resampler output sample counter overflowed");
    }
    return whole * to + remainder * to / from;
}

void StreamingResampler::advance_source_position() {
    next_source_fraction_ += static_cast<uint64_t>(from_rate_);
    next_source_sample_ +=
        next_source_fraction_ / static_cast<uint64_t>(to_rate_);
    next_source_fraction_ %= static_cast<uint64_t>(to_rate_);
}

std::vector<float> StreamingResampler::produce(bool finishing) {
    if (from_rate_ == to_rate_) {
        const size_t offset = static_cast<size_t>(
            total_output_samples_ - input_base_sample_);
        if (offset > input_.size()) {
            throw std::logic_error(
                "Streaming resampler identity state is inconsistent");
        }
        std::vector<float> output(
            input_.begin() + static_cast<std::ptrdiff_t>(offset),
            input_.end());
        total_output_samples_ += output.size();
        input_.clear();
        input_base_sample_ = total_input_samples_;
        next_source_sample_ = total_input_samples_;
        return output;
    }

    std::vector<float> output;
    const uint64_t final_count = target_output_count();
    while (total_output_samples_ < final_count) {
        if (!finishing
            && next_source_sample_
                + static_cast<uint64_t>(kernel_.half_width)
                >= total_input_samples_) {
            break;
        }
        const uint64_t phase_numerator =
            next_source_fraction_
            * static_cast<uint64_t>(Resampler::kTablePhases);
        int phase = static_cast<int>(
            phase_numerator / static_cast<uint64_t>(to_rate_));
        phase = std::min(phase, Resampler::kTablePhases - 1);
        const float* taps =
            &kernel_.table[static_cast<size_t>(phase) * kernel_.taps];
        float sum = 0.0f;
        for (int tap = -kernel_.half_width;
             tap <= kernel_.half_width; ++tap) {
            int64_t absolute =
                static_cast<int64_t>(next_source_sample_) + tap;
            absolute = std::max<int64_t>(0, absolute);
            absolute = std::min<int64_t>(
                absolute,
                static_cast<int64_t>(total_input_samples_) - 1);
            if (absolute < static_cast<int64_t>(input_base_sample_)) {
                throw std::logic_error(
                    "Streaming resampler discarded required filter history");
            }
            const uint64_t relative =
                static_cast<uint64_t>(absolute)
                - input_base_sample_;
            if (relative >= input_.size()) {
                throw std::logic_error(
                    "Streaming resampler is missing required future input");
            }
            sum += input_[static_cast<size_t>(relative)]
                * taps[tap + kernel_.half_width];
        }
        output.push_back(sum);
        ++total_output_samples_;
        advance_source_position();
    }
    prune_input();
    return output;
}

void StreamingResampler::prune_input() {
    if (input_.empty() || from_rate_ == to_rate_) return;
    const uint64_t history = static_cast<uint64_t>(kernel_.half_width);
    const uint64_t keep_from =
        next_source_sample_ > history
        ? next_source_sample_ - history : 0;
    if (keep_from <= input_base_sample_) return;
    const uint64_t requested = keep_from - input_base_sample_;
    const size_t remove = static_cast<size_t>(
        std::min<uint64_t>(requested, input_.size()));
    input_.erase(
        input_.begin(),
        input_.begin() + static_cast<std::ptrdiff_t>(remove));
    input_base_sample_ += remove;
}

}  // namespace speech_core
