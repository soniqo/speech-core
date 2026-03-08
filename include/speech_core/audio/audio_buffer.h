#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace speech_core {

/// Lock-free single-producer single-consumer ring buffer for streaming audio.
///
/// Producer (mic thread) writes samples, consumer (pipeline thread) reads.
/// Fixed capacity, overwrites oldest data on overflow.
class AudioBuffer {
public:
    /// @param capacity  Maximum number of float samples the buffer can hold
    explicit AudioBuffer(size_t capacity);

    /// Write samples into the buffer. Thread-safe for single producer.
    /// @param samples  PCM Float32 samples
    /// @param count    Number of samples
    /// @return Number of samples actually written (< count if buffer full and not overwriting)
    size_t write(const float* samples, size_t count);

    /// Read samples from the buffer. Thread-safe for single consumer.
    /// @param output  Destination buffer
    /// @param count   Maximum number of samples to read
    /// @return Number of samples actually read
    size_t read(float* output, size_t count);

    /// Number of samples available to read.
    size_t available() const;

    /// Total capacity in samples.
    size_t capacity() const { return capacity_; }

    /// Clear all buffered data.
    void clear();

private:
    std::vector<float> buffer_;
    size_t capacity_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
};

}  // namespace speech_core
