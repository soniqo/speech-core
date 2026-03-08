#include "speech_core/audio/audio_buffer.h"

#include <algorithm>
#include <cstring>

namespace speech_core {

AudioBuffer::AudioBuffer(size_t capacity)
    : buffer_(capacity), capacity_(capacity) {}

size_t AudioBuffer::write(const float* samples, size_t count) {
    size_t w = write_pos_.load(std::memory_order_relaxed);
    size_t r = read_pos_.load(std::memory_order_acquire);

    size_t avail = capacity_ - (w - r);
    size_t to_write = std::min(count, avail);

    for (size_t i = 0; i < to_write; i++) {
        buffer_[(w + i) % capacity_] = samples[i];
    }

    write_pos_.store(w + to_write, std::memory_order_release);
    return to_write;
}

size_t AudioBuffer::read(float* output, size_t count) {
    size_t r = read_pos_.load(std::memory_order_relaxed);
    size_t w = write_pos_.load(std::memory_order_acquire);

    size_t avail = w - r;
    size_t to_read = std::min(count, avail);

    for (size_t i = 0; i < to_read; i++) {
        output[i] = buffer_[(r + i) % capacity_];
    }

    read_pos_.store(r + to_read, std::memory_order_release);
    return to_read;
}

size_t AudioBuffer::available() const {
    size_t w = write_pos_.load(std::memory_order_acquire);
    size_t r = read_pos_.load(std::memory_order_acquire);
    return w - r;
}

void AudioBuffer::clear() {
    read_pos_.store(write_pos_.load(std::memory_order_relaxed),
                    std::memory_order_release);
}

}  // namespace speech_core
