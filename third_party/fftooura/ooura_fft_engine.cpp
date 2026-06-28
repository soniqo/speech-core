#include "ooura_fft_engine.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

void rdft(int n, int isgn, double* a, int* ip, double* w);

namespace speech_core::audio {

namespace {

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

}  // namespace

void OouraFftEngine::init(size_t size) {
    if (size < 2 || !is_power_of_two(size)) {
        throw std::invalid_argument("FFT size must be a power of two and at least 2");
    }

    if (size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("FFT size is too large for the Ooura FFT");
    }

    size_ = size;
    buffer_.assign(size_, 0.0);
    ip_.assign(size_ + 2, 0);
    w_.assign(size_ / 2, 0.0);
}

void OouraFftEngine::fft(const float* data, float* re, float* im) {
    if (size_ == 0) {
        throw std::logic_error("OouraFftEngine::init() must be called before fft()");
    }

    if (data == nullptr || re == nullptr || im == nullptr) {
        throw std::invalid_argument("OouraFftEngine::fft() received a null pointer");
    }

    std::copy(data, data + size_, buffer_.begin());

    rdft(static_cast<int>(size_), 1, buffer_.data(), ip_.data(), w_.data());

    const size_t half_size = size_ / 2;
    re[0] = static_cast<float>(buffer_[0]);
    im[0] = 0.0f;
    re[half_size] = static_cast<float>(buffer_[1]);
    im[half_size] = 0.0f;

    for (size_t i = 1; i < half_size; ++i) {
        re[i] = static_cast<float>(buffer_[2 * i]);
        im[i] = static_cast<float>(-buffer_[2 * i + 1]);
    }
}

void OouraFftEngine::ifft(float* data, const float* re, const float* im) {
    if (size_ == 0) {
        throw std::logic_error("OouraFftEngine::init() must be called before ifft()");
    }

    if (data == nullptr || re == nullptr || im == nullptr) {
        throw std::invalid_argument("OouraFftEngine::ifft() received a null pointer");
    }

    const size_t half_size = size_ / 2;
    buffer_[0] = re[0];
    buffer_[1] = re[half_size];

    for (size_t i = 1; i < half_size; ++i) {
        buffer_[2 * i] = re[i];
        buffer_[2 * i + 1] = -im[i];
    }

    rdft(static_cast<int>(size_), -1, buffer_.data(), ip_.data(), w_.data());

    const double scale = 2.0 / static_cast<double>(size_);
    for (size_t i = 0; i < size_; ++i) {
        data[i] = static_cast<float>(buffer_[i] * scale);
    }
}

size_t OouraFftEngine::complex_size(size_t size) {
    return size / 2 + 1;
}

}  // namespace speech_core::audio
