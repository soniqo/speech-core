#pragma once

#include <cstddef>
#include <vector>

namespace speech_core::audio {

/// Thin adapter around Ooura's real DFT layout.
class OouraFftEngine {
public:
    OouraFftEngine() = default;

    /// Allocate work buffers for a real-valued FFT of `size` samples.
    ///
    /// `size` must be a power of two and at least 2.
    void init(size_t size);

    /// Transform real samples to a non-redundant complex spectrum.
    ///
    /// `re` and `im` must contain `complex_size(size)` elements.
    void fft(const float* data, float* re, float* im);

    /// Reconstruct real samples from a non-redundant complex spectrum.
    ///
    /// The inverse transform is scaled internally, so fft() followed by ifft()
    /// returns the original samples.
    void ifft(float* data, const float* re, const float* im);

    static size_t complex_size(size_t size);

private:
    OouraFftEngine(const OouraFftEngine&) = delete;
    OouraFftEngine& operator=(const OouraFftEngine&) = delete;

    size_t size_ = 0;
    std::vector<double> buffer_;
    std::vector<int> ip_;
    std::vector<double> w_;
};

}  // namespace speech_core::audio
