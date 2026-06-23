#pragma once

#include <cstddef>

namespace speech_core::audio {

/// Ooura-backed real FFT adapter.
/// Operates on real signals — returns complex spectrum [0..N/2].
/// Non-power-of-two sizes are zero-padded internally to the next power of two.

void fft_real(const float* input, size_t n,
              float* out_real, float* out_imag);

void ifft_real(const float* in_real, const float* in_imag, size_t n,
               float* output);

}  // namespace speech_core::audio
