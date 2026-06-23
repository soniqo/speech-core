#include "speech_core/audio/fft.h"

#include "ooura_fft_engine.h"

#include <algorithm>
#include <vector>

namespace speech_core::audio {

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void fft_real(const float* input, size_t n,
              float* out_real, float* out_imag)
{
    if (n == 0) return;

    const size_t bins = n / 2 + 1;
    if (n == 1) {
        out_real[0] = input[0];
        out_imag[0] = 0.0f;
        return;
    }

    const size_t transform_size = next_pow2(n);
    const size_t transform_bins = OouraFftEngine::complex_size(transform_size);

    std::vector<float> padded(transform_size, 0.0f);
    std::copy(input, input + n, padded.begin());

    std::vector<float> re(transform_bins, 0.0f);
    std::vector<float> im(transform_bins, 0.0f);

    OouraFftEngine engine;
    engine.init(transform_size);
    engine.fft(padded.data(), re.data(), im.data());

    std::copy_n(re.data(), bins, out_real);
    std::copy_n(im.data(), bins, out_imag);
}

void ifft_real(const float* in_real, const float* in_imag, size_t n,
               float* output)
{
    if (n == 0) return;

    if (n == 1) {
        output[0] = in_real[0];
        return;
    }

    const size_t bins = n / 2 + 1;
    const size_t transform_size = next_pow2(n);
    const size_t transform_bins = OouraFftEngine::complex_size(transform_size);

    std::vector<float> re(transform_bins, 0.0f);
    std::vector<float> im(transform_bins, 0.0f);
    std::copy_n(in_real, bins, re.data());
    std::copy_n(in_imag, bins, im.data());

    std::vector<float> reconstructed(transform_size, 0.0f);

    OouraFftEngine engine;
    engine.init(transform_size);
    engine.ifft(reconstructed.data(), re.data(), im.data());

    std::copy_n(reconstructed.data(), n, output);
}

}  // namespace speech_core::audio
