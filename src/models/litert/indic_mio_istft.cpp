#include "indic_mio_istft.h"

#include "kiss_fftr.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace speech_core {

IndicMioIstft::IndicMioIstft(int n_fft, int hop_length)
    : n_fft_(n_fft), hop_(hop_length) {
    if (n_fft <= 0 || hop_length <= 0 || n_fft % 2 != 0 || hop_length > n_fft) {
        throw std::runtime_error("IndicMioIstft: invalid n_fft/hop");
    }
    // torch.hann_window(win, periodic=True): 0.5 - 0.5*cos(2*pi*i/win).
    window_.resize(static_cast<size_t>(n_fft));
    const double two_pi = 2.0 * 3.14159265358979323846;
    for (int i = 0; i < n_fft; ++i) {
        window_[static_cast<size_t>(i)] =
            static_cast<float>(0.5 - 0.5 * std::cos(two_pi * i / n_fft));
    }
    cfg_ = kiss_fftr_alloc(n_fft, /*inverse_fft=*/1, nullptr, nullptr);
    if (!cfg_) throw std::runtime_error("IndicMioIstft: kiss_fftr_alloc failed");
}

IndicMioIstft::~IndicMioIstft() {
    kiss_fftr_free(static_cast<kiss_fftr_cfg>(cfg_));
}

std::vector<float> IndicMioIstft::synthesize(const float* real, const float* imag,
                                             int frames) const {
    if (frames <= 0) return {};
    const int nb = bins();
    const int pad = (n_fft_ - hop_) / 2;
    const size_t out_size = static_cast<size_t>(frames - 1) * hop_ + n_fft_;

    std::vector<float> y(out_size, 0.0f);
    std::vector<float> envelope(out_size, 0.0f);
    std::vector<kiss_fft_cpx> spec(static_cast<size_t>(nb));
    std::vector<kiss_fft_scalar> time(static_cast<size_t>(n_fft_));

    // Envelope: overlap-added squared window (constant per frame count).
    for (int t = 0; t < frames; ++t) {
        const size_t base = static_cast<size_t>(t) * hop_;
        for (int i = 0; i < n_fft_; ++i) {
            const float w = window_[static_cast<size_t>(i)];
            envelope[base + i] += w * w;
        }
    }

    for (int t = 0; t < frames; ++t) {
        // Column t of the bin-major [nb, frames] layout.
        for (int b = 0; b < nb; ++b) {
            const size_t idx = static_cast<size_t>(b) * frames + t;
            spec[static_cast<size_t>(b)].r = real[idx];
            spec[static_cast<size_t>(b)].i = imag[idx];
        }
        kiss_fftri(static_cast<kiss_fftr_cfg>(cfg_), spec.data(), time.data());
        // kissfft's inverse real FFT is unnormalized (scaled by n_fft);
        // torch.fft.irfft(norm="backward") divides by n — match it here.
        const float inv_n = 1.0f / static_cast<float>(n_fft_);
        const size_t base = static_cast<size_t>(t) * hop_;
        for (int i = 0; i < n_fft_; ++i) {
            y[base + i] += time[static_cast<size_t>(i)] * inv_n *
                           window_[static_cast<size_t>(i)];
        }
    }

    // Trim (win-hop)/2 from both ends and normalize by the window envelope
    // (the upstream asserts envelope > 1e-11 inside the kept region; with a
    // Hann window and hop <= n_fft/2 that always holds after the edge trim).
    std::vector<float> out(out_size - 2 * static_cast<size_t>(pad));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = y[i + pad] / envelope[i + pad];
    }
    return out;
}

}  // namespace speech_core
