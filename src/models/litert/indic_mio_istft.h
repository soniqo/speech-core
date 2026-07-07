#pragma once

#include <cstddef>
#include <vector>

namespace speech_core {

/// Host-side inverse STFT for the Indic-Mio LiteRT audio decoder.
///
/// The decoder graph ships in "spec" shape (real/imag STFT frames) because
/// MioCodec's ISTFT head cannot be lowered to TFLite (complex tensors). This
/// reproduces the upstream "same"-padding ISTFT exactly (miocodec
/// module/istft_head.py): per-frame inverse rFFT × Hann window, overlap-add,
/// division by the window envelope, and (win-hop)/2 edge trim. kissfft
/// (third_party/kissfft) handles the non-power-of-two n_fft (1920 = 2^7·3·5;
/// the in-tree Ooura engine is radix-2 only).
///
/// Numerics are pinned by tests/data/indic_mio_istft_* fixtures generated
/// from the upstream torch implementation.
class IndicMioIstft {
public:
    /// n_fft is also the window length (Hann, periodic=True torch convention).
    IndicMioIstft(int n_fft, int hop_length);
    ~IndicMioIstft();

    IndicMioIstft(const IndicMioIstft&) = delete;
    IndicMioIstft& operator=(const IndicMioIstft&) = delete;

    int bins() const { return n_fft_ / 2 + 1; }

    /// Reconstruct audio from `frames` STFT columns.
    /// `real`/`imag` are [bins, frames] row-major (bin-major — the decoder
    /// graph's output layout [1, 961, T]). Returns
    /// (frames-1)*hop + n_fft - 2*((n_fft-hop)/2) samples.
    std::vector<float> synthesize(const float* real, const float* imag,
                                  int frames) const;

private:
    int n_fft_;
    int hop_;
    std::vector<float> window_;
    void* cfg_;  // kiss_fftr_cfg (inverse)
};

}  // namespace speech_core
