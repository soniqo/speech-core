#pragma once

#include <cstddef>
#include <vector>

namespace speech_core::audio {

/// Compute log-mel spectrogram from raw audio.
/// Returns flattened [num_mel_bins, num_frames] in channels-first layout
/// (row = mel bin, column = time frame).
///
/// Optional parameters (default to the original behaviour):
///   slaney_norm  — area-normalise each triangular filter by its bandwidth
///   log_floor    — additive floor before log: log(x + floor)
///   center       — pad signal by n_fft/2 on each side (reflect mode)
///   torch_stft_layout — frame exactly like torch.stft when
///       win_length < n_fft: frames are n_fft samples long
///       (num_frames = 1 + (len - n_fft)/hop, i.e. ~56 samples earlier
///       per frame than the legacy win_length slicing at 400/512) and
///       the Hann window is PERIODIC (denominator N, torch default)
///       instead of symmetric (N-1). Off by default: the legacy layout
///       is what the LiteRT wrappers (Nemotron/Parakeet paths) were
///       validated against; the Steno family opts in because its
///       training front-end is torch.stft and the layout offset was
///       measured flipping word-boundary tokens (mel corr 0.978 vs the
///       reference processor before, low bins diverging hardest).
///   center_pad_zeros — with center: pad with ZEROS (torch.stft
///       pad_mode="constant") instead of reflect. The Parakeet/NeMo
///       training front-end pads constant; reflect stays the default
///       for the paths validated against it.
///   symmetric_torch_window — with torch_stft_layout: build the Hann
///       window with the SYMMETRIC denominator (N-1, i.e.
///       torch.hann_window(periodic=False)) instead of periodic. The
///       Parakeet feature extractor uses periodic=False; Steno's uses
///       the periodic default.
std::vector<float> mel_spectrogram(
    const float* audio, size_t length,
    int sample_rate, int n_fft, int hop_length,
    int win_length, int num_mel_bins,
    bool slaney_norm = false,
    float log_floor = 1e-10f,
    bool center = false,
    bool torch_stft_layout = false,
    bool center_pad_zeros = false,
    bool symmetric_torch_window = false);

}  // namespace speech_core::audio
