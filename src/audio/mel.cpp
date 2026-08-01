#include "speech_core/audio/mel.h"

#include "speech_core/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace speech_core::audio {

// MSVC's <cmath> doesn't define M_PI without _USE_MATH_DEFINES.
static constexpr float kPi = 3.14159265358979323846f;

// HTK mel scale (used when slaney_norm=false).
static float htk_hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
static float htk_mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Slaney mel scale (used when slaney_norm=true):
//   Linear below 1000 Hz:  mel = 3 * f / 200
//   Log above 1000 Hz:     mel = 15 + 27 * log(f/1000) / log(6.4)
static constexpr float kSlaneyBreakHz = 1000.0f;
static constexpr float kSlaneyBreakMel = 15.0f;      // 3 * 1000 / 200
static const float kSlaneyLogStep = 27.0f / std::log(6.4f);  // ≈ 14.536

static float slaney_hz_to_mel(float hz) {
    if (hz < kSlaneyBreakHz)
        return 3.0f * hz / 200.0f;
    return kSlaneyBreakMel + std::log(hz / kSlaneyBreakHz) * kSlaneyLogStep;
}
static float slaney_mel_to_hz(float mel) {
    if (mel < kSlaneyBreakMel)
        return 200.0f * mel / 3.0f;
    return kSlaneyBreakHz * std::exp((mel - kSlaneyBreakMel) / kSlaneyLogStep);
}

static std::vector<float> mel_filterbank(
    int num_mel_bins, int n_fft, int sample_rate, bool slaney_norm)
{
    int num_bins = n_fft / 2 + 1;

    // Choose mel scale: Slaney (torchaudio default) when slaney_norm is on,
    // HTK otherwise (backward compat).
    auto hz2mel = slaney_norm ? slaney_hz_to_mel : htk_hz_to_mel;
    auto mel2hz = slaney_norm ? slaney_mel_to_hz : htk_mel_to_hz;

    float mel_low = hz2mel(0.0f);
    float mel_high = hz2mel(static_cast<float>(sample_rate) / 2.0f);

    std::vector<float> mel_points(num_mel_bins + 2);
    // Hz centres of each mel point (for Slaney norm later).
    std::vector<float> hz_points(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; i++) {
        float mel = mel_low + (mel_high - mel_low) * i / (num_mel_bins + 1);
        hz_points[i] = mel2hz(mel);
    }

    // Convert to FFT bin indices
    std::vector<float> bin_freqs(num_mel_bins + 2);
    for (int i = 0; i < num_mel_bins + 2; i++) {
        bin_freqs[i] = hz_points[i] * n_fft / sample_rate;
    }

    // Triangular filters [num_mel_bins * num_bins]
    std::vector<float> fb(num_mel_bins * num_bins, 0.0f);
    if (slaney_norm) {
        // Construct in double precision in Hz space, exactly as
        // librosa.filters.mel does (float64 ramps, cast at the end). The
        // float32 bin-space construction below shifts triangle edges by
        // ~1e-3 bins, which the log amplifies to 0.1..0.3 in near-floor
        // top bins — measured against the Parakeet reference extractor.
        // Init-time only; the HTK path keeps the legacy arithmetic so the
        // LiteRT-validated outputs stay byte-stable.
        std::vector<double> hz_d(num_mel_bins + 2);
        for (int i = 0; i < num_mel_bins + 2; i++) {
            hz_d[i] = static_cast<double>(hz_points[i]);
        }
        const double bin_hz = static_cast<double>(sample_rate)
                              / static_cast<double>(n_fft);
        for (int m = 0; m < num_mel_bins; m++) {
            const double left = hz_d[m];
            const double center = hz_d[m + 1];
            const double right = hz_d[m + 2];
            const double enorm = (right > left) ? 2.0 / (right - left) : 0.0;
            for (int f = 0; f < num_bins; f++) {
                const double f_hz = f * bin_hz;
                double w = 0.0;
                if (f_hz >= left && f_hz <= center && center > left) {
                    w = (f_hz - left) / (center - left);
                } else if (f_hz > center && f_hz <= right && right > center) {
                    w = (right - f_hz) / (right - center);
                }
                fb[m * num_bins + f] = static_cast<float>(w * enorm);
            }
        }
        return fb;
    }
    for (int m = 0; m < num_mel_bins; m++) {
        float left = bin_freqs[m];
        float center = bin_freqs[m + 1];
        float right = bin_freqs[m + 2];

        for (int f = 0; f < num_bins; f++) {
            float ff = static_cast<float>(f);
            if (ff >= left && ff <= center && center > left) {
                fb[m * num_bins + f] = (ff - left) / (center - left);
            } else if (ff > center && ff <= right && right > center) {
                fb[m * num_bins + f] = (right - ff) / (right - center);
            }
        }
    }
    return fb;
}

std::vector<float> mel_spectrogram(
    const float* audio, size_t length,
    int sample_rate, int n_fft, int hop_length,
    int win_length, int num_mel_bins,
    bool slaney_norm, float log_floor, bool center,
    bool torch_stft_layout, bool center_pad_zeros,
    bool symmetric_torch_window)
{
    // Optional center padding: pad signal by n_fft/2 on each side. Reflect
    // mode matches torchaudio center=True defaults; center_pad_zeros
    // matches torch.stft(pad_mode="constant") — the Parakeet/NeMo
    // training front-end.
    std::vector<float> padded;
    const float* sig = audio;
    size_t sig_len = length;

    if (center) {
        int pad = n_fft / 2;
        sig_len = length + 2 * static_cast<size_t>(pad);
        padded.assign(sig_len, 0.0f);

        std::copy(audio, audio + length, padded.begin() + pad);
        if (!center_pad_zeros) {
            // Left reflect padding: padded[pad-1-i] = audio[i+1]
            for (int i = 0; i < pad; ++i) {
                int src = std::min(i + 1, static_cast<int>(length) - 1);
                padded[pad - 1 - i] = audio[src];
            }
            // Right reflect padding
            for (int i = 0; i < pad; ++i) {
                int src = std::max(static_cast<int>(length) - 2 - i, 0);
                padded[pad + static_cast<int>(length) + i] = audio[src];
            }
        }
        sig = padded.data();
    }

    int num_bins = n_fft / 2 + 1;
    // torch.stft frames are n_fft samples long regardless of win_length
    // (the window is applied inside); the legacy layout slices win_length
    // samples, which at 400/512 starts every frame 56 samples later and
    // yields one extra frame.
    const int frame_span = torch_stft_layout ? n_fft : win_length;
    int num_frames = static_cast<int>((sig_len - static_cast<size_t>(frame_span))
                                      / hop_length) + 1;
    if (num_frames <= 0) return {};

    auto fb = mel_filterbank(num_mel_bins, n_fft, sample_rate, slaney_norm);

    // Hann window. torch.hann_window default is PERIODIC (denominator N);
    // the legacy symmetric form (N-1) stays for the LiteRT-validated paths,
    // and symmetric_torch_window selects it under the torch layout too
    // (torch.hann_window(periodic=False) — the Parakeet extractor).
    std::vector<float> window(win_length);
    const bool periodic = torch_stft_layout && !symmetric_torch_window;
    const float denom = static_cast<float>(
        periodic ? win_length : win_length - 1);
    for (int i = 0; i < win_length; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * kPi
                    * static_cast<float>(i) / denom));
    }
    // torch.stft centres a shorter window inside the n_fft frame.
    const int win_offset = torch_stft_layout ? (n_fft - win_length) / 2 : 0;

    // STFT + mel
    std::vector<float> mel(num_mel_bins * num_frames);
    std::vector<float> frame(n_fft, 0.0f);
    std::vector<float> spec_re(num_bins), spec_im(num_bins);

    for (int t = 0; t < num_frames; t++) {
        // Windowed frame (zero-padded if win_length < n_fft)
        std::fill(frame.begin(), frame.end(), 0.0f);
        for (int i = 0; i < win_length; i++) {
            frame[win_offset + i] = sig[t * hop_length + win_offset + i] * window[i];
        }

        fft_real(frame.data(), n_fft, spec_re.data(), spec_im.data());

        // Power spectrum → mel → log
        for (int m = 0; m < num_mel_bins; m++) {
            float sum = 0.0f;
            for (int f = 0; f < num_bins; f++) {
                float power = spec_re[f] * spec_re[f]
                            + spec_im[f] * spec_im[f];
                sum += power * fb[m * num_bins + f];
            }
            mel[m * num_frames + t] = std::log(sum + log_floor);
        }
    }

    return mel;
}

}  // namespace speech_core::audio
