#include "speech_core/audio/wespeaker_fbank.h"

#include "speech_core/audio/fft.h"

#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace speech_core {
namespace audio {

namespace {

constexpr int kNumMelBins   = 80;
constexpr int kFrameLenSamp = 400;  // 25 ms @ 16 kHz
constexpr int kHopSamp      = 160;  // 10 ms @ 16 kHz
constexpr int kNFft         = 512;  // next pow2 of the frame length
constexpr int kSampleRate   = 16000;

// Kaldi-style mel frequency conversion (HTK formulation).
float mel_of(float hz)  { return 1127.0f * std::log(1.0f + hz / 700.0f); }
float hz_of(float mel)  { return 700.0f * (std::exp(mel / 1127.0f) - 1.0f); }

// Triangular mel filterbank of `num_bins` filters over an `n_fft`-point STFT at
// `sr` Hz, spanning [0, sr/2]. NOT area-normalised — kaldi's default fbank uses
// raw triangles with unit peak.
std::vector<float> kaldi_filterbank(int num_bins, int n_fft, int sr) {
    int fft_bins = n_fft / 2 + 1;
    float mel_low  = mel_of(0.0f);
    float mel_high = mel_of(static_cast<float>(sr) / 2.0f);

    std::vector<float> center_mel(num_bins + 2);
    for (int i = 0; i < num_bins + 2; ++i)
        center_mel[i] = mel_low + (mel_high - mel_low) * i / (num_bins + 1);

    std::vector<float> center_hz(num_bins + 2);
    for (int i = 0; i < num_bins + 2; ++i) center_hz[i] = hz_of(center_mel[i]);

    std::vector<float> bin_hz(fft_bins);
    for (int f = 0; f < fft_bins; ++f)
        bin_hz[f] = static_cast<float>(f) * sr / static_cast<float>(n_fft);

    std::vector<float> fb(static_cast<size_t>(num_bins) * fft_bins, 0.0f);
    for (int m = 0; m < num_bins; ++m) {
        float lo = center_hz[m], ce = center_hz[m + 1], hi = center_hz[m + 2];
        for (int f = 0; f < fft_bins; ++f) {
            float h = bin_hz[f];
            if (h >= lo && h <= ce && ce > lo) {
                fb[m * fft_bins + f] = (h - lo) / (ce - lo);
            } else if (h > ce && h <= hi && hi > ce) {
                fb[m * fft_bins + f] = (hi - h) / (hi - ce);
            }
        }
    }
    return fb;
}

}  // namespace

std::vector<float> wespeaker_fbank(const float* audio, size_t length, int frames) {
    const int fft_bins = kNFft / 2 + 1;
    const int bins     = kNumMelBins;
    const int win_len  = kFrameLenSamp;
    const int hop      = kHopSamp;

    const size_t required =
        static_cast<size_t>(win_len) + static_cast<size_t>(frames - 1) * hop;

    std::vector<float> padded(required, 0.0f);
    if (length >= required) {
        std::copy(audio, audio + required, padded.begin());
    } else if (length > 0) {
        for (size_t off = 0; off < required; off += length) {
            size_t n = std::min(length, required - off);
            std::copy(audio, audio + n, padded.begin() + off);
        }
    }

    std::vector<float> window(win_len);
    for (int i = 0; i < win_len; ++i)
        window[i] = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) * i /
                                             (win_len - 1));

    static const std::vector<float> fb = kaldi_filterbank(bins, kNFft, kSampleRate);

    std::vector<float> fbank(static_cast<size_t>(frames) * bins, 0.0f);
    std::vector<float> frame_buf(kNFft, 0.0f);
    std::vector<float> spec_re(fft_bins), spec_im(fft_bins);

    for (int t = 0; t < frames; ++t) {
        std::fill(frame_buf.begin(), frame_buf.end(), 0.0f);

        // kaldi "remove_dc_offset": subtract the per-frame mean before windowing.
        float mean = 0.0f;
        for (int i = 0; i < win_len; ++i) mean += padded[t * hop + i];
        mean /= static_cast<float>(win_len);
        for (int i = 0; i < win_len; ++i)
            frame_buf[i] = (padded[t * hop + i] - mean) * window[i];

        fft_real(frame_buf.data(), kNFft, spec_re.data(), spec_im.data());

        for (int m = 0; m < bins; ++m) {
            float sum = 0.0f;
            for (int f = 0; f < fft_bins; ++f) {
                float power = spec_re[f] * spec_re[f] + spec_im[f] * spec_im[f];
                sum += power * fb[m * fft_bins + f];
            }
            fbank[t * bins + m] = std::log(std::max(sum, 1e-10f));
        }
    }
    return fbank;
}

}  // namespace audio
}  // namespace speech_core
