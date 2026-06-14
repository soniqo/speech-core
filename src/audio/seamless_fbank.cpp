#include "speech_core/audio/seamless_fbank.h"

#include "speech_core/audio/fft.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace speech_core::audio {

namespace {

// --- Reference constants (transformers.SeamlessM4TFeatureExtractor,
//     facebook/w2v-bert-2.0; torchaudio.compliance.kaldi.fbank agrees) ---
constexpr int   kSampleRate   = 16000;
constexpr int   kFrameLength  = 400;     // 25 ms
constexpr int   kHopLength    = 160;     // 10 ms
constexpr int   kFftLength    = 512;     // next_pow2(400)
constexpr int   kFreqBins     = kFftLength / 2 + 1;  // 257
constexpr int   kNumMel       = 80;
constexpr int   kStackedDim   = kNumMel * 2;          // 160
constexpr float kPreemph      = 0.97f;
constexpr float kFMin         = 20.0f;
constexpr float kFMax         = 8000.0f;
// float32 epsilon — the pre-log mel floor used by both HF and torchaudio.
constexpr float kMelFloor     = 1.1920928955078125e-07f;
// Sidon's own pipeline (src/sidon/cleansing/audio.py) uses 1e-5 in the CMVN
// denominator; the canonical HF extractor uses 1e-7. We match Sidon since the
// model was trained/used through Sidon's front-end. The difference is
// negligible for ordinary speech variance.
constexpr float kNormEps      = 1e-5f;

// Kaldi mel scale (natural-log form), NOT the HTK 2595*log10 variant.
inline double hz_to_mel(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }
inline double mel_to_hz(double m) { return 700.0 * (std::exp(m / 1127.0) - 1.0); }

// Precomputed, reused front-end tables (built once, immutable thereafter).
struct FbankTables {
    std::vector<float> window;     // Povey window [kFrameLength]
    std::vector<float> mel_fb;     // [kNumMel, kFreqBins], row-major

    FbankTables() {
        // Povey window = symmetric Hann(400) ^ 0.85.
        window.resize(kFrameLength);
        const double pi = 3.14159265358979323846;
        for (int n = 0; n < kFrameLength; ++n) {
            double hann = 0.5 - 0.5 * std::cos(2.0 * pi * n / (kFrameLength - 1));
            window[n] = static_cast<float>(std::pow(hann, 0.85));
        }

        // Kaldi triangular mel filterbank over 256 FFT bins (the Nyquist bin,
        // index 256, gets weight 0 — i.e. the filterbank is built on 256 bins
        // then implicitly zero-padded to 257). norm=None: triangles peak at 1.
        mel_fb.assign(static_cast<size_t>(kNumMel) * kFreqBins, 0.0f);
        const int    num_fft_bins = kFftLength / 2;             // 256
        const double fft_bin_hz   = static_cast<double>(kSampleRate) / kFftLength;  // 31.25 Hz
        const double mel_min      = hz_to_mel(kFMin);
        const double mel_max      = hz_to_mel(kFMax);
        const double mel_delta    = (mel_max - mel_min) / (kNumMel + 1);

        for (int m = 0; m < kNumMel; ++m) {
            const double left   = mel_min + m * mel_delta;
            const double center = mel_min + (m + 1) * mel_delta;
            const double right  = mel_min + (m + 2) * mel_delta;
            for (int k = 0; k < num_fft_bins; ++k) {
                const double mel = hz_to_mel(fft_bin_hz * k);
                double w = 0.0;
                if (mel > left && mel < right) {
                    const double up   = (mel - left) / (center - left);
                    const double down = (right - mel) / (right - center);
                    w = up < down ? up : down;
                    if (w < 0.0) w = 0.0;
                }
                mel_fb[static_cast<size_t>(m) * kFreqBins + k] =
                    static_cast<float>(w);
            }
            // bin kFreqBins-1 (Nyquist) stays 0.
        }
        (void)mel_to_hz;  // kept for documentation/parity; not needed at runtime.
    }
};

const FbankTables& tables() {
    static const FbankTables t;  // thread-safe init (C++11 magic statics)
    return t;
}

}  // namespace

std::vector<float> seamless_log_mel(
    const float* audio, size_t length, int sample_rate, int& out_frames)
{
    out_frames = 0;
    if (sample_rate != kSampleRate) {
        throw std::invalid_argument(
            "seamless_log_mel: expects 16 kHz audio; resample before calling");
    }
    if (audio == nullptr || length < static_cast<size_t>(kFrameLength)) {
        return {};
    }

    // snip_edges / center=False frame count.
    const int num_frames =
        1 + static_cast<int>((length - kFrameLength) / kHopLength);
    if (num_frames < 2) return {};  // need ≥2 frames to stack one output frame.

    const FbankTables& tb = tables();

    // --- log-mel, time-major [num_frames, kNumMel] ---
    std::vector<float> logmel(static_cast<size_t>(num_frames) * kNumMel);

    std::vector<float> frame(kFrameLength);
    std::vector<float> fft_in(kFftLength);
    std::vector<float> spec_re(kFreqBins);
    std::vector<float> spec_im(kFreqBins);

    for (int t = 0; t < num_frames; ++t) {
        const float* src = audio + static_cast<size_t>(t) * kHopLength;

        // Scale to Kaldi int16 range + remove DC offset (per frame).
        double mean = 0.0;
        for (int i = 0; i < kFrameLength; ++i) {
            frame[i] = src[i] * 32768.0f;
            mean += frame[i];
        }
        mean /= kFrameLength;
        for (int i = 0; i < kFrameLength; ++i) {
            frame[i] = static_cast<float>(frame[i] - mean);
        }

        // Pre-emphasis 0.97 with replicate-pad of sample 0, then Povey window.
        // Walk high→low so x[i-1] is still the un-emphasised value.
        for (int i = kFrameLength - 1; i > 0; --i) {
            frame[i] -= kPreemph * frame[i - 1];
        }
        frame[0] -= kPreemph * frame[0];  // = (1 - 0.97) * frame[0]
        for (int i = 0; i < kFrameLength; ++i) {
            fft_in[i] = frame[i] * tb.window[i];
        }
        for (int i = kFrameLength; i < kFftLength; ++i) fft_in[i] = 0.0f;

        // Power spectrum (257 bins).
        fft_real(fft_in.data(), kFftLength, spec_re.data(), spec_im.data());

        float* row = logmel.data() + static_cast<size_t>(t) * kNumMel;
        for (int m = 0; m < kNumMel; ++m) {
            const float* fb = tb.mel_fb.data() + static_cast<size_t>(m) * kFreqBins;
            float acc = 0.0f;
            for (int k = 0; k < kFreqBins; ++k) {
                const float p = spec_re[k] * spec_re[k] + spec_im[k] * spec_im[k];
                acc += p * fb[k];
            }
            if (acc < kMelFloor) acc = kMelFloor;
            row[m] = std::log(acc);
        }
    }

    // --- CMVN: per mel bin over time, zero-mean unit-variance (ddof = 1) ---
    for (int m = 0; m < kNumMel; ++m) {
        double sum = 0.0;
        for (int t = 0; t < num_frames; ++t) {
            sum += logmel[static_cast<size_t>(t) * kNumMel + m];
        }
        const double mu = sum / num_frames;
        double var = 0.0;
        for (int t = 0; t < num_frames; ++t) {
            const double d = logmel[static_cast<size_t>(t) * kNumMel + m] - mu;
            var += d * d;
        }
        // Sample variance (N-1); for a single frame fall back to 0.
        var = (num_frames > 1) ? var / (num_frames - 1) : 0.0;
        const double inv_std = 1.0 / std::sqrt(var + kNormEps);
        for (int t = 0; t < num_frames; ++t) {
            float& x = logmel[static_cast<size_t>(t) * kNumMel + m];
            x = static_cast<float>((x - mu) * inv_std);
        }
    }

    // --- ×2 stacking: drop trailing odd frame, concat consecutive pairs ---
    const int T = num_frames / 2;
    std::vector<float> stacked(static_cast<size_t>(T) * kStackedDim);
    for (int t = 0; t < T; ++t) {
        const float* even = logmel.data() + static_cast<size_t>(2 * t) * kNumMel;
        const float* odd  = logmel.data() + static_cast<size_t>(2 * t + 1) * kNumMel;
        float* dst = stacked.data() + static_cast<size_t>(t) * kStackedDim;
        for (int m = 0; m < kNumMel; ++m) dst[m] = even[m];
        for (int m = 0; m < kNumMel; ++m) dst[kNumMel + m] = odd[m];
    }

    out_frames = T;
    return stacked;
}

}  // namespace speech_core::audio
