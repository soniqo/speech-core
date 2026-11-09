#include "speech_core/audio/mel.h"

#include "speech_core/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
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

static std::vector<float> hann_window(int win_length, bool periodic) {
    std::vector<float> window(win_length);
    const float denom = static_cast<float>(
        periodic ? win_length : win_length - 1);
    for (int i = 0; i < win_length; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * kPi
                    * static_cast<float>(i) / denom));
    }
    return window;
}

// Power spectrum -> mel -> log of one windowed n_fft frame, written to
// out[m * stride] for each mel bin m. Shared by the whole-buffer and the
// streaming paths so a frame is the same arithmetic in both.
static void log_mel_frame(
    const float* frame, int n_fft, const std::vector<float>& fb,
    int num_mel_bins, float log_floor,
    std::vector<float>& spec_re, std::vector<float>& spec_im,
    float* out, size_t stride)
{
    const int num_bins = n_fft / 2 + 1;
    fft_real(frame, n_fft, spec_re.data(), spec_im.data());
    for (int m = 0; m < num_mel_bins; m++) {
        float sum = 0.0f;
        for (int f = 0; f < num_bins; f++) {
            float power = spec_re[f] * spec_re[f]
                        + spec_im[f] * spec_im[f];
            sum += power * fb[m * num_bins + f];
        }
        out[static_cast<size_t>(m) * stride] = std::log(sum + log_floor);
    }
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
    const bool periodic = torch_stft_layout && !symmetric_torch_window;
    const std::vector<float> window = hann_window(win_length, periodic);
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
        log_mel_frame(frame.data(), n_fft, fb, num_mel_bins, log_floor,
                      spec_re, spec_im, mel.data() + t,
                      static_cast<size_t>(num_frames));
    }

    return mel;
}

// ---------------------------------------------------------------------------
// StreamingMelSpectrogram
// ---------------------------------------------------------------------------

StreamingMelSpectrogram::StreamingMelSpectrogram(const Config& config)
    : config_(config)
{
    if (config.n_fft <= 0 || config.hop_length <= 0 || config.win_length <= 0
        || config.win_length > config.n_fft || config.num_mel_bins <= 0
        || config.sample_rate <= 0) {
        throw std::invalid_argument("StreamingMelSpectrogram: invalid geometry");
    }
    filterbank_ = mel_filterbank(
        config.num_mel_bins, config.n_fft, config.sample_rate, config.slaney_norm);
    window_ = hann_window(config.win_length, /*periodic=*/false);
    frame_.assign(static_cast<size_t>(config.n_fft), 0.0f);
    spec_re_.assign(static_cast<size_t>(config.n_fft / 2 + 1), 0.0f);
    spec_im_.assign(static_cast<size_t>(config.n_fft / 2 + 1), 0.0f);
}

void StreamingMelSpectrogram::reset() {
    buffer_.clear();
    buffer_start_ = 0;
    samples_pushed_ = 0;
    next_frame_ = 0;
    previous_sample_ = 0.0f;
    finished_ = false;
}

size_t StreamingMelSpectrogram::push(
    const float* samples, size_t length, std::vector<float>& frames)
{
    if (finished_) throw std::logic_error("StreamingMelSpectrogram: reset after finish");
    if (samples == nullptr || length == 0) return 0;

    buffer_.reserve(buffer_.size() + length);
    for (size_t i = 0; i < length; ++i) {
        const float x = samples[i];
        if (config_.pre_emphasis == 0.0f || samples_pushed_ + i == 0) {
            buffer_.push_back(x);
        } else {
            buffer_.push_back(x - config_.pre_emphasis * previous_sample_);
        }
        previous_sample_ = x;
    }
    samples_pushed_ += length;

    return produce_frames(frames, std::numeric_limits<size_t>::max(), false);
}

size_t StreamingMelSpectrogram::finish(size_t total_frames, std::vector<float>& frames) {
    if (total_frames < next_frame_ || (samples_pushed_ == 0 && total_frames != 0)) {
        throw std::invalid_argument("StreamingMelSpectrogram: invalid final frame count");
    }
    if (finished_) {
        if (total_frames != next_frame_)
            throw std::logic_error("StreamingMelSpectrogram: already finished");
        return 0;
    }
    const size_t produced = produce_frames(frames, total_frames, true);
    finished_ = true;
    buffer_.clear();
    buffer_start_ = samples_pushed_;
    return produced;
}

size_t StreamingMelSpectrogram::produce_frames(
    std::vector<float>& frames, size_t frame_limit, bool final)
{

    const std::int64_t pad = config_.n_fft / 2;
    const int window_offset = config_.torch_stft_layout
        ? (config_.n_fft - config_.win_length) / 2 : 0;
    const std::int64_t pushed = static_cast<std::int64_t>(samples_pushed_);
    const std::int64_t base = static_cast<std::int64_t>(buffer_start_);
    size_t produced = 0;
    while (next_frame_ < frame_limit) {
        // Stream index under the frame's first window tap, in the padded
        // coordinates mel_spectrogram uses: frame t starts t*hop - n_fft/2.
        const std::int64_t origin =
            static_cast<std::int64_t>(next_frame_) * config_.hop_length - pad + window_offset;
        if (!final && origin + config_.win_length > pushed) break;
        // Left of the stream mel_spectrogram reflects, padded[pad-1-k] =
        // x[k+1], which reads up to index -origin.
        if (!final && !config_.center_pad_zeros && origin < 0 && -origin >= pushed) break;

        std::fill(frame_.begin(), frame_.end(), 0.0f);
        for (int i = 0; i < config_.win_length; ++i) {
            const std::int64_t index = origin + i;
            if (index < 0 && config_.center_pad_zeros) continue;
            const std::int64_t source = index < 0 ? -index : index;
            if (source >= pushed) continue;  // final right padding, after pre-emphasis
            frame_[static_cast<size_t>(window_offset + i)] =
                buffer_[static_cast<size_t>(source - base)]
                * window_[static_cast<size_t>(i)];
        }
        const size_t at = frames.size();
        frames.resize(at + static_cast<size_t>(config_.num_mel_bins));
        log_mel_frame(frame_.data(), config_.n_fft, filterbank_,
                      config_.num_mel_bins, config_.log_floor,
                      spec_re_, spec_im_, frames.data() + at, 1);
        ++next_frame_;
        ++produced;
    }

    // Keep what the next frame reads: from its first tap, or from the start
    // while the reflected head is still needed. Trimming only past n_fft of
    // slack keeps a push from shifting the buffer every time.
    const std::int64_t next_origin =
        static_cast<std::int64_t>(next_frame_) * config_.hop_length - pad + window_offset;
    const std::int64_t keep_from = std::min(pushed, std::max<std::int64_t>(0, next_origin));
    if (!final && keep_from - base > config_.n_fft) {
        buffer_.erase(buffer_.begin(),
                      buffer_.begin() + static_cast<std::ptrdiff_t>(keep_from - base));
        buffer_start_ = static_cast<size_t>(keep_from);
    }
    return produced;
}

}  // namespace speech_core::audio
