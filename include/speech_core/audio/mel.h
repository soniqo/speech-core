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

/// Incremental form of mel_spectrogram(center=true) in its legacy layout, for
/// audio that arrives in pieces: optional pre-emphasis, reflect-centred frames
/// of win_length samples under a symmetric Hann window, the same filterbank and
/// log floor.
///
/// A frame is produced as soon as every sample it reads has arrived, and not
/// before, so it never reads the right-edge padding a whole-buffer call adds at
/// the end of its input. It is then the same arithmetic over the same samples
/// as the whole-buffer frame of the same index, for any buffer holding them.
/// Only the samples a later frame still reads are kept, so a frame costs the
/// same however long the stream has run.
///
/// The left edge is reflected as mel_spectrogram does, which needs n_fft/2 + 1
/// samples before the first frame. torch_stft_layout and center_pad_zeros
/// opt into the Nemotron/NeMo centred-window, constant-padding contract.
/// finish() can complete the right edge of a final encoder window.
/// Not thread-safe.
class StreamingMelSpectrogram {
public:
    struct Config {
        int sample_rate = 16000;
        int n_fft = 512;
        int hop_length = 160;
        int win_length = 400;
        int num_mel_bins = 128;
        bool slaney_norm = false;
        float log_floor = 1e-10f;
        /// y[n] = x[n] - k * x[n-1], passing the stream's first sample through;
        /// the one-sample memory carries across pushes. 0 disables it.
        float pre_emphasis = 0.0f;
        /// Centre the symmetric win_length window inside an n_fft frame,
        /// matching torch.stft. False retains the legacy frame layout.
        bool torch_stft_layout = false;
        /// Zero-pad the left edge instead of reflecting the input.
        bool center_pad_zeros = false;
    };

    explicit StreamingMelSpectrogram(const Config& config);

    /// Forget the stream; the next sample pushed is sample 0 again.
    void reset();

    /// Append `length` samples. Each frame they complete is appended to
    /// `frames` in time order as `num_mel_bins` values — frame-major
    /// [frames x bins], unlike mel_spectrogram's channels-first result.
    /// Returns how many frames were appended.
    size_t push(const float* samples, size_t length, std::vector<float>& frames);

    /// Finish at exactly total_frames, zero-padding the pre-emphasised signal
    /// on the right. Padding never passes through the pre-emphasis filter or
    /// counts as captured samples. Reset before pushing more audio.
    size_t finish(size_t total_frames, std::vector<float>& frames);

    size_t samples_pushed() const { return samples_pushed_; }
    size_t frames_produced() const { return next_frame_; }
    /// Samples held for frames still to come.
    size_t buffered_samples() const { return buffer_.size(); }

private:
    size_t produce_frames(std::vector<float>& frames, size_t frame_limit, bool final);
    Config config_;
    std::vector<float> filterbank_;
    std::vector<float> window_;
    std::vector<float> frame_;
    std::vector<float> spec_re_;
    std::vector<float> spec_im_;
    std::vector<float> buffer_;  // pre-emphasised samples from buffer_start_
    size_t buffer_start_ = 0;
    size_t samples_pushed_ = 0;
    size_t next_frame_ = 0;
    float previous_sample_ = 0.0f;
    bool finished_ = false;
};

}  // namespace speech_core::audio
