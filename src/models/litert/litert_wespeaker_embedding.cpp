#include "speech_core/models/litert_wespeaker_embedding.h"

#include "speech_core/audio/fft.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace speech_core {

namespace {

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

LiteRTWeSpeakerEmbedding::LiteRTWeSpeakerEmbedding(const std::string& model_path, bool hw_accel) {
    LiteRTEngine::get().load(model_path, hw_accel, &model_, &compiled_);
}

LiteRTWeSpeakerEmbedding::~LiteRTWeSpeakerEmbedding() {
    if (compiled_) LiteRtDestroyCompiledModel(compiled_);
    if (model_)    LiteRtDestroyModel(model_);
}

std::vector<float> LiteRTWeSpeakerEmbedding::compute_fbank(const float* audio, size_t length) {
    // Fixed-shape kaldi-style log-mel fbank: kNumFrames × kNumMelBins (≈3 s at
    // 10 ms hop). Shorter input is tiled; longer is truncated.
    //   frame 25 ms (400) / hop 10 ms (160), Hamming, dither 0, no energy,
    //   per-frame DC removal. n_fft = next pow2 of frame_length = 512.
    const int n_fft    = 512;
    const int fft_bins = n_fft / 2 + 1;
    const int frames   = kNumFrames;
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
        window[i] = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) * i / (win_len - 1));

    static std::vector<float> fb;
    if (fb.empty()) fb = kaldi_filterbank(bins, n_fft, 16000);

    std::vector<float> fbank(static_cast<size_t>(frames) * bins, 0.0f);
    std::vector<float> frame_buf(n_fft, 0.0f);
    std::vector<float> spec_re(fft_bins), spec_im(fft_bins);

    for (int t = 0; t < frames; ++t) {
        std::fill(frame_buf.begin(), frame_buf.end(), 0.0f);

        // kaldi "remove_dc_offset": subtract the per-frame mean before windowing.
        float mean = 0.0f;
        for (int i = 0; i < win_len; ++i) mean += padded[t * hop + i];
        mean /= static_cast<float>(win_len);
        for (int i = 0; i < win_len; ++i)
            frame_buf[i] = (padded[t * hop + i] - mean) * window[i];

        audio::fft_real(frame_buf.data(), n_fft, spec_re.data(), spec_im.data());

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

std::vector<float> LiteRTWeSpeakerEmbedding::embed(
    const float* audio, size_t length, int sample_rate)
{
    if (sample_rate != 16000) throw std::runtime_error("WeSpeaker expects 16kHz input");

    auto fbank = compute_fbank(audio, length);  // [kNumFrames * kNumMelBins]

    auto env     = LiteRTEngine::get().env();
    auto t_fbank = make_type(kLiteRtElementTypeFloat32, {1, kNumFrames, kNumMelBins});
    auto t_emb   = make_type(kLiteRtElementTypeFloat32, {1, kEmbeddingDim});

    LiteRtHostBuffer in_fbank(env, t_fbank, fbank.size() * sizeof(float), fbank.data());
    LiteRtHostBuffer out_emb (env, t_emb,   kEmbeddingDim * sizeof(float));

    LiteRtTensorBuffer ins [1] = { in_fbank.raw() };
    LiteRtTensorBuffer outs[1] = { out_emb.raw() };
    litert_check(LiteRtRunCompiledModel(compiled_, 0, 1, ins, 1, outs), "WeSpeaker Run");

    std::vector<float> embedding(kEmbeddingDim);
    out_emb.read(embedding.data(), kEmbeddingDim * sizeof(float));

    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) for (float& v : embedding) v /= norm;

    return embedding;
}

}  // namespace speech_core
