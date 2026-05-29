#include "speech_core/models/litert_pyannote_segmentation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace speech_core {

LiteRTPyannoteSegmentation::LiteRTPyannoteSegmentation(const std::string& model_path, bool hw_accel)
    : state_(kStateSize, 0.0f), chunk_buffer_(16000, 0.0f)
{
    LiteRTEngine::get().load(model_path, hw_accel, &model_, &compiled_);
}

LiteRTPyannoteSegmentation::~LiteRTPyannoteSegmentation() {
    if (compiled_) LiteRtDestroyCompiledModel(compiled_);
    if (model_)    LiteRtDestroyModel(model_);
}

void LiteRTPyannoteSegmentation::process_chunk(
    const float* audio, size_t length, std::vector<float>& window_posteriors)
{
    const int chunk = cfg_.chunk_samples;
    if (static_cast<int>(length) < chunk) {
        std::copy(audio, audio + length, chunk_buffer_.begin());
        std::fill(chunk_buffer_.begin() + length, chunk_buffer_.end(), 0.0f);
        audio = chunk_buffer_.data();
    }

    auto env     = LiteRTEngine::get().env();
    auto t_audio = make_type(kLiteRtElementTypeFloat32, {1, 1, chunk});
    auto t_state = make_type(kLiteRtElementTypeFloat32,
                             {kStateDim0, kStateDim1, kStateDim2, kStateDim3});
    auto t_post  = make_type(kLiteRtElementTypeFloat32,
                             {1, cfg_.frames_per_chunk, cfg_.num_powerset_classes});

    LiteRtHostBuffer in_audio (env, t_audio, static_cast<size_t>(chunk) * sizeof(float), audio);
    LiteRtHostBuffer in_state (env, t_state, kStateSize * sizeof(float), state_.data());
    LiteRtHostBuffer out_state(env, t_state, kStateSize * sizeof(float));
    LiteRtHostBuffer out_post (env, t_post,
        static_cast<size_t>(cfg_.frames_per_chunk) * cfg_.num_powerset_classes * sizeof(float));

    // Output order (prod-verified soniqo export): out[0]=lstm_state, out[1]=posteriors.
    LiteRtTensorBuffer ins [2] = { in_audio.raw(),  in_state.raw() };
    LiteRtTensorBuffer outs[2] = { out_state.raw(), out_post.raw() };
    litert_check(LiteRtRunCompiledModel(compiled_, 0, 2, ins, 2, outs), "Pyannote Run");

    const int n = cfg_.frames_per_chunk * cfg_.num_powerset_classes;
    std::vector<float> post(static_cast<size_t>(n));
    out_post .read(post.data(),   static_cast<size_t>(n) * sizeof(float));
    out_state.read(state_.data(), kStateSize * sizeof(float));

    window_posteriors.insert(window_posteriors.end(), post.begin(), post.end());
}

std::vector<SegmentationWindow>
LiteRTPyannoteSegmentation::segment(const float* audio, size_t length, int sample_rate) {
    if (sample_rate != cfg_.sample_rate)
        throw std::runtime_error("PyannoteSegmentation expects 16kHz");

    const int    chunk             = cfg_.chunk_samples;
    const int    chunks_per_window = cfg_.chunks_per_window;
    const size_t win_samples       = static_cast<size_t>(chunk) * chunks_per_window;
    const size_t step_samples      = static_cast<size_t>(cfg_.window_step * cfg_.sample_rate);
    const float  sr_f              = static_cast<float>(cfg_.sample_rate);

    std::vector<SegmentationWindow> results;

    // Reset LSTM state at each window start, run chunks_per_window chunks with
    // carried state, collect the posteriors, then slide the window.
    for (size_t off = 0; off + win_samples <= length; off += step_samples) {
        std::fill(state_.begin(), state_.end(), 0.0f);

        std::vector<float> window_posteriors;
        window_posteriors.reserve(static_cast<size_t>(chunks_per_window) *
                                  cfg_.frames_per_chunk * cfg_.num_powerset_classes);

        for (int k = 0; k < chunks_per_window; ++k) {
            size_t chunk_off = off + static_cast<size_t>(k) * chunk;
            size_t chunk_len = std::min<size_t>(chunk, length - chunk_off);
            process_chunk(audio + chunk_off, chunk_len, window_posteriors);
        }

        const int total_frames = chunks_per_window * cfg_.frames_per_chunk;

        SegmentationWindow wr;
        wr.start_time = static_cast<float>(off) / sr_f;
        wr.end_time   = static_cast<float>(off + win_samples) / sr_f;
        wr.posteriors = std::move(window_posteriors);
        decode_powerset(wr.posteriors.data(), total_frames, wr.speaker_activity);

        results.push_back(std::move(wr));
    }

    return results;
}

void LiteRTPyannoteSegmentation::decode_powerset(
    const float* posteriors, int num_frames, std::vector<float>& speaker_activity)
{
    const int K = cfg_.max_local_speakers;
    const int C = cfg_.num_powerset_classes;

    speaker_activity.assign(static_cast<size_t>(num_frames) * K, 0.0f);

    // Powerset over {∅, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}} → per-speaker activity.
    for (int f = 0; f < num_frames; ++f) {
        const float* p = posteriors + static_cast<size_t>(f) * C;

        float mx = *std::max_element(p, p + C);
        float sum = 0.0f;
        float probs[7];
        for (int c = 0; c < C; ++c) { probs[c] = std::exp(p[c] - mx); sum += probs[c]; }
        for (int c = 0; c < C; ++c) probs[c] /= sum;

        float* a = speaker_activity.data() + static_cast<size_t>(f) * K;
        a[0] = probs[1] + probs[4] + probs[5];
        a[1] = probs[2] + probs[4] + probs[6];
        a[2] = probs[3] + probs[5] + probs[6];
    }
}

}  // namespace speech_core
