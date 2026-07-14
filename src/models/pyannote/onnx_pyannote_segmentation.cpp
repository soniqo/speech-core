#include "speech_core/models/onnx_pyannote_segmentation.h"

#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace speech_core {

OnnxPyannoteSegmentation::OnnxPyannoteSegmentation(const std::string& model_path,
                                                   bool hw_accel)
{
    api_     = OnnxEngine::get().api();
    session_ = OnnxEngine::get().load(model_path, hw_accel);

    window_samples_ = static_cast<int>(cfg_.window_duration * cfg_.sample_rate);

    // Resolve the frame count FROM THE GRAPH. The LiteRT bundle of this same model
    // emits 560 frames per 10 s window (ten 1 s chunks x 56); this graph emits 589.
    // Hardcoding either would shift every timestamp by ~5% — a failure that leaves
    // transcripts looking correct while speaker boundaries drift.
    OrtTypeInfo* info = nullptr;
    ort_check(api_, api_->SessionGetOutputTypeInfo(session_, 0, &info));
    const OrtTensorTypeAndShapeInfo* shape = nullptr;
    ort_check(api_, api_->CastTypeInfoToTensorInfo(info, &shape));
    size_t dims = 0;
    api_->GetDimensionsCount(shape, &dims);
    std::vector<int64_t> out_shape(dims);
    api_->GetDimensions(shape, out_shape.data(), dims);
    api_->ReleaseTypeInfo(info);

    if (dims != 3 || out_shape[2] != cfg_.num_powerset_classes) {
        throw std::runtime_error(
            "Pyannote ONNX: expected output [1, frames, 7], got a " +
            std::to_string(dims) + "-D tensor");
    }
    frames_per_window_ = static_cast<int>(out_shape[1]);
    if (frames_per_window_ <= 0) {
        throw std::runtime_error(
            "Pyannote ONNX: frame count is dynamic in the graph; this model is "
            "expected to pin it for a fixed 10 s window");
    }
}

OnnxPyannoteSegmentation::~OnnxPyannoteSegmentation() {
    if (session_) api_->ReleaseSession(session_);
}

void OnnxPyannoteSegmentation::decode_powerset(
    const float* posteriors, int num_frames,
    std::vector<float>& speaker_activity) const
{
    const int K = cfg_.max_local_speakers;
    const int C = cfg_.num_powerset_classes;

    speaker_activity.assign(static_cast<size_t>(num_frames) * K, 0.0f);

    for (int f = 0; f < num_frames; ++f) {
        const float* p = posteriors + static_cast<size_t>(f) * C;

        // Graph emits log-probabilities. Max-subtract + exp + normalise recovers
        // the probabilities exactly (it is idempotent on log-probs), so this is
        // the same arithmetic the LiteRT decoder does on the same model.
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

std::vector<SegmentationWindow> OnnxPyannoteSegmentation::segment(
    const float* audio, size_t length, int sample_rate)
{
    if (sample_rate != cfg_.sample_rate) {
        throw std::runtime_error("Pyannote expects 16 kHz input");
    }

    std::vector<SegmentationWindow> results;
    if (!audio || length == 0) return results;

    auto* mem = OnnxEngine::get().cpu_memory();

    const size_t win_samples  = static_cast<size_t>(window_samples_);
    const size_t step_samples =
        static_cast<size_t>(cfg_.window_step * cfg_.sample_rate);
    const float  sr_f         = static_cast<float>(cfg_.sample_rate);

    const char* in_names[]  = {"audio"};
    const char* out_names[] = {"posteriors"};
    const int64_t in_shape[] = {1, 1, static_cast<int64_t>(win_samples)};

    std::vector<float> window(win_samples);

    // Windows are emitted for every hop that starts inside the audio, so the tail
    // is covered even when it is shorter than a window (zero-padded). Dropping a
    // short tail would silently lose the last speaker turn of every meeting.
    for (size_t off = 0; off == 0 || off < length; off += step_samples) {
        const size_t n = (off < length) ? std::min(win_samples, length - off) : 0;
        if (n == 0 && off != 0) break;

        std::fill(window.begin(), window.end(), 0.0f);
        std::copy(audio + off, audio + off + n, window.begin());

        OrtValue* t_audio = nullptr;
        ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
            mem, window.data(), window.size() * sizeof(float),
            in_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &t_audio));

        OrtValue* inputs[]  = {t_audio};
        OrtValue* outputs[] = {nullptr};
        ort_check(api_, api_->Run(session_, nullptr, in_names, inputs, 1,
                                  out_names, 1, outputs));

        float* post = nullptr;
        ort_check(api_, api_->GetTensorMutableData(outputs[0],
                                                   reinterpret_cast<void**>(&post)));

        SegmentationWindow wr;
        wr.start_time = static_cast<float>(off) / sr_f;
        wr.end_time   = static_cast<float>(off + win_samples) / sr_f;
        wr.posteriors.assign(
            post, post + static_cast<size_t>(frames_per_window_) *
                             cfg_.num_powerset_classes);
        decode_powerset(wr.posteriors.data(), frames_per_window_,
                        wr.speaker_activity);

        api_->ReleaseValue(outputs[0]);
        api_->ReleaseValue(t_audio);

        results.push_back(std::move(wr));

        if (off + win_samples >= length) break;
    }

    return results;
}

}  // namespace speech_core
