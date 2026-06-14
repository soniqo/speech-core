#include "speech_core/models/onnx_sidon_restorer.h"

#include "speech_core/audio/resampler.h"
#include "speech_core/audio/seamless_fbank.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace speech_core {

namespace {

// Read a single OrtValue tensor's float data and total element count.
// Returns a copy so the source OrtValue can be released immediately after.
std::vector<float> copy_tensor_floats(const OrtApi* api, OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    ort_check(api, api->GetTensorTypeAndShape(value, &info));
    size_t count = 0;
    OrtStatus* s = api->GetTensorShapeElementCount(info, &count);
    api->ReleaseTensorTypeAndShapeInfo(info);
    ort_check(api, s);

    float* data = nullptr;
    ort_check(api, api->GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
    return std::vector<float>(data, data + count);
}

// EnhancerInterface adapter over OnnxSidonRestorer. Resamples the 48 kHz
// restoration back to the caller's rate and writes exactly `length` samples.
class SidonEnhancerAdapter : public EnhancerInterface {
public:
    explicit SidonEnhancerAdapter(OnnxSidonRestorer& r) : restorer_(r) {}

    void enhance(const float* audio, size_t length, int sample_rate,
                 float* output) override {
        restorer_.restore_in_place(audio, length, sample_rate, output);
    }

    int input_sample_rate() const override { return restorer_.input_sample_rate(); }

private:
    OnnxSidonRestorer& restorer_;
};

}  // namespace

OnnxSidonRestorer::OnnxSidonRestorer(const std::string& predictor_path,
                                     const std::string& vocoder_path,
                                     bool hw_accel)
{
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    predictor_session_ = engine.load(predictor_path, hw_accel);
    vocoder_session_   = engine.load(vocoder_path, hw_accel);
}

OnnxSidonRestorer::~OnnxSidonRestorer() {
    if (vocoder_session_)   api_->ReleaseSession(vocoder_session_);
    if (predictor_session_) api_->ReleaseSession(predictor_session_);
}

std::vector<float> OnnxSidonRestorer::run_predictor(const float* features,
                                                    int frames) {
    auto* mem = OnnxEngine::get().cpu_memory();

    const int64_t shape[] = {1, frames, kInputFeatDim};
    OrtValue* in = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, const_cast<float*>(features),
        static_cast<size_t>(frames) * kInputFeatDim * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));

    const char* in_names[]  = {"input_features"};
    const char* out_names[] = {"features"};
    OrtValue* inputs[]  = {in};
    OrtValue* outputs[] = {nullptr};

    OrtStatus* run = api_->Run(predictor_session_, nullptr,
                               in_names, inputs, 1,
                               out_names, 1, outputs);
    api_->ReleaseValue(in);
    ort_check(api_, run);

    std::vector<float> hidden = copy_tensor_floats(api_, outputs[0]);
    api_->ReleaseValue(outputs[0]);
    return hidden;
}

std::vector<float> OnnxSidonRestorer::run_vocoder(const float* features,
                                                  int frames) {
    auto* mem = OnnxEngine::get().cpu_memory();

    const int64_t shape[] = {1, frames, kHiddenDim};
    OrtValue* in = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        mem, const_cast<float*>(features),
        static_cast<size_t>(frames) * kHiddenDim * sizeof(float),
        shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in));

    const char* in_names[]  = {"features"};
    const char* out_names[] = {"audio"};
    OrtValue* inputs[]  = {in};
    OrtValue* outputs[] = {nullptr};

    OrtStatus* run = api_->Run(vocoder_session_, nullptr,
                               in_names, inputs, 1,
                               out_names, 1, outputs);
    api_->ReleaseValue(in);
    ort_check(api_, run);

    std::vector<float> audio = copy_tensor_floats(api_, outputs[0]);
    api_->ReleaseValue(outputs[0]);
    return audio;
}

std::vector<float> OnnxSidonRestorer::restore(const float* audio, size_t length,
                                              int sample_rate) {
    if (audio == nullptr || length == 0) return {};

    // 1. Resample to the model's 16 kHz front-end rate if needed.
    const float* src = audio;
    size_t src_len = length;
    std::vector<float> resampled;
    if (sample_rate != input_sample_rate()) {
        resampled = Resampler::resample(audio, length, sample_rate,
                                        input_sample_rate());
        src = resampled.data();
        src_len = resampled.size();
    }

    // 2. SeamlessM4T log-mel front-end -> input_features[1, T, 160].
    int frames = 0;
    std::vector<float> feats =
        audio::seamless_log_mel(src, src_len, input_sample_rate(), frames);
    if (frames == 0 || feats.empty()) return {};

    // 3. Predictor: input_features -> cleansed features[1, T, 1024].
    std::vector<float> hidden = run_predictor(feats.data(), frames);

    // 4. Vocoder: features -> 48 kHz audio[1, M].
    return run_vocoder(hidden.data(), frames);
}

void OnnxSidonRestorer::restore_in_place(const float* audio, size_t length,
                                         int sample_rate, float* output) {
    std::vector<float> restored = restore(audio, length, sample_rate);

    // Bring the 48 kHz restoration back to the caller's rate, then fit it to
    // exactly `length` samples (truncate / zero-pad).
    std::vector<float> at_rate;
    const float* fitted = restored.data();
    size_t fitted_len = restored.size();
    if (!restored.empty() && sample_rate != output_sample_rate()) {
        at_rate = Resampler::resample(restored.data(), restored.size(),
                                      output_sample_rate(), sample_rate);
        fitted = at_rate.data();
        fitted_len = at_rate.size();
    }

    const size_t n = std::min(fitted_len, length);
    if (n > 0) std::memcpy(output, fitted, n * sizeof(float));
    if (n < length) {
        std::memset(output + n, 0, (length - n) * sizeof(float));
    }
}

std::unique_ptr<EnhancerInterface> OnnxSidonRestorer::as_enhancer() {
    return std::make_unique<SidonEnhancerAdapter>(*this);
}

}  // namespace speech_core
