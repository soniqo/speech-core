#include "speech_core/models/deepfilter.h"

#include "deepfilter_dsp.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace speech_core {
namespace {

deepfilter_dsp::Config to_dsp_config(const DeepFilterEnhancer::Config& cfg) {
    deepfilter_dsp::Config out;
    out.fft_size = cfg.fft_size;
    out.hop_size = cfg.hop_size;
    out.erb_bands = cfg.erb_bands;
    out.df_bins = cfg.df_bins;
    out.df_order = cfg.df_order;
    out.df_lookahead = cfg.df_lookahead;
    out.freq_bins = cfg.freq_bins;
    out.sample_rate = cfg.sample_rate;
    return out;
}

class OrtValueGuard {
public:
    OrtValueGuard(const OrtApi* api, OrtValue* value) : api_(api), value_(value) {}
    ~OrtValueGuard() {
        if (value_) api_->ReleaseValue(value_);
    }
    OrtValueGuard(const OrtValueGuard&) = delete;
    OrtValueGuard& operator=(const OrtValueGuard&) = delete;
    OrtValue* get() const { return value_; }

private:
    const OrtApi* api_;
    OrtValue* value_;
};

class OrtShapeGuard {
public:
    OrtShapeGuard(const OrtApi* api, OrtTensorTypeAndShapeInfo* info)
        : api_(api), info_(info) {}
    ~OrtShapeGuard() { api_->ReleaseTensorTypeAndShapeInfo(info_); }
    OrtTensorTypeAndShapeInfo* get() const { return info_; }

private:
    const OrtApi* api_;
    OrtTensorTypeAndShapeInfo* info_;
};

void require_shape(const OrtApi* api, OrtValue* value,
                   const std::vector<int64_t>& expected,
                   const char* output_name) {
    OrtTensorTypeAndShapeInfo* raw_info = nullptr;
    ort_check(api, api->GetTensorTypeAndShape(value, &raw_info));
    OrtShapeGuard info(api, raw_info);

    size_t rank = 0;
    ort_check(api, api->GetDimensionsCount(info.get(), &rank));
    std::vector<int64_t> actual(rank);
    if (rank > 0) ort_check(api, api->GetDimensions(info.get(), actual.data(), rank));
    ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ort_check(api, api->GetTensorElementType(info.get(), &element_type));
    if (actual != expected || element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::string message = "DeepFilterNet3: unexpected ";
        message += output_name;
        message += " tensor contract";
        throw std::runtime_error(message);
    }
}

bool approximately_equal(float a, float b, float tolerance = 1e-5f) {
    return std::fabs(a - b) <= tolerance;
}

}  // namespace

DeepFilterEnhancer::DeepFilterEnhancer(
    const std::string& model_path,
    const std::string& auxiliary_path,
    bool hw_accel) {
    load_auxiliary(auxiliary_path);
    auto& engine = OnnxEngine::get();
    api_ = engine.api();
    session_ = engine.load(model_path, hw_accel);
}

DeepFilterEnhancer::~DeepFilterEnhancer() {
    if (session_) api_->ReleaseSession(session_);
}

void DeepFilterEnhancer::load_auxiliary(const std::string& path) {
    const deepfilter_dsp::Config dsp_cfg = to_dsp_config(cfg_);
    erb_widths_ = deepfilter_dsp::make_erb_widths(dsp_cfg);
    window_ = deepfilter_dsp::make_vorbis_window(cfg_.fft_size);

    // The first published binary used overlapping, normalized triangular
    // matrices. libdf actually uses disjoint ERB widths (mean on analysis,
    // repetition on synthesis). Keep accepting the constructor argument for
    // existing bundles, but only use it as a compatibility check; canonical
    // tables above are cheap to derive and cannot suffer layout drift.
    if (path.empty()) return;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOGI("DeepFilterNet3 auxiliary file unavailable; using built-in libdf tables: %s",
             path.c_str());
        return;
    }

    const size_t matrix_values = static_cast<size_t>(cfg_.freq_bins) * cfg_.erb_bands;
    const size_t expected_values = matrix_values * 2 + static_cast<size_t>(cfg_.fft_size);
    const std::streamoff byte_count = file.tellg();
    if (byte_count != static_cast<std::streamoff>(expected_values * sizeof(float))) {
        LOGI("DeepFilterNet3 auxiliary layout is incompatible; using built-in libdf tables");
        return;
    }
    file.seekg(0);
    std::vector<float> values(expected_values);
    file.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!file) {
        LOGI("DeepFilterNet3 auxiliary file is truncated; using built-in libdf tables");
        return;
    }

    bool canonical = true;
    int frequency_offset = 0;
    for (int band = 0; band < cfg_.erb_bands && canonical; ++band) {
        const int width = erb_widths_[static_cast<size_t>(band)];
        for (int frequency = 0; frequency < cfg_.freq_bins; ++frequency) {
            const bool in_band = frequency >= frequency_offset &&
                                 frequency < frequency_offset + width;
            const float expected_forward = in_band ? 1.0f / static_cast<float>(width) : 0.0f;
            const float expected_inverse = in_band ? 1.0f : 0.0f;
            const size_t forward_index =
                static_cast<size_t>(frequency) * cfg_.erb_bands + band;
            const size_t inverse_index = matrix_values +
                static_cast<size_t>(band) * cfg_.freq_bins + frequency;
            if (!approximately_equal(values[forward_index], expected_forward) ||
                !approximately_equal(values[inverse_index], expected_inverse)) {
                canonical = false;
                break;
            }
        }
        frequency_offset += width;
    }
    const size_t window_offset = matrix_values * 2;
    for (int i = 0; i < cfg_.fft_size && canonical; ++i) {
        if (!approximately_equal(values[window_offset + static_cast<size_t>(i)],
                                 window_[static_cast<size_t>(i)])) {
            canonical = false;
        }
    }
    if (!canonical) {
        LOGI("DeepFilterNet3 auxiliary values do not match libdf; using built-in tables");
    }
}

void DeepFilterEnhancer::enhance(
    const float* audio, size_t length, int sample_rate, float* output) {
    if (sample_rate != cfg_.sample_rate) {
        throw std::invalid_argument("DeepFilterNet3 requires 48000 Hz input");
    }
    if (length == 0) return;
    if (!audio || !output) {
        throw std::invalid_argument("DeepFilterNet3 received a null audio buffer");
    }

    const deepfilter_dsp::Config dsp_cfg = to_dsp_config(cfg_);
    std::vector<float> spec_real;
    std::vector<float> spec_imag;
    deepfilter_dsp::analyze(audio, length, dsp_cfg, window_, spec_real, spec_imag);
    const int num_frames = deepfilter_dsp::frame_count(length, dsp_cfg);

    std::vector<float> feat_erb;
    std::vector<float> feat_spec;
    deepfilter_dsp::compute_features(spec_real, spec_imag, num_frames,
                                     dsp_cfg, erb_widths_, feat_erb, feat_spec);

    auto* memory = OnnxEngine::get().cpu_memory();
    const int64_t frames = num_frames;
    const int64_t erb_shape[] = {1, 1, frames, cfg_.erb_bands};
    const int64_t spec_shape[] = {1, 2, frames, cfg_.df_bins};

    OrtValue* raw_erb = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        memory, feat_erb.data(), feat_erb.size() * sizeof(float),
        erb_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw_erb));
    OrtValueGuard tensor_erb(api_, raw_erb);

    OrtValue* raw_spec = nullptr;
    ort_check(api_, api_->CreateTensorWithDataAsOrtValue(
        memory, feat_spec.data(), feat_spec.size() * sizeof(float),
        spec_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &raw_spec));
    OrtValueGuard tensor_spec(api_, raw_spec);

    const char* input_names[] = {"feat_erb", "feat_spec"};
    const char* output_names[] = {"erb_mask", "df_coefs"};
    OrtValue* inputs[] = {tensor_erb.get(), tensor_spec.get()};
    OrtValue* raw_outputs[] = {nullptr, nullptr};
    OrtStatus* run_status = api_->Run(session_, nullptr,
                                      input_names, inputs, 2,
                                      output_names, 2, raw_outputs);
    if (run_status) {
        for (OrtValue* value : raw_outputs) {
            if (value) api_->ReleaseValue(value);
        }
        ort_check(api_, run_status);
    }
    if (!raw_outputs[0] || !raw_outputs[1]) {
        for (OrtValue* value : raw_outputs) {
            if (value) api_->ReleaseValue(value);
        }
        throw std::runtime_error("DeepFilterNet3: ONNX graph returned a null output");
    }
    OrtValueGuard mask_output(api_, raw_outputs[0]);
    OrtValueGuard coefficients_output(api_, raw_outputs[1]);

    require_shape(api_, mask_output.get(),
                  {1, 1, frames, cfg_.erb_bands}, "erb_mask");
    require_shape(api_, coefficients_output.get(),
                  {1, cfg_.df_order, frames, cfg_.df_bins, 2}, "df_coefs");

    float* erb_mask = nullptr;
    ort_check(api_, api_->GetTensorMutableData(mask_output.get(),
                                               reinterpret_cast<void**>(&erb_mask)));
    float* df_coefs = nullptr;
    ort_check(api_, api_->GetTensorMutableData(coefficients_output.get(),
                                               reinterpret_cast<void**>(&df_coefs)));

    std::vector<float> enhanced_real;
    std::vector<float> enhanced_imag;
    deepfilter_dsp::apply_network_output(
        spec_real, spec_imag, erb_mask, df_coefs, num_frames,
        dsp_cfg, erb_widths_, enhanced_real, enhanced_imag);
    deepfilter_dsp::synthesize(enhanced_real, enhanced_imag, num_frames,
                              dsp_cfg, window_, output, length);
}

}  // namespace speech_core
