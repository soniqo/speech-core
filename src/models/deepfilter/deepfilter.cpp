#include "speech_core/models/deepfilter.h"

#include "deepfilter_dsp.h"
#include "speech_core/models/onnx_engine.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
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

#include "deepfilter_streaming_graph.inc"

OrtSession* load_streaming_session(const std::string& model_path) {
#if ORT_API_VERSION >= 18
    // The graph's external tensors point into the validated batch ONNX file.
    // Read-only reuse keeps existing downloads and offline inference intact.
    std::ifstream file(std::filesystem::u8path(model_path), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(kSourceModelBytes)) {
        throw std::runtime_error("DeepFilterNet3 streaming requires the published v0.5.6 FP32 model");
    }
    file.seekg(0);
    std::vector<char> source(kSourceModelBytes);
    file.read(source.data(), static_cast<std::streamsize>(source.size()));
    if (!file) throw std::runtime_error("DeepFilterNet3: truncated source model");
    uint64_t fingerprint = UINT64_C(14695981039346656037);
    for (unsigned char byte : source) {
        fingerprint = (fingerprint ^ byte) * UINT64_C(1099511628211);
    }
    if (fingerprint != kSourceModelFingerprint) {
        throw std::runtime_error("DeepFilterNet3 streaming requires the published v0.5.6 FP32 model");
    }

    auto& engine = OnnxEngine::get();
    const auto* api = engine.api();
    OrtSessionOptions* raw_options = nullptr;
    ort_check(api, api->CreateSessionOptions(&raw_options));
    auto release_options = [api](OrtSessionOptions* p) { api->ReleaseSessionOptions(p); };
    std::unique_ptr<OrtSessionOptions, decltype(release_options)> options(raw_options, release_options);
    ort_check(api, api->SetSessionGraphOptimizationLevel(options.get(), ORT_ENABLE_ALL));
    // Single-hop inference has small GEMMs; a CPU session avoids transfer and
    // thread-pool overhead for every 10 ms frame and its feedback tensors.
    ort_check(api, api->SetIntraOpNumThreads(options.get(), 1));
    const auto external_path = to_ort_path("deepfilter_source.onnx");
    const ORTCHAR_T* external_names[] = {external_path.c_str()};
    char* external_buffers[] = {source.data()};
    const size_t external_lengths[] = {source.size()};
    ort_check(api, api->AddExternalInitializersFromFilesInMemory(
        options.get(), external_names, external_buffers, external_lengths, 1));
    OrtSession* session = nullptr;
    ort_check(api, api->CreateSessionFromArray(engine.env(), kStreamingGraph,
                                              sizeof(kStreamingGraph), options.get(), &session));
    return session;
#else
    (void)model_path;
    throw std::runtime_error("DeepFilterNet3 streaming requires ONNX Runtime >= 1.18");
#endif
}

template <size_t N>
struct OrtValues {
    explicit OrtValues(const OrtApi* ort_api) : api(ort_api) {}
    ~OrtValues() {
        for (auto* value : values) if (value) api->ReleaseValue(value);
    }
    const OrtApi* api;
    std::array<OrtValue*, N> values{};
};

}  // namespace

struct DeepFilterEnhancer::StreamState {
    StreamState(const std::string& path, const Config& config)
        : cfg(config), dsp(to_dsp_config(cfg)),
          widths(deepfilter_dsp::make_erb_widths(to_dsp_config(cfg))),
          hop(static_cast<size_t>(cfg.hop_size)),
          enhanced_real(static_cast<size_t>(cfg.freq_bins)),
          enhanced_imag(enhanced_real.size()), raw_hop(hop.size()) {
        for (size_t i = 0; i < state.size(); ++i) state[i].resize(kStateSizes[i]);
        for (auto& frame : real) frame.resize(static_cast<size_t>(cfg.freq_bins));
        for (auto& frame : imag) frame.resize(static_cast<size_t>(cfg.freq_bins));
        reset();
        session = load_streaming_session(path);
    }

    ~StreamState() { if (session) api->ReleaseSession(session); }

    void reset() {
        dsp.reset();
        for (auto& value : state) std::fill(value.begin(), value.end(), 0.0f);
        for (auto& value : real) std::fill(value.begin(), value.end(), 0.0f);
        for (auto& value : imag) std::fill(value.begin(), value.end(), 0.0f);
        std::fill(hop.begin(), hop.end(), 0.0f);
        pending = 0;
        input_samples = 0;
        frames = 0;
        synthesized = 0;
        flushed = false;
        output.assign(4 * hop.size(), 0.0f);
    }

    void process_hop(bool graph_padding = false) {
        const uint64_t current = frames++;
        const size_t slot = static_cast<size_t>(current % real.size());
        dsp.analyze_hop(hop.data(), real[slot], imag[slot], feat_erb, feat_spec);
        if (current < static_cast<uint64_t>(cfg.df_lookahead)) return;
        if (graph_padding) {
            // Match the batch export's final two zero feature frames. Silence
            // normalized through libdf is different from a zero feature tensor.
            std::fill(feat_erb.begin(), feat_erb.end(), 0.0f);
            std::fill(feat_spec.begin(), feat_spec.end(), 0.0f);
        }

        OrtValues<10> inputs(api);
        OrtValues<10> outputs(api);
        const int64_t erb_shape[] = {1, 1, 1, cfg.erb_bands};
        const int64_t spec_shape[] = {1, 2, 1, cfg.df_bins};
        auto tensor = [&](std::vector<float>& data, const int64_t* shape, size_t rank, size_t index) {
            ort_check(api, api->CreateTensorWithDataAsOrtValue(
                OnnxEngine::get().cpu_memory(), data.data(), data.size() * sizeof(float),
                shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs.values[index]));
        };
        tensor(feat_erb, erb_shape, 4, 0);
        tensor(feat_spec, spec_shape, 4, 1);
        std::array<const char*, 10> input_names{"feat_erb", "feat_spec"};
        std::array<const char*, 10> output_names{"erb_mask", "df_coefs"};
        for (size_t i = 0; i < state.size(); ++i) {
            input_names[i + 2] = kStateInputs[i];
            output_names[i + 2] = kStateOutputs[i];
            tensor(state[i], kStateShapes[i].data(), kStateRanks[i], i + 2);
        }
        ort_check(api, api->Run(session, nullptr, input_names.data(), inputs.values.data(),
                                inputs.values.size(), output_names.data(), outputs.values.size(),
                                outputs.values.data()));
        for (auto* value : outputs.values) {
            if (!value) throw std::runtime_error("DeepFilterNet3: null streaming graph output");
        }
        require_shape(api, outputs.values[0], {1, 1, 1, cfg.erb_bands}, "erb_mask");
        require_shape(api, outputs.values[1], {1, cfg.df_order, 1, cfg.df_bins, 2}, "df_coefs");
        for (size_t i = 0; i < state.size(); ++i) {
            require_shape(api, outputs.values[i + 2],
                          {kStateShapes[i].begin(), kStateShapes[i].begin() + kStateRanks[i]},
                          kStateOutputs[i]);
        }
        auto data = [&](size_t index) {
            float* value = nullptr;
            ort_check(api, api->GetTensorMutableData(outputs.values[index], reinterpret_cast<void**>(&value)));
            return value;
        };
        const float* mask = data(0);
        const float* coefs = data(1);
        const size_t target = static_cast<size_t>((current - cfg.df_lookahead) % real.size());
        int frequency = 0;
        for (int band = 0; band < cfg.erb_bands; ++band) {
            for (int j = 0; j < widths[static_cast<size_t>(band)]; ++j, ++frequency) {
                const size_t f = static_cast<size_t>(frequency);
                enhanced_real[f] = real[target][f] * mask[band];
                enhanced_imag[f] = imag[target][f] * mask[band];
            }
        }
        for (int f = 0; f < cfg.df_bins; ++f) {
            float re = 0.0f;
            float im = 0.0f;
            for (int tap = 0; tap < cfg.df_order; ++tap) {
                if (current + static_cast<uint64_t>(tap) < static_cast<uint64_t>(cfg.df_order - 1)) continue;
                const size_t source = static_cast<size_t>((current + tap - (cfg.df_order - 1)) % real.size());
                const size_t coefficient = (static_cast<size_t>(tap) * cfg.df_bins + f) * 2;
                re += real[source][static_cast<size_t>(f)] * coefs[coefficient] -
                      imag[source][static_cast<size_t>(f)] * coefs[coefficient + 1];
                im += real[source][static_cast<size_t>(f)] * coefs[coefficient + 1] +
                      imag[source][static_cast<size_t>(f)] * coefs[coefficient];
            }
            enhanced_real[static_cast<size_t>(f)] = re;
            enhanced_imag[static_cast<size_t>(f)] = im;
        }
        dsp.synthesize_hop(enhanced_real, enhanced_imag, raw_hop.data());
        if (synthesized++ > 0) output.insert(output.end(), raw_hop.begin(), raw_hop.end());
        for (size_t i = 0; i < state.size(); ++i) std::copy_n(data(i + 2), state[i].size(), state[i].data());
    }

    Config cfg;
    const OrtApi* api = OnnxEngine::get().api();
    OrtSession* session = nullptr;
    deepfilter_dsp::StreamingDSP dsp;
    std::vector<int> widths;
    std::array<std::vector<float>, 8> state;
    std::array<std::vector<float>, 5> real;
    std::array<std::vector<float>, 5> imag;
    std::vector<float> hop;
    std::vector<float> feat_erb;
    std::vector<float> feat_spec;
    std::vector<float> enhanced_real;
    std::vector<float> enhanced_imag;
    std::vector<float> raw_hop;
    std::deque<float> output;
    size_t pending = 0;
    uint64_t input_samples = 0;
    uint64_t frames = 0;
    uint64_t synthesized = 0;
    bool flushed = false;
};

DeepFilterEnhancer::DeepFilterEnhancer(
    const std::string& model_path,
    const std::string& auxiliary_path,
    bool hw_accel) : model_path_(model_path) {
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

void DeepFilterEnhancer::enhance_stream(
    const float* audio, size_t length, int sample_rate, float* output) {
    if (sample_rate != cfg_.sample_rate) throw std::invalid_argument("DeepFilterNet3 requires 48000 Hz input");
    if (length == 0) return;
    if (!audio || !output) throw std::invalid_argument("DeepFilterNet3 received a null audio buffer");
    if (!stream_) stream_ = std::make_unique<StreamState>(model_path_, cfg_);
    auto& s = *stream_;
    if (s.flushed) throw std::logic_error("DeepFilterNet3: reset() required after flush_stream()");
    if (length > std::numeric_limits<uint64_t>::max() - s.input_samples) {
        throw std::overflow_error("DeepFilterNet3: stream sample count overflow");
    }
    try {
        for (size_t i = 0; i < length; ++i) {
            s.hop[s.pending++] = audio[i];
            ++s.input_samples;
            if (s.pending == s.hop.size()) {
                s.pending = 0;
                s.process_hop();
            }
            if (s.output.empty()) throw std::logic_error("DeepFilterNet3: streaming output underrun");
            output[i] = s.output.front();
            s.output.pop_front();
        }
    } catch (...) {
        s.reset();
        throw;
    }
}

std::vector<float> DeepFilterEnhancer::flush_stream() {
    if (!stream_ || stream_->input_samples == 0 || stream_->flushed) return {};
    auto& s = *stream_;
    try {
        if (s.pending > 0) {
            std::fill(s.hop.begin() + s.pending, s.hop.end(), 0.0f);
            s.pending = 0;
            s.process_hop();
        }
        std::fill(s.hop.begin(), s.hop.end(), 0.0f);
        const uint64_t analysis_frames = s.input_samples / s.hop.size() + 2;
        while (s.frames < analysis_frames) s.process_hop();
        s.process_hop(true);
        s.process_hop(true);
        std::vector<float> tail(stream_latency_samples());
        if (s.output.size() < tail.size()) throw std::logic_error("DeepFilterNet3: streaming tail underrun");
        for (auto& sample : tail) {
            sample = s.output.front();
            s.output.pop_front();
        }
        s.output.clear();
        s.flushed = true;
        return tail;
    } catch (...) {
        s.reset();
        throw;
    }
}

void DeepFilterEnhancer::reset() {
    if (stream_) stream_->reset();
}

}  // namespace speech_core
