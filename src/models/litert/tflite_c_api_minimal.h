#pragma once

// Minimal stable TensorFlow Lite C ABI used by the Kokoro wrapper. libLiteRt
// 2.1.x exports these symbols alongside the newer LiteRtCompiledModel API.
// Keeping the declarations private avoids vendoring TFLite's much larger
// public header tree just for fixed-shape interpreter calls.

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define SPEECH_TFL_CAPI __declspec(dllimport)
#else
#define SPEECH_TFL_CAPI
#endif

extern "C" {

struct TfLiteModel;
struct TfLiteInterpreterOptions;
struct TfLiteInterpreter;
struct TfLiteTensor;
struct TfLiteOpaqueDelegate;
struct TfLiteXNNPackDelegateWeightsCache;

// LiteRT 2.1.5's public XNNPACK delegate options ABI.
struct TfLiteXNNPackDelegateOptions {
    std::int32_t num_threads;
    std::uint32_t runtime_flags;
    std::uint32_t flags;
    TfLiteXNNPackDelegateWeightsCache* weights_cache;
    bool handle_variable_ops;
    const char* weight_cache_file_path;
    int weight_cache_file_descriptor;
    void* weight_cache_provider;
    bool weight_cache_lock_memory;
};

enum TfLiteStatus {
    kTfLiteOk = 0,
    kTfLiteError = 1,
};

SPEECH_TFL_CAPI TfLiteModel* TfLiteModelCreateFromFile(const char* model_path);
SPEECH_TFL_CAPI void TfLiteModelDelete(TfLiteModel* model);

SPEECH_TFL_CAPI TfLiteInterpreterOptions* TfLiteInterpreterOptionsCreate();
SPEECH_TFL_CAPI void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);
SPEECH_TFL_CAPI void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, std::int32_t num_threads);
SPEECH_TFL_CAPI void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options, TfLiteOpaqueDelegate* delegate);

SPEECH_TFL_CAPI TfLiteXNNPackDelegateOptions
TfLiteXNNPackDelegateOptionsDefault();
SPEECH_TFL_CAPI TfLiteOpaqueDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options);
SPEECH_TFL_CAPI void TfLiteXNNPackDelegateDelete(
    TfLiteOpaqueDelegate* delegate);

SPEECH_TFL_CAPI TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model,
    const TfLiteInterpreterOptions* optional_options);
SPEECH_TFL_CAPI void TfLiteInterpreterDelete(TfLiteInterpreter* interpreter);
SPEECH_TFL_CAPI std::int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter);
SPEECH_TFL_CAPI std::int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter);
SPEECH_TFL_CAPI TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, std::int32_t input_index);
SPEECH_TFL_CAPI const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, std::int32_t output_index);
SPEECH_TFL_CAPI TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter);
SPEECH_TFL_CAPI TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

SPEECH_TFL_CAPI std::int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);
SPEECH_TFL_CAPI std::int32_t TfLiteTensorDim(
    const TfLiteTensor* tensor, std::int32_t dim_index);
SPEECH_TFL_CAPI std::size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);
SPEECH_TFL_CAPI const char* TfLiteTensorName(const TfLiteTensor* tensor);
SPEECH_TFL_CAPI TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, std::size_t input_data_size);
SPEECH_TFL_CAPI TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* output_tensor, void* output_data,
    std::size_t output_data_size);

}  // extern "C"

#undef SPEECH_TFL_CAPI
