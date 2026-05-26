#pragma once

#include <tensorflow/lite/c/c_api.h>

#include <stdexcept>
#include <string>

#ifdef __ANDROID__
#include <android/log.h>
#ifndef LOG_TAG
#define LOG_TAG "Speech"
#endif
#ifndef LOGI
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#endif
#ifndef LOGE
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif
#else
#include <cstdio>
#ifndef LOGI
#define LOGI(...) do { fprintf(stderr, "[speech] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
#ifndef LOGE
#define LOGE(...) do { fprintf(stderr, "[speech ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
#endif

namespace speech_core {

inline void litert_check(TfLiteStatus status, const char* what) {
    if (status != kTfLiteOk) {
        throw std::runtime_error(std::string("LiteRT: ") + what + " failed");
    }
}

/// Shared LiteRT loader. Creates a TfLiteInterpreter from a .tflite path.
/// hw_accel is currently a thread-count hint; NNAPI/GPU delegates are not yet wired.
/// Returns the loaded interpreter and (out-param) the model handle the caller
/// must free with TfLiteModelDelete after the interpreter.
class LiteRTEngine {
public:
    static LiteRTEngine& get() {
        static LiteRTEngine instance;
        return instance;
    }

    /// Load a .tflite into an interpreter. Caller owns both returned handles
    /// and must call TfLiteInterpreterDelete before TfLiteModelDelete.
    TfLiteInterpreter* load(const std::string& path,
                            bool hw_accel,
                            TfLiteModel** out_model) {
        TfLiteModel* model = TfLiteModelCreateFromFile(path.c_str());
        if (!model) {
            throw std::runtime_error("LiteRT: failed to load model from " + path);
        }

        TfLiteInterpreterOptions* opts = TfLiteInterpreterOptionsCreate();
        // 4 threads when caller asks for hw accel, 2 otherwise. NNAPI/GPU delegates
        // are not yet wired through the C API — track follow-up in docs/models.md.
        TfLiteInterpreterOptionsSetNumThreads(opts, hw_accel ? 4 : 2);

        LOGI("Loading LiteRT model: %s (threads=%d)",
             path.substr(path.find_last_of('/') + 1).c_str(),
             hw_accel ? 4 : 2);

        TfLiteInterpreter* interp = TfLiteInterpreterCreate(model, opts);
        TfLiteInterpreterOptionsDelete(opts);

        if (!interp) {
            TfLiteModelDelete(model);
            throw std::runtime_error("LiteRT: failed to create interpreter for " + path);
        }

        litert_check(TfLiteInterpreterAllocateTensors(interp), "AllocateTensors");

        *out_model = model;
        return interp;
    }

private:
    LiteRTEngine() = default;
    LiteRTEngine(const LiteRTEngine&) = delete;
    LiteRTEngine& operator=(const LiteRTEngine&) = delete;
};

}  // namespace speech_core
