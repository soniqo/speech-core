#pragma once

#include <onnxruntime_c_api.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "Speech"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <cstdio>
#define LOGI(...) do { fprintf(stderr, "[speech] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, "[speech ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

inline void ort_check(const OrtApi* api, OrtStatus* status) {
    if (status != nullptr) {
        const char* msg = api->GetErrorMessage(status);
        std::string err(msg);
        api->ReleaseStatus(status);
        throw std::runtime_error("ORT: " + err);
    }
}

/// ORT's `CreateSession` takes `const ORTCHAR_T*`, which is `wchar_t*` on
/// Windows and `char*` everywhere else. On Windows we therefore must widen
/// the UTF-8 path string. This helper is a no-op on POSIX (where ORTCHAR_T
/// is char) and a UTF-8 → UTF-16 conversion on Windows. Caller passes the
/// returned object to CreateSession via .c_str().
#ifdef _WIN32
using OrtPathStr = std::wstring;
inline OrtPathStr to_ort_path(const std::string& s) {
    if (s.empty()) return L"";
    int wlen = ::MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                     static_cast<int>(s.size()),
                                     nullptr, 0);
    OrtPathStr w(static_cast<size_t>(wlen), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(),
                          static_cast<int>(s.size()),
                          w.data(), wlen);
    return w;
}
#else
using OrtPathStr = std::string;
inline OrtPathStr to_ort_path(const std::string& s) { return s; }
#endif

/// NVIDIA GPU execution providers, selected on desktop/server (Linux/Windows)
/// builds. Has no effect on Android (which uses NNAPI/QNN) or on a CPU-only ORT
/// runtime — both fall back to CPU at runtime. CUDA is the general-purpose GPU
/// path; TensorRT additionally fuses + builds an optimized engine (slower first
/// inference, faster steady-state), with CUDA + CPU layered beneath it.
enum class OrtGpuProvider {
    None,       ///< CPU (or Android NNAPI/QNN). Default.
    Cuda,       ///< CUDAExecutionProvider.
    TensorRT,   ///< TensorrtExecutionProvider (with CUDA + CPU fallback beneath).
};

/// Singleton ONNX Runtime environment shared across all models.
class OnnxEngine {
public:
    static OnnxEngine& get() {
        static OnnxEngine instance;
        return instance;
    }

    const OrtApi* api() const { return api_; }
    OrtEnv* env() const { return env_; }

    /// True if any model fell back from NNAPI to CPU during session creation.
    bool had_nnapi_fallback() const { return nnapi_fallback_; }
    const std::string& nnapi_fallback_reason() const { return nnapi_fallback_reason_; }

    /// The GPU provider actually in use after the resolve step (build option +
    /// env var + runtime availability). OrtGpuProvider::None when running on
    /// CPU/NNAPI/QNN. Lets callers log / emit telemetry for the effective backend.
    OrtGpuProvider gpu_provider() const { return gpu_provider_; }

    /// True if a model requested a GPU EP but session creation fell back to CPU.
    bool had_gpu_fallback() const { return gpu_fallback_; }
    const std::string& gpu_fallback_reason() const { return gpu_fallback_reason_; }

    OrtSession* load(const std::string& path, bool nnapi = true) {
        OrtSessionOptions* opts = nullptr;
        ort_check(api_, api_->CreateSessionOptions(&opts));
        api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
        api_->SetIntraOpNumThreads(opts, 2);

        // A GPU EP (CUDA / TensorRT), when resolved on a desktop/server build,
        // takes priority over NNAPI/QNN. The `nnapi` argument is the model
        // wrapper's "use hardware acceleration" flag — when it is false (e.g.
        // Parakeet's FP32 decoder-joint, which must stay numerically exact) we
        // honour it and keep the session on CPU, GPU present or not.
        bool gpu_attempted = false;
        if (nnapi && gpu_provider_ != OrtGpuProvider::None) {
            gpu_attempted = try_append_gpu(opts, path);
            if (!gpu_attempted) {
                // CUDA was resolved at startup (libs visible, env let it run)
                // but the per-session append failed — most often a missing
                // cuDNN side-load. The session will fall through to whatever
                // QNN/NNAPI/CPU path follows; mark this as a fallback so
                // callers can detect "expected GPU, got CPU" without parsing
                // logs. CreateSession's own failure path (below) sets the
                // same flag with the OrtStatus error message.
                gpu_fallback_ = true;
                if (gpu_fallback_reason_.empty()) {
                    gpu_fallback_reason_ =
                        "GPU EP append failed at session creation "
                        "(commonly: cuDNN runtime not on PATH); see [speech] log";
                }
            }
        }

        if (nnapi && !gpu_attempted) {
            LOGI("Loading model with hardware acceleration: %s",
                 path.substr(path.find_last_of('/') + 1).c_str());
#ifdef __ANDROID__
            const char* keys[] = {"nnapi_flags"};
            const char* values[] = {"0"};
            OrtStatus* s = api_->SessionOptionsAppendExecutionProvider(
                opts, "NNAPI", keys, values, 1);
#else
            const char* keys[] = {"backend_path"};
            const char* values[] = {"libQnnHtp.so"};
            OrtStatus* s = api_->SessionOptionsAppendExecutionProvider(
                opts, "QNN", keys, values, 1);
#endif
            if (s != nullptr) {
                LOGI("Hardware EP unavailable, using CPU");
                api_->ReleaseStatus(s);
            }
        }

        OrtSession* session = nullptr;
        OrtPathStr ort_path = to_ort_path(path);
        OrtStatus* create_status = api_->CreateSession(env_, ort_path.c_str(), opts, &session);

        // If session creation fails with a hardware EP (NNAPI/QNN/GPU), retry CPU-only.
        if (create_status != nullptr && nnapi) {
            const char* msg = api_->GetErrorMessage(create_status);
            LOGI("Hardware-EP session failed (%s), retrying CPU-only", msg);
            if (gpu_attempted) {
                gpu_fallback_ = true;
                gpu_fallback_reason_ = msg;
            } else {
                nnapi_fallback_ = true;
                nnapi_fallback_reason_ = msg;
            }
            api_->ReleaseStatus(create_status);
            api_->ReleaseSessionOptions(opts);

            opts = nullptr;
            ort_check(api_, api_->CreateSessionOptions(&opts));
            api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
            api_->SetIntraOpNumThreads(opts, 4);

            ort_check(api_, api_->CreateSession(env_, ort_path.c_str(), opts, &session));
        } else if (create_status != nullptr) {
            // CPU-only also failed — propagate the error
            const char* msg = api_->GetErrorMessage(create_status);
            std::string err(msg);
            api_->ReleaseStatus(create_status);
            api_->ReleaseSessionOptions(opts);
            throw std::runtime_error("ORT: " + err);
        }

        api_->ReleaseSessionOptions(opts);
        return session;
    }

    OrtMemoryInfo* cpu_memory() const { return mem_; }

    ~OnnxEngine() {
        if (mem_) api_->ReleaseMemoryInfo(mem_);
        if (env_) api_->ReleaseEnv(env_);
    }

private:
    OnnxEngine() {
        api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_check(api_, api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "speech", &env_));
        ort_check(api_, api_->CreateCpuMemoryInfo(
            OrtArenaAllocator, OrtMemTypeDefault, &mem_));
        gpu_provider_ = resolve_gpu_provider();
    }

    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    /// Resolve which (if any) GPU EP to use. Three gates, all must pass:
    ///   1. Build: SPEECH_CORE_WITH_CUDA must be defined (the CMake option that
    ///      links a CUDA-enabled ORT). Without it we never touch GPU symbols.
    ///   2. Request: env SPEECH_CORE_ORT_PROVIDER in {cuda,tensorrt,trt,gpu}.
    ///      "cpu"/"none" → CPU. Unset defaults to CUDA *because* the build is
    ///      already GPU-capable; availability (gate 3) still protects the
    ///      no-GPU-present case, so the same binary degrades to CPU cleanly.
    ///   3. Availability: the requested EP is listed by GetAvailableProviders()
    ///      — i.e. the linked libonnxruntime was actually built with it. A
    ///      CPU-only runtime reports neither, so we degrade to CPU silently.
    /// On Android the whole thing is compiled out (NNAPI/QNN own acceleration).
    OrtGpuProvider resolve_gpu_provider() {
#if defined(SPEECH_CORE_WITH_CUDA) && !defined(__ANDROID__)
        OrtGpuProvider requested = OrtGpuProvider::Cuda;  // default on a GPU build
        const char* env = std::getenv("SPEECH_CORE_ORT_PROVIDER");
        if (env != nullptr && env[0] != '\0') {
            std::string v(env);
            for (auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (v == "cpu" || v == "none") return OrtGpuProvider::None;
            else if (v == "tensorrt" || v == "trt") requested = OrtGpuProvider::TensorRT;
            else if (v == "cuda" || v == "gpu") requested = OrtGpuProvider::Cuda;
            else {
                LOGI("Unknown SPEECH_CORE_ORT_PROVIDER='%s', using CPU", env);
                return OrtGpuProvider::None;
            }
        }

        const bool cuda_avail = provider_available("CUDAExecutionProvider");
        const bool trt_avail = provider_available("TensorrtExecutionProvider");

        if (requested == OrtGpuProvider::TensorRT) {
            if (trt_avail) { LOGI("ORT GPU provider: TensorRT"); return OrtGpuProvider::TensorRT; }
            if (cuda_avail) { LOGI("TensorRT EP not built into ORT, using CUDA"); return OrtGpuProvider::Cuda; }
            LOGI("No GPU EP available in this ORT build, using CPU");
            return OrtGpuProvider::None;
        }
        if (cuda_avail) { LOGI("ORT GPU provider: CUDA"); return OrtGpuProvider::Cuda; }
        LOGI("CUDA EP not available in this ORT build, using CPU");
        return OrtGpuProvider::None;
#else
        return OrtGpuProvider::None;
#endif
    }

    /// Query GetAvailableProviders() for an EP name. Works on every ORT build
    /// (the API itself is always present); a CPU-only runtime simply won't list
    /// CUDA/TensorRT, which is how the CPU-fallback path stays automatic.
    bool provider_available(const char* name) {
        char** providers = nullptr;
        int count = 0;
        OrtStatus* s = api_->GetAvailableProviders(&providers, &count);
        if (s != nullptr) {
            api_->ReleaseStatus(s);
            return false;
        }
        bool found = false;
        for (int i = 0; i < count; ++i) {
            if (std::strcmp(providers[i], name) == 0) { found = true; break; }
        }
        OrtStatus* rs = api_->ReleaseAvailableProviders(providers, count);
        if (rs != nullptr) api_->ReleaseStatus(rs);  // documented to never fail
        return found;
    }

    /// Append the resolved GPU EP to `opts`. Returns true if a GPU EP was
    /// appended (so the caller skips NNAPI/QNN). A single failure here is
    /// non-fatal: CreateSession's own fallback in load() re-tries CPU-only.
    /// We use the V2 opaque-options API (CreateCUDAProviderOptions /
    /// SessionOptionsAppendExecutionProvider_CUDA_V2), which the header
    /// documents as *returning failure* on a non-CUDA-enabled build rather
    /// than crashing — the basis of the silent CPU fallback.
    bool try_append_gpu(OrtSessionOptions* opts, const std::string& path) {
#if defined(SPEECH_CORE_WITH_CUDA) && !defined(__ANDROID__)
        const char* fname = path.c_str() + path.find_last_of('/') + 1;
        bool appended = false;

        if (gpu_provider_ == OrtGpuProvider::TensorRT) {
            OrtTensorRTProviderOptionsV2* trt = nullptr;
            OrtStatus* cs = api_->CreateTensorRTProviderOptions(&trt);
            if (cs == nullptr && trt != nullptr) {
                // Cache built TensorRT engines on disk so the (expensive) engine
                // build only happens on first run, not every cold start. Pick the
                // OS temp dir from env (TMPDIR on Unix, TEMP on Windows) so the
                // path is valid on both platforms; fall back to a sane default.
                const char* tmp = std::getenv("TMPDIR");
                if (tmp == nullptr) tmp = std::getenv("TEMP");
                if (tmp == nullptr) tmp = std::getenv("TMP");
#ifdef _WIN32
                std::string cache = std::string(tmp ? tmp : ".") + "\\speech_core_trt_cache";
#else
                std::string cache = std::string(tmp ? tmp : "/tmp") + "/speech_core_trt_cache";
#endif
                const char* keys[] = {"trt_engine_cache_enable", "trt_engine_cache_path"};
                const char* vals[] = {"1", cache.c_str()};
                OrtStatus* us = api_->UpdateTensorRTProviderOptions(trt, keys, vals, 2);
                if (us != nullptr) api_->ReleaseStatus(us);  // non-fatal: keep defaults
                OrtStatus* as = api_->SessionOptionsAppendExecutionProvider_TensorRT_V2(opts, trt);
                api_->ReleaseTensorRTProviderOptions(trt);
                if (as != nullptr) {
                    LOGI("TensorRT EP append failed (%s), trying CUDA beneath",
                         api_->GetErrorMessage(as));
                    api_->ReleaseStatus(as);
                } else {
                    LOGI("Appended TensorRT EP: %s", fname);
                    appended = true;
                }
            } else if (cs != nullptr) {
                api_->ReleaseStatus(cs);
            }
            // Layer CUDA beneath TensorRT so any op TensorRT can't take still
            // runs on GPU rather than dropping all the way to CPU.
        }

        OrtCUDAProviderOptionsV2* cuda = nullptr;
        OrtStatus* cs = api_->CreateCUDAProviderOptions(&cuda);
        if (cs != nullptr) {
            // Build is not actually CUDA-enabled (or OOM) — bail.
            api_->ReleaseStatus(cs);
            return appended;  // TRT may already have appended above
        }
        OrtStatus* as = api_->SessionOptionsAppendExecutionProvider_CUDA_V2(opts, cuda);
        api_->ReleaseCUDAProviderOptions(cuda);
        if (as != nullptr) {
            LOGI("CUDA EP append failed (%s)", api_->GetErrorMessage(as));
            api_->ReleaseStatus(as);
            return appended;
        }
        LOGI("Appended CUDA EP: %s", fname);
        return true;
#else
        (void)opts; (void)path;
        return false;
#endif
    }

    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* mem_ = nullptr;
    bool nnapi_fallback_ = false;
    std::string nnapi_fallback_reason_;
    OrtGpuProvider gpu_provider_ = OrtGpuProvider::None;
    bool gpu_fallback_ = false;
    std::string gpu_fallback_reason_;
};
