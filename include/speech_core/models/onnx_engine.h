#pragma once

#include <onnxruntime_c_api.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

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

/// Optional session-options customization callback. Called by `load()`
/// before `CreateSession`; if it returns true, NNAPI/QNN paths are skipped.
using SessionOptionsHook =
    std::function<bool(OrtSessionOptions*, const std::string&, bool)>;

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

    /// True if the SessionOptionsHook took over at least one loaded session.
    bool has_gpu_provider() const { return has_gpu_provider_; }

    /// Install a SessionOptionsHook. Call once at process init before model
    /// loads; not thread-safe with concurrent loads.
    static void set_session_options_hook(SessionOptionsHook hook) {
        get().hook_ = std::move(hook);
    }

    /// Resolve the intra-op thread count for ORT sessions.
    /// Default is 2 — empirically optimal on Parakeet's profile of many
    /// short calls per utterance (decoder-joint runs ~50 times, each on
    /// tiny tensors; thread-pool overhead per call > parallel-GEMM win).
    /// Measured on the LibriSpeech-100 bench (LiteRT v2.1, ORT 1.26 CUDA):
    ///   intra=2  CPU 28.77x RTF / CUDA 32.91x RTF (baseline)
    ///   intra=16 CPU 24.25x RTF / CUDA 26.49x RTF (15–20% slower)
    /// The env var stays available for long-utterance / large-batch
    /// workloads where the parallel win does dominate.
    static int resolve_intra_threads(int override_threads = 0) {
        if (override_threads > 0) return override_threads;
        if (const char* env = std::getenv("SPEECH_CORE_ORT_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 2;
    }

    OrtSession* load(const std::string& path, bool nnapi = true,
                     bool capture_hint = false, int intra_threads = 0) {
        OrtSessionOptions* opts = nullptr;
        ort_check(api_, api_->CreateSessionOptions(&opts));
        // Optimization level — default ORT_ENABLE_ALL, lowered via
        // SPEECH_CORE_ORT_OPT_LEVEL=disable_all/basic/extended. For INT8
        // dynamic-quantized models the heavier passes (DequantizeLinearFusion,
        // ConstantFolding over Q/DQ pairs) can materialise FP32 weights
        // host-side at session-load time, pushing peak RSS by several GB —
        // dropping to ORT_DISABLE_ALL on those sessions keeps the INT8
        // weights compact.
        GraphOptimizationLevel opt_level = ORT_ENABLE_ALL;
        if (const char* lvl = std::getenv("SPEECH_CORE_ORT_OPT_LEVEL")) {
            std::string v(lvl);
            for (auto& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (v == "disable_all" || v == "none" || v == "0") opt_level = ORT_DISABLE_ALL;
            else if (v == "basic")     opt_level = ORT_ENABLE_BASIC;
            else if (v == "extended")  opt_level = ORT_ENABLE_EXTENDED;
            else if (v == "all")       opt_level = ORT_ENABLE_ALL;
        }
        ort_check(api_, api_->SetSessionGraphOptimizationLevel(opts, opt_level));
        ort_check(api_, api_->SetIntraOpNumThreads(
            opts, resolve_intra_threads(intra_threads)));
        // EXPERIMENT: share a single CPU arena across all sessions. Each
        // session.use_env_allocators=1 routes its CPU allocations through
        // the env's registered arena instead of creating a per-session one.
        // VoxCPM2 bundle = 4 sessions; deduplicating arena overhead saves
        // ~0.5-1 GB. arena_extend_strategy=kSameAsRequested keeps growth
        // tight (no 1.5x speculative bumps).
        const char* env_alloc = std::getenv("SPEECH_CORE_USE_ENV_ALLOCATORS");
        if (!env_alloc || env_alloc[0] != '0') {
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_env_allocators", "1"));
        }
        // EXPERIMENT: turn off prepacking (weights re-arranged for faster
        // GEMM). For INT8 dynamic-quant models, prepacking allocates a CPU
        // buffer holding the prepacked weights even after upload. Off by
        // default — SPEECH_CORE_DISABLE_PREPACKING=1.
        if (const char* d = std::getenv("SPEECH_CORE_DISABLE_PREPACKING")) {
            if (d[0] == '1') {
                ort_check(api_, api_->AddSessionConfigEntry(
                    opts, "session.disable_prepacking", "1"));
            }
        }
        // EXPERIMENT: fuse DequantizeLinear + MatMul into MatMulNBits.
        // The INT8 Q/DQ + MatMul graph keeps FP16 dequantized weights
        // in CPU memory (~6 GB excess); MatMulNBits dequantizes inside
        // the GPU kernel with no CPU shadow. SPEECH_CORE_DQ_MATMULNBITS=1.
        if (const char* d = std::getenv("SPEECH_CORE_DQ_MATMULNBITS")) {
            if (d[0] == '1') {
                ort_check(api_, api_->AddSessionConfigEntry(
                    opts, "session.enable_dq_matmulnbits_fusion", "1"));
            }
        }
        // EXPERIMENT: disable QDQ constant folding so the dequantize step
        // doesn't materialize FP16 weights at session-load time.
        if (const char* d = std::getenv("SPEECH_CORE_DISABLE_QDQ_FOLD")) {
            if (d[0] == '1') {
                ort_check(api_, api_->AddSessionConfigEntry(
                    opts, "session.disable_qdq_constant_folding", "1"));
            }
        }
        // For quantized models on CUDA EP, ORT can otherwise stage the
        // dequantized weights through a CPU buffer that never gets freed,
        // pushing host RAM by 10+ GB for a 7-GB INT8 temporal model.
        // session.use_device_allocator_for_initializers=1 tells ORT to
        // allocate initializer buffers directly from the device allocator,
        // skipping the CPU staging. Gated by env so non-CUDA sessions opt out.
        // SPEECH_CORE_DEVICE_INITIALIZERS=0 disables, default on.
        if (const char* d = std::getenv("SPEECH_CORE_DEVICE_INITIALIZERS")) {
            if (d[0] != '0' && d[0] != '\0') {
                ort_check(api_, api_->AddSessionConfigEntry(
                    opts, "session.use_device_allocator_for_initializers", "1"));
            }
        } else {
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_device_allocator_for_initializers", "1"));
        }
        // Disable per-session memory arena + memory pattern to test whether
        // they were pre-allocating the host shadow buffer. Off by default —
        // SPEECH_CORE_DISABLE_MEM_ARENA=1 / SPEECH_CORE_DISABLE_MEM_PATTERN=1.
        if (const char* d = std::getenv("SPEECH_CORE_DISABLE_MEM_ARENA")) {
            if (d[0] == '1') {
                ort_check(api_, api_->DisableCpuMemArena(opts));
            }
        }
        if (const char* d = std::getenv("SPEECH_CORE_DISABLE_MEM_PATTERN")) {
            if (d[0] == '1') {
                ort_check(api_, api_->DisableMemPattern(opts));
            }
        }
        // EXPERIMENT: BF16 models route weights through a Cast(BF16->FP32)
        // node per initializer. ORT's ConstantFolding optimizer would
        // collapse each Cast into an FP32 constant at session-load time,
        // erasing the BF16 storage savings. Gate this behind an env var
        // so non-BF16 models still get all optimizations.
        if (const char* skip = std::getenv("SPEECH_CORE_ORT_DISABLE_OPTIMIZERS")) {
            if (skip[0] != '\0') {
                ort_check(api_, api_->AddSessionConfigEntry(
                    opts, "session.disable_specific_optimizers", skip));
            }
        }

        bool gpu_attempted = false;
        if (nnapi && hook_) {
            gpu_attempted = hook_(opts, path, capture_hint);
            if (gpu_attempted) has_gpu_provider_ = true;
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

        if (create_status != nullptr && nnapi) {
            const char* msg = api_->GetErrorMessage(create_status);
            LOGI("Hardware-EP session failed (%s), retrying CPU-only", msg);
            if (!gpu_attempted) {
                nnapi_fallback_ = true;
                nnapi_fallback_reason_ = msg;
            }
            api_->ReleaseStatus(create_status);
            api_->ReleaseSessionOptions(opts);

            opts = nullptr;
            ort_check(api_, api_->CreateSessionOptions(&opts));
            ort_check(api_, api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));
            ort_check(api_, api_->SetIntraOpNumThreads(
                opts, resolve_intra_threads(intra_threads)));
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_env_allocators", "1"));
            if (const char* skip = std::getenv("SPEECH_CORE_ORT_DISABLE_OPTIMIZERS")) {
                if (skip[0] != '\0') {
                    ort_check(api_, api_->AddSessionConfigEntry(
                        opts, "session.disable_specific_optimizers", skip));
                }
            }

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
        // Register a single CPU arena on the env. Each session opts in via
        // session.use_env_allocators=1 (set in load() above), so the 4
        // VoxCPM2 sessions share one arena instead of holding four. The
        // arena_extend_strategy=1 (kSameAsRequested) avoids the default
        // kNextPowerOfTwo growth — at ~8 GB initializers per session, the
        // 1.5x speculative bump would balloon RSS.
        OrtArenaCfg* arena_cfg = nullptr;
        ort_check(api_, api_->CreateArenaCfg(0, 1, -1, -1, &arena_cfg));
        ort_check(api_, api_->CreateAndRegisterAllocator(env_, mem_, arena_cfg));
        api_->ReleaseArenaCfg(arena_cfg);
    }

    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtMemoryInfo* mem_ = nullptr;
    bool nnapi_fallback_ = false;
    std::string nnapi_fallback_reason_;

    bool has_gpu_provider_ = false;
    SessionOptionsHook hook_;
};
