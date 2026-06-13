#pragma once

#include <onnxruntime_c_api.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
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

    /// Resolve the intra-op thread count for ORT sessions.
    /// Default is 2 — empirically optimal on Parakeet's profile of many
    /// short calls per utterance (decoder-joint runs ~50 times, each on
    /// tiny tensors; thread-pool overhead per call > parallel-GEMM win).
    /// Measured on the LibriSpeech-100 bench (LiteRT v2.1, ORT 1.26 CUDA):
    ///   intra=2  CPU 28.77x RTF / CUDA 32.91x RTF (baseline)
    ///   intra=16 CPU 24.25x RTF / CUDA 26.49x RTF (15–20% slower)
    /// The env var stays available for long-utterance / large-batch
    /// workloads where the parallel win does dominate.
    static int resolve_intra_threads() {
        if (const char* env = std::getenv("SPEECH_CORE_ORT_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) return v;
        }
        return 2;
    }

    OrtSession* load(const std::string& path, bool nnapi = true,
                     bool enable_cuda_graph = false) {
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
        api_->SetSessionGraphOptimizationLevel(opts, opt_level);
        api_->SetIntraOpNumThreads(opts, resolve_intra_threads());
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

        // A GPU EP (CUDA / TensorRT), when resolved on a desktop/server build,
        // takes priority over NNAPI/QNN. The `nnapi` argument is the model
        // wrapper's "use hardware acceleration" flag — when it is false (e.g.
        // Parakeet's FP32 decoder-joint, which must stay numerically exact) we
        // honour it and keep the session on CPU, GPU present or not.
        bool gpu_attempted = false;
        if (nnapi && gpu_provider_ != OrtGpuProvider::None) {
            gpu_attempted = try_append_gpu(opts, path, enable_cuda_graph);
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

        // If session creation fails with a hardware EP (NNAPI/QNN/GPU), retry.
        //
        // Three-tier fallback:
        //   (1) GPU + graph capture (current path)  — fastest when ORT can
        //       partition every node to the CUDAExecutionProvider.
        //   (2) GPU without graph capture           — happens when some op
        //       in the graph isn't CUDA-partitionable (Memcpy nodes appear
        //       at the GPU↔CPU boundary). The session still runs on GPU
        //       for the supported ops, just without the single-launch win.
        //   (3) CPU-only                            — last resort.
        //
        // Pre-fix we jumped straight from (1) to (3), which silently moved
        // the autoregressive token_step loop entirely off the GPU and made
        // RTF worse than a non-graph GPU run. The graph-capture failure has
        // a specific ORT error string we can detect to fork between (2)
        // and (3).
        bool graph_capture_failure = false;
        if (create_status != nullptr && gpu_attempted && enable_cuda_graph) {
            const char* msg = api_->GetErrorMessage(create_status);
            if (msg && std::strstr(msg, "graph capture")) {
                graph_capture_failure = true;
            }
        }

        if (create_status != nullptr && graph_capture_failure) {
            const char* msg = api_->GetErrorMessage(create_status);
            LOGI("CUDA Graph capture rejected by ORT (%s); retrying GPU EP "
                 "without graph capture", msg);
            api_->ReleaseStatus(create_status);
            api_->ReleaseSessionOptions(opts);

            opts = nullptr;
            ort_check(api_, api_->CreateSessionOptions(&opts));
            api_->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
            api_->SetIntraOpNumThreads(opts, resolve_intra_threads());
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_env_allocators", "1"));
            if (const char* skip = std::getenv("SPEECH_CORE_ORT_DISABLE_OPTIMIZERS")) {
                if (skip[0] != '\0') {
                    ort_check(api_, api_->AddSessionConfigEntry(
                        opts, "session.disable_specific_optimizers", skip));
                }
            }
            // Append GPU EP again, this time with enable_cuda_graph=false.
            if (!try_append_gpu(opts, path, /*enable_cuda_graph=*/false)) {
                // GPU append still failed for a non-graph reason — fall
                // through to CPU-only below.
                create_status = api_->CreateSession(env_, ort_path.c_str(),
                                                    opts, &session);
            } else {
                create_status = api_->CreateSession(env_, ort_path.c_str(),
                                                    opts, &session);
            }
        }

        // Three-tier fallback when the hardware-EP session create fails:
        //   (1) Originally appended GPU EP (TRT or CUDA + CUDA fallback) ✗
        //   (2) CUDA EP only — skip TRT EP entirely.  Many TRT EP
        //       failures (UNSUPPORTED_NODE_ATTR cascades, INVALID_NODE on
        //       shape ops like /Tile, /OneHot used by transformer-class
        //       graphs) throw inside TRT EP's GetSupportedList and abort
        //       the whole session even though CUDA EP was layered
        //       beneath. Retrying without TRT EP keeps the model on
        //       GPU instead of falling all the way to CPU.
        //   (3) CPU-only — last resort if even CUDA fails (no CUDA libs,
        //       cuDNN side-load missing, etc.)
        //
        // We only attempt (2) when (a) the requested GPU provider was
        // TensorRT (CUDA failures don't benefit from retry without TRT
        // since CUDA was already the only EP), and (b) the failure was
        // not a graph-capture failure (already handled above), and
        // (c) the session was the hardware-EP path (nnapi=true).
        bool trt_to_cuda_retry = (create_status != nullptr && nnapi &&
                                  gpu_attempted &&
                                  gpu_provider_ == OrtGpuProvider::TensorRT &&
                                  !graph_capture_failure);
        if (trt_to_cuda_retry) {
            const char* msg = api_->GetErrorMessage(create_status);
            LOGI("TensorRT EP session failed (%s), retrying with CUDA EP only",
                 msg);
            api_->ReleaseStatus(create_status);
            create_status = nullptr;
            api_->ReleaseSessionOptions(opts);

            opts = nullptr;
            ort_check(api_, api_->CreateSessionOptions(&opts));
            api_->SetSessionGraphOptimizationLevel(opts, opt_level);
            api_->SetIntraOpNumThreads(opts, resolve_intra_threads());
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_env_allocators", "1"));
            ort_check(api_, api_->AddSessionConfigEntry(
                opts, "session.use_device_allocator_for_initializers", "1"));
            if (const char* skip = std::getenv("SPEECH_CORE_ORT_DISABLE_OPTIMIZERS")) {
                if (skip[0] != '\0') {
                    ort_check(api_, api_->AddSessionConfigEntry(
                        opts, "session.disable_specific_optimizers", skip));
                }
            }

            // Temporarily override gpu_provider_ so try_append_gpu only
            // appends CUDA EP — not TRT + CUDA underneath.
            OrtGpuProvider saved = gpu_provider_;
            gpu_provider_ = OrtGpuProvider::Cuda;
            bool cuda_attempted = try_append_gpu(opts, path,
                                                 /*enable_cuda_graph=*/false);
            gpu_provider_ = saved;

            if (cuda_attempted) {
                create_status = api_->CreateSession(env_, ort_path.c_str(),
                                                    opts, &session);
                if (create_status == nullptr) {
                    gpu_fallback_ = true;
                    gpu_fallback_reason_ =
                        "TensorRT EP failed, served on CUDA EP";
                    LOGI("CUDA EP retry succeeded for %s",
                         path.substr(path.find_last_of('/') + 1).c_str());
                }
            }
        }

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
            api_->SetIntraOpNumThreads(opts, resolve_intra_threads());
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
        gpu_provider_ = resolve_gpu_provider();
    }

    OnnxEngine(const OnnxEngine&) = delete;
    OnnxEngine& operator=(const OnnxEngine&) = delete;

public:
    /// Whether the SPEECH_CORE_CUDA_GRAPH env var is set to "1" (default off).
    ///
    /// IMPORTANT — currently inert scaffolding. Returning true here does NOT
    /// engage CUDA Graph capture in any wrapper:
    ///   * No wrapper currently consults this flag to gate dynamic-shape paths
    ///     (despite the original docstring promising they would).
    ///   * ORT's op partitioning still blocks capture for voxcpm2-token-step:
    ///     the exported graph has 36 Memcpy bridges between CPU and GPU
    ///     subgraphs, and capture requires every op to live on the GPU.
    /// The only observable effect is the "CUDA Graph capture enabled for: ..."
    /// log line at session load via try_append_gpu (see UpdateCUDAProviderOptions
    /// in this file). Kept so future wrappers can wire shape-stability gates
    /// once the export is fully CUDA-resident.
    bool cuda_graph_enabled() const {
        const char* env = std::getenv("SPEECH_CORE_CUDA_GRAPH");
        return env != nullptr && env[0] == '1';
    }

private:
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
    bool try_append_gpu(OrtSessionOptions* opts, const std::string& path,
                        bool enable_cuda_graph = false) {
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
                // Base options: engine cache always on; tuning knobs opt-in.
                std::vector<const char*> keys = {
                    "trt_engine_cache_enable", "trt_engine_cache_path"};
                std::vector<const char*> vals = {"1", cache.c_str()};

                // SPEECH_CORE_TRT_MAX_WORKSPACE_MB — per-tactic build
                // workspace ceiling (bytes). Default: unbounded (TRT picks).
                // TRT's tactic-search engine build materializes candidate
                // engines into this workspace and runs throughput micro-
                // benchmarks. For large transformers (e.g. VoxCPM2 prefill
                // ~0.6B params, ~hundreds of MatMul/QDQ nodes), the search
                // space is huge and peak host RAM during compile can exceed
                // 30 GiB — Salad's 16 GiB pod SIGKILLs the worker mid-build.
                // Capping the workspace to 2048 MB keeps build memory in
                // budget at the cost of TRT possibly picking suboptimal
                // tactics. Empirically VoxCPM2 prefill fits its 16 GiB pod
                // at 2048 MB; tune up if the engine choice regresses RTF.
                std::string ws_str;
                if (const char* ws = std::getenv("SPEECH_CORE_TRT_MAX_WORKSPACE_MB")) {
                    long long mb = std::atoll(ws);
                    if (mb > 0) {
                        unsigned long long bytes =
                            static_cast<unsigned long long>(mb) * 1024ULL * 1024ULL;
                        ws_str = std::to_string(bytes);
                        keys.push_back("trt_max_workspace_size");
                        vals.push_back(ws_str.c_str());
                    }
                }

                // SPEECH_CORE_TRT_BUILDER_OPT_LEVEL — TRT tactic search
                // depth, 0 (cheapest) to 5 (deepest). Default 3. Lower =
                // faster engine build, less memory, possibly suboptimal
                // kernel choices. 0-1 for memory-constrained one-shot
                // builds (CI, first cold-start); 2-3 for warm production
                // where build time isn't on the critical path. Combine
                // with TRT_MAX_WORKSPACE_MB for tight memory envelopes.
                if (const char* lvl = std::getenv("SPEECH_CORE_TRT_BUILDER_OPT_LEVEL")) {
                    if (lvl[0] != '\0') {
                        keys.push_back("trt_builder_optimization_level");
                        vals.push_back(lvl);
                    }
                }

                // SPEECH_CORE_TRT_OP_TYPES_TO_EXCLUDE — comma-separated
                // list of ONNX op types the partitioner should NEVER
                // attempt. Default: empty (try all).
                //
                // Why: TRT's ONNX importer probes every op type and
                // raises UNSUPPORTED_NODE_ATTR for ones it can't handle
                // (e.g. ScatterND with `reduction=` attribute, opset 16,
                // used in VoxCPM2's KV-cache write path). ORT correctly
                // falls those nodes back to CUDA EP, but each rejected
                // import allocates + tears down a parser context first.
                // For graphs with dozens of such nodes the cumulative
                // host RAM trips the kernel OOM-killer before serving.
                //
                // Listing them up front routes those nodes straight to
                // CUDA from the first analysis pass — smaller TRT
                // subgraphs, less compile memory, no error spam.
                if (const char* exc = std::getenv("SPEECH_CORE_TRT_OP_TYPES_TO_EXCLUDE")) {
                    if (exc[0] != '\0') {
                        keys.push_back("trt_op_types_to_exclude");
                        vals.push_back(exc);
                    }
                }

                OrtStatus* us = api_->UpdateTensorRTProviderOptions(
                    trt, keys.data(), vals.data(), keys.size());
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
        // CUDA Graph capture: a single Graph Launch replaces ~1400 individual
        // kernel launches per token_step iteration, eliminating the launch
        // overhead bottleneck for small-tensor transformer steps. Requires
        // fixed input shapes (we have them) and IoBinding (also in place).
        // Without these two, ORT silently falls back to per-call launches.
        if (enable_cuda_graph) {
            const char* keys[] = {"enable_cuda_graph"};
            const char* vals[] = {"1"};
            OrtStatus* us = api_->UpdateCUDAProviderOptions(cuda, keys, vals, 1);
            if (us != nullptr) {
                LOGI("CUDA Graph option update failed (%s); continuing without graph capture",
                     api_->GetErrorMessage(us));
                api_->ReleaseStatus(us);
            } else {
                LOGI("CUDA Graph capture enabled for: %s", fname);
            }
        }
        // VRAM-saving CUDA EP knobs. All opt-in via env.
        //   SPEECH_CORE_GPU_MEM_LIMIT_GB=12       -> hard cap on EP arena
        //   SPEECH_CORE_CUDA_ARENA_STRATEGY=kSameAsRequested|kNextPowerOfTwo
        //   SPEECH_CORE_CUDNN_NO_MAX_WS=1         -> conservative cuDNN workspace
        //   SPEECH_CORE_CUDNN_HEURISTIC=1         -> heuristic conv algo (less workspace)
        std::vector<const char*> ep_keys;
        std::vector<const char*> ep_vals;
        std::string mem_limit_str;
        if (const char* gb = std::getenv("SPEECH_CORE_GPU_MEM_LIMIT_GB")) {
            unsigned long long bytes = static_cast<unsigned long long>(std::atof(gb) * (1024ULL*1024ULL*1024ULL));
            mem_limit_str = std::to_string(bytes);
            ep_keys.push_back("gpu_mem_limit");
            ep_vals.push_back(mem_limit_str.c_str());
        }
        if (const char* s = std::getenv("SPEECH_CORE_CUDA_ARENA_STRATEGY")) {
            if (s[0] != '\0') {
                ep_keys.push_back("arena_extend_strategy");
                ep_vals.push_back(s);  // "kSameAsRequested" or "kNextPowerOfTwo"
            }
        }
        if (const char* s = std::getenv("SPEECH_CORE_CUDNN_NO_MAX_WS")) {
            if (s[0] == '1') {
                ep_keys.push_back("cudnn_conv_use_max_workspace");
                ep_vals.push_back("0");
            }
        }
        if (const char* s = std::getenv("SPEECH_CORE_CUDNN_HEURISTIC")) {
            if (s[0] == '1') {
                ep_keys.push_back("cudnn_conv_algo_search");
                ep_vals.push_back("HEURISTIC");
            }
        }
        if (!ep_keys.empty()) {
            OrtStatus* us = api_->UpdateCUDAProviderOptions(cuda, ep_keys.data(), ep_vals.data(), ep_keys.size());
            if (us != nullptr) {
                LOGI("CUDA EP options update failed (%s); using defaults", api_->GetErrorMessage(us));
                api_->ReleaseStatus(us);
            } else {
                LOGI("Applied %zu CUDA EP option(s) on %s", ep_keys.size(), fname);
            }
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
