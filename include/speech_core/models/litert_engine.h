#pragma once

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_types.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
#define LOGI(...) do { std::fprintf(stderr, "[speech] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
#ifndef LOGE
#define LOGE(...) do { std::fprintf(stderr, "[speech ERROR] "); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } while(0)
#endif
#endif

namespace speech_core {

inline void litert_check(LiteRtStatus status, const char* what) {
    if (status != kLiteRtStatusOk) {
        throw std::runtime_error(std::string("LiteRT: ") + what
                                 + " failed (status=" + std::to_string(status) + ")");
    }
}

/// Build a ranked tensor type from element dtype + static dims (rank ≤ 8).
inline LiteRtRankedTensorType make_type(LiteRtElementType dtype,
                                         std::initializer_list<int32_t> dims) {
    LiteRtRankedTensorType t{};
    t.element_type       = dtype;
    t.layout.rank        = static_cast<unsigned int>(dims.size());
    t.layout.has_strides = false;
    size_t i = 0;
    for (int32_t d : dims) t.layout.dimensions[i++] = d;
    return t;
}

/// Total element count from a layout. Assumes all dims are static.
inline size_t layout_element_count(const LiteRtLayout& l) {
    size_t n = 1;
    for (unsigned int i = 0; i < l.rank; ++i) {
        n *= static_cast<size_t>(l.dimensions[i]);
    }
    return n;
}

/// RAII wrapper around a *managed* host-memory TensorBuffer.
/// We tried wrapping caller-owned memory via `LiteRtCreateTensorBufferFromHostMemory`
/// but it returns `kLiteRtStatusErrorMemoryAllocationFailure` for the sizes
/// VoxCPM2 uses — LiteRT v2.1.x's host-memory backend has stricter alignment
/// requirements than std::vector::data() guarantees. The managed-allocation
/// path picks aligned memory internally; we just lock/write before Run and
/// lock/read after. Buffer lifetime is per-Invoke (cheap to construct).
class LiteRtHostBuffer {
public:
    /// Allocate a managed host-memory tensor buffer of the given size and
    /// (optionally) seed it with the contents of `seed` (e.g. an input).
    LiteRtHostBuffer(LiteRtEnvironment env,
                     const LiteRtRankedTensorType& type,
                     size_t bytes,
                     const void* seed = nullptr) : bytes_(bytes) {
        litert_check(LiteRtCreateManagedTensorBuffer(
                         env, kLiteRtTensorBufferTypeHostMemory, &type, bytes, &buf_),
                     "CreateManagedTensorBuffer");
        if (seed) write(seed, bytes);
    }
    ~LiteRtHostBuffer() { if (buf_) LiteRtDestroyTensorBuffer(buf_); }

    LiteRtHostBuffer(const LiteRtHostBuffer&)            = delete;
    LiteRtHostBuffer& operator=(const LiteRtHostBuffer&) = delete;
    LiteRtHostBuffer(LiteRtHostBuffer&& o) noexcept
        : buf_(o.buf_), bytes_(o.bytes_) { o.buf_ = nullptr; }
    LiteRtHostBuffer& operator=(LiteRtHostBuffer&&)      = delete;

    /// Copy `bytes` from `src` into this buffer's host memory.
    void write(const void* src, size_t bytes) {
        void* p = nullptr;
        litert_check(LiteRtLockTensorBuffer(buf_, &p, kLiteRtTensorBufferLockModeWrite),
                     "LockTensorBuffer(write)");
        std::memcpy(p, src, bytes);
        litert_check(LiteRtUnlockTensorBuffer(buf_), "UnlockTensorBuffer");
    }

    /// Copy `bytes` from this buffer's host memory into `dst`.
    void read(void* dst, size_t bytes) const {
        void* p = nullptr;
        litert_check(LiteRtLockTensorBuffer(buf_, &p, kLiteRtTensorBufferLockModeRead),
                     "LockTensorBuffer(read)");
        std::memcpy(dst, p, bytes);
        litert_check(LiteRtUnlockTensorBuffer(buf_), "UnlockTensorBuffer");
    }

    size_t             byte_size() const { return bytes_; }
    LiteRtTensorBuffer raw()       const { return buf_; }

private:
    LiteRtTensorBuffer buf_   = nullptr;
    size_t             bytes_ = 0;
};

/// Process-wide LiteRT environment + per-model load helper.
///
/// Backed by `libLiteRt.{so,dll,dylib}` from Google's `ai-edge-litert` package
/// (extracted from the PyPI wheel by `scripts/fetch_litert.sh` locally; pulled
/// the same way in CI). Replaces the legacy `libtensorflowlite_c` path — the
/// old TFLite C API in our v2.18-v2.20 source builds couldn't load >2 GB
/// models (VoxCPM2's text_prefill is 2.08 GB).
class LiteRTEngine {
public:
    static LiteRTEngine& get() {
        static LiteRTEngine instance;
        return instance;
    }

    LiteRtEnvironment env() {
        if (!env_) {
            litert_check(LiteRtCreateEnvironment(0, nullptr, &env_), "CreateEnvironment");
        }
        return env_;
    }

    /// Load a `.tflite` and compile it for CPU execution.
    ///
    /// `out_model` and `out_compiled` are caller-owned. Free in reverse order:
    /// `LiteRtDestroyCompiledModel(compiled)` first, then `LiteRtDestroyModel(model)`.
    void load(const std::string& path,
              bool /*hw_accel*/,
              LiteRtModel* out_model,
              LiteRtCompiledModel* out_compiled) {
        LOGI("Loading LiteRT model: %s",
             path.substr(path.find_last_of('/') + 1).c_str());

        LiteRtModel m = nullptr;
        // LiteRT v2.1.5's LiteRtCreateModelFromFile fails on Windows for files
        // ≥ 2 GiB ("Failed to get file size" — 32-bit stat overflow). VoxCPM2's
        // token-step graph is 2.04 GiB, so the file API can't load it. Use
        // LiteRtCreateModelFromBuffer (size_t buffer_size is 64-bit) for big
        // files, falling back to the file API otherwise so most loads stay
        // unchanged. The buffer is zero-copy and must outlive the model, so
        // we retain it in the engine singleton. We also cache the buffer by
        // path: a test suite that reloads the same VoxCPM2 graphs across six
        // wrapper instances would otherwise sink 6 × ~4.5 GiB ≈ 27 GiB of
        // RAM (CI Linux runners have ~7 GiB and SIGKILL'd at 9 min). Caching
        // caps it at one copy per path. Threshold is well under 2 GiB so the
        // prefill graph (1.94 GiB) also routes through the safer path on
        // Windows.
        constexpr std::uint64_t kBufferThreshold = std::uint64_t{1} << 30;  // 1 GiB
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            throw std::runtime_error("LiteRT: cannot open " + path);
        }
        const std::uint64_t size = static_cast<std::uint64_t>(f.tellg());
        if (size > kBufferThreshold) {
            auto it = retained_buffers_.find(path);
            const std::vector<char>* buf_ptr = nullptr;
            if (it != retained_buffers_.end()) {
                buf_ptr = it->second.get();
            } else {
                auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(size));
                f.seekg(0);
                f.read(buf->data(), static_cast<std::streamsize>(size));
                if (!f) {
                    throw std::runtime_error("LiteRT: read failed for " + path);
                }
                buf_ptr = buf.get();
                retained_buffers_.emplace(path, std::move(buf));
            }
            litert_check(LiteRtCreateModelFromBuffer(buf_ptr->data(), buf_ptr->size(), &m),
                         "CreateModelFromBuffer");
        } else {
            f.close();
            litert_check(LiteRtCreateModelFromFile(path.c_str(), &m), "CreateModelFromFile");
        }

        // Build compile options with the CPU accelerator. LiteRT rejects a
        // NULL options pointer (kLiteRtStatusErrorInvalidArgument). GPU/NPU
        // delegates aren't shipped with the desktop wheel — CPU is what every
        // platform consistently has.
        LiteRtOptions opts = nullptr;
        litert_check(LiteRtCreateOptions(&opts), "CreateOptions");
        LiteRtStatus s = LiteRtSetOptionsHardwareAccelerators(opts, kLiteRtHwAcceleratorCpu);
        if (s != kLiteRtStatusOk) {
            LiteRtDestroyOptions(opts);
            LiteRtDestroyModel(m);
            litert_check(s, "SetOptionsHardwareAccelerators");
        }

        LiteRtCompiledModel c = nullptr;
        s = LiteRtCreateCompiledModel(env(), m, opts, &c);
        LiteRtDestroyOptions(opts);
        if (s != kLiteRtStatusOk) {
            LiteRtDestroyModel(m);
            litert_check(s, "CreateCompiledModel");
        }
        *out_model    = m;
        *out_compiled = c;
    }

    /// Release the retained file buffer for `path`, if any.
    ///
    /// The caller MUST have already destroyed any `LiteRtModel` created from
    /// this path -- the model holds a zero-copy pointer into the buffer, so
    /// freeing the buffer while a model still references it is undefined
    /// behaviour. This is the lazy-unload path used by the VoxCPM2 wrapper
    /// to drop the ~1.9 GiB text_prefill graph between synthesize() calls,
    /// freeing ~2 GiB of node headroom on the prod CCX23 (16 GiB total,
    /// otherwise too cramped to host inference + Postgres + NATS + MinIO +
    /// realtime-worker side-by-side). No-op when `path` was below the
    /// CreateModelFromBuffer threshold (1 GiB) and therefore was never
    /// retained.
    void release_buffer(const std::string& path) {
        retained_buffers_.erase(path);
    }

private:
    LiteRTEngine() = default;
    ~LiteRTEngine() { if (env_) LiteRtDestroyEnvironment(env_); }
    LiteRTEngine(const LiteRTEngine&)            = delete;
    LiteRTEngine& operator=(const LiteRTEngine&) = delete;

    LiteRtEnvironment env_ = nullptr;
    // Backing storage for models loaded via LiteRtCreateModelFromBuffer,
    // keyed by file path. LiteRT retains a zero-copy pointer into each
    // buffer for the model's lifetime, so buffers must outlive any models
    // created from them. The engine is a singleton, so this naturally
    // lives until process exit. Keying by path means re-loading the same
    // file reuses the existing buffer instead of allocating another copy
    // (matters for test suites that re-instantiate big-model wrappers).
    std::unordered_map<std::string, std::unique_ptr<std::vector<char>>> retained_buffers_;
};

}  // namespace speech_core
