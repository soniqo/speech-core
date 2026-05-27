// Vendored from litert/build_common/build_config.h.in.
// Configures LiteRT for CPU-only client use (no GPU or NPU delegates).
// We don't ship those delegate runtimes — the engine selects CPU by default.

#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU 1
#define LITERT_BUILD_CONFIG_DISABLE_NPU 1

#if LITERT_BUILD_CONFIG_DISABLE_GPU
#define LITERT_DISABLE_GPU
#endif

#if LITERT_BUILD_CONFIG_DISABLE_NPU
#define LITERT_DISABLE_NPU
#endif

#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
