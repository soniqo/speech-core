#!/usr/bin/env bash
# Cross-compile liblitert-lm.so for Android arm64 from the
# google-ai-edge/LiteRT-LM source tree. Google does not publish an Android
# build of liblitert-lm via PyPI (macOS wheel only) or Google Maven (only the
# Kotlin facade), so consumers that need the raw C ABI (speech-core's
# LiteRTFunctionGemmaLLM) build it themselves from the pinned upstream tag.
#
# The companion fetch_litert_lm.sh extracts the macOS dylib from the
# litert-lm-api PyPI wheel; this script is its Android sibling. Outputs land
# in the same ${LITERT_LM_DIR}/${ANDROID_ABI}/ layout the top-level CMake
# expects (see the SPEECH_CORE_WITH_LITERT_LM block in CMakeLists.txt).
#
# Usage:
#     scripts/build_litert_lm_android.sh [output_dir] [tag]
#
# Defaults: output_dir = build/litert_lm ; tag = v0.13.1
#
# Required env:
#     ANDROID_NDK_HOME   Path to NDK r28b+ (the LiteRT-LM Bazel rules expect
#                        an NDK source.properties at this path).
#
# Optional env:
#     LITERT_LM_SRC      Where to keep the cloned LiteRT-LM checkout. Defaults
#                        to $HOME/.cache/speech-core/litert-lm-src so the Bazel
#                        cache (~10 GB after a full build) sits outside the
#                        speech-core tree.
#     JAVA_HOME          Bazel needs a JDK; if unset and Homebrew openjdk@21
#                        is present, we point JAVA_HOME at it.
#     ALSO_X86_64        If set to "1", also cross-compile for x86_64.
#                        (Google's macOS-arm64 emulator does NOT host x86_64
#                        system images, so this is mainly for x86_64 Linux
#                        Android devices / CI runners.) Doubles the build time.
#     EMULATOR_SAFE      If set to "1", disable XNNPack assembly + KleidiAI
#                        SME kernels. Required to run on the Android emulator
#                        hosted on Apple Silicon via HVF — the guest CPU
#                        presents as implementer 0x61 (Apple) and HVF
#                        mistraps Apple-specific assembly + the SME `rdsvl`
#                        instruction in KleidiAI's kai_common_sme_asm.S
#                        (SIGILL at hardware-config init). Costs 3-5× decode
#                        rate but unblocks emulator dev-loop. Real devices
#                        (any Snapdragon/Exynos/etc.) do NOT need this.
#
# Build deps (one-time install):
#     brew install bazelisk openjdk@21 git-lfs
#     brew install --cask android-ndk
#
# The build pulls in TensorFlow, ABSL, Skia, Dawn, sentencepiece, and a Rust
# toolchain (llguidance, tokenizers_cpp). Expect 30-90 min cold; ~5 min warm.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/litert_lm}"
TAG="${2:-v0.13.1}"
SRC="${LITERT_LM_SRC:-$HOME/.cache/speech-core/litert-lm-src}"
ALSO_X86_64="${ALSO_X86_64:-0}"

# ---------------------------------------------------------------------------
# Prereq checks
# ---------------------------------------------------------------------------

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
    if [[ -d "/opt/homebrew/share/android-ndk" ]]; then
        export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
        echo "[build_litert_lm_android] ANDROID_NDK_HOME=$ANDROID_NDK_HOME (from Homebrew)"
    else
        echo "[build_litert_lm_android] ANDROID_NDK_HOME not set — install with 'brew install --cask android-ndk' (r29+) or download r28b+ from https://developer.android.com/ndk/downloads" >&2
        exit 1
    fi
fi

if [[ ! -f "$ANDROID_NDK_HOME/source.properties" ]]; then
    echo "[build_litert_lm_android] $ANDROID_NDK_HOME/source.properties not found — point ANDROID_NDK_HOME at the directory that contains source.properties" >&2
    exit 1
fi

if [[ -z "${JAVA_HOME:-}" ]] && [[ -d "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home" ]]; then
    export JAVA_HOME="/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "[build_litert_lm_android] JAVA_HOME=$JAVA_HOME (from Homebrew openjdk@21)"
fi

for tool in bazelisk git git-lfs; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "[build_litert_lm_android] missing '$tool' — brew install bazelisk openjdk@21 git-lfs" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Clone or update LiteRT-LM at the pinned tag
# ---------------------------------------------------------------------------

mkdir -p "$(dirname "$SRC")"

if [[ ! -d "$SRC/.git" ]]; then
    echo "[build_litert_lm_android] Cloning google-ai-edge/LiteRT-LM@$TAG into $SRC"
    git clone --quiet --branch "$TAG" --depth=1 \
        https://github.com/google-ai-edge/LiteRT-LM "$SRC"
else
    cd "$SRC"
    CURRENT="$(git describe --tags --exact-match 2>/dev/null || echo unknown)"
    if [[ "$CURRENT" != "$TAG" ]]; then
        echo "[build_litert_lm_android] Existing checkout at $CURRENT — switching to $TAG"
        git fetch --quiet --depth=1 origin "refs/tags/$TAG:refs/tags/$TAG"
        git checkout --quiet "tags/$TAG"
    else
        echo "[build_litert_lm_android] Reusing existing checkout at $TAG"
    fi
fi

cd "$SRC"
git lfs install --local --skip-smudge >/dev/null
echo "[build_litert_lm_android] git lfs pull (prebuilt accelerators ~400 MB)"
git lfs pull

# ---------------------------------------------------------------------------
# Patch c/BUILD: v0.13.1 ships only cc_library(:engine) — the cc_binary
# shared-library target was added on main after the tag was cut. We append
# the same target back so we can build a .so without bumping the tag (and
# losing the byte-for-byte match with speech-core's vendored C header).
# Idempotent: skip if our marker is already present.
# ---------------------------------------------------------------------------

if ! grep -q "Added by speech-core/scripts/build_litert_lm_android.sh" c/BUILD; then
    echo "[build_litert_lm_android] Patching c/BUILD with cc_binary :litert-lm target"
    # Linker version script: export ONLY litert_lm_* symbols. Anonymous
    # version (no name) so consumers don't pick up a DT_VERNEED entry.
    # Combined with --exclude-libs,ALL this drops 13,938 leaked symbols
    # (protobuf, absl, libc++, TFLite internals) down to the 89 we declare.
    cat > c/litert_lm.lds <<'LDS_SCRIPT'
{
    global:
        litert_lm_*;
    local:
        *;
};
LDS_SCRIPT
    cat >> c/BUILD <<'BUILD_PATCH'

# Added by speech-core/scripts/build_litert_lm_android.sh — the v0.13.1 tag
# was published without a cc_binary shared-library target (main has one).
# We add it back here so we can cross-compile liblitert-lm.so for Android
# without bumping past the tag that matches speech-core's vendored C header.
cc_binary(
    name = "litert-lm",
    srcs = [
        "engine.cc",
        "engine.h",
    ],
    additional_linker_inputs = [":litert_lm.lds"],
    copts = ["-fvisibility=hidden"],
    features = ["-legacy_whole_archive"],
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:ios": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
            "-Wl,--no-undefined",
            "-Wl,--strip-all",
            "-Wl,--gc-sections",
            "-Wl,-Bsymbolic",
            # Hide ALL symbols from statically-linked dependencies AND
            # restrict exports to litert_lm_* via the version script.
            # Without both, protobuf / absl / libc++ symbols leak out and:
            #   (a) clash with the system copies Android loads transitively
            #       via libEGL.so etc., causing a SIGSEGV in
            #       /system/lib64/libprotobuf-cpp-lite.so global ctor; and
            #   (b) cause speech-core's test binary to dynamic-link libc++
            #       symbols (e.g. std::__ndk1::bad_function_call) from
            #       liblitert-lm.so instead of its own static libc++.
            "-Wl,--exclude-libs,ALL",
            "-Wl,--version-script=$(location :litert_lm.lds)",
            "-Wl,-rpath,$$ORIGIN",
        ],
    }),
    linkshared = 1,
    deps = ENGINE_COMMON_DEPS + [
        "//runtime/core:engine_impl",
    ],
)
BUILD_PATCH
fi

# ---------------------------------------------------------------------------
# Bazel build
# ---------------------------------------------------------------------------

build_one_abi() {
    local CONFIG="$1"
    local ABI="$2"

    # Optional: build a .so that runs on Apple-HVF Android emulators (where
    # the guest CPU presents as implementer 0x61 / Apple and HVF passthrough
    # mistraps Apple-specific assembly + the SME `rdsvl` instruction in
    # KleidiAI's `kai_common_sme_asm.S`). Setting EMULATOR_SAFE=1 drops to
    # XNNPack's portable C kernels and bypasses KleidiAI; real devices don't
    # need this and pay a 3-5× decode-rate penalty for the C path.
    local EXTRA_DEFINES=()
    if [[ "${EMULATOR_SAFE:-0}" == "1" ]]; then
        EXTRA_DEFINES+=(
            --define=xnn_enable_assembly=false
            --define=xnn_enable_kleidiai=false
        )
        echo "[build_litert_lm_android] EMULATOR_SAFE=1 — disabling XNNPack assembly + KleidiAI for Apple-HVF emulator compatibility"
    fi

    echo "[build_litert_lm_android] bazelisk build --config=$CONFIG //c:litert-lm"
    # Force hermetic Python 3.13 — host Python may be 3.9 (rules_python only
    # ships requirement locks for 3.10+).
    bazelisk build --config="$CONFIG" \
        --repo_env=HERMETIC_PYTHON_VERSION=3.13 \
        "${EXTRA_DEFINES[@]}" \
        //c:litert-lm

    # //c:litert-lm is a cc_binary with linkshared=1 — on Linux/Android the
    # output filename is "litert-lm" (no lib prefix, no .so suffix) because
    # Bazel's cc_binary doesn't auto-suffix shared binaries the way
    # cc_library does. Find whichever shape Bazel produced.
    local BUILT
    BUILT="$(find bazel-bin/c -maxdepth 1 -type f \
        \( -name 'liblitert-lm.so' -o -name 'litert-lm' \) | head -n1)"
    if [[ -z "$BUILT" ]]; then
        echo "[build_litert_lm_android] could not locate built liblitert-lm in bazel-bin/c/" >&2
        ls -la bazel-bin/c/ >&2
        exit 1
    fi

    local DEST="$OUT/$ABI"
    mkdir -p "$DEST"
    cp -f "$BUILT" "$DEST/liblitert-lm.so"

    # Strip debug info to keep the .so as small as possible — speech-core
    # consumers can resymbolicate via the original bazel-bin/ if needed.
    local STRIP="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/llvm-strip"
    if [[ ! -x "$STRIP" ]]; then
        # Apple Silicon NDK toolchain uses darwin-x86_64 directory name even on arm64.
        STRIP="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-strip"
    fi
    if [[ -x "$STRIP" ]]; then
        "$STRIP" --strip-unneeded "$DEST/liblitert-lm.so" 2>/dev/null || true
    fi

    # Copy the prebuilt accelerators that ship with the repo so consumers can
    # opt into GPU acceleration and FunctionGemma's constrained-decoding tool
    # grammar (libGemmaModelConstraintProvider.so) without a second fetch.
    local PREBUILT_DIR="prebuilt/${ABI//-/_}"
    if [[ "$ABI" == "arm64-v8a" ]]; then PREBUILT_DIR="prebuilt/android_arm64"; fi
    if [[ "$ABI" == "x86_64"    ]]; then PREBUILT_DIR="prebuilt/android_x86_64"; fi
    if [[ -d "$PREBUILT_DIR" ]]; then
        cp -f "$PREBUILT_DIR"/*.so "$DEST/" 2>/dev/null || true
    fi

    echo "[build_litert_lm_android] Wrote $ABI:"
    ls -la "$DEST/" | awk 'NR>1 {printf "  %10s  %s\n", $5, $NF}'
}

build_one_abi "android_arm64" "arm64-v8a"

if [[ "$ALSO_X86_64" == "1" ]]; then
    build_one_abi "android_x86_64" "x86_64"
fi

# ---------------------------------------------------------------------------
# Symbol-export smoke check — speech-core's LiteRTFunctionGemmaLLM calls 22
# litert_lm_* C symbols. Confirm they survived static linking + -fvisibility=hidden.
# ---------------------------------------------------------------------------

NM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-nm"
if [[ -x "$NM" ]]; then
    SO="$OUT/arm64-v8a/liblitert-lm.so"
    EXPORTED="$("$NM" --defined-only --dynamic --extern-only "$SO" 2>/dev/null \
        | awk '/ T / {print $3}' | grep -c '^litert_lm_' || true)"
    echo "[build_litert_lm_android] $SO exports $EXPORTED litert_lm_* symbols (expect ~70)"
    if [[ "$EXPORTED" -lt 22 ]]; then
        echo "[build_litert_lm_android] WARNING: fewer than 22 litert_lm_* symbols exported — speech-core's wrapper needs at least the 22 it calls (engine_settings_create, conversation_send_message, json_response_*, ...). Re-check c/BUILD." >&2
    fi
fi

echo "[build_litert_lm_android] Done. Point speech-core's cmake at -DLITERT_LM_DIR=$OUT"
