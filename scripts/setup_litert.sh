#!/usr/bin/env bash
# Build the LiteRT (TFLite) C API shared library from TensorFlow sources,
# then arrange it under $OUT/{include,lib} so it can be used as LITERT_DIR.
#
# Usage:
#     scripts/setup_litert.sh [output_dir] [tf_version]
#
# Defaults: output_dir = build/litert ; tf_version = v2.18.0
#
# The script is ~30 min on a fast machine (Bazel-equivalent CMake build with
# eigen, xnnpack, abseil, ruy, fft2d, farmhash, flatbuffers). Pin a release
# tag rather than HEAD to keep CI reproducible.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/litert}"
TF_VERSION="${2:-v2.18.0}"

TF_SRC="$OUT/tensorflow-${TF_VERSION}"
BUILD_DIR="$OUT/build-${TF_VERSION}"

mkdir -p "$OUT"

if [[ ! -d "$TF_SRC/.git" ]]; then
    echo "[setup_litert] Cloning TensorFlow ${TF_VERSION}..."
    git clone --depth=1 --branch="${TF_VERSION}" \
        https://github.com/tensorflow/tensorflow "$TF_SRC"
fi

echo "[setup_litert] Configuring TFLite C build..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# NNAPI/GPU/Metal off — we ship CPU only in this initial wrapper. They can be
# turned on per-platform when the wrappers learn to wire delegates.
cmake "$TF_SRC/tensorflow/lite/c" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_C_BUILD_SHARED_LIBS=ON \
    -DTFLITE_ENABLE_NNAPI=OFF \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_METAL=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

echo "[setup_litert] Building tensorflowlite_c (this takes ~20-30 minutes)..."
cmake --build . --target tensorflowlite_c -j

mkdir -p "$OUT/include/tensorflow/lite/c" "$OUT/include/tensorflow/lite/core/c" "$OUT/lib"

# Copy public C headers (only what the wrappers consume).
# builtin_ops.h is transitively included by core/c/{c_api,common}.h, so it
# has to ship alongside them.
cp "$TF_SRC/tensorflow/lite/builtin_ops.h"         "$OUT/include/tensorflow/lite/"
cp "$TF_SRC/tensorflow/lite/c/c_api.h"             "$OUT/include/tensorflow/lite/c/"
cp "$TF_SRC/tensorflow/lite/c/c_api_types.h"       "$OUT/include/tensorflow/lite/c/"
cp "$TF_SRC/tensorflow/lite/c/common.h"            "$OUT/include/tensorflow/lite/c/"
cp "$TF_SRC/tensorflow/lite/core/c/c_api.h"        "$OUT/include/tensorflow/lite/core/c/"
cp "$TF_SRC/tensorflow/lite/core/c/c_api_types.h"  "$OUT/include/tensorflow/lite/core/c/"
cp "$TF_SRC/tensorflow/lite/core/c/common.h"       "$OUT/include/tensorflow/lite/core/c/"

# Library
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp "$BUILD_DIR/libtensorflowlite_c.dylib" "$OUT/lib/"
else
    cp "$BUILD_DIR/libtensorflowlite_c.so" "$OUT/lib/"
fi

echo ""
echo "LiteRT installed at: $OUT"
echo "Build with: cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$OUT"
