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

mkdir -p "$OUT/include/tensorflow/lite" "$OUT/lib"

# Copy every .h under tensorflow/lite/{c,core/c,core/async/c} plus the top-level
# builtin_ops.h. Globbing is safer than enumerating individual files — the C
# API has transitive includes (builtin_ops.h, core/async/c/types.h, …) that
# changed between TF releases.
cp "$TF_SRC/tensorflow/lite/builtin_ops.h" "$OUT/include/tensorflow/lite/"
for subdir in c core/c core/async/c; do
    mkdir -p "$OUT/include/tensorflow/lite/$subdir"
    cp "$TF_SRC/tensorflow/lite/$subdir"/*.h "$OUT/include/tensorflow/lite/$subdir/"
done

# Library
if [[ "$OSTYPE" == "darwin"* ]]; then
    cp "$BUILD_DIR/libtensorflowlite_c.dylib" "$OUT/lib/"
else
    cp "$BUILD_DIR/libtensorflowlite_c.so" "$OUT/lib/"
fi

echo ""
echo "LiteRT installed at: $OUT"
echo "Build with: cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$OUT"
