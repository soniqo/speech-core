#!/bin/bash
set -euo pipefail

ORT_VERSION="1.19.0"
OS="${OS:-$(uname -s)}"
ARCH="${1:-$(uname -m)}"

# Script lives at examples/linux/setup_linux.sh — repo root is two levels up.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORT_DIR="${ROOT}/ort-linux"

echo "=== speech-linux setup (${OS} ${ARCH}) ==="

if [ ! -f "${ORT_DIR}/include/onnxruntime_c_api.h" ]; then
    echo "Downloading ONNX Runtime ${ORT_VERSION} for ${OS} ${ARCH}..."

    case "${OS}-${ARCH}" in
        Linux-aarch64|Linux-arm64)
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz"
            ORT_LIB_GLOB="libonnxruntime.so*"
            ;;
        Linux-x86_64|Linux-amd64)
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
            ORT_LIB_GLOB="libonnxruntime.so*"
            ;;
        Darwin-arm64|Darwin-aarch64)
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
            ORT_LIB_GLOB="libonnxruntime*.dylib"
            ;;
        Darwin-x86_64)
            ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-x86_64-${ORT_VERSION}.tgz"
            ORT_LIB_GLOB="libonnxruntime*.dylib"
            ;;
        *)
            echo "Unsupported platform: ${OS}-${ARCH}"
            exit 1
            ;;
    esac

    TMP_DIR=$(mktemp -d)
    curl -L -o "${TMP_DIR}/ort.tgz" "${ORT_URL}"

    mkdir -p "${ORT_DIR}"
    tar xf "${TMP_DIR}/ort.tgz" -C "${TMP_DIR}"

    # Find extracted dir
    ORT_EXTRACTED=$(find "${TMP_DIR}" -maxdepth 1 -name "onnxruntime-*" -type d | head -1)

    mkdir -p "${ORT_DIR}/include" "${ORT_DIR}/lib"
    cp "${ORT_EXTRACTED}"/include/*.h "${ORT_DIR}/include/"
    cp "${ORT_EXTRACTED}"/lib/${ORT_LIB_GLOB} "${ORT_DIR}/lib/"

    rm -rf "${TMP_DIR}"
    echo "ONNX Runtime installed to ${ORT_DIR}"
else
    echo "ONNX Runtime already installed"
fi

echo ""
echo "Build with:"
echo "  cmake -B build \\"
echo "      -DCMAKE_BUILD_TYPE=Release \\"
echo "      -DSPEECH_CORE_WITH_ONNX=ON \\"
echo "      -DSPEECH_CORE_BUILD_EXAMPLES=ON \\"
echo "      -DORT_DIR=${ORT_DIR}"
echo "  cmake --build build"
