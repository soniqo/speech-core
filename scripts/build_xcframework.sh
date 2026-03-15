#!/bin/bash
set -euo pipefail

# Build SpeechCore.xcframework from the C++ static library.
# Produces a zipped xcframework ready for SPM binary target distribution.
#
# Platforms: macOS (arm64+x86_64), iOS (arm64), iOS Simulator (arm64+x86_64)
#
# Usage:
#   ./scripts/build_xcframework.sh [--output DIR]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output) OUTPUT_DIR="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

NCPU="$(sysctl -n hw.ncpu)"

# --- macOS ---

echo "==> Building macOS (arm64)..."
BUILD_MACOS_ARM64="${ROOT_DIR}/build-macos-arm64"
cmake -B "$BUILD_MACOS_ARM64" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DSPEECH_CORE_BUILD_TESTS=OFF
cmake --build "$BUILD_MACOS_ARM64" --config Release -j "$NCPU"

echo "==> Building macOS (x86_64)..."
BUILD_MACOS_X86="${ROOT_DIR}/build-macos-x86_64"
cmake -B "$BUILD_MACOS_X86" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
    -DSPEECH_CORE_BUILD_TESTS=OFF
cmake --build "$BUILD_MACOS_X86" --config Release -j "$NCPU"

echo "==> Creating macOS universal binary..."
BUILD_MACOS_UNIVERSAL="${ROOT_DIR}/build-macos-universal"
mkdir -p "$BUILD_MACOS_UNIVERSAL"
lipo -create \
    "$BUILD_MACOS_ARM64/libspeech_core.a" \
    "$BUILD_MACOS_X86/libspeech_core.a" \
    -output "$BUILD_MACOS_UNIVERSAL/libspeech_core.a"

# --- iOS device ---

echo "==> Building iOS device (arm64)..."
IOS_SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
BUILD_IOS="${ROOT_DIR}/build-ios-arm64"
cmake -B "$BUILD_IOS" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$IOS_SDK" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
    -DSPEECH_CORE_BUILD_TESTS=OFF
cmake --build "$BUILD_IOS" --config Release -j "$NCPU"

# --- iOS Simulator ---

echo "==> Building iOS Simulator (arm64)..."
SIM_SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"
BUILD_SIM_ARM64="${ROOT_DIR}/build-sim-arm64"
cmake -B "$BUILD_SIM_ARM64" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_SYSROOT="$SIM_SDK" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
    -DSPEECH_CORE_BUILD_TESTS=OFF
cmake --build "$BUILD_SIM_ARM64" --config Release -j "$NCPU"

echo "==> Building iOS Simulator (x86_64)..."
BUILD_SIM_X86="${ROOT_DIR}/build-sim-x86_64"
cmake -B "$BUILD_SIM_X86" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DCMAKE_OSX_SYSROOT="$SIM_SDK" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
    -DSPEECH_CORE_BUILD_TESTS=OFF
cmake --build "$BUILD_SIM_X86" --config Release -j "$NCPU"

echo "==> Creating iOS Simulator universal binary..."
BUILD_SIM_UNIVERSAL="${ROOT_DIR}/build-sim-universal"
mkdir -p "$BUILD_SIM_UNIVERSAL"
lipo -create \
    "$BUILD_SIM_ARM64/libspeech_core.a" \
    "$BUILD_SIM_X86/libspeech_core.a" \
    -output "$BUILD_SIM_UNIVERSAL/libspeech_core.a"

# --- Headers ---

echo "==> Preparing headers..."
HEADERS_DIR="${ROOT_DIR}/build-headers"
mkdir -p "$HEADERS_DIR"
cp "$ROOT_DIR/include/speech_core/speech_core_c.h" "$HEADERS_DIR/"

cat > "$HEADERS_DIR/module.modulemap" << 'EOF'
module CSpeechCore {
    header "speech_core_c.h"
    link "c++"
    export *
}
EOF

# --- XCFramework ---

echo "==> Creating xcframework..."
XCFW_PATH="${OUTPUT_DIR}/SpeechCore.xcframework"
rm -rf "$XCFW_PATH"
mkdir -p "$OUTPUT_DIR"

xcodebuild -create-xcframework \
    -library "$BUILD_MACOS_UNIVERSAL/libspeech_core.a" \
    -headers "$HEADERS_DIR" \
    -library "$BUILD_IOS/libspeech_core.a" \
    -headers "$HEADERS_DIR" \
    -library "$BUILD_SIM_UNIVERSAL/libspeech_core.a" \
    -headers "$HEADERS_DIR" \
    -output "$XCFW_PATH"

echo "==> Zipping xcframework..."
cd "$OUTPUT_DIR"
ZIP_NAME="SpeechCore.xcframework.zip"
rm -f "$ZIP_NAME"
zip -ry "$ZIP_NAME" "SpeechCore.xcframework"

CHECKSUM=$(swift package compute-checksum "$ZIP_NAME")
echo ""
echo "==> Done!"
echo "    XCFramework: $XCFW_PATH"
echo "    Archive:     $OUTPUT_DIR/$ZIP_NAME"
echo "    Checksum:    $CHECKSUM"
echo ""
echo "    Use in Package.swift:"
echo "    .binaryTarget("
echo "        name: \"CSpeechCore\","
echo "        url: \"https://github.com/soniqo/speech-core/releases/download/vX.Y.Z/$ZIP_NAME\","
echo "        checksum: \"$CHECKSUM\""
echo "    )"
