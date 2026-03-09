#!/bin/bash
set -euo pipefail

# Build SpeechCore.xcframework from the C++ static library.
# Produces a zipped xcframework ready for SPM binary target distribution.
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

echo "==> Building speech-core (arm64)..."
BUILD_ARM64="${ROOT_DIR}/build-arm64"
cmake -B "$BUILD_ARM64" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build "$BUILD_ARM64" --config Release -j "$(sysctl -n hw.ncpu)"

echo "==> Building speech-core (x86_64)..."
BUILD_X86="${ROOT_DIR}/build-x86_64"
cmake -B "$BUILD_X86" -S "$ROOT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=x86_64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
cmake --build "$BUILD_X86" --config Release -j "$(sysctl -n hw.ncpu)"

echo "==> Creating universal binary..."
BUILD_UNIVERSAL="${ROOT_DIR}/build-universal"
mkdir -p "$BUILD_UNIVERSAL"
lipo -create \
    "$BUILD_ARM64/libspeech_core.a" \
    "$BUILD_X86/libspeech_core.a" \
    -output "$BUILD_UNIVERSAL/libspeech_core.a"

echo "==> Preparing headers..."
HEADERS_DIR="${BUILD_UNIVERSAL}/Headers"
mkdir -p "$HEADERS_DIR"
cp "$ROOT_DIR/include/speech_core/speech_core_c.h" "$HEADERS_DIR/"

cat > "$HEADERS_DIR/module.modulemap" << 'EOF'
module CSpeechCore {
    header "speech_core_c.h"
    link "c++"
    export *
}
EOF

echo "==> Creating xcframework..."
XCFW_PATH="${OUTPUT_DIR}/SpeechCore.xcframework"
rm -rf "$XCFW_PATH"
mkdir -p "$OUTPUT_DIR"

xcodebuild -create-xcframework \
    -library "$BUILD_UNIVERSAL/libspeech_core.a" \
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
