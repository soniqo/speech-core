#!/usr/bin/env bash
# Fetch Microsoft's native CPU runtime when the GitHub release archive is unavailable.
set -euo pipefail

VERSION="${1:?Usage: fetch_onnxruntime_nuget.sh <version> <runtime-id> <output-directory>}"
RUNTIME_ID="${2:?Missing NuGet runtime identifier}"
DESTINATION="${3:?Missing output directory}"
if [[ "$VERSION" != "1.19.0" ]]; then
    echo "No pinned ONNX Runtime NuGet package for version $VERSION" >&2
    exit 1
fi
case "$RUNTIME_ID" in
    linux-x64|linux-arm64|osx-x64|osx-arm64|win-x64) ;;
    *) echo "Unsupported NuGet runtime identifier $RUNTIME_ID" >&2; exit 1 ;;
esac

TEMPORARY="$(mktemp -d)"
trap 'rm -rf -- "$TEMPORARY"' EXIT
ARCHIVE="$TEMPORARY/onnxruntime.nupkg"
curl -fL --retry 3 -o "$ARCHIVE" \
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/$VERSION/microsoft.ml.onnxruntime.$VERSION.nupkg"
EXPECTED="60d6d0681256b83e2474fa554e976a74bb795d644e9e33d83d6bc7e5d98e0a3a"
if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL="$(sha256sum "$ARCHIVE")"
else
    ACTUAL="$(shasum -a 256 "$ARCHIVE")"
fi
if [[ "${ACTUAL%% *}" != "$EXPECTED" ]]; then
    echo "ONNX Runtime NuGet package checksum mismatch" >&2
    exit 1
fi

NATIVE="runtimes/$RUNTIME_ID/native"
(
    cd "$TEMPORARY"
    cmake -E tar xf "$ARCHIVE" -- 'build/native/include/*.h' "$NATIVE/*" LICENSE ThirdPartyNotices.txt
)
NATIVE="$TEMPORARY/$NATIVE"
mkdir -p "$DESTINATION/lib" "$DESTINATION/include"

case "$RUNTIME_ID" in
    linux-*)
        cp "$NATIVE/libonnxruntime.so" "$DESTINATION/lib/libonnxruntime.so.$VERSION"
        ln -sfn "libonnxruntime.so.$VERSION" "$DESTINATION/lib/libonnxruntime.so.1"
        ln -sfn libonnxruntime.so.1 "$DESTINATION/lib/libonnxruntime.so"
        ;;
    osx-*)
        cp "$NATIVE/libonnxruntime.dylib" "$DESTINATION/lib/libonnxruntime.$VERSION.dylib"
        ln -sfn "libonnxruntime.$VERSION.dylib" "$DESTINATION/lib/libonnxruntime.dylib"
        ;;
    win-x64)
        cp "$NATIVE/onnxruntime.dll" "$NATIVE/onnxruntime_providers_shared.dll" \
            "$NATIVE/onnxruntime.lib" "$DESTINATION/lib/"
        ;;
esac
cp "$TEMPORARY/LICENSE" "$TEMPORARY/ThirdPartyNotices.txt" "$DESTINATION/"
cp "$TEMPORARY"/build/native/include/*.h "$DESTINATION/include/"
