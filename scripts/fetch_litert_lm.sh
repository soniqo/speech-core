#!/usr/bin/env bash
# Extract liblitert-lm.{so,dylib,dll} from Google's `litert-lm` PyPI wheel.
# Drops the shared library into $OUT, ready for cmake -DLITERT_LM_DIR=$OUT.
#
# The .litertlm bundle format is loaded by this higher-level runtime, NOT by
# the libLiteRt C API that fetch_litert.sh extracts. Use both together:
# speech_core_models_litert needs libLiteRt for .tflite STT/VAD/TTS models AND
# liblitert-lm for .litertlm tool-calling LLMs like FunctionGemma.
#
# Usage:
#     scripts/fetch_litert_lm.sh [output_dir] [litert-lm-api_version]
# Defaults: output_dir = build/litert_lm ; version = 0.13.1
#
# The runtime ships in the litert-lm-api wheel; the litert-lm meta-package
# pulls in additional Python tooling we do not need at runtime.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/litert_lm}"
VERSION="${2:-0.13.1}"

mkdir -p "$OUT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PY="${PYTHON:-python3}"
echo "[fetch_litert_lm] Using $PY ($("$PY" -V 2>&1))"
echo "[fetch_litert_lm] Downloading litert-lm-api==$VERSION to $TMP"
"$PY" -m pip download --no-deps --dest "$TMP" "litert-lm-api==$VERSION" \
    --quiet --disable-pip-version-check

WHEEL="$(ls "$TMP"/litert_lm_api-*.whl 2>/dev/null | head -n1)"
if [[ -z "$WHEEL" ]]; then
    echo "[fetch_litert_lm] no wheel downloaded — check that litert-lm-api==$VERSION exists for this platform" >&2
    exit 1
fi
echo "[fetch_litert_lm] Extracting $WHEEL"

UNZ="$TMP/unzip"
mkdir -p "$UNZ"
unzip -q "$WHEEL" -d "$UNZ"

# Google's wheel ships the lib under `litert_lm/` inside the wheel.
LIB="$(find "$UNZ" -maxdepth 3 \
    \( -name 'liblitert-lm.so' -o -name 'liblitert-lm.dylib' -o -name 'liblitert-lm.dll' \) \
    | head -n1)"
if [[ -z "$LIB" ]]; then
    echo "[fetch_litert_lm] could not find liblitert-lm in the wheel" >&2
    ls -la "$UNZ" >&2
    exit 1
fi
cp "$LIB" "$OUT/"

# Google's macOS dylib ships with the Linux install name (`@rpath/liblitert-lm.so`)
# baked in, so linkers resolve it to a missing .so at runtime. Fix the install
# name to match the real filename so CMake's IMPORTED_LOCATION + the binary's
# default rpath find the library.
if [[ "$LIB" == *liblitert-lm.dylib ]] && command -v install_name_tool >/dev/null 2>&1; then
    install_name_tool -id "@rpath/liblitert-lm.dylib" "$OUT/liblitert-lm.dylib"
fi

# Windows: generate import lib from DLL export table if MSVC tooling is on PATH.
if [[ "$LIB" == *liblitert-lm.dll ]]; then
    if command -v dumpbin >/dev/null 2>&1 && command -v lib >/dev/null 2>&1; then
        echo "[fetch_litert_lm] Generating liblitert-lm.lib import library"
        pushd "$OUT" >/dev/null
        dumpbin /EXPORTS liblitert-lm.dll | awk '/^ +[0-9]+/ { print $4 }' > exports.def
        printf 'EXPORTS\n%s' "$(cat exports.def)" > liblitert-lm.def
        lib /MACHINE:X64 /DEF:liblitert-lm.def /OUT:liblitert-lm.lib >/dev/null
        rm -f exports.def liblitert-lm.def
        popd >/dev/null
    fi
fi

echo "[fetch_litert_lm] Wrote $OUT/$(basename "$LIB")"
