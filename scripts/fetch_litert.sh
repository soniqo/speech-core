#!/usr/bin/env bash
# Extract libLiteRt.{so,dylib,dll} from Google's `ai-edge-litert` PyPI wheel.
# Drops the shared library into $OUT, ready for cmake -DLITERT_DIR=$OUT.
#
# Usage:
#     scripts/fetch_litert.sh [output_dir] [ai-edge-litert_version]
# Defaults: output_dir = build/litert ; version = 2.1.5

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/litert}"
VERSION="${2:-2.1.5}"

mkdir -p "$OUT"

# pip + system python is enough; no virtualenv required because we throw the
# install away after extracting the .dylib/.so/.dll.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ai-edge-litert wheels are published per CPython tag. v2.1.5 needs Python
# 3.10+ — point PYTHON at a newer interpreter on systems where python3 is older.
PY="${PYTHON:-python3}"
echo "[fetch_litert] Using $PY ($("$PY" -V 2>&1))"
echo "[fetch_litert] Downloading ai-edge-litert==$VERSION to $TMP"
"$PY" -m pip download --no-deps --dest "$TMP" "ai-edge-litert==$VERSION" \
    --quiet --disable-pip-version-check

WHEEL="$(ls "$TMP"/ai_edge_litert-*.whl | head -n1)"
if [[ -z "$WHEEL" ]]; then
    echo "[fetch_litert] no wheel downloaded — check that ai-edge-litert==$VERSION exists for this platform" >&2
    exit 1
fi
echo "[fetch_litert] Extracting $WHEEL"

# Wheels are zip files; pull just the runtime library.
UNZ="$TMP/unzip"
mkdir -p "$UNZ"
unzip -q "$WHEEL" -d "$UNZ"

# Find the runtime library in the wheel (path is ai_edge_litert/libLiteRt.*).
LIB="$(find "$UNZ" -maxdepth 3 -name 'libLiteRt.so' -o -name 'libLiteRt.dylib' -o -name 'LiteRt.dll' | head -n1)"
if [[ -z "$LIB" ]]; then
    echo "[fetch_litert] could not find libLiteRt in the wheel" >&2
    ls -la "$UNZ" >&2
    exit 1
fi
cp "$LIB" "$OUT/"

# Windows wheels also ship an import library; copy it if present so MSVC links cleanly.
IMPLIB="$(find "$UNZ" -maxdepth 3 -name 'LiteRt.lib' | head -n1 || true)"
if [[ -n "$IMPLIB" ]]; then
    cp "$IMPLIB" "$OUT/"
fi

echo ""
echo "libLiteRt installed at: $OUT"
echo "Build with: cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$OUT"
