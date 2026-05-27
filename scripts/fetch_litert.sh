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

# Find the runtime library — Linux/macOS use the lib prefix, Windows ships
# libLiteRt.dll too (Google's wheel keeps the prefix on all platforms).
LIB="$(find "$UNZ" -maxdepth 3 \
    \( -name 'libLiteRt.so' -o -name 'libLiteRt.dylib' -o -name 'libLiteRt.dll' \) \
    | head -n1)"
if [[ -z "$LIB" ]]; then
    echo "[fetch_litert] could not find libLiteRt in the wheel" >&2
    ls -la "$UNZ" >&2
    exit 1
fi
cp "$LIB" "$OUT/"

# Windows: the wheel doesn't ship an import library. Generate one from the
# DLL's export table using MSVC's dumpbin + lib (must be in PATH — use the
# ilammy/msvc-dev-cmd action in CI, or run from a Developer Command Prompt
# locally). Skipped silently on non-Windows.
if [[ "$LIB" == *libLiteRt.dll ]]; then
    if command -v dumpbin >/dev/null 2>&1 && command -v lib >/dev/null 2>&1; then
        echo "[fetch_litert] Generating libLiteRt.lib import library"
        cd "$OUT"

        # 1. Dump exports and extract just the symbol names.
        # dumpbin /EXPORTS output has 4 header lines then rows like:
        #     1    0 00012345 LiteRtCreateEnvironment
        # We want column 4. Python is easier to debug than awk on multi-platform.
        dumpbin //EXPORTS libLiteRt.dll > litert_exports.txt
        "$PY" -c "
import sys
syms, start = [], False
for line in open('litert_exports.txt', encoding='utf-8', errors='replace'):
    t = line.split()
    if not start:
        if len(t) >= 4 and t[0]=='ordinal' and t[1]=='hint' and t[2]=='RVA' and t[3]=='name':
            start = True
        continue
    if not t: continue
    if len(t) < 4: continue
    if t[0].lower() == 'summary': break
    if not t[0].isdigit(): continue
    name = t[3]
    if name and name not in ('=',): syms.append(name)
print(f'parsed {len(syms)} exports', file=sys.stderr)
with open('libLiteRt.def', 'w', encoding='ascii') as f:
    f.write('LIBRARY libLiteRt\nEXPORTS\n')
    for s in syms:
        f.write('    ' + s + '\n')
"
        # 2. Build the import lib from the .def file.
        lib //MACHINE:X64 //DEF:libLiteRt.def //OUT:libLiteRt.lib
        rm -f litert_exports.txt libLiteRt.def libLiteRt.exp
        cd - >/dev/null
    else
        echo "[fetch_litert] WARNING: dumpbin/lib not in PATH; skipping .lib generation."
        echo "                  (Run from an MSVC Developer Command Prompt to get one.)"
    fi
fi

echo ""
echo "libLiteRt installed at: $OUT"
echo "Build with: cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$OUT"
