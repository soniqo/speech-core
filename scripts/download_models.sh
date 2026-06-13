#!/usr/bin/env bash
# Download ONNX models for testing the speech_core_models target.
# Usage: ./download_models.sh [output_dir]
#
# By default models are placed in scripts/models/ alongside this script.
# Tests pick the directory up via the SPEECH_MODEL_DIR environment variable:
#   SPEECH_MODEL_DIR=scripts/models ctest --test-dir build

set -euo pipefail

BASE_URL="https://huggingface.co/soniqo"
# Default output: repo-relative scripts/models when run from a writable
# checkout; the per-user cache dir when run from a read-only install location
# (e.g. /usr/bin/speech_download_models from the .deb). $SPEECH_MODEL_DIR
# overrides either — it is also where the installed CLIs look by default.
DEFAULT_OUT="$(dirname "$0")/models"
if [ ! -w "$(dirname "$0")" ]; then
    DEFAULT_OUT="${XDG_CACHE_HOME:-$HOME/.cache}/speech-core/models"
fi
OUT="${1:-${SPEECH_MODEL_DIR:-$DEFAULT_OUT}}"
mkdir -p "$OUT/voices"

FILES=(
    "Silero-VAD-v5-ONNX/silero-vad.onnx"
    "Parakeet-TDT-0.6B-ONNX/parakeet-encoder-int8.onnx"
    "Parakeet-TDT-0.6B-ONNX/parakeet-decoder-joint-int8.onnx"
    "Parakeet-TDT-0.6B-ONNX/vocab.json"
    "Kokoro-82M-ONNX/kokoro-e2e.onnx"
    "Kokoro-82M-ONNX/kokoro-e2e.onnx.data"
    "Kokoro-82M-ONNX/vocab_index.json"
    "Kokoro-82M-ONNX/us_gold.json"
    "Kokoro-82M-ONNX/us_silver.json"
    "Kokoro-82M-ONNX/dict_fr.json"
    "Kokoro-82M-ONNX/dict_es.json"
    "Kokoro-82M-ONNX/dict_it.json"
    "Kokoro-82M-ONNX/dict_pt.json"
    "Kokoro-82M-ONNX/dict_hi.json"
    "Kokoro-82M-ONNX/voices/af_heart.bin"
    "DeepFilterNet3-ONNX/deepfilter.onnx"
    "DeepFilterNet3-ONNX/deepfilter-auxiliary.bin"
)

for entry in "${FILES[@]}"; do
    repo="${entry%%/*}"
    rel="${entry#*/}"
    dest="$OUT/${rel#*/voices/}"
    # Handle voices/ subpath
    case "$rel" in
        voices/*) dest="$OUT/$rel" ;;
        *) dest="$OUT/$(basename "$rel")" ;;
    esac

    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "[skip] $rel (already exists)"
        continue
    fi

    url="$BASE_URL/$repo/resolve/main/$rel"
    echo "[fetch] $rel"
    # Tolerate 404s so a single missing file (e.g. DeepFilterNet3-ONNX/deepfilter.onnx
    # is not yet published) doesn't kill the whole download. The corresponding
    # test_* function in test_models.cpp skips at runtime when its files are absent.
    if ! curl -fL --retry 3 -o "$dest" "$url"; then
        echo "[warn] $rel not available (HTTP error) — leaving missing"
        rm -f "$dest"
    fi
done

echo ""
echo "Models downloaded to: $OUT"
echo "Run tests with: SPEECH_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure"
