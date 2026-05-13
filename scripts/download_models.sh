#!/usr/bin/env bash
# Download ONNX models for testing the speech_core_models target.
# Usage: ./download_models.sh [output_dir]
#
# By default models are placed in scripts/models/ alongside this script.
# Tests pick the directory up via the SPEECH_MODEL_DIR environment variable:
#   SPEECH_MODEL_DIR=scripts/models ctest --test-dir build

set -euo pipefail

BASE_URL="https://huggingface.co/aufklarer"
OUT="${1:-$(dirname "$0")/models}"
mkdir -p "$OUT/voices"

FILES=(
    "Silero-VAD-v5-ONNX/silero-vad.onnx"
    "Parakeet-TDT-v3-ONNX/parakeet-encoder-int8.onnx"
    "Parakeet-TDT-v3-ONNX/parakeet-decoder-joint-int8.onnx"
    "Parakeet-TDT-v3-ONNX/vocab.json"
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
    curl -fL --retry 3 -o "$dest" "$url"
done

echo ""
echo "Models downloaded to: $OUT"
echo "Run tests with: SPEECH_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure"
