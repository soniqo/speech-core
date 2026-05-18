#!/usr/bin/env bash
# Download LiteRT (.tflite) models for testing the speech_core_models_litert target.
# Usage: ./download_models_litert.sh [output_dir]
#
# By default models are placed in scripts/models-litert/ alongside this script.
# Tests pick the directory up via the SPEECH_LITERT_MODEL_DIR environment variable:
#   SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build

set -euo pipefail

BASE_URL="https://huggingface.co/soniqo"
OUT="${1:-$(dirname "$0")/models-litert}"
mkdir -p "$OUT"

FILES=(
    "Silero-VAD-v5-LiteRT/silero-vad.tflite"
    "Silero-VAD-v5-LiteRT/config.json"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/parakeet-encoder.tflite"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/parakeet-decoder-joint.tflite"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/vocab.json"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/config.json"
)

for entry in "${FILES[@]}"; do
    repo="${entry%%/*}"
    rel="${entry#*/}"
    # Parakeet and Silero both have a config.json — disambiguate by prefixing
    # with a short model tag so they don't overwrite each other.
    case "$repo" in
        Silero-VAD-v5-LiteRT)
            dest="$OUT/silero-${rel}"
            [[ "$rel" == "silero-vad.tflite" ]] && dest="$OUT/silero-vad.tflite"
            ;;
        Parakeet-TDT-0.6B-v3-LiteRT-INT8)
            dest="$OUT/parakeet-${rel}"
            [[ "$rel" == parakeet-* ]] && dest="$OUT/${rel}"
            ;;
        *)
            dest="$OUT/${rel}"
            ;;
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
echo "LiteRT models downloaded to: $OUT"
echo "Run tests with: SPEECH_LITERT_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure"
