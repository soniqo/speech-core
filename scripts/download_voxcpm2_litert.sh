#!/usr/bin/env bash
# Download VoxCPM2 LiteRT model bundle (mixed int8/fp16, ~6.4 GB) for the
# LiteRTVoxCPM2Tts skeleton test. token-step is FP16 (INT8 there broke
# sibilants — the AR acoustic path needs >=16-bit); text-prefill is INT8
# (context-only, clean). The all-INT8 variant was decommissioned. Kept
# separate from download_models_litert.sh because the
# bundle is too large to fit comfortably in the nightly's actions/cache
# budget (10 GB per repo, shared).
#
# Usage:
#     scripts/download_voxcpm2_litert.sh [output_dir]
#
# Output defaults to scripts/models-voxcpm2/. Point the existing LiteRT
# test runner at it via:
#     SPEECH_LITERT_MODEL_DIR=scripts/models-voxcpm2 ctest --test-dir build

set -euo pipefail

BASE_URL="https://huggingface.co/soniqo/VoxCPM2-LiteRT/resolve/main"
OUT="${1:-$(dirname "$0")/models-voxcpm2}"
mkdir -p "$OUT"

FILES=(
    "voxcpm2-text-prefill.tflite"
    "voxcpm2-token-step.tflite"
    "voxcpm2-audio-encoder.tflite"
    "voxcpm2-audio-decoder.tflite"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "generation_config.json"
    "config.json"
)

for rel in "${FILES[@]}"; do
    dest="$OUT/$rel"
    if [[ -f "$dest" && -s "$dest" ]]; then
        echo "[skip] $rel (already exists)"
        continue
    fi
    echo "[fetch] $rel"
    if ! curl -fL --retry 3 -o "$dest" "$BASE_URL/$rel"; then
        echo "[warn] $rel not available (HTTP error) — leaving missing"
        rm -f "$dest"
    fi
done

echo ""
echo "VoxCPM2 LiteRT bundle downloaded to: $OUT"
echo "Run the skeleton load test with:"
echo "  SPEECH_LITERT_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure -R test_litert_models"
