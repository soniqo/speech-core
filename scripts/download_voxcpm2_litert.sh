#!/usr/bin/env bash
# Download the VoxCPM2 LiteRT bundle for the LiteRTVoxCPM2Tts test. The repo
# root holds the ARM/'selective' bundle (fp16 token-step + fp32 LocDiT, ~8.7 GB);
# on x86_64 this script instead pulls the fp32-token-step bundle from the repo's
# fp32-p16/ subdir (~13 GB) — the fp16 token-step over-generates on x86 (its
# stop-margin rounds the wrong way under x86 XNNPACK so the stop token never
# fires). Kept separate from download_models_litert.sh because the bundle is too
# large for the nightly's actions/cache budget (10 GB per repo, shared).
#
# Usage:
#     scripts/download_voxcpm2_litert.sh [output_dir]
#
# Output defaults to scripts/models-voxcpm2/. Point the existing LiteRT
# test runner at it via:
#     SPEECH_LITERT_MODEL_DIR=scripts/models-voxcpm2 ctest --test-dir build

set -euo pipefail

BASE_URL="https://huggingface.co/soniqo/VoxCPM2-LiteRT/resolve/main"
# x86_64 pulls the fp32-token-step variant (fp16 over-generates on x86); ARM
# uses the repo-root 'selective'. Local layout stays flat — point
# SPEECH_LITERT_MODEL_DIR at $OUT either way.
case "$(uname -m)" in
    x86_64|amd64) URL_SUBDIR="fp32-p16/" ;;
    *)            URL_SUBDIR="" ;;
esac
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
    if ! curl -fL --retry 3 -o "$dest" "$BASE_URL/$URL_SUBDIR$rel"; then
        echo "[warn] $rel not available (HTTP error) — leaving missing"
        rm -f "$dest"
    fi
done

echo ""
echo "VoxCPM2 LiteRT bundle downloaded to: $OUT"
echo "Run the skeleton load test with:"
echo "  SPEECH_LITERT_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure -R test_litert_models"
