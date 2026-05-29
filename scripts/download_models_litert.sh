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

# Optional HuggingFace auth — soniqo/* repos may be private. When HF_TOKEN is
# set, send it as a bearer token; when unset, fetch anonymously. Backward
# compatible with the previous (anonymous) behaviour.
AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

FILES=(
    "Silero-VAD-v5-LiteRT/silero-vad.tflite"
    "Silero-VAD-v5-LiteRT/config.json"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/parakeet-encoder.tflite"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/parakeet-decoder-joint.tflite"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/vocab.json"
    "Parakeet-TDT-0.6B-v3-LiteRT-INT8/config.json"
    # Diarization + multilingual ASR — fetched for the upstreamed cloud
    # wrappers (WeSpeaker / Pyannote / Omnilingual). Best-effort: if a repo is
    # private and HF_TOKEN is unset, the fetch warns and the gated test skips.
    "Pyannote-Segmentation-LiteRT/pyannote-segmentation.tflite"
    "WeSpeaker-ResNet34-LM-LiteRT/wespeaker-resnet34.tflite"
    "Omnilingual-ASR-CTC-300M-LiteRT/omnilingual-ctc-300m.tflite"
    "Omnilingual-ASR-CTC-300M-LiteRT/tokenizer.model"
    # Nemotron Speech Streaming — cache-aware streaming RNN-T (3 graphs).
    "Nemotron-Speech-Streaming-LiteRT/nemotron-streaming-encoder.tflite"
    "Nemotron-Speech-Streaming-LiteRT/nemotron-streaming-decoder.tflite"
    "Nemotron-Speech-Streaming-LiteRT/nemotron-streaming-joint.tflite"
    "Nemotron-Speech-Streaming-LiteRT/vocab.json"
    "Nemotron-Speech-Streaming-LiteRT/config.json"
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
        Nemotron-Speech-Streaming-LiteRT)
            # .tflite are already nemotron-prefixed; vocab/config would collide
            # with Parakeet's in the shared dir, so prefix those.
            dest="$OUT/${rel}"
            [[ "$rel" == "vocab.json" || "$rel" == "config.json" ]] && dest="$OUT/nemotron-${rel}"
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
    # Best-effort: a missing/forbidden file (e.g. a private repo with no
    # HF_TOKEN) warns and continues so the rest still download. Gated tests
    # skip cleanly when a model is absent. The ${AUTH[@]+...} form is
    # nounset-safe for the empty-array (no-token) case on old bash.
    if ! curl -fL --retry 3 ${AUTH[@]+"${AUTH[@]}"} -o "$dest" "$url"; then
        echo "[warn] could not fetch $rel (set HF_TOKEN if the repo is private) — skipping"
        rm -f "$dest"
    fi
done

echo ""
echo "LiteRT models downloaded to: $OUT"
echo "Run tests with: SPEECH_LITERT_MODEL_DIR=$OUT ctest --test-dir build --output-on-failure"
