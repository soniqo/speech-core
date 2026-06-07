#!/usr/bin/env bash
# Download a PersonaPlex ONNX bundle for OnnxPersonaPlex. Three layouts:
#   fp16   17 GB  -- temporal FP16 + depformer FP16  (best for low host RAM)
#   int8   13 GB  -- temporal INT8/FP32-KV + depformer FP32  (best for disk)
#   mixed  11 GB  -- temporal INT8/FP32-KV + depformer FP16  (smallest)
#
# Variant is selected by env PERSONAPLEX_VARIANT or first arg (default 'mixed').
# Destination is scripts/personaplex-<variant>/.
#
# This script assumes the bundle has been uploaded to
# `soniqo/PersonaPlex-7B-ONNX` under tag <variant>/. Until that upload exists,
# use the local conversion path documented in
# speech-models/models/personaplex/export/NOTES.md.

set -euo pipefail

VARIANT="${1:-${PERSONAPLEX_VARIANT:-mixed}}"
case "$VARIANT" in
    fp16|int8|mixed) ;;
    *) echo "unknown variant '$VARIANT' (expected fp16, int8, or mixed)" >&2; exit 2 ;;
esac

REPO="${PERSONAPLEX_HF_REPO:-soniqo/PersonaPlex-7B-ONNX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/personaplex-${VARIANT}"

mkdir -p "$DEST"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: 'hf' CLI not found; install via 'pip install -U huggingface_hub[cli]'" >&2
    exit 1
fi

echo "Downloading PersonaPlex bundle variant '$VARIANT' from $REPO -> $DEST"

# Each ONNX graph is split graph + .onnx.data sidecar. We list explicit files
# rather than --include="*" so users can see what arrives.
COMMON=(
    mimi_encoder.onnx mimi_encoder.onnx.data
    mimi_decoder.onnx mimi_decoder.onnx.data
    depformer_step.onnx depformer_step.onnx.data
    tokenizer_spm_32k_3.model
    system_prompts.bin
    config.json
)

case "$VARIANT" in
    fp16|int8|mixed)
        TEMPORAL_FILES=(temporal_step.onnx temporal_step.onnx.data)
        ;;
esac

for f in "${COMMON[@]}" "${TEMPORAL_FILES[@]}"; do
    hf download "$REPO" "${VARIANT}/$f" --revision main \
        --local-dir "$DEST" --local-dir-use-symlinks False
done

# Voices live in a per-bundle subdir
hf download "$REPO" --include "${VARIANT}/voices/*" --revision main \
    --local-dir "$DEST" --local-dir-use-symlinks False

# Flatten <variant>/ into the destination root (HF preserves the prefix)
if [ -d "$DEST/$VARIANT" ]; then
    cp -r "$DEST/$VARIANT"/* "$DEST/"
    rm -rf "$DEST/$VARIANT"
fi

echo
echo "Done. Bundle at $DEST. Use:"
echo "  run_personaplex \"$DEST\" 50 tests/data/test_audio.wav VARF2"
