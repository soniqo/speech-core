#!/usr/bin/env bash
# Download a PersonaPlex ONNX bundle for OnnxPersonaPlex from
# soniqo/PersonaPlex-7B-ONNX. Four variants are available; pick the
# one that matches your memory budget and quality target:
#
#   int8-nb-dep_gint8   9.4 GB  RECOMMENDED SHIP DEFAULT
#                               - INT8 MatMulNBits temporal + custom INT8 depformer
#                               - 1.4 GB host RAM, 12.1 GB VRAM, RTF 1.12x
#                               - cos 0.998 vs FP32, excellent quality
#
#   int4-nb-dep_gint8   7.6 GB  Smallest disk
#                               - INT4 MatMulNBits temporal + custom INT8 depformer
#                               - 1.4 GB host RAM, 9.6 GB VRAM, RTF 1.12x
#                               - cos 0.877 vs FP32, visibly degraded but coherent
#
#   mixed              11 GB    Quality+VRAM Pareto winner
#                               - INT8 dynamic temporal + FP16 depformer
#                               - 7.9 GB host RAM, 6.6 GB VRAM, RTF 3.5x
#                               - cos 0.990 vs FP32, lowest VRAM
#                               - Topical responses ("We're concerned about it.")
#
#   fp16               17 GB    Maximum quality
#                               - FP16 temporal + FP16 depformer + FP32 mimi
#                               - 1.5 GB host RAM, 18.3 GB VRAM, RTF 5.3x
#                               - cos 0.9999 vs FP32, near-perfect
#
# Variant is selected by env PERSONAPLEX_VARIANT or first arg (default
# 'int8-nb-dep_gint8'). Destination is scripts/personaplex-<variant>/.

set -euo pipefail

VARIANT="${1:-${PERSONAPLEX_VARIANT:-int8-nb-dep_gint8}}"
case "$VARIANT" in
    fp16|mixed|int8-nb-dep_gint8|int4-nb-dep_gint8) ;;
    *) echo "unknown variant '$VARIANT' (expected fp16, mixed, int8-nb-dep_gint8, or int4-nb-dep_gint8)" >&2; exit 2 ;;
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
hf download "$REPO" --include "${VARIANT}/*" \
    --local-dir "$DEST" --revision main

# Flatten <variant>/ into the destination root (HF preserves the prefix)
if [ -d "$DEST/$VARIANT" ]; then
    # Use rsync if available for atomic move; otherwise mv + cp
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "$DEST/$VARIANT/" "$DEST/"
    else
        cp -r "$DEST/$VARIANT"/* "$DEST/"
        # voices/ is a subdir; cp -r handles it
        if [ -d "$DEST/$VARIANT/voices" ]; then
            mkdir -p "$DEST/voices"
            cp -r "$DEST/$VARIANT/voices"/* "$DEST/voices/" 2>/dev/null || true
        fi
    fi
    rm -rf "$DEST/$VARIANT"
fi

echo
echo "Done. Bundle at $DEST"
echo "Use:"
echo "  build/Release/run_personaplex \"$DEST\" 50 tests/data/test_audio.wav VARF2"
echo
echo "Or in C++:"
echo "  OnnxPersonaPlex pp("
echo "      \"$DEST/mimi_encoder.onnx\","
echo "      \"$DEST/mimi_decoder.onnx\","
echo "      \"$DEST/temporal_step.onnx\","
echo "      \"$DEST/depformer_step.onnx\","
echo "      \"$DEST/tokenizer_spm_32k_3.model\","
echo "      \"$DEST/voices\");"
