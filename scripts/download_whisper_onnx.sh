#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-turbo}"
PRECISION="${2:-int8}"
DEST="${3:-scripts/models/whisper-${MODEL}}"

case "$MODEL" in
  small)    REPO="soniqo/Whisper-Small-ONNX"; PREFIX="small" ;;
  medium)   REPO="soniqo/Whisper-Medium-ONNX"; PREFIX="medium" ;;
  large-v3) REPO="soniqo/Whisper-Large-v3-ONNX"; PREFIX="large-v3" ;;
  turbo)    REPO="soniqo/Whisper-Large-v3-Turbo-ONNX"; PREFIX="turbo" ;;
  *)
    echo "Usage: $0 [small|medium|large-v3|turbo] [int8|fp16|fp32|all] [dest]" >&2
    exit 2
    ;;
esac

case "$PRECISION" in
  int8|fp16|fp32|all) ;;
  *)
    echo "precision must be int8, fp16, fp32, or all" >&2
    exit 2
    ;;
esac

mkdir -p "$DEST"

AUTH_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download() {
  local name="$1"
  local required="${2:-1}"
  local url="https://huggingface.co/${REPO}/resolve/main/${name}"
  local out="${DEST}/${name}"
  if [ -f "$out" ]; then
    echo "exists ${out}"
    return
  fi
  echo "download ${REPO}/${name}"
  if ! curl -L --fail --retry 3 "${AUTH_ARGS[@]}" -o "${out}.part" "$url"; then
    rm -f "${out}.part"
    if [ "$required" = "1" ]; then
      exit 1
    fi
    echo "optional missing ${name}"
    return
  fi
  mv "${out}.part" "$out"
}

download README.md 0
download manifest.json 1
download "${PREFIX}-tokens.txt" 1

download_precision() {
  local p="$1"
  case "$p" in
    int8)
      download "${PREFIX}-encoder.int8.onnx" 1
      download "${PREFIX}-decoder.int8.onnx" 1
      ;;
    fp16)
      download "${PREFIX}-encoder.fp16.onnx" 1
      download "${PREFIX}-encoder.fp16.onnx.data" 1
      download "${PREFIX}-decoder.fp16.onnx" 1
      download "${PREFIX}-decoder.fp16.onnx.data" 1
      ;;
    fp32)
      download "${PREFIX}-encoder.onnx" 1
      download "${PREFIX}-decoder.onnx" 1
      download "${PREFIX}-encoder.weights" 0
      download "${PREFIX}-decoder.weights" 0
      ;;
  esac
}

if [ "$PRECISION" = "all" ]; then
  download_precision int8
  download_precision fp16
  download_precision fp32
else
  download_precision "$PRECISION"
fi

echo
echo "Whisper ONNX bundle ready: ${DEST}"
echo "Example:"
echo "  SPEECH_WHISPER_ONNX_DIR=${DEST} SPEECH_MODEL_DIR=${DEST} ctest --test-dir build --output-on-failure"
