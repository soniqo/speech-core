#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-turbo}"
PRECISION="${2:-int8}"
DEST="${3:-scripts/models/whisper-${MODEL}}"

case "$MODEL" in
  small)    REPO="soniqo/Whisper-Small-ONNX"; PREFIX="small"; DEFAULT_REVISION="99d37bc353d7ac6a8164e18fadda91e28a997c2f" ;;
  medium)   REPO="soniqo/Whisper-Medium-ONNX"; PREFIX="medium"; DEFAULT_REVISION="main" ;;
  large-v3) REPO="soniqo/Whisper-Large-v3-ONNX"; PREFIX="large-v3"; DEFAULT_REVISION="main" ;;
  turbo)    REPO="soniqo/Whisper-Large-v3-Turbo-ONNX"; PREFIX="turbo"; DEFAULT_REVISION="main" ;;
  *)
    echo "Usage: $0 [small|medium|large-v3|turbo] [int8|fp16|fp32|all] [dest]" >&2
    exit 2
    ;;
esac

REVISION="${SPEECH_WHISPER_ONNX_REVISION:-$DEFAULT_REVISION}"

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
  local remote_name="$1"
  local required="${2:-1}"
  local local_name="${3:-$remote_name}"
  local force="${4:-0}"
  local url="https://huggingface.co/${REPO}/resolve/${REVISION}/${remote_name}"
  local out="${DEST}/${local_name}"
  if [ -f "$out" ] && [ "$force" != "1" ]; then
    echo "exists ${out}"
    return
  fi
  echo "download ${REPO}@${REVISION}/${remote_name}"
  # Expanding an empty array is an "unbound variable" error under set -u in
  # bash 3.2, which is what /bin/bash still is on macOS — so without HF_TOKEN
  # set, this script used to die on its first download. The +"${...}" form
  # expands to nothing instead.
  if ! curl -L --fail --retry 3 ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
       -o "${out}.part" "$url"; then
    rm -f "${out}.part"
    if [ "$required" = "1" ]; then
      exit 1
    fi
    echo "optional missing ${remote_name}"
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
      if [ "$MODEL" = "small" ]; then
        download "external-v2/${PREFIX}-encoder.int8.onnx.data" 1 "${PREFIX}-encoder.int8.onnx.data"
        download "external-v2/${PREFIX}-encoder.int8.onnx" 1 "${PREFIX}-encoder.int8.onnx" 1
        download "external-v2/${PREFIX}-decoder.int8.onnx.data" 1 "${PREFIX}-decoder.int8.onnx.data"
        download "external-v2/${PREFIX}-decoder.int8.onnx" 1 "${PREFIX}-decoder.int8.onnx" 1
      else
        download "${PREFIX}-encoder.int8.onnx" 1
        download "${PREFIX}-decoder.int8.onnx" 1
      fi
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
