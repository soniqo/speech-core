#!/usr/bin/env bash
set -euo pipefail

PRECISION="${1:-int8}"
DEST="${2:-scripts/models/canary-180m-flash}"
REPO="soniqo/Canary-180M-Flash-ONNX"

case "$PRECISION" in
  int8|fp32|all) ;;
  *)
    echo "Usage: $0 [int8|fp32|all] [dest]" >&2
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
  # The +"${...}" guard keeps set -u from tripping over an empty array, which
  # is an error in the bash 3.2 that ships with macOS.
  if ! curl -L --fail --retry 3 ${AUTH_ARGS[@]+"${AUTH_ARGS[@]}"} \
       -o "${out}.part" "$url"; then
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
# config.json carries the decode contract the wrapper reads at construction.
download config.json 1
download vocab.json 1

download_precision() {
  case "$1" in
    int8)
      download canary-encoder-int8.onnx 1
      download canary-decoder-int8.onnx 1
      ;;
    fp32)
      download canary-encoder.onnx 1
      download canary-decoder.onnx 1
      ;;
  esac
}

if [ "$PRECISION" = "all" ]; then
  download_precision int8
  download_precision fp32
else
  download_precision "$PRECISION"
fi

echo
echo "Canary ONNX bundle ready: ${DEST}"
