#!/usr/bin/env bash
# Official three-stage DeepFilterNet3 bundle used by Rust streaming mode.
set -euo pipefail
destination="${1:?Usage: fetch_deepfilter_rust.sh <output-directory>}"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
curl -fL --retry 3 -o "$temporary/model.tar.gz" \
  'https://raw.githubusercontent.com/Rikorose/DeepFilterNet/d375b2d8309e0935d165700c91da9de862a99c31/models/DeepFilterNet3_onnx.tar.gz'
expected='c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616'
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$temporary/model.tar.gz")"
else
  actual="$(shasum -a 256 "$temporary/model.tar.gz")"
fi
if [[ "${actual%% *}" != "$expected" ]]; then
  echo 'DeepFilterNet3 bundle checksum mismatch' >&2
  exit 1
fi
tar -xzf "$temporary/model.tar.gz" -C "$temporary" \
  tmp/export/enc.onnx tmp/export/erb_dec.onnx tmp/export/df_dec.onnx tmp/export/config.ini
mkdir -p "$destination"
for name in enc.onnx erb_dec.onnx df_dec.onnx config.ini; do
  cp "$temporary/tmp/export/$name" "$destination/$name"
done
