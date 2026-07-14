#!/usr/bin/env bash
# Download the fixed-Alba Pocket TTS bundle consumed by OnnxPocketTts.
# Usage: bash scripts/download_pocket_tts_onnx.sh [output_dir]

set -euo pipefail

REPO="${POCKET_TTS_HF_REPO:-soniqo/Pocket-TTS-100M-ONNX-INT8}"
REVISION="${POCKET_TTS_REVISION:-v1.0.0}"
DEST="${1:-scripts/models/pocket-tts-onnx}"
BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"

FILES=(
    LICENSE
    README.md
    decoder.int8.onnx
    encoder.onnx
    lm_flow.int8.onnx
    lm_main.int8.onnx
    manifest.json
    text_conditioner.onnx
    token_scores.json
    tokenizer.model
    vocab.json
)

mkdir -p "$DEST"

for name in "${FILES[@]}"; do
    output="$DEST/$name"
    if [ -s "$output" ]; then
        echo "[skip] $name"
        continue
    fi
    echo "[fetch] ${REPO}@${REVISION}/$name"
    curl -fL --retry 3 --retry-all-errors \
        -o "$output.part" "$BASE_URL/$name"
    mv "$output.part" "$output"
done

if command -v sha256sum >/dev/null 2>&1; then
    hash_file() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    echo "error: sha256sum or shasum is required to verify the release" >&2
    exit 1
fi

verify() {
    expected="$1"
    name="$2"
    actual="$(hash_file "$DEST/$name")"
    if [ "$actual" != "$expected" ]; then
        echo "error: SHA-256 mismatch for $name" >&2
        exit 1
    fi
    echo "[verified] $name"
}

verify 9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411 LICENSE
verify 8f5b926dcf7bf1940796af9ac9be0239fea65ee22bf1a0e0a3ddae542990cacb README.md
verify ed8d050fc5da275cfca88224b7d2cc29fde7c23e133618f346e98ac505b3d862 decoder.int8.onnx
verify 2194513df47271ece9f8d2d571facd57b24d96a1d912fdc13bf0e10c69318e85 encoder.onnx
verify 8d627d235c44a597da908e1085ebe241cbbe358964c502c5a5063d18851a5529 lm_flow.int8.onnx
verify bfc0c7e7e3d72864fa3bb2ee499f62f21ddc1474b885f5f3ca570f8be73e787e lm_main.int8.onnx
verify 5eeff5278cfb1e9b627d972512ddf1e2dfacf61827103d9fb9c53707b38d0968 manifest.json
verify 5217b8474621af91127cfef891714337ae8cba106710ce04a426ec4bd56bbd1e text_conditioner.onnx
verify 3baa6ef7d57bac245271e33f96161f3ac60038d753f13f3fdbe24a7d2422ad6b token_scores.json
verify d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6 tokenizer.model
verify a2673c232cf49dd6eb1ad850e7c7682f6443c2ab64040d1150e9d8f2a7e3587b vocab.json

echo
echo "Pocket TTS ONNX bundle ready: $DEST"
echo "Run tests with:"
echo "  SPEECH_POCKET_TTS_BUNDLE=$DEST ctest --test-dir build --output-on-failure"
