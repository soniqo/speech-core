#!/usr/bin/env bash
# Download one German Kokoro voice bundle (kikiri-tts fine-tune) into its own
# directory. Each voice is a separate weight set with its own
# kokoro-e2e.onnx.data and vocab_index.json, so a bundle must not share a
# directory with the other voice or with the multilingual Kokoro-82M-ONNX files.
#
# Usage: ./download_kokoro_german_onnx.sh [martin|victoria] [dest]
#
# The pinned revisions are the packed-lstm branch of each repository, whose
# graphs match upstream Kokoro's durations; main keeps the earlier graphs.
#
# test_models reads $SPEECH_KOKORO_GERMAN_DIR, defaulting to
# $SPEECH_MODEL_DIR/kokoro-german, which is where the Martin bundle lands.
set -euo pipefail

VOICE="${1:-martin}"
case "$VOICE" in
  martin)
    REPO="soniqo/Kokoro-82M-German-Martin-ONNX"
    REVISION="${SPEECH_KOKORO_GERMAN_MARTIN_REVISION:-2f930df2e95cfc4a785e708eeea029a8e4f484d0}"
    VOICE_FILE="dm_martin.bin"
    DEFAULT_DEST="scripts/models/kokoro-german"
    ;;
  victoria)
    REPO="soniqo/Kokoro-82M-German-Victoria-ONNX"
    REVISION="${SPEECH_KOKORO_GERMAN_VICTORIA_REVISION:-7a6c7a111aa2b45ee8a8d902c9a1eb17d5d52d06}"
    VOICE_FILE="df_victoria.bin"
    DEFAULT_DEST="scripts/models/kokoro-german-victoria"
    ;;
  *)
    echo "Usage: $0 [martin|victoria] [dest]" >&2
    exit 2
    ;;
esac
DEST="${2:-$DEFAULT_DEST}"

mkdir -p "$DEST/voices"

AUTH_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_ARGS=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

download() {
  local remote_name="$1"
  local required="${2:-1}"
  local url="https://huggingface.co/${REPO}/resolve/${REVISION}/${remote_name}"
  local out="${DEST}/${remote_name}"
  if [ -f "$out" ]; then
    echo "exists ${out}"
    return
  fi
  echo "download ${REPO}@${REVISION}/${remote_name}"
  # The +"${...}" guard keeps set -u from tripping over an empty array, which
  # is an error in the bash 3.2 that ships with macOS.
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
download config.json 0
download vocab_index.json 1
download dict_de.json 1
download "voices/${VOICE_FILE}" 1
# Both graphs read the shared FP16 sidecar. Fetch it first so a graph never
# lands without its weights; the realtime graph comes last because callers
# test for it to decide whether the bundle is complete.
download kokoro-e2e.onnx.data 1
download kokoro-e2e.onnx 1
download kokoro-e2e-realtime.onnx 1

echo
echo "German Kokoro bundle (${VOICE}) ready: ${DEST}"
