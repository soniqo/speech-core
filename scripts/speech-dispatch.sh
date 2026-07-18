#!/bin/sh
# speech — multiplexer for the speech_* CLI tools, mirroring the verb-style
# interface of speech-swift's `speech` CLI (brew install soniqo/tap/speech)
# so cross-platform docs can say `speech transcribe recording.wav` everywhere.
#
# Dispatches to the sibling speech_* executables in the same directory, so it
# works from /usr/bin (the .deb), a brew Cellar, or an unpacked tarball alike.
set -e
here="$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")"
cmd="${1:-}"
[ $# -gt 0 ] && shift

run_tool() {
    tool="$1"
    shift
    if [ ! -x "$here/$tool" ]; then
        echo "speech: '$cmd' is not available in this package ($tool is missing)" >&2
        exit 127
    fi
    exec "$here/$tool" "$@"
}

case "$cmd" in
    transcribe)        run_tool speech_transcribe "$@" ;;
    speak|synthesize)
        # speech-swift form: speech speak "<text>" [out.wav]
        # speech_synthesize wants: [model_dir] <out.wav> "<text>" [language]
        if [ $# -ge 1 ]; then
            text="$1"; shift
            out="${1:-speech-out.wav}"; [ $# -ge 1 ] && shift
            run_tool speech_synthesize "$out" "$text" "$@"
        fi
        run_tool speech_synthesize ;;
    phonemize)         run_tool speech_phonemize "$@" ;;
    serve|server)      run_tool speech-server "$@" ;;
    clone)             run_tool speech_voxcpm2_clone "$@" ;;
    demo)              run_tool speech_demo "$@" ;;
    download-models)
        case "${1:-}" in
            litert)   shift; run_tool speech_download_models_litert "$@" ;;
            voxcpm2)  shift; run_tool speech_download_voxcpm2 "$@" ;;
            *)               run_tool speech_download_models "$@" ;;
        esac ;;
    -h|--help|help|"")
        cat <<USAGE
usage: speech <command> [args]

commands:
  transcribe <input.wav>             speech-to-text (Parakeet + Silero VAD)
  speak "<text>" [out.wav]           text-to-speech (Kokoro)
  serve [--host HOST] [--port PORT]  OpenAI-compatible local audio server
  clone <ref.wav> "<text>" <out.wav> voice cloning (VoxCPM2)
  phonemize "<text>" [language]      text -> phonemes (Kokoro phonemizer)
  demo [--transcribe-only]           live ALSA mic loop
  download-models [litert|voxcpm2]   fetch models to ~/.cache/speech-core

Each command is also available as a standalone speech_<command> binary
(speech-server for serve) with extra options. Model directories
default to \$SPEECH_MODEL_DIR / \$SPEECH_LITERT_MODEL_DIR, else the per-user
cache populated by download-models.
USAGE
        ;;
    *)
        echo "speech: unknown command '$cmd' (try: speech --help)" >&2
        exit 2 ;;
esac
