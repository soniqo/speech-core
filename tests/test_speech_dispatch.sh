#!/bin/sh
set -eu

dispatcher="${1:?usage: test_speech_dispatch.sh <speech-dispatch.sh>}"
tmp="${TMPDIR:-/tmp}/speech-dispatch-test.$$"
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir -p "$tmp"
cp "$dispatcher" "$tmp/speech"
chmod +x "$tmp/speech"

for tool in \
    speech_transcribe speech_synthesize speech_phonemize speech-server \
    speech_voxcpm2_clone speech_demo speech_download_models \
    speech_download_models_litert speech_download_voxcpm2
do
    # shellcheck disable=SC2016 # The generated stub expands these at runtime.
    printf '%s\n' '#!/bin/sh' 'printf "%s\\n" "$(basename "$0")" "$@"' > "$tmp/$tool"
    chmod +x "$tmp/$tool"
done

assert_output() {
    expected="$1"
    shift
    actual="$("$tmp/speech" "$@")"
    if [ "$actual" != "$expected" ]; then
        printf 'command failed: speech' >&2
        printf ' %s' "$@" >&2
        printf '\nexpected:\n%s\nactual:\n%s\n' "$expected" "$actual" >&2
        exit 1
    fi
}

"$tmp/speech" --help | grep -q 'download-models \[litert|voxcpm2\]'

assert_output 'speech_transcribe
recording.wav' transcribe recording.wav

assert_output 'speech_synthesize
speech-out.wav
Hello world' speak 'Hello world'

assert_output 'speech_synthesize
hello.wav
Bonjour
fr' synthesize Bonjour hello.wav fr

assert_output 'speech_phonemize
bonjour
fr' phonemize bonjour fr

assert_output 'speech-server
--host
127.0.0.1
--port
8090' serve --host 127.0.0.1 --port 8090

assert_output 'speech_voxcpm2_clone
ref.wav
Cloned text
clone.wav' clone ref.wav 'Cloned text' clone.wav

assert_output 'speech_demo
--transcribe-only
--device
hw:1' demo --transcribe-only --device hw:1

assert_output 'speech_download_models' download-models
assert_output 'speech_download_models_litert
/models/litert' download-models litert /models/litert
assert_output 'speech_download_voxcpm2
/models/voxcpm2' download-models voxcpm2 /models/voxcpm2

rm "$tmp/speech_voxcpm2_clone"
set +e
missing_output="$("$tmp/speech" clone ref.wav text out.wav 2>&1)"
missing_status=$?
set -e
if [ "$missing_status" -ne 127 ] || ! printf '%s' "$missing_output" | grep -q 'not available in this package'; then
    printf 'missing-command diagnostic failed (status %s): %s\n' \
        "$missing_status" "$missing_output" >&2
    exit 1
fi

set +e
unknown_output="$("$tmp/speech" unknown 2>&1)"
unknown_status=$?
set -e
if [ "$unknown_status" -ne 2 ] || ! printf '%s' "$unknown_output" | grep -q "unknown command 'unknown'"; then
    printf 'unknown-command diagnostic failed (status %s): %s\n' \
        "$unknown_status" "$unknown_output" >&2
    exit 1
fi

printf 'speech dispatcher tests passed\n'
