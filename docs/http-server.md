# OpenAI-compatible local audio server

`speech-server` exposes Kokoro ONNX text-to-speech and Parakeet ONNX
speech-to-text through OpenAI-compatible request shapes. It runs locally on
Linux and Windows and does not send text or audio to a remote service.

```text
POST /v1/audio/speech
POST /v1/audio/transcriptions
```

The server is included in speech-core release packages. Parakeet is loaded only
on the first transcription request, so TTS-only startup time and resident model
memory remain unchanged.

## Install and start

### Linux

Install the `.deb` or unpack the `.tar.gz`, then download the ONNX model set:

```bash
speech download-models
speech serve
```

The standalone executable accepts the same options:

```bash
speech-server --host 127.0.0.1 --port 8080
```

### Windows x64

Download and extract `speech-0.0.11-windows-x64.zip`, then run these commands
from its `bin` directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\speech_download_models.ps1
.\speech-server.exe
```

The model downloader uses `%LOCALAPPDATA%\speech-core\models` by default. Pass
an explicit directory to the downloader and server when a different cache is
needed:

```powershell
.\speech_download_models.ps1 D:\speech-models
.\speech-server.exe --model-dir D:\speech-models
```

## Speech synthesis

The JSON body follows the OpenAI speech endpoint shape:

This is a local, uncompressed subset of the
[OpenAI request contract](https://platform.openai.com/docs/api-reference/audio/createSpeech):
the hosted API defaults to MP3 and supports additional codecs, while
`speech-server` defaults to WAV and rejects compressed formats so the package
does not need a codec runtime.

| Field | Required | Behavior |
|---|:---:|---|
| `model` | ✓ | `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`, `gpt-4o-mini-tts-2025-12-15`, `kokoro`, `kokoro-82m`, or `kokoro-82m-onnx` |
| `input` | ✓ | Non-empty UTF-8 text, at most 4,096 characters |
| `voice` | ✓ | A bundled Kokoro voice ID string or `{ "id": "af_heart" }` object (up to 128 ASCII letters, digits, `_`, or `-`); `alloy` maps to `af_alloy` and other generic OpenAI voice names map to `af_heart` |
| `response_format` |  | `wav` by default, or headerless `pcm`; compressed formats are rejected |
| `speed` |  | `1.0` by default; accepted range is `0.25...4.0` |
| `language` |  | `en` by default; names and common codes map to Kokoro's supported language codes |
| `instructions` |  | Not supported by Kokoro; a non-empty value is rejected rather than silently ignored |
| `stream_format` |  | `audio` (default) is accepted; `sse` is not supported |

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "tts-1",
    "input": "Hello from speech-core.",
    "voice": "alloy",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

WAV responses contain mono PCM16-LE at the model's output sample rate. `pcm`
returns the same PCM16-LE samples without a container header. Endpoint
validation, authentication, and synthesis errors use the nested OpenAI
envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": null,
    "param": null
  }
}
```

`GET /health` returns `{"status":"ok"}`. Request text is not written to server
logs.

The model downloader installs `af_alloy`, `af_bella`, `af_heart`, `af_nicole`,
`af_sky`, `am_adam`, `am_michael`, `bf_emma`, and `bm_george`. A native voice
ID selects the matching file from the model directory's `voices` folder.

## Speech transcription

The transcription endpoint accepts the standard multipart form shape. The
`file` must be an uncompressed RIFF/WAVE file containing PCM16, PCM24, PCM32,
or Float32 audio. Mono and multichannel input from 8 kHz through 384 kHz is
accepted; the adapter downmixes and resamples it to the Parakeet model's 16 kHz
mono Float32 input.

| Field | Required | Behavior |
|---|:---:|---|
| `file` | ✓ | One WAV file, at most 25 MiB and 10 minutes |
| `model` | ✓ | A non-empty OpenAI model name such as `whisper-1`; the local server always routes it to the installed Parakeet model |
| `response_format` |  | `json` by default, or `text`, `verbose_json`, `srt`, or `vtt` |
| `language` |  | Accepted for client compatibility; Parakeet auto-detects language and cannot force it |
| `prompt` |  | Accepted for client compatibility; the Parakeet export has no prompt channel |
| `temperature` |  | Accepted in the OpenAI range `0...1`; greedy Parakeet decoding does not sample |

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F 'file=@recording.wav;type=audio/wav' \
  -F 'model=whisper-1'
```

The default JSON response is `{"text":"..."}`. `verbose_json`, SRT, and VTT
contain one utterance-level segment because the bundled Parakeet interface does
not expose word or segment timestamps. Compressed inputs such as MP3, M4A, and
WebM are rejected so the package does not need an additional codec runtime.

## Authentication and network exposure

The default bind address is `127.0.0.1`. Configure a bearer token with the
environment, which avoids exposing it in the process list:

```bash
export SPEECH_SERVER_API_KEY='replace-with-a-random-value'
speech serve --host 0.0.0.0
```

Clients then send `Authorization: Bearer replace-with-a-random-value` to either
audio endpoint.
`--api-key` is also accepted for short-lived local sessions. A non-loopback
bind is rejected when no key is configured unless
`--allow-unauthenticated-remote` is passed explicitly. That override should be
used only on a trusted, isolated network.

The server speaks plaintext HTTP. For access beyond a trusted, isolated
network, keep bearer authentication enabled and place it behind a
TLS-terminating reverse proxy and firewall.

The server keeps speech-synthesis requests at the existing 1 MiB limit and caps
uploaded transcription files at 25 MiB. It caps synthesis text and transcription
prompts at 4,096 characters. Kokoro remains resident, Parakeet loads lazily, and
all inference is serialized through one model lock. Text and transcripts are
not written to server logs.

## Build from source

Point `ORT_DIR` at an extracted ONNX Runtime distribution and enable the
sidecar:

```bash
cmake -B build-server -DCMAKE_BUILD_TYPE=Release \
  -DSPEECH_CORE_WITH_ONNX=ON \
  -DSPEECH_CORE_BUILD_HTTP_SERVER=ON \
  -DORT_DIR=/path/to/onnxruntime
cmake --build build-server --parallel --target speech_server
```

The model-independent HTTP contract tests are part of the default CTest suite
and do not download model weights. A build with ONNX additionally compiles the
production `speech-server` executable and a real-model transcription endpoint
test. Set `SPEECH_MODEL_DIR` to a downloaded model bundle to run that E2E path;
it skips when the bundle is unavailable.
