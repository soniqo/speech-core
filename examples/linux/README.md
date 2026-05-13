# Linux Example

Reference Linux build of speech-core — `libspeech.so` (small C ABI), ALSA demo CLI, transcribe/synthesize/phonemize tools, integration tests.

Targets embedded ARM64 platforms (Yocto, automotive — Qualcomm SA8295P / SA8255P) and any Linux dev box for quick smoke-testing the model wrappers.

## Build

```bash
# From the top of speech-core:
./examples/linux/setup_linux.sh       # download ONNX Runtime into ort-linux/
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DSPEECH_CORE_BUILD_EXAMPLES=ON \
    -DORT_DIR=ort-linux
cmake --build build

# Run unit + integration tests (integration test needs model files)
scripts/download_models.sh
SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure
```

### Docker

For a reproducible Linux build without setting up a local toolchain:

```bash
# Build the image (from repo root)
docker build -f examples/linux/Dockerfile -t speech-core-linux .

# Run tests with no models (unit + skip integration)
docker run --rm speech-core-linux

# Run tests with real models mounted
scripts/download_models.sh
docker run --rm -v "$(pwd)/scripts/models:/models" speech-core-linux \
    ctest --test-dir /work/build --output-on-failure
```

The same path runs in CI via `.github/workflows/ci.yml`'s `linux-examples` job on every PR.

## What CI does and does not cover

| Path | Where |
|---|---|
| Orchestration build + unit tests on macOS + Linux | `unit-tests` job |
| Full ONNX-on + examples build on Linux x86_64, libspeech.so + ALSA linkage, ctest | `linux-examples` job |
| aarch64 cross-compile (build-only, validates the Yocto/automotive toolchain path) | `linux-examples-aarch64` job |
| **Manual pre-release** — speech_demo against a real ALSA capture device | not automatable on hosted runners |
| **Manual pre-release** — QNN provider on Qualcomm hardware (SA8295P / SA8255P) | needs target hardware |

Build artefacts:

| Target | What |
|---|---|
| `libspeech.so` / `.dylib` | C ABI library — see `include/speech.h` |
| `speech_demo` | ALSA mic → pipeline → audio out demo (no-op stub when ALSA missing) |
| `speech_transcribe` | STT-only CLI over the C ABI; reads PCM from stdin or WAV |
| `speech_synthesize` | Kokoro TTS CLI — writes `.wav` of a phrase |
| `speech_phonemize` | Dumps the phoneme string + token IDs for input text |
| `speech_linux_test` | C-ABI integration test (skips without `SPEECH_MODEL_DIR`) |

## C API

```c
#include <speech.h>

void on_event(const speech_event_t* event, void* ctx) {
    if (event->type == SPEECH_EVENT_TRANSCRIPTION)
        printf("STT: %s\n", event->text);
}

int main(void) {
    speech_config_t cfg = speech_config_default();
    cfg.model_dir = "/opt/speech/models";

    speech_pipeline_t p = speech_create(cfg, on_event, NULL);
    speech_start(p);

    float buf[512];
    while (read_audio(buf, 512)) {
        speech_push_audio(p, buf, 512);
    }

    speech_destroy(p);
}
```

### Functions

| Function | Description |
|---|---|
| `speech_config_default()` | Default config (INT8, CPU, 400 ms silence threshold) |
| `speech_create(config, callback, ctx)` | Load models, create pipeline. Returns `NULL` on failure |
| `speech_start(pipeline)` | Start processing audio |
| `speech_push_audio(pipeline, samples, count)` | Feed PCM Float32 at 16 kHz |
| `speech_resume_listening(pipeline)` | Resume after TTS playback |
| `speech_destroy(pipeline)` | Free all resources |
| `speech_version()` | Version string |

### Events

| Event | Fields | Description |
|---|---|---|
| `SPEECH_EVENT_READY` | — | Pipeline initialized |
| `SPEECH_EVENT_SPEECH_STARTED` | — | VAD detected speech |
| `SPEECH_EVENT_SPEECH_ENDED` | — | VAD detected silence |
| `SPEECH_EVENT_PARTIAL_TRANSCRIPTION` | `text`, `confidence` | Streaming partial |
| `SPEECH_EVENT_TRANSCRIPTION` | `text`, `confidence`, `stt_duration_ms` | Final transcription |
| `SPEECH_EVENT_RESPONSE_AUDIO` | `audio_data`, `audio_data_length` | TTS PCM16 chunk (24 kHz) |
| `SPEECH_EVENT_RESPONSE_DONE` | `tts_duration_ms` | TTS complete |
| `SPEECH_EVENT_ERROR` | `text` | Error message |

### Configuration

```c
speech_config_t cfg = speech_config_default();
cfg.model_dir = "/opt/speech/models";  // required
cfg.use_int8 = true;                   // INT8 quantised models (default)
cfg.use_qnn = true;                    // Qualcomm QNN EP (Hexagon DSP)
cfg.enable_enhancer = true;            // DeepFilterNet3 noise cancellation
cfg.transcribe_only = true;            // STT only, no TTS echo
cfg.min_silence_duration = 0.4f;       // seconds before end-of-speech
```

## Cross-Compilation (Yocto)

```bash
# Source Yocto SDK environment
source /opt/poky/environment-setup-aarch64-poky-linux

cmake -B build \
    -DCMAKE_TOOLCHAIN_FILE=examples/linux/toolchain-aarch64.cmake \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DSPEECH_CORE_BUILD_EXAMPLES=ON \
    -DORT_DIR=/path/to/ort-linux-aarch64

cmake --build build
```

## QNN (Qualcomm Hexagon DSP)

For hardware acceleration on SA8295P / SA8255P:

1. Build ONNX Runtime with the QNN EP, or use Qualcomm's prebuilt
2. Place `libQnnHtp.so` in the library path
3. Set `cfg.use_qnn = true`

The pipeline falls back to CPU if QNN is unavailable.

## Implementation notes

`src/speech.cpp` is intentionally small (~170 lines). It constructs
`speech_core::SileroVad`, `ParakeetStt`, `KokoroTts` (and optionally
`DeepFilterEnhancer`) directly — those classes implement the speech_core
interfaces, so no C-vtable adapters are needed. The wrapper exists only to
expose a stable C ABI for non-C++ callers (automotive / Yocto integrators).

All inference runs on-device. No network required after the initial model
download. Models are hosted under the [aufklarer/](https://huggingface.co/aufklarer)
HF org — see the top-level `docs/models.md` for filenames and sizes.

## Thread Safety

- `speech_push_audio()` is thread-safe (single producer)
- The event callback fires from an internal worker thread
- Do not call `speech_destroy()` from the event callback
