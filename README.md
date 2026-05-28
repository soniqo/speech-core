# Speech Core

Voice agent pipeline engine in C++17 for **Linux, Windows, and Android** (plus Apple platforms via a prebuilt XCFramework). Provides the orchestration layer for real-time conversational AI — state machine, turn detection, interruption handling, speech queuing — with **zero ML dependencies in the core**.

Model inference is opt-in through two interchangeable, independent backends:

- **ONNX Runtime** (`SPEECH_CORE_WITH_ONNX`) — Silero VAD, Parakeet STT, Kokoro TTS, DeepFilterNet3.
- **LiteRT** (`SPEECH_CORE_WITH_LITERT`) — Silero VAD, Parakeet STT, VoxCPM2 TTS. Backed by Google's `ai-edge-litert` runtime (`libLiteRt`).

Consumers can enable either, both, or neither — and can always bring their own implementations of the abstract interfaces (STT, TTS, LLM, VAD, Enhancer); speech-swift does this with CoreML/MLX on Apple platforms.

## Supported models

| Model | Interface | ONNX | LiteRT |
|---|---|:---:|:---:|
| Silero VAD v5 | VAD | ✓ | ✓ |
| Parakeet TDT v3 (0.6B) | STT | ✓ | ✓ |
| Kokoro 82M | TTS | ✓ | — |
| DeepFilterNet3 | Enhancer | ✓ | — |
| VoxCPM2 (2B) | TTS | — | ✓ |

(Kokoro / DeepFilterNet3 don't have LiteRT exports yet, and VoxCPM2 has no ONNX export — wrappers land as `speech-models` ships the artifacts.)

## Backends & platforms

| Backend | Library | Runtime dep | Platforms | Setup |
|---|---|---|---|---|
| ONNX | `speech_core_models` | `onnxruntime` | Linux, macOS, Android | `ORT_DIR` from an ONNX Runtime release |
| LiteRT | `speech_core_models_litert` | `libLiteRt` | Linux x86_64, Windows x86_64, Android, macOS arm64 | `scripts/fetch_litert.sh` (extracts from the `ai-edge-litert` PyPI wheel) |

Both backends run CPU inference today; GPU / NNAPI / QNN delegate selection is a follow-up.

## Architecture

```
┌──────────────────────────────────────────────┐
│            speech-core (always built)         │
│                                              │
│  VoicePipeline / TurnDetector / SpeechQueue  │  orchestration
│  StreamingVAD / AudioBuffer / Resampler      │
│                                              │
│  STTInterface  TTSInterface  LLMInterface    │  abstract interfaces
│  VADInterface  EnhancerInterface  AEC        │
└──────────────────────────────────────────────┘
              ▲                       ▲
              │ implements (optional) │
              │                       │
┌─────────────┴──────────┐  ┌─────────┴────────────────┐
│ speech_core_models     │  │ speech_core_models_litert │
│ (SPEECH_CORE_WITH_ONNX)│  │ (SPEECH_CORE_WITH_LITERT) │
│                        │  │                          │
│ SileroVad      : VAD   │  │ LiteRTSileroVad   : VAD  │
│ ParakeetStt    : STT   │  │ LiteRTParakeetStt : STT  │
│ KokoroTts      : TTS   │  │ LiteRTVoxCPM2Tts  : TTS  │
│ DeepFilterEnhancer:Enh │  │                          │
└────────────────────────┘  └──────────────────────────┘
```

See [`docs/interfaces.md`](docs/interfaces.md) and [`docs/models.md`](docs/models.md) for details.

## Pipeline Modes

| Mode | Flow | Use case |
|---|---|---|
| **VoicePipeline** | audio → VAD → STT → LLM → TTS → audio | Full voice agent |
| **Echo** | audio → VAD → STT → TTS → audio | Testing |
| **TranscribeOnly** | audio → VAD → STT → text | Transcription only |

See [`docs/pipeline.md`](docs/pipeline.md) for state machine, turn detection, interruption handling, and configuration.

## Components

### Pipeline (`include/speech_core/pipeline/`)

| File | Purpose |
|---|---|
| `voice_pipeline.h` | Main orchestrator — connects STT, LLM, TTS via abstract interfaces |
| `turn_detector.h` | Wraps StreamingVAD, adds end-of-utterance detection and interruption logic |
| `speech_queue.h` | Priority queue for TTS outputs with cancel, interrupt, resume |
| `conversation_context.h` | Message history and turn tracking |
| `agent_config.h` | Pipeline configuration (thresholds, timeouts, model selection) |

### VAD (`include/speech_core/vad/`)

| File | Purpose |
|---|---|
| `streaming_vad.h` | 4-state hysteresis state machine (silence / pendingSpeech / speech / pendingSilence) |
| `vad_config.h` | Onset/offset thresholds, min speech/silence durations |

### Audio (`include/speech_core/audio/`)

| File | Purpose |
|---|---|
| `audio_buffer.h` | Lock-free ring buffer for streaming mic input |
| `resampler.h` | Windowed-sinc resampler with anti-aliasing and precomputed kernel cache |
| `pcm_codec.h` | Float32 / PCM16-LE / base64 conversions |
| `fft.h` | Radix-2 FFT for real signals |
| `mel.h` | Log-mel spectrogram (HTK / Slaney scales) |
| `stft.h` | Forward / inverse STFT with overlap-add |

### Models (`include/speech_core/models/`, optional)

ONNX wrappers (`SPEECH_CORE_WITH_ONNX`):

| File | Implements | Notes |
|---|---|---|
| `silero_vad.h` | `VADInterface` | Silero VAD v5, 512 samples @ 16 kHz |
| `parakeet_stt.h` | `STTInterface` | Parakeet TDT v3, batch + streaming, language detection |
| `kokoro_tts.h` | `TTSInterface` | Kokoro 82M, 24 kHz, eSpeak-free phonemizer (9 languages) |
| `deepfilter.h` | `EnhancerInterface` | DeepFilterNet3, 48 kHz noise cancellation |
| `onnx_engine.h` | (internal) | ORT singleton, NNAPI/QNN/CPU provider auto-selection |

LiteRT wrappers (`SPEECH_CORE_WITH_LITERT`):

| File | Implements | Notes |
|---|---|---|
| `litert_silero_vad.h` | `VADInterface` | Silero VAD v5 |
| `litert_parakeet_stt.h` | `STTInterface` | Parakeet TDT v3, INT8 encoder |
| `litert_voxcpm2_tts.h` | `TTSInterface` | VoxCPM2 (2B), 48 kHz, 4-graph pipeline |
| `voxcpm2_tokenizer.h` | (internal) | Hand-rolled BPE tokenizer, pure C++17 |
| `litert_engine.h` | (internal) | LiteRT environment + CompiledModel + TensorBuffer RAII |

See [`docs/models.md`](docs/models.md) for usage.

### Protocol (`include/speech_core/protocol/`)

| File | Purpose |
|---|---|
| `events.h` | Event type definitions (speech_started, transcript, audio_delta, etc.) |

### Interfaces (`include/speech_core/interfaces.h`)

Abstract classes any backend implements: `STTInterface`, `TTSInterface`, `LLMInterface`, `VADInterface`, `EnhancerInterface`, `EchoCancellerInterface`. See the header for signatures.

### Tools (`include/speech_core/tools/`)

Tool calling via LLM function calls. See [`docs/tools.md`](docs/tools.md).

### C API (`include/speech_core/speech_core_c.h`)

C wrapper for FFI — vtable-based interface bridging for Swift, Kotlin, etc. See [`docs/c-api.md`](docs/c-api.md).

## Build

Default build (orchestration only, no ML deps):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
cd build && ctest
```

With the **ONNX** backend:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DORT_DIR=/path/to/onnxruntime
cmake --build build
```

`ORT_DIR` must contain `include/onnxruntime_c_api.h` and a platform shared library (`libonnxruntime.dylib` on macOS, `libonnxruntime.so` on Linux, `lib/${ANDROID_ABI}/libonnxruntime.so` on Android). Builds `libspeech_core_models.a` (links `speech_core` + `onnxruntime`).

With the **LiteRT** backend:

```bash
scripts/fetch_litert.sh build/litert        # extracts libLiteRt from the ai-edge-litert wheel
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR=$PWD/build/litert
cmake --build build
```

`LITERT_DIR` points at the directory holding `libLiteRt.{so,dylib,dll}`; headers are vendored in `third_party/litert/`. Builds `libspeech_core_models_litert.a` (links `speech_core` + `litert`). The two backends are independent and can be enabled together.

To run the model integration tests (downloads model files):

```bash
scripts/download_models.sh                                   # ONNX (~1.2 GB)
SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure

scripts/download_models_litert.sh                            # LiteRT Silero + Parakeet
scripts/download_voxcpm2_litert.sh                           # VoxCPM2 bundle (~4.6 GB, optional)
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

See [`docs/models.md`](docs/models.md) for the full test setup.

### Examples

Add `-DSPEECH_CORE_BUILD_EXAMPLES=ON` to build the Linux example: a small C ABI library (`libspeech.so`), an ALSA demo, three CLI tools (`speech_transcribe`, `speech_synthesize`, `speech_phonemize`), and a C-ABI integration test. See [`examples/linux/README.md`](examples/linux/README.md) for details.

## Design Principles

- **ML inference is opt-in.** The orchestration core is pure C++17 with no ML deps. The ONNX and LiteRT backends are compiled in only when explicitly requested.
- **No platform dependencies in the core** — pure C++17, no OS-specific APIs. The backend wrappers use platform features (NNAPI on Android, etc.) but only when enabled.
- **No network I/O** — no sockets, no HTTP, no WebSocket.
- **No audio I/O** — audio buffer and resampler operate on float arrays.
- **Callback-driven** — pipeline emits events via `std::function` callbacks.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
