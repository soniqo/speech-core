# Speech Core

Voice agent pipeline engine in C++. Provides the orchestration layer for real-time conversational AI — state machine, turn detection, interruption handling, and speech queuing — plus a set of optional ONNX Runtime reference implementations (Silero VAD, Parakeet STT, Kokoro TTS, DeepFilterNet3 enhancer).

The orchestration core has zero ML dependencies. Consumers either bring their own model implementations of the abstract interfaces (STT, TTS, LLM, VAD, Enhancer), or compile in the reference implementations via `SPEECH_CORE_WITH_ONNX=ON`.

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
                       ▲
                       │ implements (optional)
                       │
┌──────────────────────┴───────────────────────┐
│   Reference models (SPEECH_CORE_WITH_ONNX)   │
│                                              │
│   SileroVad         : VADInterface           │
│   ParakeetStt       : STTInterface           │
│   KokoroTts         : TTSInterface           │
│   DeepFilterEnhancer: EnhancerInterface      │
└──────────────────────────────────────────────┘
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

ONNX Runtime reference implementations, compiled in only when `SPEECH_CORE_WITH_ONNX=ON`.

| File | Implements | Notes |
|---|---|---|
| `silero_vad.h` | `VADInterface` | Silero VAD v5, 512 samples @ 16 kHz |
| `parakeet_stt.h` | `STTInterface` | Parakeet TDT v3, batch + streaming, language detection |
| `kokoro_tts.h` | `TTSInterface` | Kokoro 82M, 24 kHz, eSpeak-free phonemizer (9 languages) |
| `deepfilter.h` | `EnhancerInterface` | DeepFilterNet3, 48 kHz noise cancellation |
| `onnx_engine.h` | (internal) | ORT singleton, NNAPI/QNN/CPU provider auto-selection |

See [`docs/models.md`](docs/models.md) for usage.

### Protocol (`include/speech_core/protocol/`)

| File | Purpose |
|---|---|
| `events.h` | Event type definitions (speech_started, transcript, audio_delta, etc.) |

### Interfaces (`include/speech_core/interfaces.h`)

Abstract classes:

```cpp
class STTInterface {
    virtual TranscriptionResult transcribe(const float* audio, size_t length, int sample_rate) = 0;
    virtual int input_sample_rate() const = 0;
};

class TTSInterface {
    virtual void synthesize(const std::string& text, const std::string& language,
                            TTSChunkCallback on_chunk) = 0;
    virtual int output_sample_rate() const = 0;
    virtual void cancel() {}
};

class LLMInterface {
    virtual LLMResponse chat(const std::vector<Message>& messages,
                             LLMTokenCallback on_token) = 0;
    virtual void set_tools(const std::vector<ToolDefinition>& tools) {}
    virtual void cancel() {}
};

class VADInterface {
    virtual float process_chunk(const float* samples, size_t length) = 0;
    virtual void reset() = 0;
    virtual int input_sample_rate() const = 0;
    virtual size_t chunk_size() const = 0;
};
```

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

With ONNX Runtime reference models:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DORT_DIR=/path/to/onnxruntime
cmake --build build
```

This builds two static libraries:

- `libspeech_core.a` — orchestration core, no ML deps
- `libspeech_core_models.a` — ONNX model wrappers, links `speech_core` + `onnxruntime`

`ORT_DIR` must contain `include/onnxruntime_c_api.h` and a platform shared library (`libonnxruntime.dylib` on macOS, `libonnxruntime.so` on Linux, `lib/${ANDROID_ABI}/libonnxruntime.so` on Android).

To run the model integration tests (requires ~1.2 GB of model files):

```bash
scripts/download_models.sh
SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure
```

See [`docs/models.md`](docs/models.md) for the full test setup.

## Design Principles

- **ML inference is opt-in.** The orchestration core is pure C++17 with no ML deps. ONNX Runtime models are compiled in only when explicitly requested.
- **No platform dependencies in the core** — pure C++17, no OS-specific APIs. The ORT-backed models use platform features (NNAPI on Android, QNN elsewhere) but only when enabled.
- **No network I/O** — no sockets, no HTTP, no WebSocket.
- **No audio I/O** — audio buffer and resampler operate on float arrays.
- **Callback-driven** — pipeline emits events via `std::function` callbacks.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
