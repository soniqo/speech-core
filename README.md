# Speech Core

Voice agent pipeline engine in C++. Provides the orchestration layer for real-time conversational AI — state machine, turn detection, interruption handling, and speech queuing.

ML inference is **not** in this library. Consumers implement the abstract interfaces (STT, TTS, LLM, VAD) with their own models.

## Architecture

```
                    ┌───────────────────────────┐
                    │       speech-core          │
                    │                           │
                    │   VoicePipeline           │  STT -> LLM -> TTS orchestration
                    │   TurnDetector            │  VAD-driven turn boundaries
                    │   SpeechQueue             │  Priority queue, cancel/resume
                    │   StreamingVAD            │  Hysteresis state machine
                    │   AudioBuffer             │  Ring buffer, resampler, PCM
                    │                           │
                    │   STTInterface            │  Abstract speech-to-text
                    │   TTSInterface            │  Abstract text-to-speech
                    │   LLMInterface            │  Abstract language model
                    │   VADInterface            │  Abstract voice activity detection
                    │   EnhancerInterface       │  Abstract speech enhancement
                    │   EchoCancellerInterface  │  Abstract echo cancellation (AEC)
                    │                           │
                    └───────────────────────────┘
```

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

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
cd build && ctest
```

## Design Principles

- **No ML inference** — this library never loads models or runs neural networks.
- **No platform dependencies** — pure C++17, no OS-specific APIs.
- **No network I/O** — no sockets, no HTTP, no WebSocket.
- **No audio I/O** — audio buffer and resampler operate on float arrays.
- **Callback-driven** — pipeline emits events via `std::function` callbacks.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
