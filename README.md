# Speech Core

Voice agent pipeline engine in C++. Provides the orchestration layer for real-time conversational AI — state machine, turn detection, interruption handling, speech queuing, and protocol handling.

ML inference is **not** in this library. Consumers implement the abstract interfaces (STT, TTS, LLM, VAD) with their own models.

## Architecture

```
                    ┌───────────────────────────┐
                    │       speech-core          │
                    │                           │
                    │   VoicePipeline           │  STT -> LLM -> TTS orchestration
                    │   TurnDetector            │  VAD-driven turn boundaries
                    │   SpeechQueue             │  Priority queue, cancel/resume
                    │   RealtimeProtocol        │  OpenAI Realtime API messages
                    │   StreamingVAD            │  Hysteresis state machine
                    │   AudioBuffer             │  Ring buffer, resampler, PCM
                    │                           │
                    │   STTInterface            │  Abstract speech-to-text
                    │   TTSInterface            │  Abstract text-to-speech
                    │   LLMInterface            │  Abstract language model
                    │   VADInterface            │  Abstract voice activity detection
                    │   EnhancerInterface       │  Abstract speech enhancement
                    │                           │
                    └───────────────────────────┘
```

## Pipeline Modes

| Mode | Flow | Use case |
|---|---|---|
| **VoicePipeline** | audio → VAD → STT → LLM → TTS → audio | Full voice agent with conversational AI |
| **Echo** | audio → VAD → STT → TTS → audio | Testing — speaks back what the user said |
| **TranscribeOnly** | audio → VAD → STT → text | Speech-to-text only, no response |

In **VoicePipeline** mode, the LLM decides when to call tools — if it returns tool call requests, the pipeline executes them and calls the LLM again with the results.

See [`docs/pipeline.md`](docs/pipeline.md) for detailed state machine, turn detection, and interruption handling documentation.

## Pipeline State Machine

The voice pipeline manages the full conversational loop:

```
    ┌──────┐  VAD: speech_started   ┌───────────┐
    │ IDLE ├───────────────────────►│ LISTENING  │
    └──┬───┘                        └─────┬─────┘
       ▲                                  │ VAD: speech_ended
       │                            ┌─────▼──────────┐
       │                            │ TRANSCRIBING    │
       │                            └─────┬──────────┘
       │                                  │ STT result
       │                            ┌─────▼──────────┐
       │                            │ THINKING        │
       │                            └─────┬──────────┘
       │                                  │ LLM tokens
       │                            ┌─────▼──────────┐
       │  TTS done / interrupted    │ SPEAKING        │
       └────────────────────────────┴────────────────┘
```

**Interruption handling**: when the user speaks while the agent is in SPEAKING state, the pipeline cancels TTS output and transitions back to LISTENING. False-interruption recovery pauses playback briefly and resumes if the user stops within a configurable window.

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
| `resampler.h` | Sample rate conversion (e.g. 24kHz to 16kHz for STT) |
| `pcm_codec.h` | Float32 / PCM16-LE / base64 conversions |

### Protocol (`include/speech_core/protocol/`)

| File | Purpose |
|---|---|
| `realtime_protocol.h` | OpenAI Realtime API message parser and serializer |
| `events.h` | Event type definitions (speech_started, transcript, audio_delta, etc.) |

### Interfaces (`include/speech_core/interfaces.h`)

Abstract classes:

```cpp
class STTInterface {
    virtual TranscriptionResult transcribe(const float* audio, size_t len, int sampleRate) = 0;
};

class TTSInterface {
    virtual void synthesize(const std::string& text,
                           std::function<void(const float*, size_t, bool)> onChunk) = 0;
};

class LLMInterface {
    virtual void chat(const std::vector<Message>& messages,
                     std::function<void(const std::string&, bool)> onToken) = 0;
};

class VADInterface {
    virtual float processChunk(const float* samples, size_t len) = 0;
};
```

### Tools (`include/speech_core/tools/`)

| File | Purpose |
|---|---|
| `tool_types.h` | `ToolDefinition` and `ToolResult` structs |
| `tool_registry.h` | Registry — add tools programmatically or load from JSON |
| `intent_matcher.h` | Regex pattern matching on transcripts (case-insensitive) |
| `tool_executor.h` | Shell command execution with cooldown enforcement |

Tool definitions:

```json
[
  {
    "name": "tell_time",
    "description": "Tell the current time",
    "triggers": ["what time", "current time"],
    "command": "date '+%I:%M %p'",
    "timeout": 5,
    "cooldown": 30
  }
]
```

The LLM decides when to call tools. When the LLM returns a `ToolCall`, the pipeline executes the matching command, injects the result into the conversation, and calls the LLM again for a final response.

### C API (`include/speech_core/speech_core_c.h`)

C wrapper for FFI — vtable-based interface bridging for Swift, Kotlin, etc.

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
- **No network I/O** — protocol layer parses and serializes messages, but doesn't own the transport.
- **No audio I/O** — audio buffer and resampler operate on float arrays.
- **Callback-driven** — pipeline emits events via `std::function` callbacks.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
