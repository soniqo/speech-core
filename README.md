# speech-core

Cross-platform voice agent pipeline engine in C++. Provides the orchestration layer for real-time conversational AI — turn detection, interruption handling, speech queuing, and protocol handling — shared between [speech-swift](https://github.com/soniqo/speech-swift) (iOS/macOS) and speech-android (Android).

ML inference is **not** in this library. Each platform implements the abstract interfaces using its native accelerator:

| Platform | Inference | Hardware |
|---|---|---|
| iOS / macOS | CoreML | Apple Neural Engine |
| Android | Qualcomm AI Engine (QNN) | Hexagon NPU |

## Architecture

```
                        speech-core (this repo)
                    ┌───────────────────────────┐
                    │                           │
                    │   VoicePipeline           │  STT -> LLM -> TTS orchestration
                    │   TurnDetector            │  VAD-driven turn boundaries
                    │   SpeechQueue             │  Priority queue, cancel/resume
                    │   RealtimeProtocol        │  OpenAI Realtime API messages
                    │   StreamingVAD            │  Hysteresis state machine
                    │   AudioBuffer             │  Ring buffer, resampler, PCM
                    │                           │
                    │   STTInterface  ──────────┤── implemented per-platform
                    │   TTSInterface  ──────────┤── implemented per-platform
                    │   LLMInterface  ──────────┤── implemented per-platform
                    │   VADInterface  ──────────┤── implemented per-platform
                    │   EnhancerInterface ──────┤── implemented per-platform
                    │                           │
                    └─────────┬─────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
    ┌──────────────────┐           ┌──────────────────┐
    │  speech-swift    │           │  speech-android   │
    │                  │           │                   │
    │  CoreML models   │           │  QNN models       │
    │  AVAudioEngine   │           │  Oboe audio       │
    │  SwiftUI         │           │  Jetpack Compose   │
    └──────────────────┘           └──────────────────┘
```

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

Abstract classes implemented per-platform:

```cpp
class STTInterface {
    virtual std::string transcribe(const float* audio, size_t len, int sampleRate) = 0;
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

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run tests
cd build && ctest
```

### Integration

**iOS / macOS** (as .xcframework):
```bash
cmake -B build-ios \
  -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
  -DPLATFORM=OS64COMBINED \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-ios
```

Consumed by speech-swift as a binary `.xcframework` or via CMake + SPM C interop.

**Android** (as .so via NDK):
```bash
cmake -B build-android \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android
```

Consumed by speech-android via JNI.

## Project Structure

```
speech-core/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   └── speech_core/
│       ├── speech_core.h            # Top-level include
│       ├── interfaces.h             # Abstract STT/TTS/LLM/VAD/Enhancer
│       ├── pipeline/
│       │   ├── voice_pipeline.h
│       │   ├── turn_detector.h
│       │   ├── speech_queue.h
│       │   ├── conversation_context.h
│       │   └── agent_config.h
│       ├── vad/
│       │   ├── streaming_vad.h
│       │   └── vad_config.h
│       ├── audio/
│       │   ├── audio_buffer.h
│       │   ├── resampler.h
│       │   └── pcm_codec.h
│       └── protocol/
│           ├── realtime_protocol.h
│           └── events.h
├── src/
│   ├── pipeline/
│   │   ├── voice_pipeline.cpp
│   │   ├── turn_detector.cpp
│   │   ├── speech_queue.cpp
│   │   └── conversation_context.cpp
│   ├── vad/
│   │   └── streaming_vad.cpp
│   ├── audio/
│   │   ├── audio_buffer.cpp
│   │   ├── resampler.cpp
│   │   └── pcm_codec.cpp
│   └── protocol/
│       └── realtime_protocol.cpp
├── tests/
│   ├── test_streaming_vad.cpp
│   ├── test_turn_detector.cpp
│   ├── test_speech_queue.cpp
│   ├── test_audio_buffer.cpp
│   ├── test_pcm_codec.cpp
│   └── test_realtime_protocol.cpp
└── cmake/
    └── ios.toolchain.cmake
```

## Relationship to Other Repos

| Repo | Language | Role |
|---|---|---|
| [`soniqo/speech-swift`](https://github.com/soniqo/speech-swift) | Swift | iOS/macOS app — CoreML inference, AVAudioEngine, SwiftUI |
| `soniqo/speech-core` (this) | C++ | Shared pipeline engine — orchestration, VAD, protocol |
| `soniqo/speech-android` (planned) | Kotlin | Android app — QNN inference, Oboe audio, Compose |
| [`soniqo/soniqo-web`](https://github.com/soniqo/soniqo-web) | HTML | Documentation site at [soniqo.audio](https://soniqo.audio) |

## Tracking

This repo implements (or will implement) the core logic for several [speech-swift](https://github.com/soniqo/speech-swift) issues:

| Issue | Status | speech-core component |
|---|---|---|
| [#72 — Composable voice pipeline: VAD→ASR→LLM→TTS](https://github.com/soniqo/speech-swift/issues/72) | Scaffold done | `VoicePipeline`, `TurnDetector`, `SpeechQueue` |
| [#76 — Tool-calling intent engine](https://github.com/soniqo/speech-swift/issues/76) | Planned | `ToolRegistry`, `IntentMatcher` (not yet built) |
| [#80 — Streaming ASR with real-time diarization](https://github.com/soniqo/speech-swift/issues/80) | Planned | `DiarizedPipeline` variant (not yet built) |

### Remaining work for #72

- [ ] C wrapper (`speech_core_c.h`) for Swift/Kotlin FFI
- [ ] Swift adapters in speech-swift (`VoiceAgent` target)
- [ ] At least one `LLMInterface` implementation (Ollama HTTP)
- [ ] Wire into `AudioServer` WebSocket handler
- [ ] CLI command: `audio converse`
- [ ] End-to-end test: mic → VAD → ASR → LLM → TTS → speaker

## Design Principles

- **No ML inference** — this library never loads models or runs neural networks. Platform implementations do that.
- **No platform dependencies** — pure C++17, no Apple/Android/OS-specific APIs. Builds on any platform.
- **No network I/O** — protocol layer parses and serializes messages, but doesn't own the transport (WebSocket, HTTP).
- **No audio I/O** — audio buffer and resampler operate on float arrays. Mic/speaker is platform-specific.
- **Callback-driven** — pipeline emits events via `std::function` callbacks. Platform layer decides how to dispatch (main thread, async, etc.).
- **Header-only option** — small enough that header-only distribution is viable for simple integrations.

## License

Private. All rights reserved.
