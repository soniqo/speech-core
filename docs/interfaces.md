# Interfaces

speech-core defines a small set of pure-virtual C++ interfaces in `include/speech_core/interfaces.h`. The orchestration layer (`VoicePipeline`, `TurnDetector`, `SpeechQueue`) depends only on these interfaces, never on concrete implementations. This lets callers plug in any backend — ONNX Runtime, CoreML, MLX, a remote API — without modifying the core.

This file is the C++ counterpart to speech-swift's [`docs/shared-protocols.md`](https://github.com/soniqo/speech-swift/blob/main/docs/shared-protocols.md). The two libraries share the same conceptual surface; the Swift side calls them `protocol`, here they're `class … = 0`.

```
┌─────────────────────────────────────────────┐
│      speech_core (orchestration)            │
│                                             │
│  VoicePipeline ─┬─► STTInterface            │
│                 ├─► TTSInterface            │
│                 ├─► VADInterface            │
│                 ├─► EnhancerInterface       │
│                 ├─► EchoCancellerInterface  │
│                 └─► LLMInterface (optional) │
└─────────────────────────────────────────────┘
              ▲
              │ implements
              │
┌─────────────┴────────────────────────────────┐
│  Reference implementations (optional, ORT)   │
│                                              │
│  SileroVad             : VADInterface        │
│  ParakeetStt           : STTInterface        │
│  KokoroTts             : TTSInterface        │
│  DeepFilterEnhancer    : EnhancerInterface   │
└──────────────────────────────────────────────┘
```

## STTInterface — Speech-to-text

```cpp
class STTInterface {
public:
    virtual ~STTInterface() = default;

    // Batch
    virtual TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) = 0;
    virtual int input_sample_rate() const = 0;
    virtual void cancel() {}

    // Optional streaming
    virtual bool supports_streaming() const { return false; }
    virtual void begin_stream(int sample_rate) {}
    virtual PartialResult push_chunk(const float* audio, size_t length) { return {}; }
    virtual void flush_stream() {}
    virtual TranscriptionResult end_stream() { return {}; }
    virtual void cancel_stream() {}
};
```

`TranscriptionResult` carries `text`, `language` (ISO 639-1 code, empty if not detected), `confidence`, and `start_time` / `end_time`. `PartialResult` is the streaming counterpart — same fields minus timestamps.

**Reference implementation:** `ParakeetStt` (Parakeet TDT v3 via ONNX Runtime).

**Swift counterpart:** `SpeechRecognitionModel` in `AudioCommon/Protocols.swift`.

## TTSInterface — Text-to-speech

```cpp
class TTSInterface {
public:
    virtual ~TTSInterface() = default;

    virtual void synthesize(
        const std::string& text,
        const std::string& language,
        TTSChunkCallback on_chunk) = 0;
    virtual int output_sample_rate() const = 0;
    virtual void cancel() {}
};

using TTSChunkCallback = std::function<void(
    const float* samples, size_t length, bool is_final)>;
```

The callback is invoked for each audio chunk during synthesis. `is_final=true` marks the last chunk. Implementations that aren't streaming-capable can emit a single chunk with `is_final=true`.

**Reference implementation:** `KokoroTts` (Kokoro 82M via ONNX Runtime, single-chunk output).

**Swift counterpart:** `SpeechGenerationModel`.

## VADInterface — Voice activity detection (chunk-level)

```cpp
class VADInterface {
public:
    virtual ~VADInterface() = default;

    virtual float process_chunk(const float* samples, size_t length) = 0;
    virtual void reset() = 0;
    virtual int input_sample_rate() const = 0;
    virtual size_t chunk_size() const = 0;
};
```

Returns a speech probability in `[0, 1]` per chunk. The probability stream is consumed by `StreamingVAD` (in `speech_core/vad/streaming_vad.h`), which applies hysteresis and emits `SpeechStarted` / `SpeechEnded` events. The pipeline owns the `StreamingVAD`; you supply the chunk-level model.

**Reference implementation:** `SileroVad` (Silero VAD v5 via ONNX Runtime, 512 samples @ 16 kHz).

**Swift counterpart:** `StreamingVADProvider`.

## EnhancerInterface — Speech enhancement / denoising

```cpp
class EnhancerInterface {
public:
    virtual ~EnhancerInterface() = default;

    virtual void enhance(
        const float* audio, size_t length, int sample_rate,
        float* output) = 0;
    virtual int input_sample_rate() const = 0;
};
```

Pre-allocated output buffer. Caller is responsible for sample-rate matching.

**Reference implementation:** `DeepFilterEnhancer` (DeepFilterNet3 via ONNX Runtime, 48 kHz).

**Swift counterpart:** `SpeechEnhancementModel`.

## EchoCancellerInterface — Acoustic echo cancellation

```cpp
class EchoCancellerInterface {
public:
    virtual ~EchoCancellerInterface() = default;

    virtual void feed_reference(const float* samples, size_t length) = 0;
    virtual void cancel_echo(const float* input, size_t length, float* output) = 0;
    virtual int input_sample_rate() const = 0;
    virtual void reset() = 0;
};
```

Pipeline feeds TTS output via `feed_reference()` and runs `cancel_echo()` on mic input before VAD. Thread-safety: `feed_reference()` and `cancel_echo()` may be called concurrently. No reference implementation is shipped — bring your own (SpeexDSP, WebRTC AEC, platform-native).

## LLMInterface — Language model

```cpp
class LLMInterface {
public:
    virtual ~LLMInterface() = default;

    virtual LLMResponse chat(
        const std::vector<Message>& messages,
        LLMTokenCallback on_token) = 0;
    virtual void set_tools(const std::vector<ToolDefinition>& tools) {}
    virtual void cancel() {}
};
```

Only needed for the `VoicePipeline` mode (full agent loop). Not needed for `Echo` or `TranscribeOnly` modes.

## Design choices

1. **Direct inheritance.** Models inherit interfaces directly (`class SileroVad : public VADInterface`) rather than going through adapter classes. speech-swift uses extension files to keep the same separation (Swift only — C++ has no equivalent). With only one consumer (speech-core itself), the indirection isn't worth its cost; revisit if a new caller wants the models without the orchestration.

2. **No vtable boilerplate in callers.** Earlier the platform code (Linux `speech.cpp`, Android `jni_bridge.cpp`) hand-rolled `sc_vad_t` / `sc_stt_t` / `sc_tts_t` adapter structs that wrapped the model classes. With direct inheritance those adapters are unnecessary — the model pointer goes straight into `VoicePipeline`.

3. **ORT is optional.** speech-core builds without ONNX Runtime by default. Set `-DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=…` to compile in the reference implementations. Consumers that bring their own backends (e.g. speech-swift uses CoreML/MLX) get a smaller artifact.

4. **No abstract `InferenceBackend` layer.** An earlier design had `InferenceBackend` / `OnnxBackend` / `LiteRT` placeholder classes. The actual models bypassed them and talked to `OrtApi` directly, so the abstraction was dead code. Dropped. If a second backend lands, reintroduce it then.
