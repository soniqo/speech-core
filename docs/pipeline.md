# Voice Pipeline

The `VoicePipeline` is the central orchestrator. It connects STT, TTS, LLM, and VAD implementations through a state machine with turn detection, conversation tracking, and speech queuing.

## Modes

### VoicePipeline (default)

Full voice agent loop:

```
audio → VAD → STT → [tools?] → LLM → TTS → audio
```

1. VAD detects user speech via `TurnDetector`
2. On speech end, the buffered audio is sent to `STTInterface.transcribe()`
3. Transcript + conversation history are sent to `LLMInterface.chat()`
4. If the LLM returns tool calls, the pipeline executes them, injects results, and calls the LLM again
5. Final LLM response is sent to `TTSInterface.synthesize()` for audio output
6. Pipeline emits audio chunks as `ResponseAudioDelta` events

### Echo

Testing mode — skips the LLM and speaks back the transcribed text:

```
audio → VAD → STT → TTS → audio
```

### TranscribeOnly

Speech-to-text only — emits `TranscriptionCompleted` events but produces no audio response:

```
audio → VAD → STT → text
```

## State Machine

Six states with automatic transitions:

| State | Description | Transitions to |
|---|---|---|
| **Idle** | Waiting for user speech | Listening (on VAD speech_started) |
| **Listening** | User is speaking, audio being buffered | Transcribing (on VAD speech_ended) |
| **Transcribing** | STT is processing the utterance | Thinking or Idle |
| **Thinking** | LLM is generating a response | Speaking or Idle |
| **Speaking** | TTS audio is being emitted / waiting for playback to finish | Idle (on resume_listening) or Listening (on interruption) |

## Turn Detection

`TurnDetector` wraps a `VADInterface` + `StreamingVAD` hysteresis:

1. Audio is chunked to the VAD's expected size (e.g., 512 samples for Silero)
2. Each chunk produces a speech probability [0, 1]
3. `StreamingVAD` applies hysteresis: 4 states (Silence → PendingSpeech → Speech → PendingSilence) with configurable onset/offset thresholds and minimum durations
4. On confirmed speech start: begin buffering audio, emit `UserSpeechStarted`
5. On confirmed speech end: emit `UserSpeechEnded` with the buffered audio

### Force-split

If an utterance exceeds `max_utterance_duration` (default 15s), `TurnDetector` force-ends the current segment and resets the VAD. This prevents unbounded memory growth and triggers intermediate transcriptions.

### Interruption handling

When the agent is speaking (`agent_speaking_ == true`) and the user starts talking:

1. Pipeline emits an `Interruption` event
2. TTS is cancelled, speech queue is cleared
3. Pipeline transitions to Listening

**Interruption recovery**: if the user stops speaking within `interruption_recovery_timeout` (default 0.4s), an `InterruptionRecovered` event is emitted instead of processing the utterance — allowing the platform to resume playback.

## Conversation Context

`ConversationContext` maintains message history for multi-turn LLM interactions:

- Messages have roles: System, User, Assistant, Tool
- Maximum message count is configurable (default 50)
- Oldest messages (after system prompt) are trimmed when the limit is reached
- Tool results are formatted as `[tool_name] output`

## Speech Queue

`SpeechQueue` manages TTS outputs with states: Pending → Playing → Done/Cancelled.

- `enqueue()` adds a new speech item
- `next()` marks the next pending item as playing
- `cancel_all()` cancels all items (used during interruption)
- `mark_done()` completes a speech item

## Events

The pipeline emits events via the `EventCallback`:

| Event | When | Payload |
|---|---|---|
| `SpeechStarted` | VAD confirms user speech | `start_time` |
| `SpeechEnded` | User utterance finalized, STT starting | `start_time` |
| `TranscriptionCompleted` | STT returns text | `text`, `start_time` |
| `ToolCallStarted` | LLM requested a tool call | `text` (tool name) |
| `ToolCallCompleted` | Tool execution finished | `text` (output) |
| `ResponseCreated` | TTS synthesis starting | — |
| `ResponseAudioDelta` | TTS audio chunk ready | `audio_data` (PCM16) |
| `ResponseInterrupted` | User barged in during TTS | `start_time` |
| `ResponseDone` | TTS synthesis complete | — |
| `Error` | STT/LLM/TTS failure | `text` (error message) |

## Thread Safety

- `push_audio()` is mutex-protected — safe to call from any thread
- STT/LLM/TTS run on a dedicated worker thread — `push_audio()` never blocks on inference
- Events are emitted on the calling thread (push_audio events) or the worker thread (STT/TTS events) — platform dispatches to main thread as needed
- `start()`/`stop()`/`resume_listening()` are mutex-protected
- `resume_listening()` is non-blocking — post-playback guard is applied as a sample counter in the turn detector
- State reads (`state()`, `is_running()`) are atomic — lock-free

## Configuration

`AgentConfig` controls pipeline behavior:

```cpp
AgentConfig config;
config.mode = AgentConfig::Mode::Pipeline;

// VAD thresholds
config.vad.onset = 0.5f;                    // speech probability threshold
config.vad.offset = 0.35f;                  // silence probability threshold
config.vad.min_speech_duration = 0.25f;     // seconds before confirming speech
config.vad.min_silence_duration = 0.1f;     // seconds before confirming silence
config.vad.pre_speech_buffer_duration = 0.6f; // seconds of pre-onset audio to capture

// Interruption
config.allow_interruptions = true;
config.min_interruption_duration = 1.0f;    // seconds of speech before confirming barge-in
config.interruption_recovery_timeout = 0.4f; // seconds — brief interruptions recover

// Timing
config.min_speech_gap = 0.1f;              // seconds between agent speech outputs
config.max_utterance_duration = 15.0f;     // seconds — force-split long utterances
config.max_response_duration = 10.0f;      // seconds — cap TTS output (prevents hallucination)
config.post_playback_guard = 0.3f;         // seconds — suppress VAD after playback (AEC settle)

// Latency optimizations
config.eager_stt = true;                   // start STT on first silence frame (saves ~0.6s)
config.warmup_stt = true;                  // dummy transcription at pipeline start (ANE cold start)

config.language = "en";                    // STT/TTS language hint (empty = auto-detect)
```

### Eager STT

When enabled (`eager_stt = true`, default), the turn detector emits `UserSpeechEnded` as soon as the first silence frame arrives (StreamingVAD `SpeechPaused` event), instead of waiting for `min_silence_duration` to confirm the end of speech. This saves ~0.6s of latency.

If the user resumes speaking before silence is confirmed (`SpeechResumed`), the eager result is discarded and a new utterance starts fresh.

### STT Warm-up

When enabled (`warmup_stt = true`, default), the worker thread runs a dummy 0.5s silent transcription at pipeline start. First inference on CoreML / Neural Engine is slow due to cold start — warm-up brings subsequent latency from ~3s to <1s.
