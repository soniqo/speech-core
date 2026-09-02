# Voice Pipeline

The `VoicePipeline` is the central orchestrator. It connects STT, TTS, LLM, and VAD implementations through a state machine with turn detection, conversation tracking, and speech queuing.

## Modes

### VoicePipeline (default)

Full voice agent loop:

```
audio → [AEC] → [enhance] → VAD → STT → [tools?] → LLM → TTS → audio
                                                              │
                                                              └──► AEC reference
```

1. Optional echo cancellation removes TTS playback from mic signal
2. Optional speech enhancement (denoising) runs on the clean signal
3. VAD detects user speech via `TurnDetector`
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

## Passive meeting tracks

`MeetingTranscriptionTrack` is separate from the conversational
`VoicePipeline`. It implements the fixed source-local path used by passive
meeting recorders:

```text
timestamped PCM -> Silero VAD -> Nemotron revisable preview
                              -> MOSS authoritative paragraph text/activity
```

Construct one instance per capture source. PCM, VAD state, Nemotron stream,
paragraph state, and timestamps never cross instances. A MOSS runtime may be
shared when its implementation serializes inference. The default meeting
configuration closes a paragraph after 550 ms of silence, retains 200 ms of
pre/post-roll, first publishes continuous speech at ten seconds, and then
revises a rolling window capped at twenty seconds.

When MOSS returns paragraph text with no speaker activity, the track decodes
again with extra source-local audio. Whether that retry still says what the
paragraph said is application policy, so the track asks the caller:
`Config::activity_recovery_compatible` judges a retry over the same audio, and
`Config::following_recovery_compatible` a retry whose speaker-marked text runs
on past the paragraph. Both receive flattened text; the structural
preconditions stay in the engine. Leave either unset and that retry is
rejected — the original paragraph publishes unchanged. There is no default
rule.

`MeetingTrackEvent::Preview` is non-durable text. A
`MeetingTrackEvent::Revision` supplies an exact source-local replacement
interval and authoritative blocks. MOSS activity labels remain scoped to that
one result. `RecordingSpeakerIdentity` applies independent embedding evidence,
within-result different-speaker constraints, conservative short-fragment
galleries, and exact-range backfills. Ambiguous evidence remains unlabelled.

For microphone capture, feed raw microphone and the independently captured
playback reference to `TimestampedEchoCancellationStream`. Only its cleaned
output should enter the microphone meeting track. Missing timestamps, buffer
overrun, or AEC inference errors surface as failures; the class never emits
raw microphone PCM as a fallback.

## State Machine

Five states with automatic transitions:

| State | Description | Transitions to |
|---|---|---|
| **Idle** | Waiting for user speech | Listening (on VAD speech_started) |
| **Listening** | User is speaking, audio being buffered | Transcribing (on VAD speech_ended) |
| **Transcribing** | STT is processing the utterance | Thinking, Speaking (echo mode), or Idle (empty STT) |
| **Thinking** | LLM is generating a response | Speaking or Idle |
| **Speaking** | TTS audio is being emitted / waiting for playback to finish | Idle (on resume_listening) or Listening (on interruption) |

## Input Session Boundaries

`stop()` and the next `start()` form a hard input-session boundary. Buffered
turn-detector audio, the pre-speech ring, queued utterances, streaming STT
state, and late results from the previous session are discarded.

For long-lived pipelines, call `cancel_current_turn()` instead. It provides the
same turn isolation without tearing down worker threads or reloading models:

```cpp
pipeline.cancel_current_turn();
// The pipeline is still running and ready for audio from a new mic session.
```

Cancellation does not clear conversation context. It resets VAD and AEC input
state, cancels STT/LLM/TTS on a best-effort basis, and uses a turn generation
to prevent backends that ignore cancellation from emitting stale results.

## Turn Detection

`TurnDetector` wraps a `VADInterface` + `StreamingVAD` hysteresis:

1. Audio is chunked to the VAD's expected size (e.g., 512 samples for Silero)
2. Each chunk produces a speech probability [0, 1]
3. `StreamingVAD` applies hysteresis: 4 states (Silence → PendingSpeech → Speech → PendingSilence) with configurable onset/offset thresholds and minimum durations
4. On confirmed speech start: begin buffering audio, emit `UserSpeechStarted`
5. On confirmed speech end: emit `UserSpeechEnded` with the buffered audio

### End-of-turn classifier (Smart Turn)

Silence alone is a weak signal for the end of a turn: people pause mid-sentence to think, and a VAD-only detector cuts them off. `TurnDetector::set_turn_completion()` — exposed as `VoicePipeline::set_turn_completion()`, call before `start()`, `nullptr` disables — attaches an optional `TurnCompletionInterface` such as `OnnxSmartTurn` (Pipecat Smart Turn v3.2). With a classifier attached, a VAD pause becomes a question rather than a verdict:

1. On a pause — silence confirmed, or the eager-STT moment — the detector calls `turn_complete_probability()` on the audio of the turn so far (Smart Turn looks at the last 8 s). The call is synchronous on the audio thread, once per pause, roughly 20–50 ms on a laptop CPU.
2. If the probability reaches `turn_completion_threshold` (default 0.5), the turn ends as usual: `UserSpeechEnded` carries the audio, and `TurnEvent::turn_completion_probability` carries the score.
3. Otherwise the detector **holds** the turn (`turn_held()` is true): audio keeps accumulating, and if the user speaks again it continues the *same* turn — no second `UserSpeechStarted`. The next pause asks the classifier again on the whole turn.
4. If silence continues for `turn_completion_max_silence` (default 2.0 s, measured from the start of the pause; 0 = never), the turn ends anyway, so a user who trails off still gets a reply.

Eager STT respects the veto: a mid-sentence pause no longer produces an early utterance, while a pause the classifier agrees is final still fires the eager utterance at `eager_stt_delay`. `flush()` settles a held turn at end of stream, and force-split (`max_utterance_duration`) bypasses the classifier. The classifier is never consulted while the agent is speaking, so the short-speech (AEC echo) discard path is unchanged. `turn_completion_probability` is -1 whenever no classifier ran — no model attached, force-split, flush, or the silence cap expired.

### Force-split

If an utterance exceeds `max_utterance_duration` (default 15s), `TurnDetector` force-ends the current segment and resets the VAD. This prevents unbounded memory growth and triggers intermediate transcriptions.

### Interruption handling

When the agent is speaking (`agent_speaking_ == true`) and the user starts talking:

1. A deferred interruption timer starts (requires `min_interruption_duration` of continuous speech to confirm — filters AEC residual echo)
2. Once confirmed, pipeline emits an `Interruption` event
3. TTS is cancelled, speech queue is cleared
4. Pipeline transitions to Listening

**Retroactive interruption**: if the user is already speaking when `set_agent_speaking(true)` is called (e.g., user spoke during STT processing after an eager utterance), the deferred interruption timer starts immediately.

**Interruption recovery**: if the user stops speaking within `interruption_recovery_timeout` (default 0.4s), an `InterruptionRecovered` event is emitted instead of processing the utterance — allowing the platform to resume playback.

### Empty / low-confidence STT recovery

When STT returns empty text or confidence below `min_transcription_confidence`, the pipeline resets to Idle and clears `agent_speaking` + turn detector state. Without this, queued speech during TTS playback could produce an empty STT result that leaves the pipeline stuck — `agent_speaking` stays true and the turn detector has stale speech state, preventing new speech detection.

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
| `PartialTranscription` | Streaming STT partial result during speech | `text` |
| `TranscriptionCompleted` | STT returns text | `text`, `start_time`, `stt_duration_ms` |
| `ToolCallStarted` | LLM requested a tool call | `text` (tool name) |
| `ToolCallCompleted` | Tool execution finished | `text` (output) |
| `ResponseCreated` | TTS synthesis starting | `llm_duration_ms` |
| `ResponseAudioDelta` | TTS audio chunk ready | `audio_data` (PCM16) |
| `ResponseInterrupted` | User barged in during TTS | `start_time` |
| `ResponseDone` | TTS synthesis complete | `stt_duration_ms`, `llm_duration_ms`, `tts_duration_ms` |
| `Error` | STT/LLM/TTS failure | `text` (error message) |

## Thread Safety

- `push_audio()` is mutex-protected — safe to call from any thread
- STT/LLM/TTS run on a dedicated worker thread — `push_audio()` never blocks on inference
- Events are emitted on the calling thread (push_audio events) or the worker thread (STT/TTS events) — platform dispatches to main thread as needed
- Lifecycle calls synchronize with audio and worker state; callers must not
  invoke `start()` and `stop()` concurrently with each other
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
config.max_utterance_duration = 15.0f;     // seconds — force-split long utterances
config.max_response_duration = 10.0f;      // seconds — cap TTS output (prevents hallucination)
config.post_playback_guard = 0.3f;         // seconds — suppress VAD after playback (AEC settle)

// Latency optimizations
config.eager_stt = true;                   // start STT before silence confirms (saves ~0.3s)
config.eager_stt_delay = 0.3f;             // seconds in silence before eager fires (filters pauses)
config.turn_completion_threshold = 0.5f;   // end-of-turn classifier: probability that ends the turn
config.turn_completion_max_silence = 2.0f; // seconds of silence that end a vetoed turn anyway (0 = never)
config.warmup_stt = true;                  // dummy transcription at pipeline start (ANE cold start)

// Conversation history
config.max_history_messages = 50;          // max messages retained (0 = unlimited)
config.max_history_tokens = 0;             // max tokens (0 = disabled, needs token counter)
config.mask_tool_results = true;           // drop tool messages before conversation during trimming

// Streaming STT (partial transcriptions)
config.emit_partial_transcriptions = false; // opt-in, requires streaming STT model
config.partial_transcription_interval = 1.0f; // seconds between chunk pushes

config.language = "en";                    // STT/TTS language hint (empty = auto-detect)
```

The two `turn_completion_*` fields only take effect once a classifier is attached with `VoicePipeline::set_turn_completion()` before `start()` (see [End-of-turn classifier](#end-of-turn-classifier-smart-turn)); without one, turns end on VAD silence as before.

### Eager STT

When enabled (`eager_stt = true`, default), the turn detector emits `UserSpeechEnded` early — before `min_silence_duration` confirms the end of speech — saving latency equal to `min_silence_duration - eager_stt_delay`.

The `eager_stt_delay` parameter (default 0.3s) controls how long to wait in `PendingSilence` before firing the eager utterance. This filters natural mid-sentence pauses (typically 0.1–0.3s in conversational speech) while still being faster than full silence confirmation. Set to 0 to fire on the first silence frame.

If the user resumes speaking before `min_silence_duration` (i.e., the VAD fires `SpeechResumed`), the eager result is discarded and the turn is treated as one continuous utterance. If the full silence elapses, the eager utterance is committed and any subsequent speech starts a new turn.

The pipeline marks eager utterances with an `eager` flag so that new speech during STT processing is not mistaken for an interruption — it's treated as a separate utterance.

### STT Warm-up

When enabled (`warmup_stt = true`, default), the worker thread runs a dummy 0.5s silent transcription at pipeline start. First inference on CoreML / Neural Engine is slow due to cold start — warm-up brings subsequent latency from ~3s to <1s.
