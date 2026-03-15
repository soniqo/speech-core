# C API

`speech_core_c.h` provides a C wrapper for FFI integration with Swift, Kotlin, and other languages.

## Design

The C API uses a **vtable pattern** — consumers provide structs of function pointers that the C++ pipeline calls through. This avoids requiring consumers to link against C++ runtime or understand C++ ABIs.

```
Platform (Swift/Kotlin)           speech-core (C++)
┌─────────────────────┐          ┌──────────────────┐
│                     │          │                  │
│  sc_stt_vtable_t ──────────►  CSTTAdapter        │
│  sc_tts_vtable_t ──────────►  CTTSAdapter        │
│  sc_vad_vtable_t ──────────►  CVADAdapter        │
│  sc_llm_vtable_t ──────────►  CLLMAdapter        │
│                     │          │       │          │
│  sc_event_fn  ◄────────────── VoicePipeline      │
│                     │          │                  │
└─────────────────────┘          └──────────────────┘
```

## Usage

```c
// 1. Implement interface vtables
sc_stt_vtable_t stt = {
    .context = my_stt_model,
    .transcribe = my_transcribe_fn,
    .input_sample_rate = my_stt_sample_rate_fn
};

sc_tts_vtable_t tts = {
    .context = my_tts_model,
    .synthesize = my_synthesize_fn,
    .output_sample_rate = my_tts_sample_rate_fn,
    .cancel = my_tts_cancel_fn
};

sc_vad_vtable_t vad = {
    .context = my_vad_model,
    .process_chunk = my_vad_process_fn,
    .reset = my_vad_reset_fn,
    .input_sample_rate = my_vad_sample_rate_fn,
    .chunk_size = my_vad_chunk_size_fn
};

// 2. Create pipeline
sc_config_t config = sc_config_default();
config.mode = SC_MODE_PIPELINE;

sc_pipeline_t pipeline = sc_pipeline_create(
    stt, tts, NULL, vad, config,
    my_event_callback, my_context);

// 3. Start and feed audio
sc_pipeline_start(pipeline);
sc_pipeline_push_audio(pipeline, samples, count);

// 4. Clean up
sc_pipeline_stop(pipeline);
sc_pipeline_destroy(pipeline);
```

## Types

### Config

```c
sc_config_t sc_config_default(void);
```

Returns default config: Silero VAD thresholds, Echo mode, interruptions enabled, eager STT and STT warm-up on.

Key fields:
- `vad_onset`, `vad_offset` — speech/silence probability thresholds
- `min_speech_duration`, `min_silence_duration` — confirmation durations
- `allow_interruptions`, `min_interruption_duration` — barge-in control
- `interruption_recovery_timeout` — brief-interruption recovery window
- `max_utterance_duration` — force-split long utterances
- `max_response_duration` — cap TTS output (prevents hallucination)
- `post_playback_guard` — suppress VAD after playback (AEC settle)
- `eager_stt` — start STT before silence confirms (saves latency)
- `eager_stt_delay` — seconds in PendingSilence before eager fires (filters mid-sentence pauses)
- `warmup_stt` — dummy STT at pipeline start (Neural Engine cold start)
- `max_history_messages` — max conversation messages to retain (default 50, 0 = unlimited)
- `max_history_tokens` — max conversation tokens (default 0 = disabled, requires `count_tokens` on LLM vtable)
- `mask_tool_results` — drop tool messages before conversation messages during trimming (default true)
- `language` — STT/TTS language hint (empty string = auto-detect)
- `mode` — `SC_MODE_PIPELINE`, `SC_MODE_TRANSCRIBE_ONLY`, or `SC_MODE_ECHO`

### Pipeline lifecycle

```c
sc_pipeline_t sc_pipeline_create(...);
void sc_pipeline_destroy(sc_pipeline_t);
void sc_pipeline_start(sc_pipeline_t);
void sc_pipeline_stop(sc_pipeline_t);
```

### Audio input

```c
void sc_pipeline_push_audio(sc_pipeline_t, const float* samples, size_t count);
void sc_pipeline_push_text(sc_pipeline_t, const char* text);
```

### State

```c
sc_state_t sc_pipeline_state(sc_pipeline_t);  // SC_STATE_IDLE..SC_STATE_SPEAKING
bool sc_pipeline_is_running(sc_pipeline_t);
void sc_pipeline_resume_listening(sc_pipeline_t);  // signal playback done
```

States: `SC_STATE_IDLE`, `SC_STATE_LISTENING`, `SC_STATE_TRANSCRIBING`, `SC_STATE_THINKING`, `SC_STATE_SPEAKING`. The pipeline stays in `SC_STATE_SPEAKING` after TTS finishes until the platform calls `sc_pipeline_resume_listening()` after audio playback ends.

### Tools

Register tools before calling `sc_pipeline_start()`.

```c
// Callback-based tool (for platform consumers)
const char* my_time_handler(const char* name, const char* args, void* ctx) {
    return "3:14 PM";  // must remain valid until next call
}

const char* triggers[] = {"what time", "current time", NULL};
sc_tool_definition_t tool = {
    .name = "tell_time",
    .description = "Tell the current time",
    .triggers = triggers,
    .handler = my_time_handler,
    .handler_context = NULL,
    .timeout = 5,
    .cooldown = 30
};
sc_pipeline_add_tool(pipeline, tool);

// Shell-command tool (for CLI usage)
sc_tool_definition_t shell_tool = {
    .name = "greet",
    .description = "Say hello",
    .command = "echo Hello!",
    .timeout = 3,
    .cooldown = 0
};
sc_pipeline_add_tool(pipeline, shell_tool);

// Or load shell-command tools from JSON
int count = sc_pipeline_load_tools_json(pipeline, json_string);

// Remove all tools
sc_pipeline_clear_tools(pipeline);
```

The `handler` callback takes priority over `command` when both are set. The returned `const char*` must remain valid until the next call to the same handler.

### Events

```c
typedef void (*sc_event_fn)(const sc_event_t* event, void* context);
```

The event callback receives `sc_event_t` with:
- `type` — event type enum
- `text` — transcript, error message, or tool output (valid for the duration of the callback)
- `audio_data` / `audio_data_length` — PCM16 audio bytes (for `SC_EVENT_RESPONSE_AUDIO_DELTA`)
- `stt_duration_ms` — STT inference time in ms (on `TRANSCRIPTION_COMPLETED` and `RESPONSE_DONE`)
- `llm_duration_ms` — LLM generation time in ms (on `RESPONSE_CREATED` and `RESPONSE_DONE`)
- `tts_duration_ms` — TTS synthesis time in ms (on `RESPONSE_DONE`)

Event types: `SC_EVENT_SESSION_CREATED`, `SC_EVENT_SPEECH_STARTED`, `SC_EVENT_SPEECH_ENDED`, `SC_EVENT_TRANSCRIPTION_COMPLETED`, `SC_EVENT_RESPONSE_CREATED`, `SC_EVENT_RESPONSE_INTERRUPTED`, `SC_EVENT_RESPONSE_AUDIO_DELTA`, `SC_EVENT_RESPONSE_DONE`, `SC_EVENT_TOOL_CALL_STARTED`, `SC_EVENT_TOOL_CALL_COMPLETED`, `SC_EVENT_ERROR`.

All pointers in `sc_event_t` are valid only during the callback. Copy if needed.

## Null safety

All API functions handle `NULL` pipeline gracefully — they are no-ops. `sc_pipeline_state(NULL)` returns `SC_STATE_IDLE`, `sc_pipeline_is_running(NULL)` returns `false`. `sc_pipeline_clear_tools(NULL)` and `sc_pipeline_load_tools_json(NULL, ...)` are safe no-ops.
