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

Returns default config: Silero VAD thresholds, Echo mode, interruptions enabled.

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
```

### Events

```c
typedef void (*sc_event_fn)(const sc_event_t* event, void* context);
```

The event callback receives `sc_event_t` with:
- `type` — `SC_EVENT_SPEECH_STARTED`, `SC_EVENT_TRANSCRIPTION_COMPLETED`, etc.
- `text` — transcript, error message, or tool output (valid for the duration of the callback)
- `audio_data` / `audio_data_length` — PCM16 audio bytes (for `SC_EVENT_RESPONSE_AUDIO_DELTA`)

All pointers in `sc_event_t` are valid only during the callback. Copy if needed.

## Null safety

All API functions handle `NULL` pipeline gracefully — they are no-ops. `sc_pipeline_state(NULL)` returns `SC_STATE_IDLE`, `sc_pipeline_is_running(NULL)` returns `false`.
