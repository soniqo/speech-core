#include "speech_core/speech_core_c.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Mock implementations for vtable testing
// ---------------------------------------------------------------------------

static sc_transcription_result_t mock_transcribe(
    void* /*ctx*/, const float* /*audio*/, size_t /*length*/, int /*sample_rate*/)
{
    sc_transcription_result_t r = {};
    r.text = "hello world";
    r.confidence = 0.95f;
    return r;
}

static int mock_stt_sample_rate(void* /*ctx*/) { return 16000; }

static void mock_synthesize(
    void* /*ctx*/, const char* /*text*/, const char* /*language*/,
    sc_tts_chunk_fn on_chunk, void* chunk_ctx)
{
    float samples[] = {0.1f, 0.2f, 0.3f};
    on_chunk(samples, 3, true, chunk_ctx);
}

static int mock_tts_sample_rate(void* /*ctx*/) { return 24000; }
static void mock_tts_cancel(void* /*ctx*/) {}

static float mock_vad_process(void* /*ctx*/, const float* /*samples*/, size_t /*length*/) {
    return 0.1f;  // no speech
}

static void mock_vad_reset(void* /*ctx*/) {}
static int mock_vad_sample_rate(void* /*ctx*/) { return 16000; }
static size_t mock_vad_chunk_size(void* /*ctx*/) { return 512; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_config_default() {
    sc_config_t c = sc_config_default();
    assert(c.vad_onset > 0.0f);
    assert(c.vad_offset > 0.0f);
    assert(c.allow_interruptions == true);
    assert(c.mode == SC_MODE_ECHO);
    printf("  PASS: config_default\n");
}

void test_create_destroy() {
    sc_stt_vtable_t stt = {};
    stt.transcribe = mock_transcribe;
    stt.input_sample_rate = mock_stt_sample_rate;

    sc_tts_vtable_t tts = {};
    tts.synthesize = mock_synthesize;
    tts.output_sample_rate = mock_tts_sample_rate;
    tts.cancel = mock_tts_cancel;

    sc_vad_vtable_t vad = {};
    vad.process_chunk = mock_vad_process;
    vad.reset = mock_vad_reset;
    vad.input_sample_rate = mock_vad_sample_rate;
    vad.chunk_size = mock_vad_chunk_size;

    sc_config_t config = sc_config_default();

    std::vector<sc_event_type_t> received_events;

    sc_pipeline_t p = sc_pipeline_create(
        stt, tts, nullptr, vad, config,
        [](const sc_event_t* event, void* ctx) {
            auto* events = static_cast<std::vector<sc_event_type_t>*>(ctx);
            events->push_back(event->type);
        },
        &received_events);

    assert(p != nullptr);
    assert(sc_pipeline_state(p) == SC_STATE_IDLE);
    assert(sc_pipeline_is_running(p) == false);

    sc_pipeline_destroy(p);
    printf("  PASS: create_destroy\n");
}

void test_start_stop() {
    sc_stt_vtable_t stt = {};
    stt.transcribe = mock_transcribe;
    stt.input_sample_rate = mock_stt_sample_rate;

    sc_tts_vtable_t tts = {};
    tts.synthesize = mock_synthesize;
    tts.output_sample_rate = mock_tts_sample_rate;
    tts.cancel = mock_tts_cancel;

    sc_vad_vtable_t vad = {};
    vad.process_chunk = mock_vad_process;
    vad.reset = mock_vad_reset;
    vad.input_sample_rate = mock_vad_sample_rate;
    vad.chunk_size = mock_vad_chunk_size;

    sc_config_t config = sc_config_default();

    sc_pipeline_t p = sc_pipeline_create(
        stt, tts, nullptr, vad, config,
        [](const sc_event_t*, void*) {}, nullptr);

    sc_pipeline_start(p);
    assert(sc_pipeline_is_running(p) == true);
    assert(sc_pipeline_state(p) == SC_STATE_IDLE);

    sc_pipeline_stop(p);
    assert(sc_pipeline_is_running(p) == false);

    sc_pipeline_destroy(p);
    printf("  PASS: start_stop\n");
}

void test_push_text_echo() {
    sc_stt_vtable_t stt = {};
    stt.transcribe = mock_transcribe;
    stt.input_sample_rate = mock_stt_sample_rate;

    sc_tts_vtable_t tts = {};
    tts.synthesize = mock_synthesize;
    tts.output_sample_rate = mock_tts_sample_rate;
    tts.cancel = mock_tts_cancel;

    sc_vad_vtable_t vad = {};
    vad.process_chunk = mock_vad_process;
    vad.reset = mock_vad_reset;
    vad.input_sample_rate = mock_vad_sample_rate;
    vad.chunk_size = mock_vad_chunk_size;

    sc_config_t config = sc_config_default();
    config.mode = SC_MODE_ECHO;

    std::vector<sc_event_type_t> events;

    sc_pipeline_t p = sc_pipeline_create(
        stt, tts, nullptr, vad, config,
        [](const sc_event_t* event, void* ctx) {
            auto* v = static_cast<std::vector<sc_event_type_t>*>(ctx);
            v->push_back(event->type);
        },
        &events);

    sc_pipeline_start(p);
    sc_pipeline_push_text(p, "test echo");

    // Echo mode: push_text -> process_utterance -> speak -> events
    assert(!events.empty());

    // Should have ResponseCreated + ResponseAudioDelta + ResponseDone
    bool has_response_created = false;
    bool has_audio_delta = false;
    bool has_response_done = false;
    for (auto e : events) {
        if (e == SC_EVENT_RESPONSE_CREATED) has_response_created = true;
        if (e == SC_EVENT_RESPONSE_AUDIO_DELTA) has_audio_delta = true;
        if (e == SC_EVENT_RESPONSE_DONE) has_response_done = true;
    }
    assert(has_response_created);
    assert(has_audio_delta);
    assert(has_response_done);

    sc_pipeline_destroy(p);
    printf("  PASS: push_text_echo\n");
}

void test_add_tool_callback() {
    sc_stt_vtable_t stt = {};
    stt.transcribe = mock_transcribe;
    stt.input_sample_rate = mock_stt_sample_rate;

    sc_tts_vtable_t tts = {};
    tts.synthesize = mock_synthesize;
    tts.output_sample_rate = mock_tts_sample_rate;
    tts.cancel = mock_tts_cancel;

    sc_vad_vtable_t vad = {};
    vad.process_chunk = mock_vad_process;
    vad.reset = mock_vad_reset;
    vad.input_sample_rate = mock_vad_sample_rate;
    vad.chunk_size = mock_vad_chunk_size;

    sc_config_t config = sc_config_default();

    sc_pipeline_t p = sc_pipeline_create(
        stt, tts, nullptr, vad, config,
        [](const sc_event_t*, void*) {}, nullptr);

    // Register a tool with callback handler
    const char* triggers[] = {"what time", "current time", nullptr};
    sc_tool_definition_t tool = {};
    tool.name = "tell_time";
    tool.description = "Tell the current time";
    tool.triggers = triggers;
    tool.handler = [](const char* name, const char*, void*) -> const char* {
        if (strcmp(name, "tell_time") == 0) return "3:14 PM";
        return "unknown";
    };
    tool.handler_context = nullptr;
    tool.timeout = 5;
    tool.cooldown = 0;

    sc_pipeline_add_tool(p, tool);

    // Register another tool with shell command
    sc_tool_definition_t tool2 = {};
    tool2.name = "greet";
    tool2.description = "Say hello";
    const char* triggers2[] = {"hello", "hi", nullptr};
    tool2.triggers = triggers2;
    tool2.command = "echo Hello!";
    tool2.timeout = 3;
    tool2.cooldown = 0;

    sc_pipeline_add_tool(p, tool2);

    // Verify tools are registered (we can't directly query, but clear shouldn't crash)
    sc_pipeline_clear_tools(p);

    sc_pipeline_destroy(p);
    printf("  PASS: add_tool_callback\n");
}

void test_load_tools_json_c_api() {
    sc_stt_vtable_t stt = {};
    stt.transcribe = mock_transcribe;
    stt.input_sample_rate = mock_stt_sample_rate;

    sc_tts_vtable_t tts = {};
    tts.synthesize = mock_synthesize;
    tts.output_sample_rate = mock_tts_sample_rate;
    tts.cancel = mock_tts_cancel;

    sc_vad_vtable_t vad = {};
    vad.process_chunk = mock_vad_process;
    vad.reset = mock_vad_reset;
    vad.input_sample_rate = mock_vad_sample_rate;
    vad.chunk_size = mock_vad_chunk_size;

    sc_config_t config = sc_config_default();

    sc_pipeline_t p = sc_pipeline_create(
        stt, tts, nullptr, vad, config,
        [](const sc_event_t*, void*) {}, nullptr);

    const char* json = R"([
        {
            "name": "tell_time",
            "description": "Tell the current time",
            "triggers": ["what time"],
            "command": "date '+%I:%M %p'",
            "timeout": 5,
            "cooldown": 30
        }
    ])";

    int count = sc_pipeline_load_tools_json(p, json);
    assert(count == 1);

    // Invalid JSON
    assert(sc_pipeline_load_tools_json(p, "not json") == -1);

    // NULL safety
    assert(sc_pipeline_load_tools_json(nullptr, json) == -1);
    assert(sc_pipeline_load_tools_json(p, nullptr) == -1);

    sc_pipeline_clear_tools(p);
    sc_pipeline_destroy(p);
    printf("  PASS: load_tools_json_c_api\n");
}

void test_null_safety() {
    // All functions should handle NULL pipeline gracefully
    sc_pipeline_start(nullptr);
    sc_pipeline_stop(nullptr);
    sc_pipeline_push_audio(nullptr, nullptr, 0);
    sc_pipeline_push_text(nullptr, nullptr);
    sc_pipeline_clear_tools(nullptr);
    assert(sc_pipeline_state(nullptr) == SC_STATE_IDLE);
    assert(sc_pipeline_is_running(nullptr) == false);
    sc_pipeline_destroy(nullptr);
    printf("  PASS: null_safety\n");
}

int main() {
    printf("test_c_api:\n");
    test_config_default();
    test_create_destroy();
    test_start_stop();
    test_push_text_echo();
    test_add_tool_callback();
    test_load_tools_json_c_api();
    test_null_safety();
    printf("All C API tests passed.\n");
    return 0;
}
