#include "speech_core/speech_core_c.h"
#include "speech_core/pipeline/voice_pipeline.h"

#include <memory>
#include <string>
#include <vector>

using namespace speech_core;

// ---------------------------------------------------------------------------
// Adapter classes — bridge C vtable function pointers to C++ interfaces
// ---------------------------------------------------------------------------

class CSTTAdapter : public STTInterface {
    sc_stt_vtable_t vt_;
public:
    explicit CSTTAdapter(sc_stt_vtable_t vt) : vt_(vt) {}

    TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) override
    {
        auto r = vt_.transcribe(vt_.context, audio, length, sample_rate);
        return {
            r.text ? std::string(r.text) : "",
            r.confidence,
            r.start_time,
            r.end_time
        };
    }

    int input_sample_rate() const override {
        return vt_.input_sample_rate(vt_.context);
    }
};

class CTTSAdapter : public TTSInterface {
    sc_tts_vtable_t vt_;
public:
    explicit CTTSAdapter(sc_tts_vtable_t vt) : vt_(vt) {}

    void synthesize(const std::string& text, const std::string& language,
                    TTSChunkCallback on_chunk) override
    {
        // Bridge C++ std::function to C function pointer + context
        vt_.synthesize(vt_.context, text.c_str(), language.c_str(),
            [](const float* samples, size_t length, bool is_final, void* ctx) {
                auto* fn = static_cast<TTSChunkCallback*>(ctx);
                (*fn)(samples, length, is_final);
            },
            &on_chunk);
    }

    int output_sample_rate() const override {
        return vt_.output_sample_rate(vt_.context);
    }

    void cancel() override {
        if (vt_.cancel) vt_.cancel(vt_.context);
    }
};

class CVADAdapter : public VADInterface {
    sc_vad_vtable_t vt_;
public:
    explicit CVADAdapter(sc_vad_vtable_t vt) : vt_(vt) {}

    float process_chunk(const float* samples, size_t length) override {
        return vt_.process_chunk(vt_.context, samples, length);
    }

    void reset() override {
        vt_.reset(vt_.context);
    }

    int input_sample_rate() const override {
        return vt_.input_sample_rate(vt_.context);
    }

    size_t chunk_size() const override {
        return vt_.chunk_size(vt_.context);
    }
};

class CLLMAdapter : public LLMInterface {
    sc_llm_vtable_t vt_;
public:
    explicit CLLMAdapter(sc_llm_vtable_t vt) : vt_(vt) {}

    LLMResponse chat(const std::vector<Message>& messages,
                     LLMTokenCallback on_token) override
    {
        // Convert C++ messages to C array
        std::vector<sc_message_t> c_msgs(messages.size());
        for (size_t i = 0; i < messages.size(); i++) {
            c_msgs[i].role = static_cast<sc_role_t>(messages[i].role);
            c_msgs[i].content = messages[i].content.c_str();
        }

        // Bridge C++ std::function to C function pointer + context
        vt_.chat(vt_.context, c_msgs.data(), c_msgs.size(),
            [](const char* token, bool is_final, void* ctx) {
                auto* fn = static_cast<LLMTokenCallback*>(ctx);
                (*fn)(std::string(token), is_final);
            },
            &on_token);

        return {};  // C API doesn't support tool calls yet
    }

    void cancel() override {
        if (vt_.cancel) vt_.cancel(vt_.context);
    }
};

// ---------------------------------------------------------------------------
// Pipeline handle
// ---------------------------------------------------------------------------

struct sc_pipeline_s {
    std::unique_ptr<CSTTAdapter> stt;
    std::unique_ptr<CTTSAdapter> tts;
    std::unique_ptr<CLLMAdapter> llm;
    std::unique_ptr<CVADAdapter> vad;
    std::unique_ptr<VoicePipeline> pipeline;
    sc_event_fn event_fn;
    void* event_context;
};

// ---------------------------------------------------------------------------
// Event type mapping
// ---------------------------------------------------------------------------

static sc_event_type_t map_event_type(EventType type) {
    switch (type) {
        case EventType::SessionCreated:           return SC_EVENT_SESSION_CREATED;
        case EventType::SpeechStarted:            return SC_EVENT_SPEECH_STARTED;
        case EventType::SpeechEnded:              return SC_EVENT_SPEECH_ENDED;
        case EventType::TranscriptionCompleted:   return SC_EVENT_TRANSCRIPTION_COMPLETED;
        case EventType::ResponseCreated:          return SC_EVENT_RESPONSE_CREATED;
        case EventType::ResponseAudioDelta:       return SC_EVENT_RESPONSE_AUDIO_DELTA;
        case EventType::ResponseDone:             return SC_EVENT_RESPONSE_DONE;
        case EventType::Error:                    return SC_EVENT_ERROR;
        default:                                  return SC_EVENT_ERROR;
    }
}

// ---------------------------------------------------------------------------
// C API implementation
// ---------------------------------------------------------------------------

extern "C" {

sc_config_t sc_config_default(void) {
    sc_config_t c = {};
    c.vad_onset = 0.5f;
    c.vad_offset = 0.35f;
    c.min_speech_duration = 0.25f;
    c.min_silence_duration = 0.1f;
    c.allow_interruptions = true;
    c.interruption_recovery_timeout = 0.4f;
    c.max_utterance_duration = 15.0f;
    c.language = "";
    c.mode = SC_MODE_ECHO;
    return c;
}

sc_pipeline_t sc_pipeline_create(
    sc_stt_vtable_t stt,
    sc_tts_vtable_t tts,
    sc_llm_vtable_t* llm,
    sc_vad_vtable_t vad,
    sc_config_t config,
    sc_event_fn on_event,
    void* event_context)
{
    auto p = new sc_pipeline_s();
    p->event_fn = on_event;
    p->event_context = event_context;

    // Create adapters
    p->stt = std::make_unique<CSTTAdapter>(stt);
    p->tts = std::make_unique<CTTSAdapter>(tts);
    p->vad = std::make_unique<CVADAdapter>(vad);
    if (llm) {
        p->llm = std::make_unique<CLLMAdapter>(*llm);
    }

    // Convert config
    AgentConfig agent_config;
    agent_config.vad.onset = config.vad_onset;
    agent_config.vad.offset = config.vad_offset;
    agent_config.vad.min_speech_duration = config.min_speech_duration;
    agent_config.vad.min_silence_duration = config.min_silence_duration;
    agent_config.allow_interruptions = config.allow_interruptions;
    agent_config.interruption_recovery_timeout = config.interruption_recovery_timeout;
    agent_config.max_utterance_duration = config.max_utterance_duration;
    agent_config.language = config.language ? config.language : "";
    agent_config.mode = static_cast<AgentConfig::Mode>(config.mode);

    // Create pipeline
    p->pipeline = std::make_unique<VoicePipeline>(
        *p->stt, *p->tts, p->llm.get(), *p->vad,
        agent_config,
        [p](const PipelineEvent& event) {
            sc_event_t e = {};
            e.type = map_event_type(event.type);
            e.text = event.text.c_str();
            e.audio_data = event.audio_data.data();
            e.audio_data_length = event.audio_data.size();
            e.start_time = event.start_time;
            e.end_time = event.end_time;
            e.confidence = event.confidence;
            p->event_fn(&e, p->event_context);
        });

    return p;
}

void sc_pipeline_destroy(sc_pipeline_t pipeline) {
    delete pipeline;
}

void sc_pipeline_start(sc_pipeline_t pipeline) {
    if (pipeline) pipeline->pipeline->start();
}

void sc_pipeline_stop(sc_pipeline_t pipeline) {
    if (pipeline) pipeline->pipeline->stop();
}

void sc_pipeline_push_audio(sc_pipeline_t pipeline,
                            const float* samples, size_t count)
{
    if (pipeline) pipeline->pipeline->push_audio(samples, count);
}

void sc_pipeline_push_text(sc_pipeline_t pipeline, const char* text) {
    if (pipeline && text) pipeline->pipeline->push_text(std::string(text));
}

sc_state_t sc_pipeline_state(sc_pipeline_t pipeline) {
    if (!pipeline) return SC_STATE_IDLE;
    return static_cast<sc_state_t>(pipeline->pipeline->state());
}

bool sc_pipeline_is_running(sc_pipeline_t pipeline) {
    if (!pipeline) return false;
    return pipeline->pipeline->is_running();
}

}  // extern "C"
