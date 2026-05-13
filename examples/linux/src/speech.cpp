#include "speech.h"

#include <speech_core/models/deepfilter.h>
#include <speech_core/models/kokoro_tts.h>
#include <speech_core/models/parakeet_stt.h>
#include <speech_core/models/silero_vad.h>
#include <speech_core/pipeline/agent_config.h>
#include <speech_core/pipeline/voice_pipeline.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

// ---------------------------------------------------------------------------
// Pipeline handle
//
// The model wrappers in speech_core::* directly implement the speech_core
// interfaces (VADInterface, STTInterface, TTSInterface, EnhancerInterface),
// so we instantiate VoicePipeline directly. No C-vtable adapters needed.
// ---------------------------------------------------------------------------

struct speech_pipeline_s {
    std::unique_ptr<speech_core::SileroVad> vad;
    std::unique_ptr<speech_core::ParakeetStt> stt;
    std::unique_ptr<speech_core::KokoroTts> tts;
    std::unique_ptr<speech_core::DeepFilterEnhancer> enhancer;
    std::unique_ptr<speech_core::VoicePipeline> pipeline;

    speech_event_fn user_callback = nullptr;
    void* user_context = nullptr;

    // C strings handed back to the user; must outlive the callback invocation.
    std::string event_text;
};

// ---------------------------------------------------------------------------
// Event bridge: speech_core::PipelineEvent → speech_event_t
// ---------------------------------------------------------------------------

static void dispatch_event(speech_pipeline_s* h,
                           const speech_core::PipelineEvent& event) {
    if (!h->user_callback) return;

    speech_event_t out = {};
    h->event_text = event.text;
    out.text = h->event_text.c_str();
    out.confidence = event.confidence;
    out.stt_duration_ms = event.stt_duration_ms;
    out.tts_duration_ms = event.tts_duration_ms;

    // Response-audio payloads: PipelineEvent::audio_data already contains
    // PCM16 bytes (see speech_core/protocol/events.h). Forward verbatim.
    if (event.type == speech_core::EventType::ResponseAudioDelta) {
        out.audio_data = event.audio_data.data();
        out.audio_data_length = event.audio_data.size();
    }

    using ET = speech_core::EventType;
    switch (event.type) {
        case ET::SessionCreated:         out.type = SPEECH_EVENT_READY; break;
        case ET::SpeechStarted:          out.type = SPEECH_EVENT_SPEECH_STARTED; break;
        case ET::SpeechEnded:            out.type = SPEECH_EVENT_SPEECH_ENDED; break;
        case ET::PartialTranscription:   out.type = SPEECH_EVENT_PARTIAL_TRANSCRIPTION; break;
        case ET::TranscriptionCompleted: out.type = SPEECH_EVENT_TRANSCRIPTION; break;
        case ET::ResponseAudioDelta:     out.type = SPEECH_EVENT_RESPONSE_AUDIO; break;
        case ET::ResponseDone:           out.type = SPEECH_EVENT_RESPONSE_DONE; break;
        case ET::Error:                  out.type = SPEECH_EVENT_ERROR; break;
        default: return;  // skip unmapped events
    }

    h->user_callback(&out, h->user_context);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" speech_config_t speech_config_default(void) {
    return {
        .model_dir = nullptr,
        .use_int8 = true,
        .use_qnn = false,
        .enable_enhancer = false,
        .transcribe_only = false,
        .min_silence_duration = 0.4f,
    };
}

extern "C" speech_pipeline_t speech_create(speech_config_t config,
                                           speech_event_fn on_event,
                                           void* event_context)
{
    if (!config.model_dir) return nullptr;

    auto h = std::make_unique<speech_pipeline_s>();
    h->user_callback = on_event;
    h->user_context = event_context;

    std::string dir(config.model_dir);
    std::string suffix = config.use_int8 ? "-int8" : "";
    bool hw_accel = config.use_qnn;

    try {
        h->vad = std::make_unique<speech_core::SileroVad>(
            dir + "/silero-vad.onnx");
        h->stt = std::make_unique<speech_core::ParakeetStt>(
            dir + "/parakeet-encoder" + suffix + ".onnx",
            dir + "/parakeet-decoder-joint" + suffix + ".onnx",
            dir + "/vocab.json",
            hw_accel);
        // Skip TTS in transcribe-only mode — saves model load time.
        if (!config.transcribe_only) {
            h->tts = std::make_unique<speech_core::KokoroTts>(
                dir + "/kokoro-e2e.onnx",
                dir + "/voices", dir, hw_accel);
        }

        speech_core::AgentConfig sc_cfg;
        sc_cfg.vad.min_silence_duration = config.min_silence_duration;
        sc_cfg.mode = config.transcribe_only
            ? speech_core::AgentConfig::Mode::TranscribeOnly
            : speech_core::AgentConfig::Mode::Echo;

        speech_pipeline_s* raw = h.get();
        // VoicePipeline takes TTSInterface& (not pointer). In transcribe-only
        // mode the pipeline never invokes synthesize(), but the reference
        // must still bind to something — a no-op stub suffices.
        struct NullTTS : speech_core::TTSInterface {
            void synthesize(const std::string&, const std::string&,
                            speech_core::TTSChunkCallback) override {}
            int output_sample_rate() const override { return 24000; }
        };
        static NullTTS null_tts;
        speech_core::TTSInterface* tts_ptr =
            h->tts ? static_cast<speech_core::TTSInterface*>(h->tts.get())
                   : static_cast<speech_core::TTSInterface*>(&null_tts);

        h->pipeline = std::make_unique<speech_core::VoicePipeline>(
            *h->stt, *tts_ptr, nullptr, *h->vad, sc_cfg,
            [raw](const speech_core::PipelineEvent& e) { dispatch_event(raw, e); });

        // Optional enhancer
        if (config.enable_enhancer) {
            std::string aux = dir + "/deepfilter-auxiliary.bin";
            std::string df = dir + "/deepfilter" + suffix + ".onnx";
            if (FILE* f = std::fopen(df.c_str(), "r")) {
                std::fclose(f);
                h->enhancer = std::make_unique<speech_core::DeepFilterEnhancer>(
                    df, aux, hw_accel);
                h->pipeline->set_enhancer(h->enhancer.get());
            }
        }

        return h.release();

    } catch (const std::exception& e) {
        std::fprintf(stderr, "[speech] pipeline creation failed: %s\n", e.what());
        return nullptr;
    }
}

extern "C" void speech_start(speech_pipeline_t pipeline) {
    if (pipeline && pipeline->pipeline) pipeline->pipeline->start();
}

extern "C" void speech_push_audio(speech_pipeline_t pipeline,
                                  const float* samples, size_t count) {
    if (pipeline && pipeline->pipeline) {
        pipeline->pipeline->push_audio(samples, count);
    }
}

extern "C" void speech_resume_listening(speech_pipeline_t pipeline) {
    if (pipeline && pipeline->pipeline) pipeline->pipeline->resume_listening();
}

extern "C" void speech_destroy(speech_pipeline_t pipeline) {
    delete pipeline;
}

extern "C" const char* speech_version(void) {
    return "0.0.2";
}
