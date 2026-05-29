// C ABI implementation for VoxCPM2 voice cloning — see speech_core/voxcpm2_c.h.
// Thin wrapper over LiteRTVoxCPM2Tts; lives in speech_core_models_litert.

#include "speech_core/voxcpm2_c.h"
#include "speech_core/models/litert_voxcpm2_tts.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>

using speech_core::LiteRTVoxCPM2Tts;

struct sc_voxcpm2_s {
    std::unique_ptr<LiteRTVoxCPM2Tts> tts;
    std::string last_error;
};

extern "C" {

sc_voxcpm2_t sc_voxcpm2_create(const char* bundle_dir) {
    if (!bundle_dir) return nullptr;
    try {
        const std::string b = bundle_dir;
        auto* h = new sc_voxcpm2_s();
        h->tts = std::make_unique<LiteRTVoxCPM2Tts>(
            b + "/voxcpm2-text-prefill.tflite",
            b + "/voxcpm2-token-step.tflite",
            b + "/voxcpm2-audio-encoder.tflite",
            b + "/voxcpm2-audio-decoder.tflite",
            b + "/tokenizer.json",
            /*hw_accel=*/false);
        return h;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[speech ERROR] sc_voxcpm2_create: %s\n", e.what());
        return nullptr;
    }
}

void sc_voxcpm2_destroy(sc_voxcpm2_t s) { delete s; }

void sc_voxcpm2_set_instruction(sc_voxcpm2_t s, const char* instruction) {
    if (s && s->tts) s->tts->set_instruction(instruction ? instruction : "");
}

void sc_voxcpm2_set_max_steps(sc_voxcpm2_t s, int max_steps) {
    if (s && s->tts) s->tts->set_max_steps(max_steps);
}

void sc_voxcpm2_set_min_steps_before_stop(sc_voxcpm2_t s, int min_steps) {
    if (s && s->tts) s->tts->set_min_steps_before_stop(min_steps);
}

void sc_voxcpm2_set_stop_on_stop_token(sc_voxcpm2_t s, bool stop_on_stop_token) {
    if (s && s->tts) s->tts->set_stop_on_stop_token(stop_on_stop_token);
}

void sc_voxcpm2_set_seed(sc_voxcpm2_t s, uint32_t seed) {
    if (s && s->tts) s->tts->set_seed(seed);
}

int sc_voxcpm2_max_text_tokens(sc_voxcpm2_t s) {
    return (s && s->tts) ? s->tts->max_text_tokens() : 0;
}

int sc_voxcpm2_set_reference(sc_voxcpm2_t s, const float* pcm,
                             size_t length, int sample_rate) {
    if (!s || !s->tts) return -1;
    try {
        s->tts->set_reference(pcm, length, sample_rate);
        s->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        s->last_error = e.what();
        return -1;
    }
}

void sc_voxcpm2_clear_reference(sc_voxcpm2_t s) {
    if (s && s->tts) s->tts->clear_reference();
}

int sc_voxcpm2_output_sample_rate(sc_voxcpm2_t s) {
    return (s && s->tts) ? s->tts->output_sample_rate() : 0;
}

int sc_voxcpm2_synthesize(sc_voxcpm2_t s, const char* text,
                          sc_voxcpm2_chunk_fn on_chunk, void* context) {
    if (!s || !s->tts) return -1;
    if (!text || !on_chunk) {
        if (s) s->last_error = "null text or callback";
        return -1;
    }
    try {
        s->tts->synthesize(text, "auto",
            [on_chunk, context](const float* samples, size_t length, bool is_final) {
                on_chunk(samples, length, is_final, context);
            });
        s->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        s->last_error = e.what();
        return -1;
    }
}

void sc_voxcpm2_cancel(sc_voxcpm2_t s) {
    if (s && s->tts) s->tts->cancel();
}

int sc_voxcpm2_tokens_generated(sc_voxcpm2_t s) {
    return (s && s->tts) ? s->tts->tokens_generated() : 0;
}

bool sc_voxcpm2_stopped_on_stop_token(sc_voxcpm2_t s) {
    return (s && s->tts) ? s->tts->stopped_on_stop_token() : false;
}

uint32_t sc_voxcpm2_seed_used(sc_voxcpm2_t s) {
    return (s && s->tts) ? s->tts->seed_used() : 0u;
}

const char* sc_voxcpm2_last_error(sc_voxcpm2_t s) {
    return s ? s->last_error.c_str() : "";
}

}  // extern "C"
