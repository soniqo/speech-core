// C ABI implementation for Chatterbox TTS — thin wrapper over LiteRTChatterboxTts.
// Lives in speech_core_models_litert.

#include "speech_core/chatterbox_c.h"
#include "speech_core/models/litert_chatterbox_tts.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>

using speech_core::LiteRTChatterboxTts;

struct sc_chatterbox_s {
    std::unique_ptr<LiteRTChatterboxTts> tts;
    std::string last_error;
};

sc_chatterbox_t sc_chatterbox_create(const char* bundle_dir) {
    if (!bundle_dir) return nullptr;
    try {
        auto* h = new sc_chatterbox_s();
        h->tts = std::make_unique<LiteRTChatterboxTts>(std::string(bundle_dir));
        return h;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[chatterbox] create failed: %s\n", e.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "[chatterbox] create failed: unknown error\n");
        return nullptr;
    }
}

void sc_chatterbox_destroy(sc_chatterbox_t h) { delete h; }

void sc_chatterbox_set_temperature(sc_chatterbox_t h, float v) { if (h) h->tts->set_temperature(v); }
void sc_chatterbox_set_top_p(sc_chatterbox_t h, float v) { if (h) h->tts->set_top_p(v); }
void sc_chatterbox_set_min_p(sc_chatterbox_t h, float v) { if (h) h->tts->set_min_p(v); }
void sc_chatterbox_set_repetition_penalty(sc_chatterbox_t h, float v) { if (h) h->tts->set_repetition_penalty(v); }
void sc_chatterbox_set_max_tokens(sc_chatterbox_t h, int v) { if (h) h->tts->set_max_tokens(v); }
void sc_chatterbox_set_seed(sc_chatterbox_t h, uint32_t v) { if (h) h->tts->set_seed(v); }

int sc_chatterbox_output_sample_rate(sc_chatterbox_t h) { return h ? h->tts->output_sample_rate() : 0; }

int sc_chatterbox_synthesize(sc_chatterbox_t h, const char* text, const char* language,
                             sc_chatterbox_chunk_fn on_chunk, void* context) {
    if (!h) return -1;
    h->last_error.clear();
    try {
        h->tts->synthesize(text ? text : "", language ? language : "",
                           [&](const float* s, size_t n, bool fin) { if (on_chunk) on_chunk(s, n, fin, context); });
        return 0;
    } catch (const std::exception& e) {
        h->last_error = e.what(); return -1;
    } catch (...) {
        h->last_error = "unknown error"; return -1;
    }
}

void sc_chatterbox_cancel(sc_chatterbox_t h) { if (h) h->tts->cancel(); }
int sc_chatterbox_tokens_generated(sc_chatterbox_t h) { return h ? h->tts->tokens_generated() : 0; }
uint32_t sc_chatterbox_seed_used(sc_chatterbox_t h) { return h ? h->tts->seed_used() : 0; }
const char* sc_chatterbox_last_error(sc_chatterbox_t h) { return h ? h->last_error.c_str() : ""; }
