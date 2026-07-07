// C ABI implementation for Indic-Mio TTS — see speech_core/indic_mio_c.h.
// Thin wrapper over LiteRTIndicMioTts; lives in speech_core_models_litert.

#include "speech_core/indic_mio_c.h"
#include "speech_core/models/litert_indic_mio_tts.h"

#include "hf_download.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using speech_core::LiteRTIndicMioTts;

struct sc_indic_mio_s {
    std::unique_ptr<LiteRTIndicMioTts> tts;
    std::string last_error;
};

namespace {

sc_indic_mio_s* create_from_dir(const std::string& b) {
    auto* h = new sc_indic_mio_s();
    h->tts = std::make_unique<LiteRTIndicMioTts>(
        b + "/indicmio-text-prefill.tflite",
        b + "/indicmio-token-step.tflite",
        b + "/indicmio-audio-decoder.tflite",
        b + "/indicmio-ref-encoder.tflite",
        b + "/tokenizer.json",
        /*hw_accel=*/false);
    return h;
}

// Same cache-root policy as the VoxCPM2 wrapper.
std::string default_cache_dir() {
    if (const char* c = std::getenv("SPEECH_CORE_CACHE_DIR"); c && *c) return c;
#if defined(_WIN32)
    if (const char* la = std::getenv("LOCALAPPDATA"); la && *la)
        return std::string(la) + "/speech-core";
#elif defined(__APPLE__)
    if (const char* h = std::getenv("HOME"); h && *h)
        return std::string(h) + "/Library/Caches/speech-core";
#else
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x)
        return std::string(x) + "/speech-core";
    if (const char* h = std::getenv("HOME"); h && *h)
        return std::string(h) + "/.cache/speech-core";
#endif
    return "./speech-core-cache";
}

std::string sanitize_repo(const std::string& id) {
    std::string s;
    for (char c : id) {
        if (c == '/' || c == '\\') s += "__";
        else s += c;
    }
    return s;
}

}  // namespace

extern "C" {

sc_indic_mio_t sc_indic_mio_create(const char* bundle_dir) {
    if (!bundle_dir) return nullptr;
    try {
        return create_from_dir(bundle_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[speech ERROR] sc_indic_mio_create: %s\n", e.what());
        return nullptr;
    }
}

sc_indic_mio_t sc_indic_mio_create_from_pretrained(
    const char* model_id, const char* revision, const char* cache_dir,
    sc_indic_mio_progress_fn on_progress, void* progress_context) {
    if (!model_id || !*model_id) return nullptr;
    try {
        const std::string repo = model_id;
        const std::string rev = (revision && *revision) ? revision : "main";
        const std::string base =
            (cache_dir && *cache_dir) ? std::string(cache_dir) : default_cache_dir();
        const std::string dir = base + "/" + sanitize_repo(repo);

        const std::vector<std::string> kFiles = {
            "indicmio-text-prefill.tflite",
            "indicmio-token-step.tflite",
            "indicmio-audio-decoder.tflite",
            "indicmio-ref-encoder.tflite",
            "tokenizer.json",
            "config.json",
        };

        speech_core::hf::ProgressFn cb;
        if (on_progress) {
            cb = [on_progress, progress_context](const std::string& f, int idx,
                                                 int cnt, uint64_t done,
                                                 uint64_t total) {
                on_progress(f.c_str(), idx, cnt, done, total, progress_context);
            };
        }
        speech_core::hf::download_bundle(repo, rev, kFiles, dir, cb);
        return create_from_dir(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "[speech ERROR] sc_indic_mio_create_from_pretrained: %s\n",
                     e.what());
        return nullptr;
    }
}

bool sc_indic_mio_has_download_support(void) {
#ifdef SPEECH_CORE_WITH_HF_DOWNLOAD
    return true;
#else
    return false;
#endif
}

void sc_indic_mio_destroy(sc_indic_mio_t synth) { delete synth; }

int sc_indic_mio_set_reference(sc_indic_mio_t synth, const float* pcm,
                               size_t length, int sample_rate) {
    if (!synth) return 1;
    try {
        synth->tts->set_reference(pcm, length, sample_rate);
        synth->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        synth->last_error = e.what();
        return 1;
    }
}

void sc_indic_mio_clear_reference(sc_indic_mio_t synth) {
    if (synth) synth->tts->clear_reference();
}

void sc_indic_mio_set_temperature(sc_indic_mio_t synth, float temperature) {
    if (synth) synth->tts->set_temperature(temperature);
}
void sc_indic_mio_set_top_k(sc_indic_mio_t synth, int top_k) {
    if (synth) synth->tts->set_top_k(top_k);
}
void sc_indic_mio_set_top_p(sc_indic_mio_t synth, float top_p) {
    if (synth) synth->tts->set_top_p(top_p);
}
void sc_indic_mio_set_repetition_penalty(sc_indic_mio_t synth, float penalty) {
    if (synth) synth->tts->set_repetition_penalty(penalty);
}
void sc_indic_mio_set_max_new_tokens(sc_indic_mio_t synth, int max_new_tokens) {
    if (synth) synth->tts->set_max_new_tokens(max_new_tokens);
}
void sc_indic_mio_set_seed(sc_indic_mio_t synth, uint32_t seed) {
    if (synth) synth->tts->set_seed(seed);
}

int sc_indic_mio_output_sample_rate(sc_indic_mio_t synth) {
    return synth ? synth->tts->output_sample_rate() : 0;
}
int sc_indic_mio_max_text_tokens(sc_indic_mio_t synth) {
    return synth ? synth->tts->max_text_tokens() : 0;
}

int sc_indic_mio_synthesize(sc_indic_mio_t synth, const char* text,
                            sc_indic_mio_chunk_fn on_chunk, void* context) {
    if (!synth || !text || !on_chunk) return 1;
    try {
        synth->tts->synthesize(
            text, /*language=*/"",
            [on_chunk, context](const float* samples, size_t length, bool is_final) {
                on_chunk(samples, length, is_final, context);
            });
        synth->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        synth->last_error = e.what();
        return 1;
    }
}

void sc_indic_mio_cancel(sc_indic_mio_t synth) {
    if (synth) synth->tts->cancel();
}

int sc_indic_mio_tokens_generated(sc_indic_mio_t synth) {
    return synth ? synth->tts->tokens_generated() : 0;
}
bool sc_indic_mio_stopped_on_eos(sc_indic_mio_t synth) {
    return synth ? synth->tts->stopped_on_eos() : false;
}
uint32_t sc_indic_mio_seed_used(sc_indic_mio_t synth) {
    return synth ? synth->tts->seed_used() : 0;
}

const char* sc_indic_mio_last_error(sc_indic_mio_t synth) {
    return synth ? synth->last_error.c_str() : "";
}

}  // extern "C"
