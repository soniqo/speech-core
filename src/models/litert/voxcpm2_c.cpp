// C ABI implementation for VoxCPM2 voice cloning — see speech_core/voxcpm2_c.h.
// Thin wrapper over LiteRTVoxCPM2Tts; lives in speech_core_models_litert.

#include "speech_core/voxcpm2_c.h"
#include "speech_core/models/litert_voxcpm2_tts.h"

#include "hf_download.h"
#include "tts_c_options.h"

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using speech_core::LiteRTVoxCPM2Tts;
using speech_core::VoxCPM2SynthesisOptions;
using speech_core::convert_c_tts_synthesis_options;

struct sc_voxcpm2_s {
    std::unique_ptr<LiteRTVoxCPM2Tts> tts;
    std::string last_error;
};

namespace {

// Open a VoxCPM2 LiteRT bundle directory into a fresh handle. Throws on a
// missing/invalid bundle (the LiteRTVoxCPM2Tts constructor validates paths).
sc_voxcpm2_s* create_from_dir(const std::string& b) {
    auto* h = new sc_voxcpm2_s();
    h->tts = std::make_unique<LiteRTVoxCPM2Tts>(
        b + "/voxcpm2-text-prefill.tflite",
        b + "/voxcpm2-token-step.tflite",
        b + "/voxcpm2-audio-encoder.tflite",
        b + "/voxcpm2-audio-decoder.tflite",
        b + "/tokenizer.json",
        /*hw_accel=*/false);
    return h;
}

// Per-user cache root for downloaded model bundles. SPEECH_CORE_CACHE_DIR wins;
// otherwise the OS-native cache location.
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

// "soniqo/VoxCPM2-LiteRT" -> "soniqo__VoxCPM2-LiteRT" (filesystem-safe subdir).
std::string sanitize_repo(const std::string& id) {
    std::string s;
    for (char c : id) {
        if (c == '/' || c == '\\') s += "__";
        else s += c;
    }
    return s;
}

// Arch-appropriate VoxCPM2 LiteRT bundle subdir within the HF repo. On x86_64
// the fp16 token-step over-generates (its stop-margin rounds the wrong way on
// x86 XNNPACK so the stop token never fires); desktop x86 therefore pulls the
// fp32-token-step bundle from the repo's "fp32-p16/" subdir. ARM64 (native
// fp16, correct) uses the fp16 'selective' bundle at the repo root.
#if defined(__x86_64__) || defined(_M_X64)
constexpr const char* kBundleSubdir = "fp32-p16";
#else
constexpr const char* kBundleSubdir = "";
#endif

}  // namespace

extern "C" {

sc_voxcpm2_t sc_voxcpm2_create(const char* bundle_dir) {
    if (!bundle_dir) return nullptr;
    try {
        return create_from_dir(bundle_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[speech ERROR] sc_voxcpm2_create: %s\n", e.what());
        return nullptr;
    }
}

sc_voxcpm2_t sc_voxcpm2_create_from_pretrained(const char* model_id,
                                               const char* revision,
                                               const char* cache_dir,
                                               sc_voxcpm2_progress_fn on_progress,
                                               void* progress_context) {
    if (!model_id || !*model_id) return nullptr;
    try {
        const std::string repo = model_id;
        const std::string rev = (revision && *revision) ? revision : "main";
        const std::string base =
            (cache_dir && *cache_dir) ? std::string(cache_dir) : default_cache_dir();
        const std::string dir = base + "/" + sanitize_repo(repo);

        // The five files LiteRTVoxCPM2Tts needs, under the arch-appropriate
        // bundle subdir (x86 -> "fp32-p16/", ARM -> repo root). The subdir is
        // part of each fetched path, so the bundle lands in dir/<subdir>/.
        const std::string sub = *kBundleSubdir ? std::string(kBundleSubdir) + "/" : "";
        const std::vector<std::string> kFiles = {
            sub + "voxcpm2-text-prefill.tflite",
            sub + "voxcpm2-token-step.tflite",
            sub + "voxcpm2-audio-encoder.tflite",
            sub + "voxcpm2-audio-decoder.tflite",
            sub + "tokenizer.json",
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
        return create_from_dir(*kBundleSubdir ? dir + "/" + kBundleSubdir : dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[speech ERROR] sc_voxcpm2_create_from_pretrained: %s\n",
                     e.what());
        return nullptr;
    }
}

bool sc_voxcpm2_has_download_support(void) {
    return speech_core::hf::download_supported();
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

void sc_voxcpm2_set_reference_transcript(sc_voxcpm2_t s,
                                         const char* transcript) {
    if (s && s->tts) {
        s->tts->set_reference_transcript(transcript ? transcript : "");
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
    return sc_voxcpm2_synthesize_with_options(s, text, nullptr, on_chunk, context);
}

int sc_voxcpm2_synthesize_with_options(
    sc_voxcpm2_t s,
    const char* text,
    const sc_voxcpm2_synthesis_options_t* options,
    sc_voxcpm2_chunk_fn on_chunk,
    void* context) {
    if (!s || !s->tts) return -1;
    if (!text || !on_chunk) {
        if (s) s->last_error = "null text or callback";
        return -1;
    }
    VoxCPM2SynthesisOptions cpp_options;
    std::string error;
    if (!convert_c_tts_synthesis_options(options, cpp_options, error, "VoxCPM2")) {
        s->last_error = error;
        return -1;
    }
    try {
        s->tts->synthesize_with_options(text, "auto", cpp_options,
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
