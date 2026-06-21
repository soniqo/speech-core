#include "speech_core/supertonic_c.h"

#include "speech_core/models/litert_supertonic_tts.h"

#include <filesystem>
#include <new>
#include <string>

using speech_core::LiteRTSupertonicTts;

struct sc_supertonic_s {
    LiteRTSupertonicTts engine;
    std::string         last_error;

    sc_supertonic_s(const std::string& dur, const std::string& enc, const std::string& vec,
                    const std::string& voc, const std::string& tok, const std::string& vs,
                    bool hw_accel)
        : engine(dur, enc, vec, voc, tok, vs, hw_accel) {}
};

namespace {
// Process-wide last-creation-error, returned when the caller passes a NULL handle.
std::string& creation_error() {
    static std::string err;
    return err;
}
}  // namespace

extern "C" {

sc_supertonic_t sc_supertonic_create_from_paths(const char* duration_path,
                                                const char* text_encoder_path,
                                                const char* vector_estimator_path,
                                                const char* vocoder_path,
                                                const char* tokenizer_dir,
                                                const char* voice_styles_dir,
                                                bool hw_accel) {
    try {
        return new sc_supertonic_s(duration_path, text_encoder_path, vector_estimator_path,
                                   vocoder_path, tokenizer_dir, voice_styles_dir, hw_accel);
    } catch (const std::exception& e) {
        creation_error() = e.what();
        return nullptr;
    } catch (...) {
        creation_error() = "unknown error";
        return nullptr;
    }
}

sc_supertonic_t sc_supertonic_create(const char* bundle_dir) {
    namespace fs = std::filesystem;
    const fs::path b(bundle_dir ? bundle_dir : "");
    const std::string vs = (b / "voice_styles").string();
    return sc_supertonic_create_from_paths(
        (b / "duration_predictor.tflite").string().c_str(),
        (b / "text_encoder.tflite").string().c_str(),
        (b / "vector_estimator.tflite").string().c_str(),
        (b / "vocoder.tflite").string().c_str(),
        b.string().c_str(),
        vs.c_str(),
        false);
}

void sc_supertonic_destroy(sc_supertonic_t synth) { delete synth; }

void sc_supertonic_set_voice(sc_supertonic_t synth, const char* voice_id) {
    if (!synth || !voice_id) return;
    try { synth->engine.set_voice(voice_id); }
    catch (const std::exception& e) { synth->last_error = e.what(); }
}

void sc_supertonic_set_total_step(sc_supertonic_t synth, int total_step) {
    if (synth) synth->engine.set_total_step(total_step);
}

void sc_supertonic_set_speed(sc_supertonic_t synth, float speed) {
    if (synth) synth->engine.set_speed(speed);
}

void sc_supertonic_set_seed(sc_supertonic_t synth, uint32_t seed) {
    if (synth) synth->engine.set_seed(seed);
}

int sc_supertonic_output_sample_rate(sc_supertonic_t synth) {
    return synth ? synth->engine.output_sample_rate() : 0;
}

int sc_supertonic_synthesize(sc_supertonic_t synth, const char* text, const char* language,
                             sc_supertonic_chunk_fn on_chunk, void* user) {
    if (!synth || !text || !language || !on_chunk) return 1;
    try {
        synth->engine.synthesize(text, language,
            [on_chunk, user](const float* samples, size_t length, bool is_final) {
                on_chunk(samples, length, is_final, user);
            });
        return 0;
    } catch (const std::exception& e) {
        synth->last_error = e.what();
        return 2;
    } catch (...) {
        synth->last_error = "unknown error";
        return 2;
    }
}

void sc_supertonic_cancel(sc_supertonic_t synth) {
    if (synth) synth->engine.cancel();
}

const char* sc_supertonic_last_error(sc_supertonic_t synth) {
    return synth ? synth->last_error.c_str() : creation_error().c_str();
}

}  // extern "C"
