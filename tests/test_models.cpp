// Integration tests for the speech_core_models target.
//
// Each test loads real ONNX models from $SPEECH_MODEL_DIR and exercises one
// model wrapper. Skipped (with exit code 0) when SPEECH_MODEL_DIR is unset or
// the expected files are missing — keeps CI green when models aren't fetched.
//
// To run locally:
//     scripts/download_models.sh
//     cmake -B build -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/ort
//     cmake --build build
//     SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure

#include "speech_core/interfaces.h"
#include "speech_core/models/deepfilter.h"
#include "speech_core/models/kokoro_tts.h"
#include "speech_core/models/parakeet_stt.h"
#include "speech_core/models/silero_vad.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d)\n", #cond, __LINE__); \
        ++failures; \
        return; \
    } \
} while (0)

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string env_model_dir() {
    const char* env = std::getenv("SPEECH_MODEL_DIR");
    return env ? env : "";
}

// Generate a 16 kHz tone (sine) and 16 kHz silence chunk.
std::vector<float> generate_tone(int sample_rate, float freq, float seconds, float amp = 0.3f) {
    size_t n = static_cast<size_t>(seconds * sample_rate);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = amp * std::sin(2.0f * static_cast<float>(M_PI) * freq * i / sample_rate);
    }
    return out;
}

// ---------------------------------------------------------------------------

void test_silero_vad(const std::string& dir) {
    std::string model = dir + "/silero-vad.onnx";
    if (!file_exists(model)) {
        std::printf("  [skip] silero-vad.onnx not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_silero_vad ... ");

    speech_core::SileroVad vad(model);
    REQUIRE(vad.input_sample_rate() == 16000);
    REQUIRE(vad.chunk_size() == 512);

    // Silence chunk → low probability
    std::vector<float> silence(512, 0.0f);
    float p_silence = vad.process_chunk(silence.data(), silence.size());
    REQUIRE(p_silence >= 0.0f && p_silence <= 1.0f);

    // Speech-band tone → higher probability than silence
    vad.reset();
    auto tone = generate_tone(16000, 220.0f, 0.5f, 0.4f);  // ~220 Hz, voiced range
    float p_tone_max = 0.0f;
    for (size_t i = 0; i + 512 <= tone.size(); i += 512) {
        float p = vad.process_chunk(tone.data() + i, 512);
        if (p > p_tone_max) p_tone_max = p;
    }
    REQUIRE(p_tone_max > p_silence);

    // reset() clears state
    vad.reset();
    float p_after_reset = vad.process_chunk(silence.data(), silence.size());
    REQUIRE(std::abs(p_after_reset - p_silence) < 0.2f);

    std::printf("ok (silence=%.3f tone_peak=%.3f)\n", p_silence, p_tone_max);
}

// ---------------------------------------------------------------------------

void test_parakeet_stt(const std::string& dir) {
    std::string enc = dir + "/parakeet-encoder-int8.onnx";
    std::string dec = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string vocab = dir + "/vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(vocab)) {
        std::printf("  [skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_parakeet_stt ... ");

    speech_core::ParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);
    REQUIRE(stt.supports_streaming());

    // Pure silence → empty or near-empty transcription, no crash
    std::vector<float> silence(16000 * 1, 0.0f);
    auto result = stt.transcribe(silence.data(), silence.size(), 16000);
    REQUIRE(result.confidence >= 0.0f && result.confidence <= 1.0f);

    std::printf("ok (silence text=\"%s\" conf=%.3f)\n",
                result.text.c_str(), result.confidence);
}

// ---------------------------------------------------------------------------

void test_kokoro_tts(const std::string& dir) {
    std::string model = dir + "/kokoro-e2e.onnx";
    std::string voices = dir + "/voices";
    if (!file_exists(model) || !file_exists(dir + "/vocab_index.json")) {
        std::printf("  [skip] kokoro files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_kokoro_tts ... ");

    speech_core::KokoroTts tts(model, voices, dir, /*hw_accel=*/false);
    REQUIRE(tts.output_sample_rate() == 24000);

    size_t total_samples = 0;
    bool got_final = false;
    tts.synthesize("Hello world.", "en",
        [&](const float* samples, size_t len, bool is_final) {
            (void)samples;
            total_samples += len;
            if (is_final) got_final = true;
        });

    REQUIRE(got_final);
    REQUIRE(total_samples > 0);

    std::printf("ok (samples=%zu)\n", total_samples);
}

// ---------------------------------------------------------------------------

void test_deepfilter(const std::string& dir) {
    std::string model = dir + "/deepfilter.onnx";
    std::string aux = dir + "/deepfilter-auxiliary.bin";
    if (!file_exists(model) || !file_exists(aux)) {
        std::printf("  [skip] deepfilter files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_deepfilter ... ");

    speech_core::DeepFilterEnhancer enh(model, aux, /*hw_accel=*/false);
    REQUIRE(enh.input_sample_rate() == 48000);

    // 1 second of low-amplitude noise at 48 kHz
    std::vector<float> noisy(48000);
    for (size_t i = 0; i < noisy.size(); ++i) {
        noisy[i] = 0.05f * std::sin(2.0f * static_cast<float>(M_PI) * 440.0f * i / 48000.0f);
    }
    std::vector<float> clean(noisy.size(), 0.0f);
    enh.enhance(noisy.data(), noisy.size(), 48000, clean.data());

    // Output should not be all zeros and not contain NaN/Inf
    bool has_signal = false;
    bool has_nan = false;
    for (float v : clean) {
        if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
        if (std::abs(v) > 1e-6f) has_signal = true;
    }
    REQUIRE(!has_nan);
    REQUIRE(has_signal);

    std::printf("ok\n");
}

}  // namespace

int main() {
    std::string dir = env_model_dir();
    if (dir.empty()) {
        std::printf("SPEECH_MODEL_DIR not set — skipping model tests\n");
        std::printf("(run scripts/download_models.sh and re-run with SPEECH_MODEL_DIR=scripts/models)\n");
        return 0;
    }

    std::printf("Running model tests against %s\n", dir.c_str());

    test_silero_vad(dir);
    test_parakeet_stt(dir);
    test_kokoro_tts(dir);
    test_deepfilter(dir);

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll model tests passed\n");
    return 0;
}
