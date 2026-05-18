// Integration tests for the speech_core_models_litert target.
//
// Each test loads real .tflite models from $SPEECH_LITERT_MODEL_DIR and exercises
// one LiteRT wrapper. Skipped (with exit code 0) when SPEECH_LITERT_MODEL_DIR is
// unset or the expected files are missing — keeps CI green when models aren't
// fetched.
//
// To run locally:
//     scripts/download_models_litert.sh
//     cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=/path/to/litert
//     cmake --build build
//     SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure

#include "speech_core/audio/resampler.h"
#include "speech_core/interfaces.h"
#include "speech_core/models/litert_parakeet_stt.h"
#include "speech_core/models/litert_silero_vad.h"
#include "speech_core/vad/streaming_vad.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
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
    const char* env = std::getenv("SPEECH_LITERT_MODEL_DIR");
    return env ? env : "";
}

std::string test_audio_path() {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
#else
    return "tests/data/test_audio.wav";
#endif
}

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};
WavData load_wav_mono_pcm16(const std::string& path) {
    WavData out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;

    char riff[4], wave[4];
    uint32_t file_size, fmt_size;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&file_size), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) return out;

    char chunk_id[4];
    uint32_t chunk_size;
    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    bool have_fmt = false, have_data = false;

    while (f.read(chunk_id, 4)) {
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.seekg(6, std::ios::cur);
            f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (chunk_size > 16) f.seekg(chunk_size - 16, std::ios::cur);
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt || audio_format != 1 || num_channels != 1 || bits_per_sample != 16) return out;
            size_t n_samples = chunk_size / 2;
            std::vector<int16_t> pcm(n_samples);
            f.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            out.samples.resize(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                out.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
            }
            out.sample_rate = static_cast<int>(sample_rate);
            have_data = true;
            break;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }
    if (!have_data) out = {};
    return out;
}

std::vector<float> generate_tone(int sample_rate, float freq, float seconds, float amp = 0.3f) {
    size_t n = static_cast<size_t>(seconds * sample_rate);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = amp * std::sin(2.0f * static_cast<float>(M_PI) * freq * i / sample_rate);
    }
    return out;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return s;
}

// ---------------------------------------------------------------------------

void test_litert_silero_vad(const std::string& dir) {
    std::string model = dir + "/silero-vad.tflite";
    if (!file_exists(model)) {
        std::printf("  [skip] silero-vad.tflite not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_silero_vad ... ");

    speech_core::LiteRTSileroVad vad(model);
    REQUIRE(vad.input_sample_rate() == 16000);
    REQUIRE(vad.chunk_size() == 512);

    std::vector<float> silence(512, 0.0f);
    float p_silence = vad.process_chunk(silence.data(), silence.size());
    REQUIRE(p_silence >= 0.0f && p_silence <= 1.0f);

    vad.reset();
    auto tone = generate_tone(16000, 220.0f, 0.5f, 0.4f);
    float p_tone_max = 0.0f;
    for (size_t i = 0; i + 512 <= tone.size(); i += 512) {
        float p = vad.process_chunk(tone.data() + i, 512);
        if (p > p_tone_max) p_tone_max = p;
    }
    REQUIRE(p_tone_max > p_silence);

    vad.reset();
    float p_after_reset = vad.process_chunk(silence.data(), silence.size());
    REQUIRE(std::abs(p_after_reset - p_silence) < 0.2f);

    std::printf("ok (silence=%.3f tone_peak=%.3f)\n", p_silence, p_tone_max);
}

// ---------------------------------------------------------------------------
// VAD on real speech — same fixture as the ORT test (test_audio.wav).
// Asserts speech_start ∈ (3, 7) and speech_end ∈ (7, 10) seconds.
// ---------------------------------------------------------------------------

void test_litert_silero_vad_real_speech(const std::string& dir) {
    std::string model = dir + "/silero-vad.tflite";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(model)) {
        std::printf("  [skip] silero-vad.tflite not in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_silero_vad_real_speech ... ");

    speech_core::LiteRTSileroVad vad(model);

    auto audio_16k = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio_16k.empty());

    constexpr size_t kChunk = 512;
    const float chunk_dur = static_cast<float>(kChunk) / 16000.0f;
    speech_core::StreamingVAD events(speech_core::VADConfig::silero_default(), chunk_dur);

    float speech_start = -1.0f, speech_end = -1.0f;
    for (size_t i = 0; i + kChunk <= audio_16k.size(); i += kChunk) {
        float p = vad.process_chunk(audio_16k.data() + i, kChunk);
        for (const auto& ev : events.process(p)) {
            if (ev.type == speech_core::VADEvent::SpeechStarted && speech_start < 0)
                speech_start = ev.start_time;
            if (ev.type == speech_core::VADEvent::SpeechEnded)
                speech_end = ev.end_time;
        }
    }
    for (const auto& ev : events.flush()) {
        if (ev.type == speech_core::VADEvent::SpeechEnded)
            speech_end = ev.end_time;
    }

    std::printf("start=%.2fs end=%.2fs ", speech_start, speech_end);

    REQUIRE(speech_start >= 3.0f && speech_start <= 7.0f);
    REQUIRE(speech_end   >= 7.0f && speech_end   <= 10.0f);

    std::printf("ok\n");
}

// ---------------------------------------------------------------------------

void test_litert_parakeet_stt(const std::string& dir) {
    std::string enc   = dir + "/parakeet-encoder.tflite";
    std::string dec   = dir + "/parakeet-decoder-joint.tflite";
    std::string vocab = dir + "/vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(vocab)) {
        std::printf("  [skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_parakeet_stt ... ");

    speech_core::LiteRTParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);
    REQUIRE(stt.supports_streaming());

    std::vector<float> silence(16000 * 1, 0.0f);
    auto result = stt.transcribe(silence.data(), silence.size(), 16000);
    REQUIRE(result.confidence >= 0.0f && result.confidence <= 1.0f);

    std::printf("ok (silence text=\"%s\" conf=%.3f)\n",
                result.text.c_str(), result.confidence);
}

// ---------------------------------------------------------------------------
// Parakeet on real speech — runs the fixture through LiteRT STT and checks
// the transcript contains something. The fixture is short Spanish speech, so
// we only assert non-empty output rather than expected content.
// ---------------------------------------------------------------------------

void test_litert_parakeet_real_speech(const std::string& dir) {
    std::string enc   = dir + "/parakeet-encoder.tflite";
    std::string dec   = dir + "/parakeet-decoder-joint.tflite";
    std::string vocab = dir + "/vocab.json";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(vocab)) {
        std::printf("  [skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_parakeet_real_speech ... ");

    speech_core::LiteRTParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);

    auto audio_16k = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio_16k.empty());

    auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);
    std::printf("text=\"%s\" conf=%.3f lang=\"%s\" ",
                result.text.c_str(), result.confidence, result.language.c_str());

    REQUIRE(!result.text.empty());
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VAD → STT pipeline e2e — runs the fixture through LiteRTSileroVad +
// StreamingVAD to locate the speech segment, slices the audio at the detected
// boundaries (with a small pre-/post-roll), and hands it to LiteRTParakeetStt.
// Catches integration bugs between the two backends that the per-model real-
// speech tests miss: window alignment, context-buffer state, off-by-one in
// the speech-segment slice, language detection on truncated audio.
// ---------------------------------------------------------------------------

void test_litert_vad_to_stt_pipeline(const std::string& dir) {
    std::string vad_model = dir + "/silero-vad.tflite";
    std::string enc       = dir + "/parakeet-encoder.tflite";
    std::string dec       = dir + "/parakeet-decoder-joint.tflite";
    std::string vocab     = dir + "/vocab.json";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(vad_model) || !file_exists(enc) || !file_exists(dec) || !file_exists(vocab)) {
        std::printf("  [skip] pipeline needs both silero and parakeet files in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_vad_to_stt_pipeline ... ");

    auto audio_16k = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio_16k.empty());

    // Stage 1: VAD → detect speech start/end (seconds).
    speech_core::LiteRTSileroVad vad(vad_model);
    constexpr size_t kChunk = 512;
    const float chunk_dur = static_cast<float>(kChunk) / 16000.0f;
    speech_core::StreamingVAD events(speech_core::VADConfig::silero_default(), chunk_dur);

    float speech_start = -1.0f, speech_end = -1.0f;
    for (size_t i = 0; i + kChunk <= audio_16k.size(); i += kChunk) {
        float p = vad.process_chunk(audio_16k.data() + i, kChunk);
        for (const auto& ev : events.process(p)) {
            if (ev.type == speech_core::VADEvent::SpeechStarted && speech_start < 0)
                speech_start = ev.start_time;
            if (ev.type == speech_core::VADEvent::SpeechEnded)
                speech_end = ev.end_time;
        }
    }
    for (const auto& ev : events.flush()) {
        if (ev.type == speech_core::VADEvent::SpeechEnded)
            speech_end = ev.end_time;
    }
    REQUIRE(speech_start >= 0.0f);
    REQUIRE(speech_end > speech_start);

    // Stage 2: slice the detected segment with 100 ms pre/post-roll, clamped
    // to the audio bounds.
    constexpr float kPad = 0.10f;
    size_t lo = static_cast<size_t>(std::max(0.0f, speech_start - kPad) * 16000.0f);
    size_t hi = std::min(audio_16k.size(),
                         static_cast<size_t>((speech_end + kPad) * 16000.0f));
    REQUIRE(hi > lo);
    std::vector<float> segment(audio_16k.begin() + lo, audio_16k.begin() + hi);

    // Stage 3: STT on the segment.
    speech_core::LiteRTParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);
    auto result = stt.transcribe(segment.data(), segment.size(), 16000);

    std::printf("vad=[%.2fs..%.2fs] segment=%.2fs text=\"%s\" conf=%.3f lang=\"%s\" ",
                speech_start, speech_end,
                static_cast<float>(segment.size()) / 16000.0f,
                result.text.c_str(), result.confidence, result.language.c_str());

    REQUIRE(!result.text.empty());
    std::printf("ok\n");
}

}  // namespace

int main() {
    std::string dir = env_model_dir();
    if (dir.empty()) {
        std::printf("SPEECH_LITERT_MODEL_DIR not set — skipping LiteRT model tests\n");
        std::printf("(run scripts/download_models_litert.sh and re-run with "
                    "SPEECH_LITERT_MODEL_DIR=scripts/models-litert)\n");
        return 0;
    }

    std::printf("Running LiteRT model tests against %s\n", dir.c_str());

    test_litert_silero_vad(dir);
    test_litert_silero_vad_real_speech(dir);
    test_litert_parakeet_stt(dir);
    test_litert_parakeet_real_speech(dir);
    test_litert_vad_to_stt_pipeline(dir);

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll LiteRT model tests passed\n");
    return 0;
}
