// Integration tests for the speech_core_models target.
//
// Each test loads real ONNX models from $SPEECH_MODEL_DIR and exercises one
// model wrapper. Includes a TTS→STT roundtrip that catches token/vocab bugs
// the per-model smoke tests miss (e.g. the BOS/EOS regression documented in
// kokoro_phonemizer.h:23-28).
//
// Skipped (with exit code 0) when SPEECH_MODEL_DIR is unset or the expected
// files are missing — keeps CI green when models aren't fetched.
//
// To run locally:
//     scripts/download_models.sh
//     cmake -B build -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/ort
//     cmake --build build
//     SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure

#include "speech_core/audio/resampler.h"
#include "speech_core/interfaces.h"
#include "speech_core/models/deepfilter.h"
#include "speech_core/models/kokoro_tts.h"
#include "speech_core/models/parakeet_stt.h"
#include "speech_core/models/silero_vad.h"
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
    const char* env = std::getenv("SPEECH_MODEL_DIR");
    return env ? env : "";
}

/// Path to the test audio fixture, sourced from speech-swift
/// (Tests/Qwen3ASRTests/Resources/test_audio.wav, 24 kHz mono PCM16).
/// CMake passes the source dir via SPEECH_CORE_TEST_DATA_DIR.
std::string test_audio_path() {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
#else
    return "tests/data/test_audio.wav";
#endif
}

/// Minimal WAV reader — mono PCM16 little-endian only.
/// Returns audio as Float32 normalised to [-1, 1], plus the sample rate.
/// Returns empty samples on any failure (caller should skip the test).
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

    // Iterate chunks until we find "fmt " and "data".
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
            f.seekg(6, std::ios::cur);  // byte_rate (4) + block_align (2)
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
// VAD on real speech — uses the test_audio.wav fixture from speech-swift.
// The fixture contains roughly 5.2s–8.4s of speech in an otherwise quiet
// recording. The Swift test asserts speech_start ∈ (3, 7) and speech_end
// ∈ (7, 10) seconds. We mirror that here through StreamingVAD.
// ---------------------------------------------------------------------------

void test_silero_vad_real_speech(const std::string& dir) {
    std::string model = dir + "/silero-vad.onnx";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(model)) {
        std::printf("  [skip] silero-vad.onnx not in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_silero_vad_real_speech ... ");

    speech_core::SileroVad vad(model);

    auto audio_16k = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio_16k.empty());

    // Feed 512-sample chunks through StreamingVAD's event-driven wrapper.
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

// ---------------------------------------------------------------------------
// Roundtrip: Kokoro TTS → resample 24→16 kHz → Parakeet STT
//
// Catches end-to-end issues that the per-model smoke tests miss:
//   - Phonemizer BOS/EOS misalignment (kokoro_phonemizer.h:23-28 documents a
//     past bug where "Hello world" came back as "I wrote")
//   - Vocab ID off-by-ones in either model
//   - Sample-rate / Hz mismatch between TTS output and STT input
//   - Voice file loading failures producing silent or distorted audio
// ---------------------------------------------------------------------------

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return s;
}

void test_kokoro_parakeet_roundtrip(const std::string& dir) {
    std::string kokoro_model = dir + "/kokoro-e2e.onnx";
    std::string parakeet_enc = dir + "/parakeet-encoder-int8.onnx";
    std::string parakeet_dec = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string parakeet_vocab = dir + "/vocab.json";
    if (!file_exists(kokoro_model) || !file_exists(parakeet_enc)
        || !file_exists(parakeet_dec) || !file_exists(parakeet_vocab)
        || !file_exists(dir + "/vocab_index.json"))
    {
        std::printf("  [skip] roundtrip needs both kokoro and parakeet files in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_kokoro_parakeet_roundtrip ... ");

    speech_core::KokoroTts tts(kokoro_model, dir + "/voices", dir, /*hw_accel=*/false);
    speech_core::ParakeetStt stt(parakeet_enc, parakeet_dec, parakeet_vocab, /*hw_accel=*/false);

    const std::string input = "Hello world.";
    std::vector<float> audio_24k;
    tts.synthesize(input, "en",
        [&](const float* samples, size_t len, bool /*is_final*/) {
            audio_24k.insert(audio_24k.end(), samples, samples + len);
        });
    REQUIRE(!audio_24k.empty());

    // Kokoro outputs 24 kHz, Parakeet expects 16 kHz.
    auto audio_16k = speech_core::Resampler::resample(
        audio_24k.data(), audio_24k.size(), 24000, 16000);
    REQUIRE(!audio_16k.empty());

    auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);
    std::string transcript = to_lower(result.text);

    // Content-word check: at least one of "hello"/"world" should appear.
    // Strict equality is too brittle (TTS adds pauses, STT may insert filler).
    bool has_hello = transcript.find("hello") != std::string::npos;
    bool has_world = transcript.find("world") != std::string::npos;
    int matched = (has_hello ? 1 : 0) + (has_world ? 1 : 0);

    std::printf("input=\"%s\" transcript=\"%s\" matched=%d/2 ",
                input.c_str(), result.text.c_str(), matched);

    REQUIRE(matched >= 1);
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
    test_silero_vad_real_speech(dir);
    test_parakeet_stt(dir);
    test_kokoro_tts(dir);
    test_deepfilter(dir);
    test_kokoro_parakeet_roundtrip(dir);

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll model tests passed\n");
    return 0;
}
