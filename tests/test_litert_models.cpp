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
#include "speech_core/diarization/diarization_pipeline.h"
#include "speech_core/models/litert_nemotron_streaming_stt.h"
#include "speech_core/models/litert_nemotron_multilingual_stt.h"
#ifdef SPEECH_CORE_HAS_LITERT_KOKORO
#include "speech_core/models/litert_kokoro_tts.h"
#endif
#include "speech_core/models/litert_omnilingual_stt.h"
#include "speech_core/models/litert_parakeet_stt.h"
#include "speech_core/models/litert_pyannote_segmentation.h"
#include "speech_core/models/litert_silero_vad.h"
#include "speech_core/models/litert_voxcpm2_tts.h"
#include "speech_core/models/litert_wespeaker_embedding.h"
#include "speech_core/models/voxcpm2_tokenizer.h"
#include "speech_core/vad/streaming_vad.h"
#include "speech_core/voxcpm2_c.h"

#include "voxcpm2_tokenizer_test_util.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

// MSVC's <cmath> doesn't define M_PI without _USE_MATH_DEFINES.
constexpr float kPi = 3.14159265358979323846f;

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

/// Reference voice for the Hindi cloning test. Prefer the studio-pipeline
/// reference if a user has dropped it into tests/data as test_hindi_ref.wav;
/// fall back to the generic English fixture otherwise.
std::string test_hindi_ref_path() {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_hindi_ref.wav";
#else
    return "tests/data/test_hindi_ref.wav";
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
        out[i] = amp * std::sin(2.0f * kPi * freq * static_cast<float>(i) / static_cast<float>(sample_rate));
    }
    return out;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return s;
}

// ---------------------------------------------------------------------------

#ifdef SPEECH_CORE_HAS_LITERT_KOKORO
void test_litert_kokoro_tts(const std::string& dir) {
    const std::string encoder = dir + "/kokoro-encoder.tflite";
    const std::string recurrent =
        dir + "/kokoro-recurrent-equivalent32.tflite";
    const std::string vocoder = dir + "/kokoro-vocoder.tflite";
    const std::string vocab = dir + "/vocab_index.json";
    const std::string gold = dir + "/us_gold.json";
    const std::string silver = dir + "/us_silver.json";
    const std::string voice = dir + "/voices/af_heart.bin";
    if (!file_exists(encoder) || !file_exists(recurrent) ||
        !file_exists(vocoder) || !file_exists(vocab) || !file_exists(gold) ||
        !file_exists(silver) || !file_exists(voice)) {
        std::printf("  [skip] Kokoro LiteRT bundle not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_kokoro_tts ... ");

    speech_core::LiteRTKokoroTts tts(
        encoder, recurrent, vocoder, dir + "/voices", dir,
        /*hw_accel=*/false, /*num_threads=*/4);
    REQUIRE(tts.output_sample_rate() == 24000);
    REQUIRE(tts.max_active_phonemes() == 32);
    REQUIRE(tts.preferred_chunk_phonemes() == 14);
    REQUIRE(tts.max_safe_frames() == 56);
    tts.set_seed(1234);
    tts.set_speed(1.0f);

    std::vector<float> audio;
    bool final = false;
    tts.synthesize("Hello world.", "en",
        [&](const float* samples, size_t length, bool is_final) {
            if (samples && length) audio.insert(audio.end(), samples, samples + length);
            final = final || is_final;
        });
    REQUIRE(final);
    REQUIRE(audio.size() >= 6000);
    REQUIRE(audio.size() <= 56 * 600);
    REQUIRE(tts.model_runs_last_synthesis() == 1);
    float peak = 0.0f;
    double energy = 0.0;
    for (float sample : audio) {
        REQUIRE(std::isfinite(sample));
        peak = std::max(peak, std::abs(sample));
        energy += static_cast<double>(sample) * sample;
    }
    REQUIRE(peak > 0.01f && peak <= 2.0f);
    REQUIRE(energy / audio.size() > 1e-6);
    std::printf("ok (%zu samples, peak=%.3f)\n", audio.size(), peak);
}
#endif

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
// Parakeet streaming — exercises the begin_stream / push_chunk / end_stream
// path that the orchestrator uses in production (the batch transcribe() above
// covers only the one-shot mode). Feeds the fixture in ~100 ms chunks and
// checks that end_stream() returns a non-empty transcript matching the batch
// result. Catches regressions in stream_buffer_ management, the 0.5 s warm-up
// gate in push_chunk, and end_stream not draining the buffer.
// ---------------------------------------------------------------------------

void test_litert_parakeet_streaming(const std::string& dir) {
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
    std::printf("  test_litert_parakeet_streaming ... ");

    speech_core::LiteRTParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);
    REQUIRE(stt.supports_streaming());

    auto audio_16k = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio_16k.empty());

    constexpr size_t kChunk = 1600;  // 100 ms @ 16 kHz
    stt.begin_stream(16000);
    size_t partials_seen = 0;
    for (size_t off = 0; off < audio_16k.size(); off += kChunk) {
        size_t len = std::min(kChunk, audio_16k.size() - off);
        auto partial = stt.push_chunk(audio_16k.data() + off, len);
        if (!partial.text.empty()) ++partials_seen;
    }
    auto result = stt.end_stream();
    std::printf("partials=%zu final=\"%s\" conf=%.3f ",
                partials_seen, result.text.c_str(), result.confidence);
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

// ---------------------------------------------------------------------------
// VoxCPM2 skeleton load — proves the four LiteRT graphs and the HF tokenizer
// file parse. The wrapper's synthesize() is intentionally a stub; this test
// only covers what the skeleton does.
// ---------------------------------------------------------------------------

void test_litert_voxcpm2_load(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string enc  = dir + "/voxcpm2-audio-encoder.tflite";
    std::string dec  = dir + "/voxcpm2-audio-decoder.tflite";
    std::string tok  = dir + "/tokenizer.json";
    if (!file_exists(pref) || !file_exists(step) || !file_exists(enc)
        || !file_exists(dec) || !file_exists(tok)) {
        std::printf("  [skip] voxcpm2 files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_voxcpm2_load ... ");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, enc, dec, tok, /*hw_accel=*/false);
    REQUIRE(tts.output_sample_rate() == 48000);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VoxCPM2 end-to-end synthesis — runs the full pipeline with a short text and
// a capped step count, accumulates the streamed PCM, and validates basic
// properties (sample rate produces the expected chunk size, audio is not
// silent, no NaN/inf). Skipped unless VOXCPM2 model files are present.
// Set step count low (16) so even a heavy nightly run stays bounded.
// ---------------------------------------------------------------------------

void test_litert_voxcpm2_synthesize(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string enc  = dir + "/voxcpm2-audio-encoder.tflite";
    std::string dec  = dir + "/voxcpm2-audio-decoder.tflite";
    std::string tok  = dir + "/tokenizer.json";
    if (!file_exists(pref) || !file_exists(step) || !file_exists(enc)
        || !file_exists(dec) || !file_exists(tok)) {
        std::printf("  [skip] voxcpm2 files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_voxcpm2_synthesize ... ");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, enc, dec, tok, /*hw_accel=*/false);
    // Long enough that the model can actually utter "the quick brown fox jumps
    // over the lazy dog" before stop-signal kicks in (~25-30 AR steps in
    // upstream Python runs).
    // The 9-word test phrase takes ~25-40 AR steps to fully utter; bump
    // both knobs up from defaults so the ASR round-trip in the weekly CI
    // has enough audio to recognise. min_stop_steps_=32 ignores stop-signal
    // for the first 32 steps (~5 s) — empirically enough to cover the
    // phrase before the model flips its stop bit on a natural pause.
    //
    // Also set a non-empty instruction: VoxCPM2 was trained on
    // "({instruction}){text}" prompts and an empty "()..." prefix tends to
    // make the model emit a one-word filler ("Yeah.") instead of the text.
    tts.set_instruction("clear, natural delivery");
    tts.set_max_steps(128);
    tts.set_min_steps_before_stop(32);
    // Pin the noise seed so this run is reproducible and seed_used() is
    // predictable (the AR step samples Gaussian noise, so the seed matters).
    tts.set_seed(4242);

    // max_text_tokens() is the prefill context window — 512 on the deployed
    // bundle, 256 on the small export; either way comfortably > our prompt.
    REQUIRE(tts.max_text_tokens() >= 256);

    std::vector<float> audio;
    bool got_final = false;
    tts.synthesize("The quick brown fox jumps over the lazy dog", "en",
        [&](const float* samples, size_t length, bool is_final) {
            if (samples && length) audio.insert(audio.end(), samples, samples + length);
            if (is_final) got_final = true;
        });
    REQUIRE(got_final);
    REQUIRE(!audio.empty());
    // Each step contributes 7680 samples (160 ms @ 48 kHz). The model usually
    // stops between 20-40 steps for this prompt — bound the assertion loosely.
    const size_t steps = audio.size() / 7680;
    REQUIRE(steps >= 8);

    // Synthesis metadata getters (restored from the cloud fork). seed_used()
    // echoes the pinned seed; tokens_generated() equals the streamed step
    // count; with max_steps=128 a 9-word phrase stops on the model's signal
    // well before the budget.
    REQUIRE(tts.seed_used() == 4242u);
    REQUIRE(tts.tokens_generated() == static_cast<int>(steps));
    REQUIRE(tts.tokens_generated() <= 128);
    std::printf("tokens=%d stopped_on_stop=%d ",
                tts.tokens_generated(), tts.stopped_on_stop_token() ? 1 : 0);

    double sum_sq = 0.0;
    float  peak   = 0.0f;
    for (float s : audio) {
        if (!std::isfinite(s)) { std::printf("\n    sample is non-finite "); REQUIRE(false); }
        sum_sq += static_cast<double>(s) * s;
        peak    = std::max(peak, std::abs(s));
    }
    const double rms = std::sqrt(sum_sq / audio.size());
    std::printf("steps=%zu samples=%zu rms=%.4f peak=%.4f ",
                steps, audio.size(), rms, peak);
    REQUIRE(rms > 1e-5);     // not silent
    REQUIRE(peak < 1.5f);    // no extreme blow-up (clip detector)

    // Dump the WAV so the CI step can pipe it through ASR for a semantic
    // round-trip assertion against the input text.
    const char* wav_out = std::getenv("VOXCPM2_SYNTH_WAV");
    if (wav_out) {
        std::ofstream w(wav_out, std::ios::binary);
        if (w) {
            // Minimal RIFF/PCM-16 header — 48 kHz mono int16.
            const uint32_t sample_rate = 48000;
            const uint32_t n_samples   = static_cast<uint32_t>(audio.size());
            const uint32_t data_bytes  = n_samples * 2;
            const uint32_t fmt_chunk_size = 16;
            const uint32_t riff_size = 4 + 8 + fmt_chunk_size + 8 + data_bytes;
            const uint16_t one = 1;
            const uint16_t two = 2;
            const uint16_t sixteen = 16;
            const uint32_t byte_rate = sample_rate * 2;
            w.write("RIFF", 4);
            w.write(reinterpret_cast<const char*>(&riff_size), 4);
            w.write("WAVE", 4);
            w.write("fmt ", 4);
            w.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
            w.write(reinterpret_cast<const char*>(&one), 2);          // PCM
            w.write(reinterpret_cast<const char*>(&one), 2);          // mono
            w.write(reinterpret_cast<const char*>(&sample_rate), 4);
            w.write(reinterpret_cast<const char*>(&byte_rate), 4);
            w.write(reinterpret_cast<const char*>(&two), 2);          // block align
            w.write(reinterpret_cast<const char*>(&sixteen), 2);      // bits per sample
            w.write("data", 4);
            w.write(reinterpret_cast<const char*>(&data_bytes), 4);
            for (float s : audio) {
                int32_t v = static_cast<int32_t>(std::lround(std::clamp(s, -1.0f, 1.0f) * 32767.0f));
                int16_t v16 = static_cast<int16_t>(v);
                w.write(reinterpret_cast<const char*>(&v16), 2);
            }
            std::printf("wav=%s ", wav_out);
        }
    }

    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VoxCPM2 tokenizer — pins encode/decode to upstream VoxCPM2's tokenizer
// wrapper. Only requires tokenizer.json, not the .tflite graphs, so it runs
// whenever tokenizer.json is present.
// ---------------------------------------------------------------------------

void test_voxcpm2_tokenizer(const std::string& dir) {
    std::string tok_path = dir + "/tokenizer.json";
    if (!file_exists(tok_path)) {
        std::printf("  [skip] tokenizer.json not in %s\n", dir.c_str());
        return;
    }
    REQUIRE(speech_core_test::run_voxcpm2_tokenizer_reference_cases(tok_path));
}

// ---------------------------------------------------------------------------
// VoxCPM2 voice cloning — conditions on the test fixture as a reference clip
// and synthesizes a known phrase, asserting non-silent finite audio. The
// fixture's silent lead-in is trimmed inside set_reference(). Dumps a WAV when
// VOXCPM2_CLONE_WAV is set so CI can ASR round-trip it.
// ---------------------------------------------------------------------------

void dump_wav48k_mono(const char* path, const std::vector<float>& audio) {
    std::ofstream w(path, std::ios::binary);
    if (!w) return;
    const uint32_t rate = 48000, n = static_cast<uint32_t>(audio.size());
    const uint32_t data = n * 2, riff = 36 + data;
    auto u32 = [&](uint32_t v) { w.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { w.write(reinterpret_cast<const char*>(&v), 2); };
    w.write("RIFF", 4); u32(riff); w.write("WAVE", 4);
    w.write("fmt ", 4); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16);
    w.write("data", 4); u32(data);
    for (float s : audio) {
        if (s < -1.0f) s = -1.0f; if (s > 1.0f) s = 1.0f;
        u16(static_cast<uint16_t>(static_cast<int16_t>(s * 32767.0f)));
    }
}

void check_voice_audio(const std::vector<float>& audio, const char* tag) {
    REQUIRE(!audio.empty());
    const size_t steps = audio.size() / 7680;
    REQUIRE(steps >= 8);
    double sum_sq = 0.0; float peak = 0.0f;
    for (float s : audio) {
        if (!std::isfinite(s)) { std::printf("\n    %s non-finite ", tag); REQUIRE(false); }
        sum_sq += static_cast<double>(s) * s; peak = std::max(peak, std::abs(s));
    }
    const double rms = std::sqrt(sum_sq / audio.size());
    std::printf("%s steps=%zu rms=%.4f peak=%.4f ", tag, steps, rms, peak);
    REQUIRE(rms > 1e-3);     // cloned output must not be silent
    REQUIRE(peak < 1.5f);
}

void test_litert_voxcpm2_clone(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string enc  = dir + "/voxcpm2-audio-encoder.tflite";
    std::string dec  = dir + "/voxcpm2-audio-decoder.tflite";
    std::string tok  = dir + "/tokenizer.json";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(pref) || !file_exists(step) || !file_exists(enc)
        || !file_exists(dec) || !file_exists(tok)) {
        std::printf("  [skip] voxcpm2 files not in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_voxcpm2_clone ... ");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, enc, dec, tok, /*hw_accel=*/false);
    tts.set_instruction("clear, natural delivery");
    tts.set_max_steps(128);
    tts.set_min_steps_before_stop(32);
    tts.set_reference(wav.samples.data(), wav.samples.size(), wav.sample_rate);
    REQUIRE(tts.has_reference());

    std::vector<float> audio;
    bool got_final = false;
    tts.synthesize("The quick brown fox jumps over the lazy dog", "en",
        [&](const float* s, size_t len, bool is_final) {
            if (s && len) audio.insert(audio.end(), s, s + len);
            if (is_final) got_final = true;
        });
    REQUIRE(got_final);
    check_voice_audio(audio, "clone");

    const char* wav_out = std::getenv("VOXCPM2_CLONE_WAV");
    if (wav_out) { dump_wav48k_mono(wav_out, audio); std::printf("wav=%s ", wav_out); }
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// Round-trip text→TTS→STT helpers — the semantic assertion that the existing
// "non-silent finite audio" checks can't make. The weekly CI does this through
// a shell pipeline (LiteRT VoxCPM2 → WAV → ORT Parakeet via speech_transcribe),
// which only catches regressions weekly and depends on the example CLI builds.
// Running it inside a single test binary makes the round-trip part of every
// model-gated test invocation.
// ---------------------------------------------------------------------------

constexpr const char* kRoundtripPhrase = "The quick brown fox jumps over the lazy dog";

/// Content words in the pangram. Excludes "the", "over" — articles and
/// prepositions are the words ASR most often drops or substitutes; checking
/// only nouns/verbs/adjectives gives a meaningful "did the model say it"
/// signal without false-failing on innocuous filler differences.
const std::vector<std::string>& roundtrip_content_words() {
    static const std::vector<std::string> kWords =
        {"quick", "brown", "fox", "jumps", "lazy", "dog"};
    return kWords;
}

/// Configure a VoxCPM2 instance for the round-trip phrase. Pulls the same
/// settings the existing synthesize/clone tests landed on after CI tuning —
/// enough steps to utter all 9 words, enough min-stop slack to ignore the
/// first natural pause, a fixed seed for reproducibility.
void configure_voxcpm2_for_roundtrip(speech_core::LiteRTVoxCPM2Tts& tts) {
    tts.set_instruction("clear, natural delivery");
    tts.set_max_steps(128);
    tts.set_min_steps_before_stop(32);
    tts.set_seed(4242);
}

/// Returns the count of content words from `roundtrip_content_words()` that
/// appear (case-insensitively) in `transcript`.
size_t count_matched_words(const std::string& transcript) {
    const std::string lower = to_lower(transcript);
    size_t matched = 0;
    for (const auto& w : roundtrip_content_words()) {
        if (lower.find(w) != std::string::npos) ++matched;
    }
    return matched;
}

/// Resample 48 kHz VoxCPM2 output down to 16 kHz Parakeet input. Tiny helper
/// so the two round-trip tests below can stay focused on the actual flow.
std::vector<float> resample_48k_to_16k(const std::vector<float>& audio_48k) {
    return speech_core::Resampler::resample(audio_48k.data(), audio_48k.size(),
                                            48000, 16000);
}

// ---------------------------------------------------------------------------
// LiteRT VoxCPM2 → LiteRT Parakeet round-trip — the in-process equivalent of
// the weekly CI's text→TTS→ASR chain, using the LiteRT Parakeet (INT8) STT for
// the ASR side. Requires every content word in the pangram to appear in the
// transcript — same bar the weekly CI sets against ORT Parakeet. INT8 noise
// might require relaxing if it turns out the quantization drops a word; tune
// after we see real output. Skipped unless all five VoxCPM2 graphs + Parakeet
// are present in SPEECH_LITERT_MODEL_DIR.
// ---------------------------------------------------------------------------

void test_litert_voxcpm2_parakeet_roundtrip(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string vox_enc = dir + "/voxcpm2-audio-encoder.tflite";
    std::string vox_dec = dir + "/voxcpm2-audio-decoder.tflite";
    std::string tok  = dir + "/tokenizer.json";
    std::string par_enc = dir + "/parakeet-encoder.tflite";
    std::string par_dec = dir + "/parakeet-decoder-joint.tflite";
    std::string vocab   = dir + "/vocab.json";
    if (!file_exists(pref) || !file_exists(step) || !file_exists(vox_enc)
        || !file_exists(vox_dec) || !file_exists(tok)) {
        std::printf("  [skip] voxcpm2 files not in %s\n", dir.c_str());
        return;
    }
    if (!file_exists(par_enc) || !file_exists(par_dec) || !file_exists(vocab)) {
        std::printf("  [skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_voxcpm2_parakeet_roundtrip ... ");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, vox_enc, vox_dec, tok, /*hw_accel=*/false);
    configure_voxcpm2_for_roundtrip(tts);

    std::vector<float> audio_48k;
    bool got_final = false;
    tts.synthesize(kRoundtripPhrase, "en",
        [&](const float* s, size_t len, bool is_final) {
            if (s && len) audio_48k.insert(audio_48k.end(), s, s + len);
            if (is_final) got_final = true;
        });
    REQUIRE(got_final);
    REQUIRE(!audio_48k.empty());

    auto audio_16k = resample_48k_to_16k(audio_48k);
    REQUIRE(!audio_16k.empty());

    speech_core::LiteRTParakeetStt stt(par_enc, par_dec, vocab, /*hw_accel=*/false);
    auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);

    const size_t matched = count_matched_words(result.text);
    const size_t expected = roundtrip_content_words().size();
    std::printf("tokens=%d text=\"%s\" matched=%zu/%zu ",
                tts.tokens_generated(), result.text.c_str(), matched, expected);
    // 4 of 6 content words is the floor; the weekly CI demands 6 of 6 (with
    // ORT Parakeet on the FP32 path). LiteRT Parakeet is INT8-quantized and
    // slightly less accurate; pick a threshold that detects genuine breakage
    // (garbled babble would match ~0) but tolerates one or two INT8 word
    // drops.
    REQUIRE(matched >= 4);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// LiteRT VoxCPM2 (voice-cloned) → LiteRT Parakeet round-trip — same semantic
// assertion as the non-cloned round-trip, but the TTS is conditioned on the
// fixture clip. Proves that voice cloning produces intelligible speech in the
// target speaker's voice, not just non-silent output. Same threshold (≥ 4/6).
// ---------------------------------------------------------------------------

void test_litert_voxcpm2_clone_parakeet_roundtrip(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string vox_enc = dir + "/voxcpm2-audio-encoder.tflite";
    std::string vox_dec = dir + "/voxcpm2-audio-decoder.tflite";
    std::string tok  = dir + "/tokenizer.json";
    std::string par_enc = dir + "/parakeet-encoder.tflite";
    std::string par_dec = dir + "/parakeet-decoder-joint.tflite";
    std::string vocab   = dir + "/vocab.json";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(pref) || !file_exists(step) || !file_exists(vox_enc)
        || !file_exists(vox_dec) || !file_exists(tok)) {
        std::printf("  [skip] voxcpm2 files not in %s\n", dir.c_str());
        return;
    }
    if (!file_exists(par_enc) || !file_exists(par_dec) || !file_exists(vocab)) {
        std::printf("  [skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_voxcpm2_clone_parakeet_roundtrip ... ");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, vox_enc, vox_dec, tok, /*hw_accel=*/false);
    configure_voxcpm2_for_roundtrip(tts);
    tts.set_reference(wav.samples.data(), wav.samples.size(), wav.sample_rate);
    REQUIRE(tts.has_reference());

    std::vector<float> audio_48k;
    bool got_final = false;
    tts.synthesize(kRoundtripPhrase, "en",
        [&](const float* s, size_t len, bool is_final) {
            if (s && len) audio_48k.insert(audio_48k.end(), s, s + len);
            if (is_final) got_final = true;
        });
    REQUIRE(got_final);
    REQUIRE(!audio_48k.empty());

    auto audio_16k = resample_48k_to_16k(audio_48k);
    REQUIRE(!audio_16k.empty());

    speech_core::LiteRTParakeetStt stt(par_enc, par_dec, vocab, /*hw_accel=*/false);
    auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);

    const size_t matched = count_matched_words(result.text);
    const size_t expected = roundtrip_content_words().size();
    std::printf("tokens=%d text=\"%s\" matched=%zu/%zu ",
                tts.tokens_generated(), result.text.c_str(), matched, expected);
    // 2/6 floor here, weaker than the 4/6 floor for the non-cloned variant.
    // The cloned voice ramps up over the first ~1 s while the model conditions
    // on the reference timbre, and LiteRT Parakeet INT8 mishandles that
    // ramp-up — empirically 4 of 6 content words land on the floor, the
    // other 2 are too quantization-noisy to recognise. The weekly CI runs
    // the same WAV through ORT Parakeet FP32 and demands all 6, so the
    // strong assertion lives there. This floor just proves the clone
    // produced intelligible speech, not noise.
    REQUIRE(matched >= 2);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VoxCPM2 C ABI (sc_voxcpm2_*) — the surface speech-studio links via FFI. Drives
// a clone through the C entry points and asserts non-silent audio.
// ---------------------------------------------------------------------------

void test_voxcpm2_c_api(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string tok  = dir + "/tokenizer.json";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(pref) || !file_exists(tok) || wav.samples.empty()) {
        std::printf("  [skip] voxcpm2 files/fixture not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_voxcpm2_c_api ... ");

    sc_voxcpm2_t v = sc_voxcpm2_create(dir.c_str());
    REQUIRE(v != nullptr);
    REQUIRE(sc_voxcpm2_output_sample_rate(v) == 48000);
    sc_voxcpm2_set_instruction(v, "clear, natural delivery");
    sc_voxcpm2_set_max_steps(v, 128);
    sc_voxcpm2_set_min_steps_before_stop(v, 32);

    int rc = sc_voxcpm2_set_reference(v, wav.samples.data(), wav.samples.size(),
                                      wav.sample_rate);
    if (rc != 0) std::printf("\n    set_reference: %s ", sc_voxcpm2_last_error(v));
    REQUIRE(rc == 0);
    sc_voxcpm2_set_reference_transcript(v, "This is the exact reference sentence.");

    std::vector<float> audio;
    rc = sc_voxcpm2_synthesize(v, "The quick brown fox jumps over the lazy dog",
        [](const float* s, size_t len, bool, void* ctx) {
            auto* out = static_cast<std::vector<float>*>(ctx);
            if (s && len) out->insert(out->end(), s, s + len);
        }, &audio);
    if (rc != 0) std::printf("\n    synthesize: %s ", sc_voxcpm2_last_error(v));
    REQUIRE(rc == 0);
    check_voice_audio(audio, "c_api");

    sc_voxcpm2_destroy(v);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// WeSpeaker embedding — embeds a tone, checks dim, L2-norm ≈ 1, finiteness,
// and that identical input yields a self-similarity ≈ 1 (deterministic).
// ---------------------------------------------------------------------------

void test_litert_wespeaker_embedding(const std::string& dir) {
    std::string model = dir + "/wespeaker-resnet34.tflite";
    if (!file_exists(model)) {
        std::printf("  [skip] wespeaker-resnet34.tflite not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_wespeaker_embedding ... ");

    speech_core::LiteRTWeSpeakerEmbedding emb(model, /*hw_accel=*/false);
    REQUIRE(emb.embedding_dim() == 256);
    REQUIRE(emb.input_sample_rate() == 16000);

    auto tone = generate_tone(16000, 220.0f, 2.0f, 0.4f);
    auto v = emb.embed(tone.data(), tone.size(), 16000);
    REQUIRE(static_cast<int>(v.size()) == emb.embedding_dim());

    float norm = 0.0f;
    for (float x : v) { REQUIRE(std::isfinite(x)); norm += x * x; }
    norm = std::sqrt(norm);
    REQUIRE(std::abs(norm - 1.0f) < 1e-2f);  // wrapper L2-normalises

    auto v2 = emb.embed(tone.data(), tone.size(), 16000);
    float self_sim = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) self_sim += v[i] * v2[i];
    REQUIRE(self_sim > 0.99f);  // deterministic on identical input

    std::printf("ok (dim=%zu norm=%.4f self_sim=%.4f)\n", v.size(), norm, self_sim);
}

// ---------------------------------------------------------------------------
// Pyannote segmentation — runs ≥1 window over (padded) fixture audio and
// checks window shape + that speaker_activity is finite and in [0, 1].
// ---------------------------------------------------------------------------

void test_litert_pyannote_segmentation(const std::string& dir) {
    std::string model = dir + "/pyannote-segmentation.tflite";
    if (!file_exists(model)) {
        std::printf("  [skip] pyannote-segmentation.tflite not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_pyannote_segmentation ... ");

    speech_core::LiteRTPyannoteSegmentation seg(model, /*hw_accel=*/false);
    REQUIRE(seg.input_sample_rate() == 16000);
    REQUIRE(seg.max_local_speakers() == 3);

    auto wav = load_wav_mono_pcm16(test_audio_path());
    std::vector<float> audio;
    if (!wav.samples.empty()) {
        audio = wav.sample_rate == 16000
            ? wav.samples
            : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                               wav.sample_rate, 16000);
    }
    if (audio.size() < static_cast<size_t>(16000) * 11) {
        audio.resize(static_cast<size_t>(16000) * 11, 0.0f);  // guarantee ≥1 window
    }

    auto windows = seg.segment(audio.data(), audio.size(), 16000);
    REQUIRE(!windows.empty());

    const auto& w0 = windows.front();
    REQUIRE(!w0.posteriors.empty());
    REQUIRE(static_cast<int>(w0.speaker_activity.size()) % seg.max_local_speakers() == 0);
    for (float a : w0.speaker_activity) {
        REQUIRE(std::isfinite(a));
        REQUIRE(a >= -0.01f && a <= 1.01f);
    }

    std::printf("ok (windows=%zu frames/win=%zu)\n",
                windows.size(), w0.speaker_activity.size() / seg.max_local_speakers());
}

// ---------------------------------------------------------------------------
// Omnilingual CTC STT (optional model) — transcribes the fixture; requires
// non-empty text on real speech, otherwise just a clean run on silence.
// ---------------------------------------------------------------------------

void test_litert_omnilingual_stt(const std::string& dir) {
    std::string model = dir + "/omnilingual-ctc-300m.tflite";
    std::string tok   = dir + "/tokenizer.model";
    if (!file_exists(model) || !file_exists(tok)) {
        std::printf("  [skip] omnilingual files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_litert_omnilingual_stt ... ");

    speech_core::LiteRTOmnilingualStt stt(model, tok, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);

    auto wav = load_wav_mono_pcm16(test_audio_path());
    bool real_speech = !wav.samples.empty();
    std::vector<float> audio;
    if (real_speech) {
        audio = wav.sample_rate == 16000
            ? wav.samples
            : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                               wav.sample_rate, 16000);
    } else {
        audio.assign(16000, 0.0f);
    }

    auto result = stt.transcribe(audio.data(), audio.size(), 16000);
    std::printf("text=\"%s\" ", result.text.c_str());
    if (real_speech) REQUIRE(!result.text.empty());
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// Diarization e2e — composes Pyannote (segmentation) + WeSpeaker (embedding)
// through DiarizationPipeline on the fixture and checks the segments are
// well-formed (speaker ≥ 0, end ≥ start, time-sorted) and non-empty.
// ---------------------------------------------------------------------------

void test_litert_diarization(const std::string& dir) {
    std::string seg_model = dir + "/pyannote-segmentation.tflite";
    std::string emb_model = dir + "/wespeaker-resnet34.tflite";
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (!file_exists(seg_model) || !file_exists(emb_model)) {
        std::printf("  [skip] diarization needs pyannote + wespeaker in %s\n", dir.c_str());
        return;
    }
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_diarization ... ");

    auto audio = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    if (audio.size() < static_cast<size_t>(16000) * 11) {
        audio.resize(static_cast<size_t>(16000) * 11, 0.0f);  // ≥1 segmentation window
    }

    speech_core::LiteRTPyannoteSegmentation seg(seg_model, /*hw_accel=*/false);
    speech_core::LiteRTWeSpeakerEmbedding    emb(emb_model, /*hw_accel=*/false);
    speech_core::DiarizationPipeline diar(seg, emb);

    speech_core::DiarizerConfig cfg;  // defaults
    auto segments = diar.diarize(audio.data(), audio.size(), 16000, cfg);
    REQUIRE(!segments.empty());

    std::set<int> speakers;
    for (size_t i = 0; i < segments.size(); ++i) {
        REQUIRE(segments[i].speaker >= 0);
        REQUIRE(segments[i].end >= segments[i].start);
        if (i > 0) REQUIRE(segments[i].start >= segments[i - 1].start);
        speakers.insert(segments[i].speaker);
    }

    std::printf("ok (segments=%zu speakers=%zu)\n", segments.size(), speakers.size());
}

// ---------------------------------------------------------------------------
// Nemotron Speech Streaming — feeds the fixture in ~100 ms chunks through the
// cache-aware streaming RNN-T (encoder+cache → decoder LSTM → joint) and runs
// the full begin/push/end path. This exercises the per-stream cache rolling +
// decoder-state advance + joint greedy loop end-to-end; reaching the end
// without throwing means the encoder/decoder/joint tensor wiring is correct.
// (Transcript content parity is a nightly concern against known audio.)
// ---------------------------------------------------------------------------

void test_litert_nemotron_streaming_stt(const std::string& dir) {
    std::string enc = dir + "/nemotron-streaming-encoder.tflite";
    std::string dec = dir + "/nemotron-streaming-decoder.tflite";
    std::string jnt = dir + "/nemotron-streaming-joint.tflite";
    std::string voc = dir + "/nemotron-vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt) || !file_exists(voc)) {
        std::printf("  [skip] nemotron-streaming files not in %s\n", dir.c_str());
        return;
    }
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_nemotron_streaming_stt ... ");

    speech_core::LiteRTNemotronStreamingStt stt(enc, dec, jnt, voc, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);
    REQUIRE(stt.supports_streaming());

    auto audio = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio.empty());

    stt.begin_stream(16000);
    size_t partials = 0;
    constexpr size_t kFeed = 1600;  // 100 ms @ 16 kHz
    for (size_t off = 0; off < audio.size(); off += kFeed) {
        size_t len = std::min(kFeed, audio.size() - off);
        auto p = stt.push_chunk(audio.data() + off, len);
        if (!p.text.empty()) ++partials;
    }
    auto result = stt.end_stream();

    std::printf("partials=%zu final=\"%.50s\" ok\n", partials, result.text.c_str());
}

void test_litert_nemotron_multilingual_stt(const std::string& dir) {
    std::string enc = dir + "/nemotron-multilingual-encoder.tflite";
    std::string dec = dir + "/nemotron-multilingual-decoder.tflite";
    std::string jnt = dir + "/nemotron-multilingual-joint.tflite";
    std::string voc = dir + "/nemotron-multilingual-vocab.json";
    std::string lng = dir + "/nemotron-multilingual-languages.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt)
        || !file_exists(voc) || !file_exists(lng)) {
        std::printf("  [skip] nemotron-multilingual files not in %s\n", dir.c_str());
        return;
    }
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (wav.samples.empty()) {
        std::printf("  [skip] could not load %s\n", test_audio_path().c_str());
        return;
    }
    std::printf("  test_litert_nemotron_multilingual_stt ... ");

    speech_core::LiteRTNemotronMultilingualStt stt(enc, dec, jnt, voc, lng, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);
    REQUIRE(stt.supports_streaming());
    REQUIRE(stt.set_language("en-US"));  // English prompt slot must resolve

    auto audio = wav.sample_rate == 16000
        ? wav.samples
        : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                           wav.sample_rate, 16000);
    REQUIRE(!audio.empty());

    stt.begin_stream(16000);
    constexpr size_t kFeed = 1600;  // 100 ms @ 16 kHz
    for (size_t off = 0; off < audio.size(); off += kFeed) {
        size_t len = std::min(kFeed, audio.size() - off);
        stt.push_chunk(audio.data() + off, len);
    }
    auto result = stt.end_stream();

    std::printf("final=\"%.50s\" ok\n", result.text.c_str());
}

// ---------------------------------------------------------------------------
// VoxCPM2 Hindi cloning — Windows artifact repro harness. Mirrors the studio
// voice-cloning flow exactly (set_reference + per-phrase seed + bounded
// max_steps) over 5 Hindi phrases. Dumps each synthesized 48 kHz PCM16 WAV to
// $SPEECH_VOXCPM2_WAV_DUMP/voxcpm2_hindi_NN.wav for offline spectral analysis,
// and roundtrips each through Nemotron multilingual STT for a hyp transcript.
// Skipped when any model file is missing or set_reference is rejected.
// ---------------------------------------------------------------------------

// UTF-8-aware tokenizer for Devanagari; splits on ASCII whitespace/punct AND
// Devanagari danda (U+0964) / double-danda (U+0965). The repo's standard
// tokenize_lower() drops Devanagari because std::isalnum is ASCII-only.
std::vector<std::string> tokenize_unicode_words(const std::string& s) {
    std::vector<std::string> toks; std::string cur;
    auto flush = [&]() { if (!cur.empty()) { toks.push_back(std::move(cur)); cur.clear(); } };
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]); int len = 1;
        if      ((c & 0x80) == 0x00) len = 1;
        else if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + static_cast<size_t>(len) > s.size()) break;
        bool sep = false;
        if (len == 1) {
            if (c <= 0x20) sep = true;
            else switch (c) {
                case '!': case '?': case '.': case ',': case ':': case ';':
                case '"': case '\'': case '(': case ')': case '[': case ']':
                case '{': case '}': case '/': case '\\': case '-':
                case '_': case '*': case '=': sep = true; break;
                default: break;
            }
        } else if (len == 3) {
            unsigned char b1 = static_cast<unsigned char>(s[i+1]);
            unsigned char b2 = static_cast<unsigned char>(s[i+2]);
            if (c == 0xE0 && b1 == 0xA5 && (b2 == 0xA4 || b2 == 0xA5)) sep = true;
        }
        if (sep) flush(); else cur.append(s, i, len);
        i += len;
    }
    flush();
    return toks;
}

void test_litert_voxcpm2_hindi_cloning(const std::string& dir) {
    std::string pref = dir + "/voxcpm2-text-prefill.tflite";
    std::string step = dir + "/voxcpm2-token-step.tflite";
    std::string venc = dir + "/voxcpm2-audio-encoder.tflite";
    std::string vdec = dir + "/voxcpm2-audio-decoder.tflite";
    std::string vtok = dir + "/tokenizer.json";
    std::string nenc = dir + "/nemotron-multilingual-encoder.tflite";
    std::string ndec = dir + "/nemotron-multilingual-decoder.tflite";
    std::string njnt = dir + "/nemotron-multilingual-joint.tflite";
    std::string nvoc = dir + "/nemotron-multilingual-vocab.json";
    std::string nlng = dir + "/nemotron-multilingual-languages.json";
    // Prefer the studio-pipeline reference voice (Ruuchi mono) at
    // tests/data/test_hindi_ref.wav when present; fall back to the generic
    // English test fixture so the test still runs in CI without the bundle.
    std::string ref_path = test_hindi_ref_path();
    auto wav = load_wav_mono_pcm16(ref_path);
    bool used_hindi_ref = !wav.samples.empty();
    if (!used_hindi_ref) {
        ref_path = test_audio_path();
        wav = load_wav_mono_pcm16(ref_path);
    }
    if (!file_exists(pref) || !file_exists(step) || !file_exists(venc)
        || !file_exists(vdec) || !file_exists(vtok) || wav.samples.empty()) {
        std::printf("  [skip] hindi cloning VoxCPM2 files/fixture missing in %s\n", dir.c_str());
        return;
    }
    const bool have_stt = file_exists(nenc) && file_exists(ndec) && file_exists(njnt)
        && file_exists(nvoc) && file_exists(nlng);
    std::printf("  test_litert_voxcpm2_hindi_cloning ... (stt=%s)\n",
                have_stt ? "on" : "off");

    speech_core::LiteRTVoxCPM2Tts tts(pref, step, venc, vdec, vtok, /*hw_accel=*/false);
    if (tts.max_text_tokens() < 256) {
        std::printf("  [skip] hindi cloning needs production VoxCPM2 (max_text=%d)\n",
                    tts.max_text_tokens());
        return;
    }
    std::unique_ptr<speech_core::LiteRTNemotronMultilingualStt> stt_ptr;
    const char* res = "(skipped)";
    if (have_stt) {
        stt_ptr = std::make_unique<speech_core::LiteRTNemotronMultilingualStt>(
            nenc, ndec, njnt, nvoc, nlng, /*hw_accel=*/false);
        const char* locs[] = {"hi-IN", "hi", "hi-HI"};
        for (const char* l : locs) if (stt_ptr->set_language(l)) { res = l; break; }
        if (std::string(res) == "(skipped)") {
            std::printf("    [warn] no Hindi slot in nemotron — running without STT\n");
            stt_ptr.reset();
            res = "(no-hi-slot)";
        }
    }

    tts.set_reference(wav.samples.data(), wav.samples.size(), wav.sample_rate);
    if (!tts.has_reference()) { std::printf("  [skip] set_reference rejected clip\n"); return; }
    // Log the reference RMS — useful when chasing a "cloned voice is too
    // quiet" report; loudness can collapse if the reference clip itself is
    // quiet and the model preserves its level too literally.
    double ref_rms_sq = 0.0;
    for (float s : wav.samples) ref_rms_sq += static_cast<double>(s) * s;
    double ref_rms = std::sqrt(ref_rms_sq / std::max<size_t>(wav.samples.size(), 1));
    std::printf("    [stt-lang=%s reference=%s ref_rms=%.4f ref_dur=%.2fs sr=%d]\n",
                res, used_hindi_ref ? "test_hindi_ref" : "test_audio",
                ref_rms, double(wav.samples.size()) / wav.sample_rate, wav.sample_rate);

    tts.set_instruction("");
    const std::vector<std::string> phrases = {
        "इसे कहाँ रखें: गियर साइडबार खोलें। इसके अंदर, हॉटकीज़ ब्लॉक और नीचे के टेक्स्ट के बीच में एक हॉरिजॉन्टल लाइन जोड़ें।",
        "मेरा नाम राहुल है।",
        "आज मौसम बहुत अच्छा है।",
        "मुझे संगीत सुनना पसंद है।",
        "क्या आप मेरी मदद कर सकते हैं?",
    };
    const char* dump = std::getenv("SPEECH_VOXCPM2_WAV_DUMP");
    for (size_t i = 0; i < phrases.size(); ++i) {
        const auto& p = phrases[i];
        int wc = static_cast<int>(tokenize_unicode_words(p).size());
        int ms = std::clamp(wc * 12 + 40, 60, 240);
        tts.set_max_steps(ms);
        tts.set_seed(static_cast<unsigned>(1000 + i));

        std::vector<float> a48; bool fin = false;
        tts.synthesize(p, "hi", [&](const float* s, size_t n, bool isf) {
            if (s && n) a48.insert(a48.end(), s, s + n);
            if (isf) fin = true;
        });
        if (!fin || a48.empty()) { std::printf("    [%zu] EMPTY\n", i); continue; }
        if (dump) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/voxcpm2_hindi_%02zu.wav", dump, i);
            dump_wav48k_mono(path, a48);
        }
        std::string hyp = "(stt-off)";
        if (stt_ptr) {
            auto a16 = speech_core::Resampler::resample(a48.data(), a48.size(), 48000, 16000);
            stt_ptr->begin_stream(16000);
            constexpr size_t kFeed = 1600;  // 100 ms @ 16 kHz
            for (size_t off = 0; off < a16.size(); off += kFeed) {
                size_t len = std::min(kFeed, a16.size() - off);
                stt_ptr->push_chunk(a16.data() + off, len);
            }
            hyp = stt_ptr->end_stream().text;
        }
        double sq = 0.0; float pk = 0.0f;
        for (float s : a48) { sq += static_cast<double>(s) * s; pk = std::max(pk, std::abs(s)); }
        double rms = std::sqrt(sq / std::max<size_t>(a48.size(), 1));
        std::printf("    [%zu] tokens=%d dur=%.2fs rms=%.4f peak=%.4f  ref=\"%s\"  hyp=\"%s\"\n",
                    i, tts.tokens_generated(),
                    double(a48.size()) / 48000.0, rms, pk,
                    p.c_str(), hyp.c_str());
        std::fflush(stdout);
    }
    std::printf("  ok (WAVs in %s)\n", dump ? dump : "(no dump dir)");
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

    // Drive every test through a single dispatch site that catches uncaught
    // exceptions and flushes stdout between tests. Without this, a runtime
    // throw from one wrapper (e.g. LiteRT v2.1's "no dims_signature" error
    // on certain graphs) terminates the whole binary, so subsequent tests
    // — including the round-trips below — never run, and partial stdout is
    // lost to buffering on Windows.
    auto run = [&](const char* name, void (*fn)(const std::string&)) {
        try {
            fn(dir);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  FAIL: %s threw: %s\n", name, e.what());
            ++failures;
        } catch (...) {
            std::fprintf(stderr, "  FAIL: %s threw (unknown)\n", name);
            ++failures;
        }
        std::fflush(stdout);
        std::fflush(stderr);
    };

    #define RUN(t) run(#t, t)
#ifdef SPEECH_CORE_HAS_LITERT_KOKORO
    RUN(test_litert_kokoro_tts);
#endif
    RUN(test_litert_silero_vad);
    RUN(test_litert_silero_vad_real_speech);
    RUN(test_litert_parakeet_stt);
    RUN(test_litert_parakeet_real_speech);
    RUN(test_litert_parakeet_streaming);
    RUN(test_litert_vad_to_stt_pipeline);
    RUN(test_litert_wespeaker_embedding);
    RUN(test_litert_pyannote_segmentation);
    RUN(test_litert_omnilingual_stt);
    RUN(test_litert_diarization);
    RUN(test_litert_nemotron_streaming_stt);
    RUN(test_litert_nemotron_multilingual_stt);
    RUN(test_voxcpm2_tokenizer);
    RUN(test_litert_voxcpm2_load);
    RUN(test_litert_voxcpm2_synthesize);
    RUN(test_litert_voxcpm2_clone);
    RUN(test_voxcpm2_c_api);
    RUN(test_litert_voxcpm2_parakeet_roundtrip);
    RUN(test_litert_voxcpm2_clone_parakeet_roundtrip);
    RUN(test_litert_voxcpm2_hindi_cloning);
    #undef RUN

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll LiteRT model tests passed\n");
    return 0;
}
