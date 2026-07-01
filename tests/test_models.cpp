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
#include "speech_core/models/onnx_engine.h"
#include "speech_core/models/onnx_cosyvoice3_tts.h"
#include "speech_core/models/onnx_whisper_stt.h"
#include "speech_core/models/onnx_sidon_restorer.h"
#include "speech_core/models/onnx_voxcpm2_tts.h"
#include "speech_core/models/onnx_personaplex.h"
#include "speech_core/models/parakeet_stt.h"
#include "speech_core/models/nemotron_multilingual_stt.h"
#include "speech_core/models/silero_vad.h"
#include "speech_core/vad/streaming_vad.h"

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
#include <string>
#include <utility>
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

struct OnnxVoxCPM2Bundle {
    std::string dir;
    std::string prefill;
    std::string token_step;
    std::string decoder;
    std::string audio_encoder;
    std::string audio_decoder;
    std::string tokenizer;

    bool has_split() const {
        return file_exists(prefill) && file_exists(token_step);
    }

    bool has_unified() const {
        return file_exists(decoder);
    }

    bool complete() const {
        return (has_split() || has_unified())
            && file_exists(audio_encoder)
            && file_exists(audio_decoder)
            && file_exists(tokenizer);
    }
};

OnnxVoxCPM2Bundle onnx_voxcpm2_bundle() {
    // The VoxCPM2 ONNX bundle is independent of $SPEECH_MODEL_DIR — it lives
    // alongside the speech-models export workspace. Tries a sensible default
    // and falls back to the env var $SPEECH_VOXCPM2_ONNX_DIR for CI flexibility.
    const char* override_dir = std::getenv("SPEECH_VOXCPM2_ONNX_DIR");
    std::string vox_dir = override_dir ? override_dir : "/tmp/voxcpm2-onnx";
    return {
        vox_dir,
        vox_dir + "/voxcpm2-text-prefill.onnx",
        vox_dir + "/voxcpm2-token-step.onnx",
        vox_dir + "/voxcpm2-decoder.onnx",
        vox_dir + "/voxcpm2-audio-encoder.onnx",
        vox_dir + "/voxcpm2-audio-decoder.onnx",
        vox_dir + "/tokenizer.json",
    };
}

std::unique_ptr<speech_core::OnnxVoxCPM2Tts>
make_onnx_voxcpm2_tts(const OnnxVoxCPM2Bundle& bundle, bool hw_accel) {
    if (bundle.has_split()) {
        return std::make_unique<speech_core::OnnxVoxCPM2Tts>(
            bundle.prefill, bundle.token_step,
            bundle.audio_encoder, bundle.audio_decoder,
            bundle.tokenizer, hw_accel);
    }
    return std::make_unique<speech_core::OnnxVoxCPM2Tts>(
        bundle.decoder,
        bundle.audio_encoder, bundle.audio_decoder,
        bundle.tokenizer, hw_accel);
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
        out[i] = amp * std::sin(2.0f * kPi * freq * static_cast<float>(i) / static_cast<float>(sample_rate));
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

void test_onnx_whisper_stt(const std::string& dir) {
    const char* override_dir = std::getenv("SPEECH_WHISPER_ONNX_DIR");
    std::string root = override_dir ? override_dir : dir;

    struct Candidate { const char* prefix; const char* suffix; };
    const Candidate candidates[] = {
        {"turbo", ".int8"},
        {"large-v3", ".int8"},
        {"medium", ".int8"},
        {"small", ".int8"},
        {"turbo", ".fp16"},
        {"large-v3", ".fp16"},
        {"medium", ".fp16"},
        {"small", ".fp16"},
        {"turbo", ""},
        {"large-v3", ""},
        {"medium", ""},
        {"small", ""},
    };

    std::string enc;
    std::string dec;
    std::string tok;
    for (const auto& c : candidates) {
        std::string e = root + "/" + c.prefix + "-encoder" + c.suffix + ".onnx";
        std::string d = root + "/" + c.prefix + "-decoder" + c.suffix + ".onnx";
        std::string t = root + "/" + c.prefix + "-tokens.txt";
        if (file_exists(e) && file_exists(d) && file_exists(t)) {
            enc = std::move(e);
            dec = std::move(d);
            tok = std::move(t);
            break;
        }
    }
    if (enc.empty()) {
        std::printf("  [skip] whisper ONNX bundle not in %s\n", root.c_str());
        return;
    }

    std::printf("  test_onnx_whisper_stt ... ");
    speech_core::OnnxWhisperStt::Config cfg;
    cfg.language = "en";
    cfg.max_decode_tokens = 4;
    cfg.tail_padding_frames = 50;
    speech_core::OnnxWhisperStt stt(enc, dec, tok, cfg, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);

    std::vector<float> silence(16000, 0.0f);
    auto result = stt.transcribe(silence.data(), silence.size(), 16000);
    auto profile = stt.last_profile();
    REQUIRE(result.confidence >= 0.0f && result.confidence <= 1.0f);
    REQUIRE(result.language.empty() || result.language == "en");
    REQUIRE(profile.chunks == 1);
    REQUIRE(profile.total_ms >= 0.0);
    REQUIRE(profile.feature_frames > 0);

    std::printf("ok (silence text=\"%.40s\" lang=%s)\n",
                result.text.c_str(), result.language.c_str());
}

// ---------------------------------------------------------------------------

void test_nemotron_multilingual_stt(const std::string& dir) {
    std::string enc = dir + "/nemotron-multilingual-encoder.onnx";
    std::string dec = dir + "/nemotron-multilingual-decoder.onnx";
    std::string jnt = dir + "/nemotron-multilingual-joint.onnx";
    std::string voc = dir + "/nemotron-multilingual-vocab.json";
    std::string lng = dir + "/nemotron-multilingual-languages.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt)
        || !file_exists(voc) || !file_exists(lng)) {
        std::printf("  [skip] nemotron-multilingual files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_nemotron_multilingual_stt ... ");

    speech_core::NemotronMultilingualStt stt(enc, dec, jnt, voc, lng, /*hw_accel=*/false);
    REQUIRE(stt.input_sample_rate() == 16000);
    REQUIRE(stt.supports_streaming());
    REQUIRE(stt.set_language("en-US"));  // English prompt slot must resolve

    // Pure silence → no crash, valid (possibly empty) transcription.
    std::vector<float> silence(16000 * 1, 0.0f);
    auto result = stt.transcribe(silence.data(), silence.size(), 16000);
    REQUIRE(result.confidence >= 0.0f && result.confidence <= 1.0f);

    std::printf("ok (silence text=\"%.40s\")\n", result.text.c_str());
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
        noisy[i] = 0.05f * std::sin(2.0f * kPi * 440.0f * static_cast<float>(i) / 48000.0f);
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

void test_sidon_restorer(const std::string& dir) {
    std::string predictor = dir + "/sidon-predictor.onnx";
    std::string vocoder = dir + "/sidon-vocoder.onnx";
    if (!file_exists(predictor) || !file_exists(vocoder)) {
        std::printf("  [skip] sidon files not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_sidon_restorer ... ");

    speech_core::OnnxSidonRestorer rest(predictor, vocoder, /*hw_accel=*/false);
    REQUIRE(rest.input_sample_rate() == 16000);
    REQUIRE(rest.output_sample_rate() == 48000);

    // ~1 second of a tone mix at 16 kHz (the model's input rate).
    std::vector<float> in(16000);
    for (size_t i = 0; i < in.size(); ++i) {
        const float t = static_cast<float>(i) / 16000.0f;
        in[i] = 0.4f * std::sin(2.0f * kPi * 220.0f * t)
              + 0.2f * std::sin(2.0f * kPi * 880.0f * t);
    }

    std::vector<float> out = rest.restore(in.data(), in.size(), 16000);

    // Restoration upsamples 16 kHz -> 48 kHz, so expect ~3x the samples.
    REQUIRE(!out.empty());
    REQUIRE(out.size() > in.size() * 2);

    bool has_signal = false, has_nan = false;
    for (float v : out) {
        if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; }
        if (std::abs(v) > 1e-6f) has_signal = true;
    }
    REQUIRE(!has_nan);
    REQUIRE(has_signal);

    // --- 48 kHz output-length contract ---------------------------------------
    // The DAC vocoder upsamples by a fixed ×960 over the (×2-stacked) frame
    // count, and the SeamlessM4T front-end emits T = floor(num_frames/2) with
    // num_frames = 1 + (samples-400)/160. So output samples ≈ T * 960. For
    // 16000 input samples: num_frames = 98, T = 49, expected ≈ 47040. Allow a
    // small slack for the vocoder's edge handling.
    {
        const int num_frames = 1 + static_cast<int>((in.size() - 400) / 160);
        const int Texp = num_frames / 2;
        const size_t expected = static_cast<size_t>(Texp) * 960;
        const double ratio = static_cast<double>(out.size()) / expected;
        REQUIRE(ratio > 0.95 && ratio < 1.05);
    }

    // --- resample-to-16k path -------------------------------------------------
    // Feed the SAME audio resampled to 48 kHz. The restorer must resample down
    // to its 16 kHz front-end internally and produce a comparable 48 kHz length
    // (same underlying ~1 s of content -> same frame count within slack).
    {
        auto in_48k = speech_core::Resampler::resample(
            in.data(), in.size(), 16000, 48000);
        REQUIRE(!in_48k.empty());
        auto out_48in = rest.restore(in_48k.data(), in_48k.size(), 48000);
        REQUIRE(!out_48in.empty());
        const double ratio = static_cast<double>(out_48in.size()) / out.size();
        REQUIRE(ratio > 0.95 && ratio < 1.05);
        for (float v : out_48in) REQUIRE(std::isfinite(v));
    }

    // --- error / degenerate-input handling -----------------------------------
    // Null pointer and zero length must return empty, never crash.
    REQUIRE(rest.restore(nullptr, 0, 16000).empty());
    REQUIRE(rest.restore(in.data(), 0, 16000).empty());
    // A clip shorter than one analysis frame yields no features -> empty.
    {
        std::vector<float> tiny(200, 0.1f);
        REQUIRE(rest.restore(tiny.data(), tiny.size(), 16000).empty());
    }

    // --- EnhancerInterface adapter: equal-rate, equal-length, in-place --------
    // as_enhancer() resamples the 48 kHz restoration back down to the caller's
    // rate and writes EXACTLY `length` samples (truncate / zero-pad). Contract:
    // input_sample_rate() == 16000, output buffer fully written, finite values.
    {
        auto enh = rest.as_enhancer();
        REQUIRE(enh != nullptr);
        REQUIRE(enh->input_sample_rate() == 16000);

        std::vector<float> adapted(in.size(), -42.0f);  // sentinel
        enh->enhance(in.data(), in.size(), 16000, adapted.data());
        // Every sample must have been written (no sentinel survivors) and finite.
        bool any_sentinel = false, adapter_signal = false;
        for (float v : adapted) {
            REQUIRE(std::isfinite(v));
            if (v == -42.0f) any_sentinel = true;
            if (std::abs(v) > 1e-6f) adapter_signal = true;
        }
        REQUIRE(!any_sentinel);
        REQUIRE(adapter_signal);
    }

    std::printf("ok (%zu in -> %zu out @ 48k)\n", in.size(), out.size());
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

void test_onnx_engine_provider_resolution(const std::string& /*dir*/) {
    std::printf("  test_onnx_engine_provider_resolution ... ");
    auto& engine = OnnxEngine::get();
    REQUIRE(engine.api() != nullptr);
    REQUIRE(engine.has_gpu_provider() == false);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VoxCPM2 tokenizer parity — independent of the ONNX graphs, but lives in the
// ONNX test binary because speech_core_models owns the shared tokenizer in
// ONNX-only builds.
// ---------------------------------------------------------------------------

void test_voxcpm2_tokenizer(const std::string& /*dir*/) {
    OnnxVoxCPM2Bundle bundle = onnx_voxcpm2_bundle();
    if (!file_exists(bundle.tokenizer)) {
        std::printf("  [skip] tokenizer.json not in %s "
                    "(set SPEECH_VOXCPM2_ONNX_DIR to override)\n",
                    bundle.dir.c_str());
        return;
    }
    REQUIRE(speech_core_test::run_voxcpm2_tokenizer_reference_cases(bundle.tokenizer));
}

// ---------------------------------------------------------------------------
// ONNX VoxCPM2 load smoke — constructs the wrapper from the random-init
// exported graphs at /tmp/voxcpm2-onnx/ (the tiny-init bundle from
// speech-models, used until the real-weights text_prefill export lands).
// Skips cleanly when the bundle isn't present so CI stays green.
// ---------------------------------------------------------------------------

void test_onnx_voxcpm2_load(const std::string& /*dir*/) {
    OnnxVoxCPM2Bundle bundle = onnx_voxcpm2_bundle();
    if (!bundle.complete()) {
        std::printf("  [skip] ONNX VoxCPM2 bundle not in %s "
                    "(set SPEECH_VOXCPM2_ONNX_DIR to override)\n",
                    bundle.dir.c_str());
        return;
    }
    std::printf("  test_onnx_voxcpm2_load [%s] ... ",
                bundle.has_split() ? "split" : "unified");

    // Phase 1 (always): construct + introspect. Confirms all four ORT
    // sessions load, the tokenizer parses, and the I/O introspection paths
    // don't throw.
    auto tts = make_onnx_voxcpm2_tts(bundle, /*hw_accel=*/false);
    REQUIRE(tts->output_sample_rate() == 48000);
    REQUIRE(tts->max_text_tokens() > 0);
    REQUIRE(!tts->has_reference());

    // Phase 2 (production-config only): exercise the runtime audio paths
    // end-to-end with set_reference + a few AR steps + at least one
    // audio_decoder Run. We gate on max_text_tokens() >= 256 because the
    // tiny shape-validation bundle from speech-models exports with
    // max_text=32 (kFeatDim=16, kHidden=128) — those tensor dims would
    // mismatch the C++ wrapper's production constants (kFeatDim=64,
    // kPatchSize=4, kHidden=2048; see litert_voxcpm2_tts.h:139-160).
    // A real-weights export from openbmb/VoxCPM2 keeps max_text=512.
    if (tts->max_text_tokens() < 256) {
        std::printf("ok (max_text=%d, tiny config — skip runtime audio test)\n",
                    tts->max_text_tokens());
        return;
    }
    // Encoder Run: short 220 Hz ramp as reference clip.
    std::vector<float> ref(16000 * 2, 0.0f);
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] = 0.05f * std::sin(2.0f * kPi * 220.0f * static_cast<float>(i) / 16000.0f);
    }
    tts->set_reference(ref.data(), ref.size(), 16000);
    tts->set_reference_transcript("This is the exact reference sentence.");
    REQUIRE(tts->has_reference());

    // Decoder Run: cap the AR loop tight so the test stays bounded.
    tts->set_seed(4242);
    tts->set_max_steps(16);
    tts->set_min_steps_before_stop(0);
    size_t samples = 0;
    bool got_final = false;
    bool any_nonfinite = false;
    tts->synthesize("hi", "en",
        [&](const float* s, size_t n, bool is_final) {
            samples += n;
            if (is_final) got_final = true;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isfinite(s[i])) any_nonfinite = true;
            }
        });
    REQUIRE(got_final);
    REQUIRE(samples > 0);
    REQUIRE(!any_nonfinite);

    std::printf("ok (max_text=%d production tokens=%d samples=%zu)\n",
                tts->max_text_tokens(), tts->tokens_generated(), samples);
}

void test_onnx_cosyvoice3_load(const std::string& /*dir*/) {
    const char* override_dir = std::getenv("SPEECH_COSYVOICE3_ONNX_DIR");
    std::string cosy_dir = override_dir ? override_dir : "/tmp/cosyvoice3-onnx-bundle";
    const std::string prefill = cosy_dir + "/llm_prefill.onnx";
    const std::string step = cosy_dir + "/llm_step.onnx";
    const std::string flow = cosy_dir + "/flow_frontend.onnx";
    const std::string estimator = cosy_dir + "/flow.decoder.estimator.fp32.onnx";
    const std::string hift = cosy_dir + "/hift.onnx";
    const std::string vocab = cosy_dir + "/CosyVoice-BlankEN/vocab.json";
    const std::string merges = cosy_dir + "/CosyVoice-BlankEN/merges.txt";
    if (!file_exists(prefill) || !file_exists(step) || !file_exists(flow)
        || !file_exists(estimator) || !file_exists(hift)
        || !file_exists(vocab) || !file_exists(merges)) {
        std::printf("  [skip] ONNX CosyVoice3 bundle not in %s "
                    "(set SPEECH_COSYVOICE3_ONNX_DIR to override)\n",
                    cosy_dir.c_str());
        return;
    }
    std::printf("  test_onnx_cosyvoice3_load ... ");
    speech_core::OnnxCosyVoice3Tts tts(cosy_dir, /*hw_accel=*/false);
    REQUIRE(tts.output_sample_rate() == 24000);
    REQUIRE(!tts.has_conditioning());
    tts.set_flow_steps(6);
    REQUIRE(tts.flow_steps() == 6);
    tts.set_flow_steps(0);
    REQUIRE(tts.flow_steps() == 1);
    tts.set_cfg_rate(0.0f);
    REQUIRE(tts.cfg_rate() == 0.0f);
    tts.set_cfg_rate(0.7f);
    REQUIRE(tts.cfg_rate() > 0.69f && tts.cfg_rate() < 0.71f);

    speech_core::OnnxCosyVoice3Tts::Conditioning c;
    c.prompt_text_ids = {1446, 525, 264, 10950, 17847, 13, 151646};
    c.llm_prompt_speech_tokens = {1, 2, 3, 4};
    c.flow_prompt_speech_tokens = {1, 2, 3, 4};
    c.prompt_speech_feat_frames = 4;
    c.prompt_speech_feat.assign(4 * 80, 0.0f);
    c.embedding.assign(192, 0.0f);
    auto blob = speech_core::OnnxCosyVoice3Tts::encode_conditioning_blob(c);
    auto decoded = speech_core::OnnxCosyVoice3Tts::decode_conditioning_blob(
        blob.data(), blob.size());
    REQUIRE(decoded.prompt_text_ids == c.prompt_text_ids);
    REQUIRE(decoded.llm_prompt_speech_tokens == c.llm_prompt_speech_tokens);
    REQUIRE(decoded.flow_prompt_speech_tokens == c.flow_prompt_speech_tokens);
    REQUIRE(decoded.prompt_speech_feat.size() == c.prompt_speech_feat.size());
    REQUIRE(decoded.embedding.size() == c.embedding.size());
    tts.set_conditioning(std::move(decoded));
    REQUIRE(tts.has_conditioning());
    if (std::getenv("SPEECH_COSYVOICE3_E2E")) {
        tts.set_seed(1986);
        tts.set_max_steps(8);
        size_t samples = 0;
        bool got_final = false;
        bool any_nonfinite = false;
        tts.synthesize("hi", "",
            [&](const float* s, size_t n, bool is_final) {
                samples += n;
                if (is_final) got_final = true;
                for (size_t i = 0; i < n; ++i) {
                    if (!std::isfinite(s[i])) any_nonfinite = true;
                }
            });
        REQUIRE(got_final);
        REQUIRE(samples > 0);
        REQUIRE(!any_nonfinite);
        REQUIRE(tts.prefill_ms() >= 0);
        REQUIRE(tts.ar_ms() >= 0);
        REQUIRE(tts.audio_decode_ms() >= 0);
        REQUIRE(tts.flow_frontend_ms() >= 0);
        REQUIRE(tts.flow_estimator_ms() >= 0);
        REQUIRE(tts.hift_ms() >= 0);
        std::printf("ok (e2e samples=%zu tokens=%d)\n",
                    samples, tts.tokens_generated());
    } else {
        std::printf("ok\n");
    }
}

// ---------------------------------------------------------------------------
// VoxCPM2 ONNX → Parakeet ORT round-trip — mirrors test_kokoro_parakeet_roundtrip.
// Synthesizes the pangram via the ONNX VoxCPM2 wrapper, resamples to 16 kHz,
// and asserts ≥ 4 of the 6 content words land in the Parakeet ORT transcript.
// Same gating as test_onnx_voxcpm2_load (production config required).
// ---------------------------------------------------------------------------

void test_onnx_voxcpm2_parakeet_roundtrip(const std::string& dir) {
    OnnxVoxCPM2Bundle bundle = onnx_voxcpm2_bundle();
    std::string parakeet_enc  = dir + "/parakeet-encoder-int8.onnx";
    std::string parakeet_dec  = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string parakeet_voc  = dir + "/vocab.json";

    if (!bundle.complete()
        || !file_exists(parakeet_enc) || !file_exists(parakeet_dec)
        || !file_exists(parakeet_voc))
    {
        std::printf("  [skip] roundtrip needs VoxCPM2 ONNX bundle + parakeet\n");
        return;
    }

    auto tts = make_onnx_voxcpm2_tts(bundle, /*hw_accel=*/true);
    if (tts->max_text_tokens() < 256) {
        std::printf("  [skip] roundtrip needs production-config VoxCPM2 bundle "
                    "(max_text=%d)\n", tts->max_text_tokens());
        return;
    }
    std::printf("  test_onnx_voxcpm2_parakeet_roundtrip [%s] ... ",
                bundle.has_split() ? "split" : "unified");

    tts->set_seed(4242);
    tts->set_max_steps(128);
    tts->set_min_steps_before_stop(32);

    const std::string phrase = "The quick brown fox jumps over the lazy dog";
    std::vector<float> audio_48k;
    bool got_final = false;
    tts->synthesize(phrase, "en",
        [&](const float* s, size_t n, bool is_final) {
            if (s && n) audio_48k.insert(audio_48k.end(), s, s + n);
            if (is_final) got_final = true;
        });
    REQUIRE(got_final);
    REQUIRE(!audio_48k.empty());

    auto audio_16k = speech_core::Resampler::resample(
        audio_48k.data(), audio_48k.size(), 48000, 16000);
    REQUIRE(!audio_16k.empty());

    speech_core::ParakeetStt stt(parakeet_enc, parakeet_dec, parakeet_voc,
                                  /*hw_accel=*/true);
    auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);

    std::string lower = to_lower(result.text);
    const char* words[] = {"quick", "brown", "fox", "jumps", "lazy", "dog"};
    int matched = 0;
    for (const char* w : words) {
        if (lower.find(w) != std::string::npos) ++matched;
    }
    std::printf("tokens=%d text=\"%s\" matched=%d/6 ",
                tts->tokens_generated(), result.text.c_str(), matched);
    REQUIRE(matched >= 4);
    std::printf("ok\n");
}

// ---------------------------------------------------------------------------
// VoxCPM2 ONNX → Parakeet WER corpus — runs the same TTS-ASR roundtrip over
// 15 short clean-English phrases and reports per-phrase + aggregate WER.
// Lets us quantify how much INT8 quantization on the LM graphs degrades the
// synthesis vs an FP32 reference. ALSO dumps the synthesized WAVs to
// $SPEECH_VOXCPM2_WAV_DUMP (when set) so an independent ASR (e.g. sherpa
// Whisper-tiny) can verify Parakeet isn't being uniquely tolerant of any
// quantization artefacts.
//
// Gated on production-config bundle (same gate as the single-phrase
// roundtrip). Always passes if mean WER < 0.25; the wrapper-quality smoke
// test is `matched >= 4 of 6` on the pangram, which this corpus tightens.
// ---------------------------------------------------------------------------

static int levenshtein_word_distance(const std::vector<std::string>& a,
                                     const std::vector<std::string>& b) {
    const int n = (int)a.size(), m = (int)b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (int i = 0; i <= n; ++i) dp[i][0] = i;
    for (int j = 0; j <= m; ++j) dp[0][j] = j;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int del_ = dp[i-1][j] + 1;
            int ins_ = dp[i][j-1] + 1;
            int sub_ = dp[i-1][j-1] + cost;
            int best = del_ < ins_ ? del_ : ins_;
            dp[i][j] = best < sub_ ? best : sub_;
        }
    }
    return dp[n][m];
}

static std::vector<std::string> tokenize_lower(const std::string& s) {
    std::string lower = to_lower(s);
    std::vector<std::string> toks;
    std::string cur;
    for (char c : lower) {
        if (std::isalnum((unsigned char)c) || c == '\'') {
            cur.push_back(c);
        } else if (!cur.empty()) {
            toks.push_back(std::move(cur));
            cur.clear();
        }
    }
    if (!cur.empty()) toks.push_back(std::move(cur));
    return toks;
}

void test_onnx_voxcpm2_wer_corpus(const std::string& dir) {
    OnnxVoxCPM2Bundle bundle = onnx_voxcpm2_bundle();
    std::string parakeet_enc  = dir + "/parakeet-encoder-int8.onnx";
    std::string parakeet_dec  = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string parakeet_voc  = dir + "/vocab.json";

    if (!bundle.complete()
        || !file_exists(parakeet_enc) || !file_exists(parakeet_dec)
        || !file_exists(parakeet_voc))
    {
        std::printf("  [skip] wer_corpus needs VoxCPM2 ONNX bundle + parakeet\n");
        return;
    }

    auto tts = make_onnx_voxcpm2_tts(bundle, /*hw_accel=*/true);
    if (tts->max_text_tokens() < 256) {
        std::printf("  [skip] wer_corpus needs production-config bundle "
                    "(max_text=%d)\n", tts->max_text_tokens());
        return;
    }
    // Surface which inference path will execute. The IoBinding refactor in this
    // wrapper is gated on a non-CPU EP being resolved (see synthesize()'s
    // use_gpu_bind check). On CPU-only builds, on GPU builds whose ORT didn't
    // link CUDA, and on machines without an NVIDIA device, the test silently
    // falls back to the pre-existing host path -- making this print the only
    // way a reviewer or CI can tell whether the GPU+IoBinding code was actually
    // exercised, or whether a green run only validates the legacy path.
    const bool _gpu_on = OnnxEngine::get().has_gpu_provider();
    const char* _gpu_name = _gpu_on ? "GPU (via hook)" : "CPU (host path)";
    const char* _bind = _gpu_on ? "ON" : "OFF";
    std::printf("  test_onnx_voxcpm2_wer_corpus [%s provider=%s iobinding=%s] ... ",
                bundle.has_split() ? "split" : "unified", _gpu_name, _bind);
    std::fflush(stdout);

    // 15 short clean-English phrases (paraphrases of LibriSpeech-style
    // sentences). Kept under 12 words each so a 16-step AR loop completes
    // bounded; production VoxCPM2 generates 25-50 steps for these.
    const std::vector<std::string> phrases = {
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore",
        "Mister Quilter is the apostle of the middle classes",
        "He tells us that this season is the best of the year",
        "Linnell's pictures are a sort of upguards and atom paintings",
        "It was the best of times it was the worst of times",
        "Call me Ishmael",
        "All happy families are alike",
        "The pen is mightier than the sword",
        "An apple a day keeps the doctor away",
        "Practice makes perfect every single day",
        "Time flies when you are having fun",
        "Better late than never my old friend",
        "Knowledge is power and ignorance is bliss",
        "The early bird catches the worm at dawn",
    };

    // Match the LiteRT corpus test settings — VoxCPM2 was trained on
    // "({instruction}){text}" prompts; an empty instruction makes the model
    // emit filler tokens. set_reference would clone a target voice; for a
    // generic WER-vs-text test we skip the reference and rely on the
    // instruction alone to anchor the style.
    tts->set_instruction("clear, natural delivery");
    tts->set_max_steps(128);
    tts->set_min_steps_before_stop(32);

    speech_core::ParakeetStt stt(parakeet_enc, parakeet_dec, parakeet_voc,
                                  /*hw_accel=*/true);

    const char* wav_dump = std::getenv("SPEECH_VOXCPM2_WAV_DUMP");

    std::vector<double> wers;
    int total_ref_words = 0, total_errors = 0;
    std::printf("\n");
    for (size_t i = 0; i < phrases.size(); ++i) {
        const std::string& ref = phrases[i];
        tts->set_seed(static_cast<unsigned>(4242 + i));  // distinct per phrase

        std::vector<float> audio_48k;
        tts->synthesize(ref, "en",
            [&](const float* s, size_t n, bool) {
                if (s && n) audio_48k.insert(audio_48k.end(), s, s + n);
            });
        if (audio_48k.empty()) {
            std::printf("    [%zu] EMPTY audio — skip\n", i);
            continue;
        }
        auto audio_16k = speech_core::Resampler::resample(
            audio_48k.data(), audio_48k.size(), 48000, 16000);
        auto r = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);

        auto ref_toks = tokenize_lower(ref);
        auto hyp_toks = tokenize_lower(r.text);
        int dist = levenshtein_word_distance(ref_toks, hyp_toks);
        double wer = ref_toks.empty() ? 0.0 : (double)dist / ref_toks.size();
        wers.push_back(wer);
        total_errors += dist;
        total_ref_words += (int)ref_toks.size();
        std::printf("    [%2zu] WER=%5.1f%% (%d/%zu err)  ref=\"%.50s\"  hyp=\"%.50s\"\n",
                    i, wer * 100.0, dist, ref_toks.size(),
                    ref.c_str(), r.text.c_str());

        if (wav_dump) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/voxcpm2_phrase_%02zu.wav", wav_dump, i);
            std::ofstream w(path, std::ios::binary);
            if (w) {
                const uint32_t sr = 48000;
                const uint32_t n_samples = (uint32_t)audio_48k.size();
                const uint32_t data_bytes = n_samples * 2;
                const uint32_t riff = 36 + data_bytes;
                w.write("RIFF", 4); w.write((const char*)&riff, 4); w.write("WAVE", 4);
                w.write("fmt ", 4);
                uint32_t fmt_sz = 16; uint16_t fmt = 1, ch = 1, bits = 16;
                uint32_t byterate = sr * 2; uint16_t block = 2;
                w.write((const char*)&fmt_sz, 4); w.write((const char*)&fmt, 2);
                w.write((const char*)&ch, 2);    w.write((const char*)&sr, 4);
                w.write((const char*)&byterate, 4); w.write((const char*)&block, 2);
                w.write((const char*)&bits, 2);
                w.write("data", 4); w.write((const char*)&data_bytes, 4);
                for (float s : audio_48k) {
                    int v = (int)std::lround(std::clamp(s, -1.0f, 1.0f) * 32767.0f);
                    int16_t v16 = (int16_t)v;
                    w.write((const char*)&v16, 2);
                }
            }
        }
    }

    if (wers.empty()) {
        std::printf("    no phrases ran — corpus FAIL\n");
        REQUIRE(false);
        return;
    }
    double sum = 0.0;
    for (double w : wers) sum += w;
    double mean_wer = sum / wers.size();
    std::vector<double> sorted_wers = wers;
    std::sort(sorted_wers.begin(), sorted_wers.end());
    double p50 = sorted_wers[sorted_wers.size() / 2];
    double p95 = sorted_wers[(std::min)(sorted_wers.size() - 1,
                                         static_cast<size_t>(sorted_wers.size() * 0.95))];
    double overall_wer = (total_ref_words > 0)
        ? (double)total_errors / total_ref_words : 0.0;
    std::printf("    SUMMARY: n=%zu mean_wer=%.2f%% p50=%.2f%% p95=%.2f%% "
                "overall=%.2f%% (%d/%d errs)\n",
                wers.size(), mean_wer * 100, p50 * 100, p95 * 100,
                overall_wer * 100, total_errors, total_ref_words);
    REQUIRE(mean_wer < 0.25);  // generous bar — INT8 degradation is the concern
    std::printf("  ok\n");
}

// ---------------------------------------------------------------------------
// PersonaPlex 7B — load + structural smoke. Bundle (~16 GB FP16) is large so
// this test only loads sessions and confirms the IO names match the contract.
// End-to-end audio generation comes online when the bundle is uploaded to HF
// and the wrapper's deferred refinements (SentencePiece, voice prompt, delay
// pattern) are landed. See docs/models.md §"OnnxPersonaPlex".
// ---------------------------------------------------------------------------

void test_onnx_personaplex_load(const std::string& dir) {
    // Bundle layout matches download_personaplex_onnx.sh / upload_to_hf.sh.
    // The bundle root contains the raw ONNX graphs (no 'personaplex-' prefix)
    // alongside system_prompts.bin and a voices/ subdir.
    std::string enc = dir + "/mimi_encoder.onnx";
    std::string dec = dir + "/mimi_decoder.onnx";
    std::string tmp = dir + "/temporal_step.onnx";
    std::string dep = dir + "/depformer_step.onnx";
    std::string spm = dir + "/tokenizer_spm_32k_3.model";
    std::string voices = dir + "/voices";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(tmp)
        || !file_exists(dep) || !file_exists(spm)) {
        std::printf("  [skip] PersonaPlex bundle not in %s\n", dir.c_str());
        return;
    }
    std::printf("  test_onnx_personaplex_load ... ");

    speech_core::OnnxPersonaPlex pp(enc, dec, tmp, dep, spm, voices, /*hw_accel=*/false);
    REQUIRE(pp.output_sample_rate() == 24000);
    pp.set_voice("VARF2");
    pp.set_system_prompt("helpful");  // matches a key in system_prompts.bin
    pp.set_max_frames(4);  // tiny budget — just exercises the loop once
    pp.reset_session();

    // Feed 4 frames of silence at 24 kHz. With max_frames=4 and chunk_frames
    // default 25, we should get exactly one final chunk back.
    const size_t total_samples = static_cast<size_t>(1920) * 4;
    std::vector<float> silence(total_samples, 0.0f);
    int chunks = 0;
    bool got_final = false;
    size_t total_emitted = 0;
    pp.respond_stream(silence.data(), silence.size(), 24000,
        [&](const speech_core::FullDuplexChunk& c) {
            ++chunks;
            total_emitted += c.length;
            if (c.is_final) got_final = true;
        });
    REQUIRE(chunks > 0);
    REQUIRE(got_final);
    REQUIRE(total_emitted > 0);
    std::printf("ok (chunks=%d frames_generated=%d emitted=%zu)\n",
                chunks, pp.frames_generated(), total_emitted);
}

// ---------------------------------------------------------------------------

}  // namespace

int main() {
    std::string dir = env_model_dir();
    if (dir.empty()) {
        std::printf("SPEECH_MODEL_DIR not set — skipping model tests\n");
        std::printf("(run scripts/download_models.sh and re-run with SPEECH_MODEL_DIR=scripts/models)\n");
        return 0;
    }

    std::printf("Running model tests against %s\n", dir.c_str());

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
    RUN(test_onnx_engine_provider_resolution);
    RUN(test_silero_vad);
    RUN(test_silero_vad_real_speech);
    RUN(test_parakeet_stt);
    RUN(test_onnx_whisper_stt);
    RUN(test_nemotron_multilingual_stt);
    RUN(test_kokoro_tts);
    RUN(test_deepfilter);
    RUN(test_sidon_restorer);
    RUN(test_kokoro_parakeet_roundtrip);
    RUN(test_voxcpm2_tokenizer);
    RUN(test_onnx_voxcpm2_load);
    RUN(test_onnx_cosyvoice3_load);
    RUN(test_onnx_voxcpm2_parakeet_roundtrip);
    RUN(test_onnx_voxcpm2_wer_corpus);
    RUN(test_onnx_personaplex_load);
    #undef RUN

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) failed\n", failures);
        return 1;
    }
    std::printf("\nAll model tests passed\n");
    return 0;
}
