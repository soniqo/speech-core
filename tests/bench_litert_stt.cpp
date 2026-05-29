// Micro-benchmark for the LiteRT CPU STT paths (streaming Nemotron + batch
// Parakeet). NOT a correctness test — it reports throughput numbers so we can
// compare optimization prototypes before/after.
//
// Build: part of the SPEECH_CORE_WITH_LITERT target set.
// Run:
//   DYLD_LIBRARY_PATH=build/litert SPEECH_LITERT_MODEL_DIR=scripts/models-litert \
//       ./build/bench_litert_stt [warmup_runs] [measured_runs]
//
// Reports, for streaming: first-partial latency, per-chunk compute p50/p95/max,
// and stream RTF (audio_seconds / wall_seconds). For batch Parakeet: end-to-end
// wall time and RTF. Skips cleanly when models/fixture are missing.

#include "speech_core/audio/resampler.h"
#include "speech_core/models/litert_nemotron_streaming_stt.h"
#include "speech_core/models/litert_parakeet_stt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

std::string env_model_dir() {
    const char* e = std::getenv("SPEECH_LITERT_MODEL_DIR");
    return e ? e : "";
}

struct WavData { std::vector<float> samples; int sample_rate = 0; };
WavData load_wav_mono_pcm16(const std::string& path) {
    WavData out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    char riff[4], wave[4];
    uint32_t file_size; (void)file_size;
    f.read(riff, 4); f.read(reinterpret_cast<char*>(&file_size), 4); f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) return out;
    char chunk_id[4]; uint32_t chunk_size;
    uint16_t audio_format = 0, num_channels = 0, bits = 0; uint32_t rate = 0;
    bool have_fmt = false;
    while (f.read(chunk_id, 4)) {
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&rate), 4);
            f.seekg(6, std::ios::cur);
            f.read(reinterpret_cast<char*>(&bits), 2);
            if (chunk_size > 16) f.seekg(chunk_size - 16, std::ios::cur);
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt || audio_format != 1 || num_channels != 1 || bits != 16) return out;
            size_t n = chunk_size / 2;
            std::vector<int16_t> pcm(n);
            f.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            out.samples.resize(n);
            for (size_t i = 0; i < n; ++i) out.samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
            out.sample_rate = static_cast<int>(rate);
            break;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }
    return out;
}

std::string test_audio_path() {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
#else
    return "tests/data/test_audio.wav";
#endif
}

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

std::vector<float> load_audio_16k() {
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (wav.samples.empty()) return {};
    if (wav.sample_rate == 16000) return wav.samples;
    return speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                            wav.sample_rate, 16000);
}

// ---------------------------------------------------------------------------

void bench_nemotron_streaming(const std::string& dir, int warmup, int runs) {
    std::string enc = dir + "/nemotron-streaming-encoder.tflite";
    std::string dec = dir + "/nemotron-streaming-decoder.tflite";
    std::string jnt = dir + "/nemotron-streaming-joint.tflite";
    std::string voc = dir + "/nemotron-vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt) || !file_exists(voc)) {
        std::printf("[skip] nemotron streaming files not in %s\n", dir.c_str());
        return;
    }
    auto audio = load_audio_16k();
    if (audio.empty()) { std::printf("[skip] no fixture audio\n"); return; }
    const double audio_seconds = static_cast<double>(audio.size()) / 16000.0;

    std::printf("\n=== Nemotron streaming (audio=%.2fs) ===\n", audio_seconds);
    auto t_load = clk::now();
    speech_core::LiteRTNemotronStreamingStt stt(enc, dec, jnt, voc, /*hw_accel=*/false);
    std::printf("model load: %.0f ms\n", ms_since(t_load));

    // Feed audio in 80 ms chunks (1280 samples @ 16 kHz) — matches the prod
    // realtime frame cadence. Time every push_chunk individually.
    constexpr size_t kChunk = 1280;  // 80 ms

    std::vector<double> all_first_partial;
    std::vector<double> all_rtf;
    std::vector<double> chunk_ms_pooled;

    for (int r = 0; r < warmup + runs; ++r) {
        bool measured = (r >= warmup);
        stt.begin_stream(16000);
        double first_partial_ms = -1.0;
        auto t_stream = clk::now();
        double max_chunk = 0.0;
        std::vector<double> chunk_ms;
        for (size_t off = 0; off < audio.size(); off += kChunk) {
            size_t len = std::min(kChunk, audio.size() - off);
            auto t_c = clk::now();
            auto p = stt.push_chunk(audio.data() + off, len);
            double dt = ms_since(t_c);
            if (measured) {
                chunk_ms.push_back(dt);
                chunk_ms_pooled.push_back(dt);
                max_chunk = std::max(max_chunk, dt);
            }
            if (first_partial_ms < 0 && !p.text.empty()) first_partial_ms = ms_since(t_stream);
        }
        double wall_s = ms_since(t_stream) / 1000.0;
        stt.end_stream();
        if (measured) {
            all_first_partial.push_back(first_partial_ms);
            all_rtf.push_back(audio_seconds / wall_s);
            std::printf("  run %d: wall=%.2fs RTF=%.3fx first_partial=%.0fms "
                        "chunk p50=%.1f p95=%.1f max=%.1f ms\n",
                        r - warmup, wall_s, audio_seconds / wall_s, first_partial_ms,
                        pct(chunk_ms, 50), pct(chunk_ms, 95), max_chunk);
        }
    }

    std::printf("  SUMMARY nemotron: RTF=%.3fx first_partial=%.0fms "
                "chunk p50=%.1fms p95=%.1fms max=%.1fms\n",
                pct(all_rtf, 50), pct(all_first_partial, 50),
                pct(chunk_ms_pooled, 50), pct(chunk_ms_pooled, 95), pct(chunk_ms_pooled, 100));
}

void bench_parakeet_batch(const std::string& dir, int warmup, int runs) {
    std::string e = dir + "/parakeet-encoder.tflite";
    std::string d = dir + "/parakeet-decoder-joint.tflite";
    std::string v = dir + "/vocab.json";
    if (!file_exists(e) || !file_exists(d) || !file_exists(v)) {
        std::printf("[skip] parakeet files not in %s\n", dir.c_str());
        return;
    }
    auto audio = load_audio_16k();
    if (audio.empty()) { std::printf("[skip] no fixture audio\n"); return; }
    const double audio_seconds = static_cast<double>(audio.size()) / 16000.0;

    std::printf("\n=== Parakeet batch transcribe (audio=%.2fs) ===\n", audio_seconds);
    auto t_load = clk::now();
    speech_core::LiteRTParakeetStt stt(e, d, v, /*hw_accel=*/false);
    std::printf("model load: %.0f ms\n", ms_since(t_load));

    std::vector<double> all_ms;
    for (int r = 0; r < warmup + runs; ++r) {
        auto t = clk::now();
        auto res = stt.transcribe(audio.data(), audio.size(), 16000);
        double dt = ms_since(t);
        if (r >= warmup) {
            all_ms.push_back(dt);
            std::printf("  run %d: %.0f ms RTF=%.3fx text_len=%zu\n",
                        r - warmup, dt, audio_seconds / (dt / 1000.0), res.text.size());
        }
    }
    double med = pct(all_ms, 50);
    std::printf("  SUMMARY parakeet batch: %.0f ms RTF=%.3fx\n",
                med, audio_seconds / (med / 1000.0));
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = env_model_dir();
    if (dir.empty()) {
        std::printf("SPEECH_LITERT_MODEL_DIR not set — skipping bench\n");
        return 0;
    }
    int warmup = (argc > 1) ? std::atoi(argv[1]) : 1;
    int runs   = (argc > 2) ? std::atoi(argv[2]) : 3;
    std::printf("bench: warmup=%d runs=%d dir=%s\n", warmup, runs, dir.c_str());

    bench_nemotron_streaming(dir, warmup, runs);
    bench_parakeet_batch(dir, warmup, runs);
    return 0;
}
