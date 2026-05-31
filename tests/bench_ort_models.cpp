// Micro-benchmark for the ORT models (Silero VAD, Parakeet STT, Kokoro TTS).
// Runs each model on the test fixture, reports load time / latency / RTF /
// peak RSS. Pick the provider via env: SPEECH_CORE_ORT_PROVIDER=cuda|cpu.
// hw_accel is wired true so the engine actually engages the GPU EP under CUDA.
//
// Build: part of the SPEECH_CORE_WITH_ONNX target set.
// Run:
//   SPEECH_CORE_ORT_PROVIDER=cpu  SPEECH_MODEL_DIR=scripts/models ./build-cuda/Release/bench_ort_models.exe
//   SPEECH_CORE_ORT_PROVIDER=cuda SPEECH_MODEL_DIR=scripts/models ./build-cuda/Release/bench_ort_models.exe
//
// Reports machine-readable CSV lines to stdout, headed by a # comment.

#include "speech_core/audio/resampler.h"
#include "speech_core/models/kokoro_tts.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/models/parakeet_stt.h"
#include "speech_core/models/silero_vad.h"

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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#endif

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

double peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage r{};
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return static_cast<double>(r.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(r.ru_maxrss) / 1024.0;
#endif
#endif
}

bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

std::string env_model_dir() {
    const char* e = std::getenv("SPEECH_MODEL_DIR");
    return e ? e : "";
}

const char* provider_label() {
    auto p = OnnxEngine::get().gpu_provider();
    if (p == OrtGpuProvider::Cuda)     return "cuda";
    if (p == OrtGpuProvider::TensorRT) return "tensorrt";
    return "cpu";
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
    return v[(std::min)(idx, v.size() - 1)];
}

std::vector<float> load_audio_16k() {
    auto wav = load_wav_mono_pcm16(test_audio_path());
    if (wav.samples.empty()) return {};
    if (wav.sample_rate == 16000) return wav.samples;
    return speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                            wav.sample_rate, 16000);
}

void emit(const std::string& model, const std::string& metric,
          double load_ms, double wall_ms, double rtf, double rss_mb,
          const std::string& extra = "") {
    // Single CSV line. Header is printed once in main().
    std::printf("%s,%s,%s,%.0f,%.1f,%.3f,%.1f,%s\n",
                model.c_str(), provider_label(), metric.c_str(),
                load_ms, wall_ms, rtf, rss_mb, extra.c_str());
}

// ---------------------------------------------------------------------------

void bench_silero(const std::string& dir) {
    std::string m = dir + "/silero-vad.onnx";
    if (!file_exists(m)) { std::fprintf(stderr, "[skip] silero-vad.onnx\n"); return; }
    auto audio = load_audio_16k();
    if (audio.empty()) { std::fprintf(stderr, "[skip] no fixture\n"); return; }

    auto t_load = clk::now();
    speech_core::SileroVad vad(m, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);

    constexpr size_t kChunk = 512;
    // Warmup
    for (int i = 0; i < 5; ++i) {
        vad.process_chunk(audio.data(), kChunk);
    }
    vad.reset();

    std::vector<double> per_chunk;
    auto t_total = clk::now();
    for (size_t i = 0; i + kChunk <= audio.size(); i += kChunk) {
        auto t = clk::now();
        vad.process_chunk(audio.data() + i, kChunk);
        per_chunk.push_back(ms_since(t));
    }
    double wall = ms_since(t_total);
    double audio_s = static_cast<double>(audio.size()) / 16000.0;
    char ex[128];
    std::snprintf(ex, sizeof(ex), "p50=%.3fms p95=%.3fms n=%zu",
                  pct(per_chunk, 50), pct(per_chunk, 95), per_chunk.size());
    emit("silero-vad", "stream", load_ms, wall, audio_s / (wall / 1000.0), peak_rss_mb(), ex);
}

void bench_parakeet(const std::string& dir) {
    std::string e = dir + "/parakeet-encoder-int8.onnx";
    std::string d = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string v = dir + "/vocab.json";
    if (!file_exists(e) || !file_exists(d) || !file_exists(v)) {
        std::fprintf(stderr, "[skip] parakeet files\n"); return;
    }
    auto audio = load_audio_16k();
    if (audio.empty()) { std::fprintf(stderr, "[skip] no fixture\n"); return; }

    // Parakeet's decoder-joint is numerically fragile under non-CPU EPs (see
    // engine docs). We bench BOTH: hw_accel=true (encoder may go GPU, joint
    // gets CPU fallback per its hw_accel arg) and the wrapper's actual ctor
    // signature only takes one bool — so we honour what the test runner
    // would do in production: hw_accel=true for the encoder path.
    auto t_load = clk::now();
    speech_core::ParakeetStt stt(e, d, v, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);

    // Warmup
    auto res_warm = stt.transcribe(audio.data(), audio.size(), 16000);
    (void)res_warm;

    std::vector<double> runs;
    std::string last_text;
    for (int i = 0; i < 5; ++i) {
        auto t = clk::now();
        auto r = stt.transcribe(audio.data(), audio.size(), 16000);
        runs.push_back(ms_since(t));
        last_text = r.text;
    }
    double med = pct(runs, 50);
    double audio_s = static_cast<double>(audio.size()) / 16000.0;
    // Truncate transcript for CSV (replace commas to keep one field).
    for (char& c : last_text) if (c == ',') c = ' ';
    if (last_text.size() > 80) last_text.resize(80);
    char ex[256];
    std::snprintf(ex, sizeof(ex), "p50=%.0fms p95=%.0fms text=\"%s\"",
                  pct(runs, 50), pct(runs, 95), last_text.c_str());
    emit("parakeet-tdt", "batch", load_ms, med, audio_s / (med / 1000.0), peak_rss_mb(), ex);
}

void bench_kokoro(const std::string& dir) {
    std::string m = dir + "/kokoro-e2e.onnx";
    if (!file_exists(m) || !file_exists(dir + "/vocab_index.json")) {
        std::fprintf(stderr, "[skip] kokoro files\n"); return;
    }
    auto t_load = clk::now();
    speech_core::KokoroTts tts(m, dir + "/voices", dir, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);

    const std::string text = "The quick brown fox jumps over the lazy dog";

    // Warmup
    size_t warm_samples = 0;
    tts.synthesize(text, "en",
                   [&](const float*, size_t n, bool) { warm_samples += n; });

    std::vector<double> runs;
    size_t last_samples = 0;
    for (int i = 0; i < 3; ++i) {
        size_t samples = 0;
        auto t = clk::now();
        tts.synthesize(text, "en",
                       [&](const float*, size_t n, bool) { samples += n; });
        runs.push_back(ms_since(t));
        last_samples = samples;
    }
    double med = pct(runs, 50);
    double audio_s = static_cast<double>(last_samples) / 24000.0;  // Kokoro outputs 24 kHz
    char ex[160];
    std::snprintf(ex, sizeof(ex), "p50=%.0fms p95=%.0fms audio=%.2fs",
                  pct(runs, 50), pct(runs, 95), audio_s);
    emit("kokoro-tts", "synth", load_ms, med, audio_s / (med / 1000.0), peak_rss_mb(), ex);
}

// ---------------------------------------------------------------------------
// Corpus mode — read a CSV manifest (uid,wav_path,reference) and transcribe
// each utterance with Parakeet. Per-utterance latency + transcript go to
// stdout for the Python orchestrator to aggregate (WER computed there).
// ---------------------------------------------------------------------------

WavData read_wav(const std::string& path) {
    WavData out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    char riff[4], wave[4]; uint32_t file_size; (void)file_size;
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

int run_corpus(const std::string& dir, const std::string& manifest_path,
               int batch_size) {
    std::string e = dir + "/parakeet-encoder-int8.onnx";
    std::string d = dir + "/parakeet-decoder-joint-int8.onnx";
    std::string v = dir + "/vocab.json";
    if (!file_exists(e) || !file_exists(d) || !file_exists(v)) {
        std::fprintf(stderr, "parakeet files missing in %s\n", dir.c_str());
        return 1;
    }
    speech_core::ParakeetStt stt(e, d, v, /*hw_accel=*/true);
    std::ifstream m(manifest_path);
    if (!m) { std::fprintf(stderr, "cannot open manifest %s\n", manifest_path.c_str()); return 1; }
    std::printf("#uid,provider,audio_s,wall_ms,transcript\n");

    struct Item { std::string uid; std::vector<float> audio_16k; };
    std::vector<Item> queue;
    queue.reserve(batch_size > 0 ? batch_size : 1);

    auto flush = [&]() {
        if (queue.empty()) return;
        std::vector<const float*> ptrs; ptrs.reserve(queue.size());
        std::vector<size_t>       lens; lens.reserve(queue.size());
        for (auto& it : queue) {
            ptrs.push_back(it.audio_16k.data());
            lens.push_back(it.audio_16k.size());
        }
        auto t = clk::now();
        auto rs = stt.transcribe_batch(ptrs.data(), lens.data(),
                                       ptrs.size(), 16000);
        double batch_wall = ms_since(t);
        // Apportion wall time across utterances by audio duration so the
        // CSV row remains apples-to-apples vs the single-utterance bench.
        double total_audio = 0.0;
        for (auto& it : queue) total_audio += static_cast<double>(it.audio_16k.size());
        for (size_t i = 0; i < queue.size(); ++i) {
            double share = static_cast<double>(queue[i].audio_16k.size()) / (std::max)(1.0, total_audio);
            double per_wall = batch_wall * share;
            double audio_s = static_cast<double>(queue[i].audio_16k.size()) / 16000.0;
            std::string text = rs[i].text;
            for (char& ch : text) if (ch == ',') ch = ' ';
            std::printf("%s,%s,%.2f,%.1f,%s\n",
                        queue[i].uid.c_str(), provider_label(),
                        audio_s, per_wall, text.c_str());
        }
        std::fflush(stdout);
        queue.clear();
    };

    std::string line;
    while (std::getline(m, line)) {
        if (line.empty()) continue;
        size_t c1 = line.find(','); if (c1 == std::string::npos) continue;
        size_t c2 = line.find(',', c1 + 1); if (c2 == std::string::npos) continue;
        std::string uid = line.substr(0, c1);
        std::string wav_path = line.substr(c1 + 1, c2 - c1 - 1);
        auto wav = read_wav(wav_path);
        if (wav.samples.empty()) {
            std::fprintf(stderr, "skip %s: cannot read %s\n", uid.c_str(), wav_path.c_str());
            continue;
        }
        std::vector<float> audio_16k = wav.sample_rate == 16000
            ? wav.samples
            : speech_core::Resampler::resample(wav.samples.data(), wav.samples.size(),
                                               wav.sample_rate, 16000);
        if (batch_size <= 1) {
            auto t = clk::now();
            auto r = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);
            double wall = ms_since(t);
            double audio_s = static_cast<double>(audio_16k.size()) / 16000.0;
            std::string text = r.text;
            for (char& ch : text) if (ch == ',') ch = ' ';
            std::printf("%s,%s,%.2f,%.1f,%s\n", uid.c_str(), provider_label(),
                        audio_s, wall, text.c_str());
            std::fflush(stdout);
        } else {
            queue.push_back({uid, std::move(audio_16k)});
            if (static_cast<int>(queue.size()) >= batch_size) flush();
        }
    }
    if (!queue.empty()) flush();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = env_model_dir();
    if (dir.empty()) {
        std::fprintf(stderr, "SPEECH_MODEL_DIR not set\n");
        return 0;
    }
    // Corpus mode if --manifest <path> passed; else default fixture bench.
    // Optional --batch N (default 1) batches encoder calls.
    std::string manifest;
    int batch_size = 1;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest") manifest = argv[i + 1];
        else if (arg == "--batch") batch_size = std::atoi(argv[i + 1]);
    }
    if (!manifest.empty()) return run_corpus(dir, manifest, batch_size);
    std::printf("#model,provider,metric,load_ms,wall_ms,rtf,rss_mb,extra\n");
    bench_silero(dir);
    bench_parakeet(dir);
    bench_kokoro(dir);
    return 0;
}
