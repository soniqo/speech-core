// Micro-benchmark for the ORT models (Silero VAD, Parakeet STT, Whisper,
// Kokoro TTS).
// Runs each model on the test fixture, reports load time / latency / RTF /
// peak RSS.
//
// Build: part of the SPEECH_CORE_WITH_ONNX target set.
// Run:
//   SPEECH_MODEL_DIR=scripts/models ./build/Release/bench_ort_models.exe
//
// Reports machine-readable CSV lines to stdout, headed by a # comment.

#include "speech_core/audio/resampler.h"
#include "speech_core/models/kokoro_tts.h"
#include "speech_core/models/onnx_engine.h"
#include "speech_core/models/onnx_cosyvoice3_tts.h"
#include "speech_core/models/onnx_nemotron_streaming_stt.h"
#include "speech_core/models/onnx_voxcpm2_tts.h"
#include "speech_core/models/onnx_whisper_stt.h"
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
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__APPLE__)
#include <mach/mach.h>
#else
#include <sys/resource.h>
#include <unistd.h>
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

double current_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#elif defined(__APPLE__)
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0.0;
    long pages = 0;
    long resident = 0;
    if (std::fscanf(f, "%ld %ld", &pages, &resident) != 2) {
        std::fclose(f);
        return 0.0;
    }
    std::fclose(f);
    long page_size = sysconf(_SC_PAGESIZE);
    return static_cast<double>(resident) * page_size / (1024.0 * 1024.0);
#endif
}

bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

std::string env_model_dir() {
    const char* e = std::getenv("SPEECH_MODEL_DIR");
    return e ? e : "";
}

const char* provider_label() {
    return OnnxEngine::get().has_gpu_provider() ? "gpu" : "cpu";
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

// Conventional real-time factor: inference wall time divided by the amount
// of audio processed or produced. Values below 1.0 are faster than real time.
constexpr double classic_rtf(double wall_ms, double audio_s) {
    if (audio_s <= 0.0 || wall_ms < 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return (wall_ms / 1000.0) / audio_s;
}
static_assert(classic_rtf(2000.0, 4.0) == 0.5,
              "RTF must be wall seconds divided by audio seconds");

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
    emit("silero-vad", "stream", load_ms, wall,
         classic_rtf(wall, audio_s), peak_rss_mb(), ex);
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
    emit("parakeet-tdt", "batch", load_ms, med,
         classic_rtf(med, audio_s), peak_rss_mb(), ex);
}

struct WhisperBundle {
    std::string encoder;
    std::string decoder;
    std::string tokens;
    std::string label;
};

bool find_whisper_bundle(const std::string& dir, WhisperBundle* out) {
    const char* override_dir = std::getenv("SPEECH_WHISPER_ONNX_DIR");
    std::string root = override_dir ? override_dir : dir;

    struct Candidate { const char* prefix; const char* suffix; const char* label; };
    const Candidate candidates[] = {
        {"turbo", ".int8", "turbo-int8"},
        {"large-v3", ".int8", "large-v3-int8"},
        {"medium", ".int8", "medium-int8"},
        {"small", ".int8", "small-int8"},
        {"turbo", ".fp16", "turbo-fp16"},
        {"large-v3", ".fp16", "large-v3-fp16"},
        {"medium", ".fp16", "medium-fp16"},
        {"small", ".fp16", "small-fp16"},
        {"turbo", "", "turbo-fp32"},
        {"large-v3", "", "large-v3-fp32"},
        {"medium", "", "medium-fp32"},
        {"small", "", "small-fp32"},
    };
    for (const auto& c : candidates) {
        std::string e = root + "/" + c.prefix + "-encoder" + c.suffix + ".onnx";
        std::string d = root + "/" + c.prefix + "-decoder" + c.suffix + ".onnx";
        std::string t = root + "/" + c.prefix + "-tokens.txt";
        if (file_exists(e) && file_exists(d) && file_exists(t)) {
            out->encoder = std::move(e);
            out->decoder = std::move(d);
            out->tokens = std::move(t);
            out->label = c.label;
            return true;
        }
    }
    std::fprintf(stderr, "[skip] whisper ONNX bundle in %s\n", root.c_str());
    return false;
}

void emit_whisper_profile(const std::string& metric,
                          const WhisperBundle& bundle,
                          const speech_core::OnnxWhisperStt::Profile& p,
                          double load_ms,
                          double audio_s,
                          const std::string& text) {
    std::string transcript = text;
    for (char& c : transcript) if (c == ',') c = ' ';
    if (transcript.size() > 80) transcript.resize(80);
    char ex[512];
    std::snprintf(ex, sizeof(ex),
                  "variant=%s current_rss=%.1fMB first_token=%.1fms feature=%.1fms encoder=%.1fms language=%.1fms prompt=%.1fms decoder=%.1fms chunks=%d feature_frames=%d encoded_frames=%d prompt_tokens=%d generated_tokens=%d text=\"%s\"",
                  bundle.label.c_str(), current_rss_mb(), p.first_token_ms,
                  p.feature_ms, p.encoder_ms, p.language_ms,
                  p.decoder_prompt_ms, p.decoder_ms, p.chunks,
                  p.feature_frames, p.encoded_frames, p.prompt_tokens,
                  p.generated_tokens, transcript.c_str());
    emit("whisper", metric, load_ms, p.total_ms,
         classic_rtf(p.total_ms, audio_s), peak_rss_mb(), ex);
}

void bench_whisper_config(const WhisperBundle& bundle,
                          const std::string& metric,
                          speech_core::OnnxWhisperStt::Config cfg,
                          const std::vector<float>& audio) {
    auto t_load = clk::now();
    speech_core::OnnxWhisperStt stt(
        bundle.encoder, bundle.decoder, bundle.tokens, cfg, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);
    double audio_s = static_cast<double>(audio.size()) / 16000.0;

    auto cold = stt.transcribe(audio.data(), audio.size(), 16000);
    emit_whisper_profile(metric + "-cold", bundle, stt.last_profile(),
                         load_ms, audio_s, cold.text);

    std::vector<double> walls;
    speech_core::OnnxWhisperStt::Profile last_profile;
    std::string last_text;
    for (int i = 0; i < 3; ++i) {
        auto r = stt.transcribe(audio.data(), audio.size(), 16000);
        last_profile = stt.last_profile();
        walls.push_back(last_profile.total_ms);
        last_text = r.text;
    }
    double med = pct(walls, 50);
    (void)med;
    emit_whisper_profile(metric, bundle, last_profile, load_ms, audio_s, last_text);
}

void bench_whisper(const std::string& dir) {
    WhisperBundle bundle;
    if (!find_whisper_bundle(dir, &bundle)) return;
    auto audio = load_audio_16k();
    if (audio.empty()) { std::fprintf(stderr, "[skip] no fixture\n"); return; }

    const char* mode_env = std::getenv("SPEECH_WHISPER_BENCH_CONFIG");
    std::string mode = mode_env ? mode_env : "";

    if (mode.empty() || mode == "auto") {
        speech_core::OnnxWhisperStt::Config default_cfg;
        bench_whisper_config(bundle, "batch-auto", default_cfg, audio);
    }

    if (mode.empty() || mode == "en-tail50") {
        speech_core::OnnxWhisperStt::Config low_latency_cfg;
        low_latency_cfg.language = "en";
        low_latency_cfg.tail_padding_frames = 50;
        bench_whisper_config(bundle, "batch-en-tail50", low_latency_cfg, audio);
    }

    if (mode == "en-tail0") {
        speech_core::OnnxWhisperStt::Config low_latency_cfg;
        low_latency_cfg.language = "en";
        low_latency_cfg.tail_padding_frames = 0;
        bench_whisper_config(bundle, "batch-en-tail0", low_latency_cfg, audio);
    }
}

void bench_kokoro(const std::string& dir) {
    const char* model_override = std::getenv("SPEECH_KOKORO_ONNX_PATH");
    std::string m = model_override ? model_override : dir + "/kokoro-e2e.onnx";
    if (!file_exists(m) || !file_exists(dir + "/vocab_index.json")) {
        std::fprintf(stderr, "[skip] kokoro files\n"); return;
    }
    auto t_load = clk::now();
    // Keep this row an explicit CPU baseline. Hardware-provider experiments
    // belong in separate runs where provider assignment is verified.
    auto config = speech_core::KokoroTts::Config::default_for_model_path(
        m, /*hw_accel=*/false);
    const char* config_label =
        config.max_safe_output_samples > 0 ? "short-3.0s-auto" : "full-auto";
    if (const char* profile = std::getenv("SPEECH_KOKORO_REALTIME_PROFILE");
        profile && profile[0] != '\0') {
        if (std::strcmp(profile, "3") == 0 ||
            std::strcmp(profile, "3.0") == 0) {
            config = speech_core::KokoroTts::Config::short_turn_3s(
                /*hw_accel=*/false);
            config_label = "short-3.0s";
        } else if (std::strcmp(profile, "3.5") == 0) {
            config = speech_core::KokoroTts::Config::short_turn_3p5s(
                /*hw_accel=*/false);
            config_label = "short-3.5s";
        } else {
            std::fprintf(stderr,
                         "unknown SPEECH_KOKORO_REALTIME_PROFILE=%s\n",
                         profile);
            std::exit(2);
        }
    }
    speech_core::KokoroTts tts(m, dir + "/voices", dir, config);
    double load_ms = ms_since(t_load);

    const char* text_override = std::getenv("SPEECH_KOKORO_TEXT");
    const std::string text = text_override
        ? text_override
        : "The quick brown fox jumps over the lazy dog";

    // Warmup
    size_t warm_samples = 0;
    tts.synthesize(text, "en",
                   [&](const float*, size_t n, bool) { warm_samples += n; });

    std::vector<double> runs;
    std::vector<double> audio_durations;
    std::vector<double> rtfs;
    size_t last_chunks = 0;
    size_t last_finals = 0;
    bool last_finite = true;
    for (int i = 0; i < 3; ++i) {
        size_t samples = 0;
        size_t chunks = 0;
        size_t finals = 0;
        bool finite = true;
        auto t = clk::now();
        tts.synthesize(text, "en",
                       [&](const float* pcm, size_t n, bool is_final) {
                           ++chunks;
                           if (is_final) ++finals;
                           for (size_t j = 0; j < n; ++j) {
                               if (!std::isfinite(pcm[j])) finite = false;
                           }
                           samples += n;
                       });
        const double wall_ms = ms_since(t);
        const double audio_s = static_cast<double>(samples) / 24000.0;
        runs.push_back(wall_ms);
        audio_durations.push_back(audio_s);
        rtfs.push_back(classic_rtf(wall_ms, audio_s));
        last_chunks = chunks;
        last_finals = finals;
        last_finite = finite;
    }
    double med = pct(runs, 50);
    double audio_s = pct(audio_durations, 50);
    const size_t slash = m.find_last_of("/\\");
    const std::string graph = slash == std::string::npos
        ? m
        : m.substr(slash + 1);
    char ex[384];
    std::snprintf(ex, sizeof(ex),
                  "config=%s graph=%s p50=%.0fms p95=%.0fms audio_p50=%.2fs chunks=%zu finals=%zu finite=%d",
                  config_label, graph.c_str(), pct(runs, 50), pct(runs, 95), audio_s,
                  last_chunks, last_finals, last_finite ? 1 : 0);
    emit("kokoro-tts", "synth", load_ms, med, pct(rtfs, 50),
         peak_rss_mb(), ex);
}

void bench_cosyvoice3(const std::string& /*dir*/) {
    const char* override_dir = std::getenv("SPEECH_COSYVOICE3_ONNX_DIR");
    std::string cosy = override_dir ? override_dir : "/tmp/cosyvoice3-onnx-bundle";
    std::string prefill = cosy + "/llm_prefill.onnx";
    std::string step = cosy + "/llm_step.onnx";
    std::string flow = cosy + "/flow_frontend.onnx";
    std::string estimator = cosy + "/flow.decoder.estimator.fp32.onnx";
    std::string hift = cosy + "/hift.onnx";
    std::string vocab = cosy + "/CosyVoice-BlankEN/vocab.json";
    std::string merges = cosy + "/CosyVoice-BlankEN/merges.txt";
    if (!file_exists(prefill) || !file_exists(step) || !file_exists(flow)
        || !file_exists(estimator) || !file_exists(hift)
        || !file_exists(vocab) || !file_exists(merges)) {
        std::fprintf(stderr, "[skip] cosyvoice3 bundle in %s\n", cosy.c_str());
        return;
    }

    auto t_load = clk::now();
    speech_core::OnnxCosyVoice3Tts tts(cosy, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);

    speech_core::OnnxCosyVoice3Tts::Conditioning c;
    c.prompt_text_ids = {1446, 525, 264, 10950, 17847, 13, 151646};
    c.llm_prompt_speech_tokens = {1, 2, 3, 4};
    c.flow_prompt_speech_tokens = {1, 2, 3, 4};
    c.prompt_speech_feat_frames = 4;
    c.prompt_speech_feat.assign(4 * 80, 0.0f);
    c.embedding.assign(192, 0.0f);
    tts.set_conditioning(std::move(c));
    tts.set_seed(1986);
    tts.set_max_steps(64);
    tts.set_flow_steps(4);
    tts.set_cfg_rate(0.7f);

    const std::string text = "The quick brown fox jumps over the lazy dog.";

    // Warmup: populate ORT kernels and allocator state.
    {
        size_t samples = 0;
        tts.synthesize(text, "en",
                       [&](const float*, size_t n, bool) { samples += n; });
        (void)samples;
    }

    std::vector<double> walls;
    size_t last_samples = 0;
    int last_tokens = 0;
    bool last_stopped = false;
    int64_t prefill_ms = 0;
    int64_t ar_ms = 0;
    int64_t decode_ms = 0;
    int64_t flow_frontend_ms = 0;
    int64_t flow_estimator_ms = 0;
    int64_t hift_ms = 0;
    for (int i = 0; i < 3; ++i) {
        size_t samples = 0;
        auto t = clk::now();
        tts.synthesize(text, "en",
                       [&](const float*, size_t n, bool) { samples += n; });
        walls.push_back(ms_since(t));
        last_samples = samples;
        last_tokens = tts.tokens_generated();
        last_stopped = tts.stopped_on_stop_token();
        prefill_ms = tts.prefill_ms();
        ar_ms = tts.ar_ms();
        decode_ms = tts.audio_decode_ms();
        flow_frontend_ms = tts.flow_frontend_ms();
        flow_estimator_ms = tts.flow_estimator_ms();
        hift_ms = tts.hift_ms();
    }
    double med = pct(walls, 50);
    double audio_s = static_cast<double>(last_samples) / 24000.0;
    double ms_per_token = (last_tokens > 0) ? static_cast<double>(ar_ms) / last_tokens : 0.0;
    char ex[320];
    std::snprintf(ex, sizeof(ex),
                  "p50=%.0fms p95=%.0fms audio=%.2fs tokens=%d stopped=%d ms_per_token=%.1f prefill=%.0fms ar=%.0fms decode=%.0fms flow_frontend=%.0fms flow_estimator=%.0fms hift=%.0fms",
                  med, pct(walls, 95), audio_s, last_tokens,
                  last_stopped ? 1 : 0, ms_per_token,
                  static_cast<double>(prefill_ms),
                  static_cast<double>(ar_ms),
                  static_cast<double>(decode_ms),
                  static_cast<double>(flow_frontend_ms),
                  static_cast<double>(flow_estimator_ms),
                  static_cast<double>(hift_ms));
    emit("cosyvoice3-tts", "synth", load_ms, med,
         classic_rtf(med, audio_s), peak_rss_mb(), ex);
}

// ---------------------------------------------------------------------------
// VoxCPM2 ONNX TTS — the comparison the LiteRT bench shows at ~10 RTF
// (0.10x real-time throughput) / ~12 GB
// RSS / 1.55 s per AR step on CPU. The ONNX wrapper drives the same 4-graph
// pipeline; with hw_accel=true (CUDA) the per-step compute should drop
// substantially (text_prefill once + token_step ×N is where the GPU win lives).
// Picks the bundle from $SPEECH_VOXCPM2_ONNX_DIR or /tmp/voxcpm2-onnx.
// Skips when missing or tiny-config (max_text < 256).
// ---------------------------------------------------------------------------

void bench_nemotron(const std::string& dir) {
    // Nemotron-streaming bundle lives outside scripts/models — we look for it
    // via SPEECH_NEMOTRON_ONNX_DIR (the converter's `--output-dir`) with a
    // sensible fallback for the local dev path.
    std::string nem_dir;
    if (const char* p = std::getenv("SPEECH_NEMOTRON_ONNX_DIR")) nem_dir = p;
    if (nem_dir.empty()) nem_dir = dir + "/nemotron-streaming-onnx";
    std::string enc = nem_dir + "/nemotron-streaming-encoder.onnx";
    std::string dec = nem_dir + "/nemotron-streaming-decoder.onnx";
    std::string jnt = nem_dir + "/nemotron-streaming-joint.onnx";
    std::string voc = nem_dir + "/vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt) || !file_exists(voc)) {
        std::fprintf(stderr, "[skip] nemotron streaming files in %s\n", nem_dir.c_str());
        return;
    }
    auto audio = load_audio_16k();
    if (audio.empty()) { std::fprintf(stderr, "[skip] no fixture\n"); return; }

    auto t_load = clk::now();
    speech_core::OnnxNemotronStreamingStt stt(enc, dec, jnt, voc, /*hw_accel=*/true);
    double load_ms = ms_since(t_load);

    // Match the LiteRT bench's 80 ms streaming cadence (1280 samples @ 16 kHz)
    // — only the runtime differs. One warmup full pass; then 3 measured runs.
    constexpr size_t kChunk = 1280;
    {
        stt.begin_stream(16000);
        for (size_t off = 0; off < audio.size(); off += kChunk) {
            size_t len = (kChunk < audio.size() - off) ? kChunk : (audio.size() - off);
            (void)stt.push_chunk(audio.data() + off, len);
        }
        stt.end_stream();
    }

    std::vector<double> walls, chunk_ms, first_partials;
    std::string last_text;
    for (int r = 0; r < 3; ++r) {
        stt.begin_stream(16000);
        auto t_stream = clk::now();
        double first_partial_ms = -1.0;
        for (size_t off = 0; off < audio.size(); off += kChunk) {
            size_t len = (kChunk < audio.size() - off) ? kChunk : (audio.size() - off);
            auto t_c = clk::now();
            auto p = stt.push_chunk(audio.data() + off, len);
            chunk_ms.push_back(ms_since(t_c));
            if (first_partial_ms < 0 && !p.text.empty()) first_partial_ms = ms_since(t_stream);
        }
        last_text = stt.end_stream().text;
        walls.push_back(ms_since(t_stream));
        first_partials.push_back(first_partial_ms);
    }
    double wall_ms = pct(walls, 50);
    double audio_s = static_cast<double>(audio.size()) / 16000.0;
    for (char& c : last_text) if (c == ',') c = ' ';
    if (last_text.size() > 80) last_text.resize(80);
    char ex[256];
    std::snprintf(ex, sizeof(ex),
                  "p50=%.0fms p95=%.0fms chunk_p50=%.1fms chunk_p95=%.1fms first_partial=%.0fms text=\"%s\"",
                  pct(walls, 50), pct(walls, 95),
                  pct(chunk_ms, 50), pct(chunk_ms, 95),
                  pct(first_partials, 50), last_text.c_str());
    emit("nemotron-streaming", "stream", load_ms, wall_ms,
         classic_rtf(wall_ms, audio_s), peak_rss_mb(), ex);
}

void bench_voxcpm2(const std::string& /*dir*/) {
    const char* override_dir = std::getenv("SPEECH_VOXCPM2_ONNX_DIR");
    std::string vox = override_dir ? override_dir : "/tmp/voxcpm2-onnx";
    std::string decoder = vox + "/voxcpm2-decoder.onnx";
    std::string enc     = vox + "/voxcpm2-audio-encoder.onnx";
    std::string dec     = vox + "/voxcpm2-audio-decoder.onnx";
    std::string tok     = vox + "/tokenizer.json";
    if (!file_exists(decoder) || !file_exists(enc)
        || !file_exists(dec) || !file_exists(tok)) {
        std::fprintf(stderr, "[skip] voxcpm2 bundle\n");
        return;
    }

    auto t_load = clk::now();
    speech_core::OnnxVoxCPM2Tts tts(decoder, enc, dec, tok,
                                     /*hw_accel=*/true);
    double load_ms = ms_since(t_load);
    if (tts.max_text_tokens() < 256) {
        std::fprintf(stderr, "[skip] voxcpm2 bundle is tiny-config (max_text=%d)\n",
                     tts.max_text_tokens());
        return;
    }
    tts.set_seed(4242);
    tts.set_max_steps(32);
    tts.set_min_steps_before_stop(8);

    const std::string text = "The quick brown fox jumps over the lazy dog";

    // Warmup — one short run to populate ORT caches + amortize CUDA EP init.
    {
        size_t s = 0;
        tts.synthesize(text, "en",
                       [&](const float*, size_t n, bool) { s += n; });
        (void)s;
    }

    // Measured: 3 runs, take the median wall.
    std::vector<double> walls;
    size_t last_samples = 0;
    int    last_tokens  = 0;
    for (int i = 0; i < 3; ++i) {
        size_t samples = 0;
        auto t = clk::now();
        tts.synthesize(text, "en",
                       [&](const float*, size_t n, bool) { samples += n; });
        walls.push_back(ms_since(t));
        last_samples = samples;
        last_tokens  = tts.tokens_generated();
    }
    double med = pct(walls, 50);
    double audio_s = static_cast<double>(last_samples) / 48000.0;  // VoxCPM2 outputs 48 kHz
    double ms_per_step = (last_tokens > 0) ? med / last_tokens : 0.0;
    char ex[256];
    std::snprintf(ex, sizeof(ex),
                  "p50=%.0fms p95=%.0fms audio=%.2fs steps=%d ms_per_step=%.0fms",
                  med, pct(walls, 95), audio_s, last_tokens, ms_per_step);
    emit("voxcpm2-tts", "synth", load_ms, med, classic_rtf(med, audio_s),
         peak_rss_mb(), ex);
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

int run_corpus_nemotron(const std::string& dir, const std::string& manifest_path) {
    // Same dir-resolution shape as bench_nemotron: env override wins, otherwise
    // `${dir}/nemotron-streaming-onnx`. Three model files + vocab JSON.
    std::string nem_dir;
    if (const char* p = std::getenv("SPEECH_NEMOTRON_ONNX_DIR")) nem_dir = p;
    if (nem_dir.empty()) nem_dir = dir + "/nemotron-streaming-onnx";
    std::string enc = nem_dir + "/nemotron-streaming-encoder.onnx";
    std::string dec = nem_dir + "/nemotron-streaming-decoder.onnx";
    std::string jnt = nem_dir + "/nemotron-streaming-joint.onnx";
    std::string vcb = nem_dir + "/vocab.json";
    if (!file_exists(enc) || !file_exists(dec) || !file_exists(jnt) || !file_exists(vcb)) {
        std::fprintf(stderr, "nemotron files missing in %s\n", nem_dir.c_str());
        return 1;
    }
    speech_core::OnnxNemotronStreamingStt stt(enc, dec, jnt, vcb, /*hw_accel=*/true);

    std::ifstream m(manifest_path);
    if (!m) { std::fprintf(stderr, "cannot open manifest %s\n", manifest_path.c_str()); return 1; }
    std::printf("#uid,provider,audio_s,wall_ms,transcript\n");

    constexpr size_t kChunkSamples = 1280;  // 80 ms @ 16 kHz — matches encoder stride

    std::string line;
    while (std::getline(m, line)) {
        if (line.empty()) continue;
        size_t c1 = line.find(','); if (c1 == std::string::npos) continue;
        size_t c2 = line.find(',', c1 + 1); if (c2 == std::string::npos) continue;
        std::string uid      = line.substr(0, c1);
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

        std::string transcript;
        auto t = clk::now();
        stt.begin_stream(16000);
        for (size_t off = 0; off < audio_16k.size(); off += kChunkSamples) {
            size_t n = (kChunkSamples < audio_16k.size() - off)
                       ? kChunkSamples : (audio_16k.size() - off);
            auto partial = stt.push_chunk(audio_16k.data() + off, n);
            if (!partial.text.empty()) transcript += partial.text;
        }
        auto final_r = stt.end_stream();
        if (!final_r.text.empty()) transcript = final_r.text;
        double wall = ms_since(t);
        double audio_s = static_cast<double>(audio_16k.size()) / 16000.0;
        for (char& ch : transcript) if (ch == ',') ch = ' ';
        std::printf("%s,%s,%.2f,%.1f,%s\n", uid.c_str(), provider_label(),
                    audio_s, wall, transcript.c_str());
        std::fflush(stdout);
    }
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
    // --nemotron-manifest <path> routes to the Nemotron streaming corpus
    // harness instead (each utterance streamed in 80 ms chunks).
    std::string manifest;
    std::string nemotron_manifest;
    int batch_size = 1;
    for (int i = 1; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--manifest") manifest = argv[i + 1];
        else if (arg == "--nemotron-manifest") nemotron_manifest = argv[i + 1];
        else if (arg == "--batch") batch_size = std::atoi(argv[i + 1]);
    }
    if (!nemotron_manifest.empty()) return run_corpus_nemotron(dir, nemotron_manifest);
    if (!manifest.empty()) return run_corpus(dir, manifest, batch_size);
    std::printf("#model,provider,metric,load_ms,wall_ms,rtf,rss_mb,extra\n");
    if (const char* only_env = std::getenv("SPEECH_BENCH_ONLY")) {
        const std::string only(only_env);
        if (only == "silero") bench_silero(dir);
        else if (only == "parakeet") bench_parakeet(dir);
        else if (only == "whisper") bench_whisper(dir);
        else if (only == "kokoro") bench_kokoro(dir);
        else if (only == "cosyvoice3") bench_cosyvoice3(dir);
        else if (only == "voxcpm2") bench_voxcpm2(dir);
        else if (only == "nemotron") bench_nemotron(dir);
        else {
            std::fprintf(stderr,
                         "unknown SPEECH_BENCH_ONLY=%s\n",
                         only.c_str());
            return 2;
        }
        return 0;
    }
    bench_silero(dir);
    bench_parakeet(dir);
    bench_whisper(dir);
    bench_kokoro(dir);
    bench_cosyvoice3(dir);
    bench_voxcpm2(dir);
    bench_nemotron(dir);
    return 0;
}
