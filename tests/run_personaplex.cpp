// Standalone CLI: load the PersonaPlex wrapper against an on-disk bundle and
// run respond_stream with N frames of silence. Reports load time, frame
// generation rate (frames/sec), per-chunk callback timing, and final agent
// audio size. Use this to smoke-test end-to-end on CPU + CUDA EPs without
// the full ctest harness.
//
// Build (with ONNX): -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=...
// Run:               run_personaplex <bundle_dir> [num_user_frames]

#include "speech_core/models/onnx_personaplex.h"
#include "speech_core/models/parakeet_stt.h"

#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#endif

#include <cassert>
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
bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
}

// Minimal mono PCM16 WAV writer. Used to dump the agent audio so we can
// (a) listen to verify the model is producing speech-like signal and
// (b) feed it through Parakeet STT for the roundtrip transcription gate.
bool write_wav_mono_pcm16(const std::string& path,
                          const std::vector<float>& samples,
                          int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n = static_cast<uint32_t>(samples.size());
    const uint32_t byte_rate = sample_rate * 2;
    const uint32_t data_bytes = n * 2;
    const uint32_t riff_size = 36 + data_bytes;
    auto w32 = [&](uint32_t v){ f.write(reinterpret_cast<char*>(&v), 4); };
    auto w16 = [&](uint16_t v){ f.write(reinterpret_cast<char*>(&v), 2); };
    f.write("RIFF", 4); w32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1);
    w32(static_cast<uint32_t>(sample_rate)); w32(byte_rate); w16(2); w16(16);
    f.write("data", 4); w32(data_bytes);
    for (float s : samples) {
        if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
        int16_t pcm = static_cast<int16_t>(s * 32767.0f);
        w16(static_cast<uint16_t>(pcm));
    }
    return f.good();
}

// Resample naive linear (for src->16k Parakeet input). Good enough for
// the roundtrip sanity gate; speech-swift's resampler has higher quality.
std::vector<float> resample_linear(const std::vector<float>& src, int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return src;
    const double ratio = static_cast<double>(src_rate) / dst_rate;
    const size_t n_dst = static_cast<size_t>(src.size() / ratio);
    std::vector<float> out(n_dst);
    for (size_t i = 0; i < n_dst; ++i) {
        double sp = i * ratio;
        size_t i0 = static_cast<size_t>(sp);
        double f = sp - i0;
        float v0 = src[i0];
        float v1 = (i0 + 1 < src.size()) ? src[i0 + 1] : v0;
        out[i] = static_cast<float>(v0 * (1.0 - f) + v1 * f);
    }
    return out;
}

// Peak working-set size in MB (Windows). Useful for memory benchmarking
// across the FP16 / INT8 / mixed bundles.
size_t peak_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    }
#endif
    return 0;
}

// Current working-set size in MB. Used to detect transient load-time spikes.
size_t current_rss_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024 * 1024);
    }
#endif
    return 0;
}

// --- NVML dynamic-dispatch VRAM probe -----------------------------------
// We avoid linking nvml.lib (build-system invasion) by resolving the two
// functions we need from nvml.dll (always present on machines with the
// NVIDIA driver) at runtime. Returns "used VRAM in MB by all CUDA contexts
// on device 0", which on our single-process benchmark equals our usage.
// Tracks peak across calls in a static.
struct NvmlMemInfo { unsigned long long total, free, used; };
typedef int (*NVML_INIT_FN)(void);
typedef int (*NVML_SHUTDOWN_FN)(void);
typedef int (*NVML_DEVICE_GET_HANDLE_FN)(unsigned int, void**);
typedef int (*NVML_DEVICE_GET_MEMORY_FN)(void*, NvmlMemInfo*);

#if defined(_WIN32)
class NvmlProbe {
public:
    NvmlProbe() {
        dll_ = LoadLibraryA("nvml.dll");
        if (!dll_) return;
        auto p_init = reinterpret_cast<NVML_INIT_FN>(GetProcAddress(dll_, "nvmlInit_v2"));
        get_handle_ = reinterpret_cast<NVML_DEVICE_GET_HANDLE_FN>(GetProcAddress(dll_, "nvmlDeviceGetHandleByIndex_v2"));
        get_mem_    = reinterpret_cast<NVML_DEVICE_GET_MEMORY_FN>(GetProcAddress(dll_, "nvmlDeviceGetMemoryInfo"));
        shutdown_   = reinterpret_cast<NVML_SHUTDOWN_FN>(GetProcAddress(dll_, "nvmlShutdown"));
        if (!p_init || !get_handle_ || !get_mem_) return;
        if (p_init() != 0) return;
        if (get_handle_(0, &device_) != 0) return;
        ok_ = true;
    }
    ~NvmlProbe() {
        if (ok_ && shutdown_) shutdown_();
        if (dll_) FreeLibrary(dll_);
    }
    // Returns current VRAM "used" by device 0 in MB; updates peak_mb_.
    size_t sample_mb(const char* label = nullptr) {
        if (!ok_) return 0;
        NvmlMemInfo mi{};
        if (get_mem_(device_, &mi) != 0) return 0;
        size_t used_mb = static_cast<size_t>(mi.used / (1024ULL * 1024ULL));
        if (used_mb > peak_mb_) peak_mb_ = used_mb;
        if (label) std::printf("    [vram] %-22s %zu MB used (%zu MB free)\n",
                                label, used_mb, static_cast<size_t>(mi.free / (1024ULL*1024ULL)));
        return used_mb;
    }
    size_t peak_mb() const { return peak_mb_; }
    bool   available() const { return ok_; }
private:
    HMODULE dll_ = nullptr;
    void*   device_ = nullptr;
    bool    ok_ = false;
    size_t  peak_mb_ = 0;
    NVML_DEVICE_GET_HANDLE_FN  get_handle_ = nullptr;
    NVML_DEVICE_GET_MEMORY_FN  get_mem_ = nullptr;
    NVML_SHUTDOWN_FN           shutdown_ = nullptr;
};
#else
class NvmlProbe {
public:
    size_t sample_mb(const char* = nullptr) { return 0; }
    size_t peak_mb() const { return 0; }
    bool   available() const { return false; }
};
#endif

// Basic acoustic metrics on the agent audio. Speech should have non-trivial
// variance and non-trivial dynamic range; silence has neither.
struct AudioStats {
    double rms;
    float  peak;
    size_t n_nonzero;
};
AudioStats audio_stats(const std::vector<float>& s) {
    AudioStats st{0.0, 0.0f, 0};
    if (s.empty()) return st;
    double ss = 0.0;
    for (float v : s) {
        ss += static_cast<double>(v) * v;
        if (std::fabs(v) > st.peak) st.peak = std::fabs(v);
        if (v != 0.0f) ++st.n_nonzero;
    }
    st.rms = std::sqrt(ss / s.size());
    return st;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: run_personaplex <bundle_dir> [frames=4] [wav_path] [voice=NATM0]\n"
                     "  When wav_path is given, real user audio is fed in (resampled to 24 kHz).\n"
                     "  Otherwise silence is used. frames caps generation length regardless of wav.\n");
        return 2;
    }
    const std::string dir = argv[1];
    int num_user_frames = (argc >= 3) ? std::atoi(argv[2]) : 4;
    const std::string wav_path = (argc >= 4) ? argv[3] : "";
    const std::string voice    = (argc >= 5) ? argv[4] : "NATM0";

    const std::string enc = dir + "/mimi_encoder.onnx";
    const std::string dec = dir + "/mimi_decoder.onnx";
    const std::string tmp = dir + "/temporal_step.onnx";
    const std::string dep = dir + "/depformer_step.onnx";
    const std::string spm = dir + "/tokenizer_spm_32k_3.model";

    for (const auto& p : {enc, dec, tmp, dep, spm}) {
        if (!file_exists(p)) {
            std::fprintf(stderr, "Missing bundle file: %s\n", p.c_str());
            return 1;
        }
    }

    const bool hw_accel = (std::getenv("SPEECH_CORE_DISABLE_HW") == nullptr);
    std::printf("PersonaPlex bundle: %s  (hw_accel=%d)\n", dir.c_str(), (int)hw_accel);

    NvmlProbe nvml;
    if (nvml.available()) {
        std::printf("VRAM probe (NVML):\n");
        nvml.sample_mb("baseline (pre-load)");
    } else {
        std::printf("VRAM probe (NVML): unavailable\n");
    }

    auto t0 = std::chrono::steady_clock::now();
    speech_core::OnnxPersonaPlex pp(enc, dec, tmp, dep, spm, dir + "/voices", hw_accel);
    auto t1 = std::chrono::steady_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("Load: %lld ms\n", static_cast<long long>(load_ms));
    nvml.sample_mb("after session load");

    pp.set_voice(voice);
    // System prompt is matched by NAME against the pre-tokenized blob loaded
    // from <bundle>/system_prompts.bin. Known names: "helpful", "expert",
    // "warm", "direct". Override via env SPEECH_CORE_PP_PROMPT.
    const char* prompt_name = std::getenv("SPEECH_CORE_PP_PROMPT");
    pp.set_system_prompt(prompt_name ? prompt_name : "helpful");
    pp.set_max_frames(num_user_frames);
    pp.reset_session();
    nvml.sample_mb("after reset_session");

    // User audio: either a real WAV (resampled to 24 kHz) or silence.
    const size_t samples_per_frame = 1920;
    std::vector<float> user_pcm;
    if (!wav_path.empty()) {
        // Minimal mono PCM16 WAV reader. Same format as test_models.cpp's load_wav.
        std::ifstream f(wav_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "Cannot open WAV %s\n", wav_path.c_str());
            return 1;
        }
        char riff[4], wave[4]; uint32_t file_size;
        f.read(riff, 4); f.read(reinterpret_cast<char*>(&file_size), 4); f.read(wave, 4);
        if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
            std::fprintf(stderr, "Not a RIFF WAVE\n"); return 1;
        }
        uint16_t fmt_tag = 0, channels = 0, bps = 0; uint32_t sr = 0;
        std::vector<int16_t> pcm16;
        char chunk_id[4]; uint32_t chunk_size;
        while (f.read(chunk_id, 4)) {
            f.read(reinterpret_cast<char*>(&chunk_size), 4);
            if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
                f.read(reinterpret_cast<char*>(&fmt_tag), 2);
                f.read(reinterpret_cast<char*>(&channels), 2);
                f.read(reinterpret_cast<char*>(&sr), 4);
                f.seekg(6, std::ios::cur);
                f.read(reinterpret_cast<char*>(&bps), 2);
                if (chunk_size > 16) f.seekg(chunk_size - 16, std::ios::cur);
            } else if (std::memcmp(chunk_id, "data", 4) == 0) {
                pcm16.resize(chunk_size / 2);
                f.read(reinterpret_cast<char*>(pcm16.data()), chunk_size);
                break;
            } else {
                f.seekg(chunk_size, std::ios::cur);
            }
        }
        if (fmt_tag != 1 || channels != 1 || bps != 16 || pcm16.empty()) {
            std::fprintf(stderr, "WAV must be mono PCM16. Got fmt=%u ch=%u bps=%u\n",
                         fmt_tag, channels, bps);
            return 1;
        }
        std::vector<float> src(pcm16.size());
        for (size_t i = 0; i < pcm16.size(); ++i) src[i] = pcm16[i] / 32768.0f;
        // Resample to 24 kHz if needed (linear).
        if (static_cast<int>(sr) != 24000) {
            user_pcm = resample_linear(src, static_cast<int>(sr), 24000);
            std::printf("WAV: loaded %s (sr=%u, %.2f s) -> resampled to %.2f s @ 24 kHz\n",
                        wav_path.c_str(), sr, src.size() / static_cast<double>(sr),
                        user_pcm.size() / 24000.0);
        } else {
            user_pcm = std::move(src);
            std::printf("WAV: loaded %s (sr=24000, %.2f s)\n", wav_path.c_str(),
                        user_pcm.size() / 24000.0);
        }
        const size_t need = samples_per_frame * num_user_frames;
        if (user_pcm.size() < need) {
            user_pcm.resize(need, 0.0f);
        } else if (user_pcm.size() > need) {
            user_pcm.resize(need);
        }
    } else {
        std::printf("Input: silence (%d frames)\n", num_user_frames);
        user_pcm.assign(samples_per_frame * num_user_frames, 0.0f);
    }
    std::printf("Voice: %s\n", voice.c_str());

    // SPEECH_CORE_PP_MULTITURN=N → run respond_stream N times in a row WITHOUT
    // reset_session between them, simulating an N-turn dialogue. Used to probe
    // whether the wrapper's KV cache survives across turns (and whether the
    // model produces sensible follow-up responses given prior context).
    int multiturn_n = 1;
    if (const char* mt = std::getenv("SPEECH_CORE_PP_MULTITURN")) {
        int n = std::atoi(mt);
        multiturn_n = n > 1 ? n : 1;
    }

    int chunks = 0;
    size_t total_emitted = 0;
    bool got_final = false;
    std::vector<float> all_audio;
    long long total_run_ms = 0;
    int total_frames_gen = 0;
    for (int turn = 1; turn <= multiturn_n; ++turn) {
        std::printf("\n=== TURN %d/%d ===\n", turn, multiturn_n);
        auto t_run0 = std::chrono::steady_clock::now();
        pp.respond_stream(user_pcm.data(), user_pcm.size(), 24000,
            [&](const speech_core::FullDuplexChunk& c) {
                ++chunks;
                total_emitted += c.length;
                all_audio.insert(all_audio.end(), c.samples, c.samples + c.length);
                if (c.is_final) got_final = true;
                std::printf("  chunk #%d: %zu samples (%.2fs), text_tokens=%zu, final=%d\n",
                            chunks, c.length,
                            static_cast<double>(c.length) / c.sample_rate,
                            c.text_tokens.size(), (int)c.is_final);
            });
        auto t_run1 = std::chrono::steady_clock::now();
        auto turn_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_run1 - t_run0).count();
        total_run_ms += turn_ms;
        total_frames_gen = pp.frames_generated();
        std::printf("  turn %d: %lld ms, frames=%d\n", turn, (long long)turn_ms, total_frames_gen);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "after turn %d", turn);
        nvml.sample_mb(buf);
    }
    long long run_ms = total_run_ms;

    std::printf("\nrespond_stream: %lld ms\n", static_cast<long long>(run_ms));
    std::printf("  frames_generated: %d\n", pp.frames_generated());
    std::printf("  chunks:           %d\n", chunks);
    std::printf("  total samples:    %zu (%.2fs of agent audio)\n",
                total_emitted, static_cast<double>(total_emitted) / 24000.0);
    std::printf("  got_final:        %d\n", (int)got_final);
    if (pp.frames_generated() > 0) {
        double frame_ms = static_cast<double>(run_ms) / pp.frames_generated();
        std::printf("  per-frame:        %.1f ms\n", frame_ms);
        std::printf("  RTF (12.5 Hz):    %.3f (1.0 = realtime)\n", frame_ms / 80.0);
    }

    if (size_t rss = peak_rss_mb()) {
        std::printf("  peak host RAM:    %zu MB\n", rss);
    }
    if (size_t rss = current_rss_mb()) {
        std::printf("  steady host RAM:  %zu MB\n", rss);
    }
    if (nvml.available()) {
        std::printf("  peak VRAM (NVML): %zu MB\n", nvml.peak_mb());
    }

    // Audio integrity stats — silence has rms<1e-4; speech is rms>=0.01.
    auto st = audio_stats(all_audio);
    std::printf("\nAgent audio integrity:\n");
    std::printf("  RMS:       %.6f\n", st.rms);
    std::printf("  peak |x|:  %.4f\n", st.peak);
    std::printf("  non-zero:  %zu / %zu (%.2f%%)\n", st.n_nonzero, all_audio.size(),
                100.0 * st.n_nonzero / std::max<size_t>(1, all_audio.size()));

    // Write WAV
    const std::string wav_out = dir + "/personaplex_out.wav";
    if (write_wav_mono_pcm16(wav_out, all_audio, 24000)) {
        std::printf("\nWrote %s (%zu samples @ 24 kHz)\n", wav_out.c_str(), all_audio.size());
    } else {
        std::fprintf(stderr, "WARN: failed to write %s\n", wav_out.c_str());
    }

    // Optional Parakeet roundtrip: when SPEECH_CORE_PARAKEET_DIR points at a
    // Parakeet ONNX bundle, transcribe the agent audio and print the result.
    // Lets us verify the model is producing speech-shaped acoustic content
    // (even nonsense transcripts mean Mimi+temporal+depformer are at least
    // generating audio in the right manifold).
    const char* pq_env = std::getenv("SPEECH_CORE_PARAKEET_DIR");
    if (pq_env) {
        std::string pq_dir = pq_env;
        std::string pq_enc = pq_dir + "/parakeet-encoder-int8.onnx";
        std::string pq_dec = pq_dir + "/parakeet-decoder-joint-int8.onnx";
        std::string pq_vocab = pq_dir + "/vocab.json";
        if (file_exists(pq_enc) && file_exists(pq_dec) && file_exists(pq_vocab)) {
            std::printf("\nParakeet roundtrip:\n");
            speech_core::ParakeetStt stt(pq_enc, pq_dec, pq_vocab, false);
            auto audio_16k = resample_linear(all_audio, 24000, 16000);
            auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);
            std::printf("  transcript: \"%.200s\"\n", result.text.c_str());
            std::printf("  language:   %s\n", result.language.c_str());
            std::printf("  confidence: %.3f\n", result.confidence);
        } else {
            std::printf("\nSPEECH_CORE_PARAKEET_DIR set but bundle missing required files\n");
        }
    }

    return got_final ? 0 : 3;
}
