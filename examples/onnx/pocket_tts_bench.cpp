#include "speech_core/models/onnx_pocket_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = (values.size() - 1) * fraction;
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, values.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

long read_status_kib(const char* key) {
    std::FILE* stream = std::fopen("/proc/self/status", "r");
    if (!stream) return -1;
    char line[256];
    long value = -1;
    while (std::fgets(line, sizeof(line), stream)) {
        if (std::strncmp(line, key, std::strlen(key)) == 0) {
            if (std::sscanf(line + std::strlen(key), "%ld", &value) != 1) value = -1;
            break;
        }
    }
    std::fclose(stream);
    return value;
}

struct Run {
    speech_core::PocketTtsMetrics metrics;
    std::uint64_t hash = UINT64_C(14695981039346656037);
    int chunks = 0;
    double sum_squares = 0.0;
    float peak = 0.0f;
};

Run generate(speech_core::OnnxPocketTts& tts,
             const std::string& text,
             std::vector<float>* captured_audio = nullptr) {
    Run result;
    tts.synthesize(text, "en", [&](const float* samples, std::size_t count, bool final) {
        if (!final && samples && count > 0) {
            ++result.chunks;
            if (captured_audio) {
                captured_audio->insert(captured_audio->end(), samples, samples + count);
            }
            for (std::size_t index = 0; index < count; ++index) {
                result.sum_squares += static_cast<double>(samples[index]) * samples[index];
                result.peak = std::max(result.peak, std::abs(samples[index]));
            }
            const auto* bytes = reinterpret_cast<const unsigned char*>(samples);
            for (std::size_t index = 0; index < count * sizeof(float); ++index) {
                result.hash ^= bytes[index];
                result.hash *= UINT64_C(1099511628211);
            }
        }
    });
    result.metrics = tts.last_metrics();
    return result;
}

void write_u16(std::FILE* stream, std::uint16_t value) {
    const unsigned char bytes[] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff)};
    std::fwrite(bytes, 1, sizeof(bytes), stream);
}

void write_u32(std::FILE* stream, std::uint32_t value) {
    const unsigned char bytes[] = {
        static_cast<unsigned char>(value & 0xff),
        static_cast<unsigned char>((value >> 8) & 0xff),
        static_cast<unsigned char>((value >> 16) & 0xff),
        static_cast<unsigned char>((value >> 24) & 0xff)};
    std::fwrite(bytes, 1, sizeof(bytes), stream);
}

void write_wav(const std::string& path, const std::vector<float>& samples) {
    std::FILE* stream = std::fopen(path.c_str(), "wb");
    if (!stream) throw std::runtime_error("Cannot open WAV output: " + path);
    const auto data_bytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    std::fwrite("RIFF", 1, 4, stream);
    write_u32(stream, 36 + data_bytes);
    std::fwrite("WAVEfmt ", 1, 8, stream);
    write_u32(stream, 16);
    write_u16(stream, 1);
    write_u16(stream, 1);
    write_u32(stream, 24000);
    write_u32(stream, 24000 * sizeof(std::int16_t));
    write_u16(stream, sizeof(std::int16_t));
    write_u16(stream, 16);
    std::fwrite("data", 1, 4, stream);
    write_u32(stream, data_bytes);
    for (const float value : samples) {
        const float clipped = std::max(-1.0f, std::min(1.0f, value));
        const auto pcm = static_cast<std::int16_t>(std::lrint(clipped * 32767.0f));
        write_u16(stream, static_cast<std::uint16_t>(pcm));
    }
    std::fclose(stream);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "Usage: %s BUNDLE [TEXT] [THREADS] [STEPS] [WARMUP] [RUNS] [OUTPUT_WAV]\n",
            argv[0]);
        return 2;
    }

    const std::string bundle = argv[1];
    const std::string text = argc > 2 ? argv[2] : "Hello world.";
    const int threads = argc > 3 ? std::atoi(argv[3]) : 2;
    const int steps = argc > 4 ? std::atoi(argv[4]) : 4;
    const int warmup = argc > 5 ? std::atoi(argv[5]) : 1;
    const int run_count = argc > 6 ? std::atoi(argv[6]) : 10;
    const std::string output_wav = argc > 7 ? argv[7] : "";
    if (threads < 1 || steps < 1 || warmup < 0 || run_count < 1) return 2;

    try {
        speech_core::PocketTtsConfig config;
        config.intra_threads = threads;
        config.flow_steps = steps;
        config.max_frames = 100;
        config.seed = 42;

        const auto load_started = Clock::now();
        speech_core::OnnxPocketTts tts(bundle, config);
        const double load_ms = milliseconds(load_started, Clock::now());

        for (int index = 0; index < warmup; ++index) generate(tts, text);

        std::vector<Run> runs;
        std::vector<double> first_audio;
        std::vector<double> total;
        std::vector<double> rtf;
        std::vector<float> captured_audio;
        runs.reserve(static_cast<std::size_t>(run_count));
        for (int index = 0; index < run_count; ++index) {
            auto* capture = !output_wav.empty() && index == run_count - 1
                ? &captured_audio : nullptr;
            runs.push_back(generate(tts, text, capture));
            const auto& metrics = runs.back().metrics;
            first_audio.push_back(metrics.first_audio_ms);
            total.push_back(metrics.total_ms);
            const double audio_seconds = metrics.output_samples / 24000.0;
            rtf.push_back(metrics.total_ms / 1000.0 / audio_seconds);
        }

        const auto& last = runs.back();
        const long rss_kib = read_status_kib("VmRSS:");
        const long peak_kib = read_status_kib("VmHWM:");
        if (!output_wav.empty()) write_wav(output_wav, captured_audio);
        std::printf("Pocket TTS speech-core interleaved benchmark\n");
        std::printf("text=\"%s\" threads=%d steps=%d warmup=%d runs=%d\n",
                    text.c_str(), threads, steps, warmup, run_count);
        std::printf("load_ms=%.3f conditioning_ms=%.3f\n",
                    load_ms, last.metrics.conditioning_ms);
        std::printf("ttfa_p50_ms=%.3f ttfa_p90_ms=%.3f ttfa_p95_ms=%.3f\n",
                    percentile(first_audio, 0.50), percentile(first_audio, 0.90),
                    percentile(first_audio, 0.95));
        std::printf("total_p50_ms=%.3f total_p90_ms=%.3f total_p95_ms=%.3f\n",
                    percentile(total, 0.50), percentile(total, 0.90),
                    percentile(total, 0.95));
        std::printf(
            "rtf_p50=%.4f audio_seconds=%.3f chunks=%d rss_mib=%.1f "
            "peak_rss_mib=%.1f peak=%.4f rms=%.4f hash=%016llx\n",
            percentile(rtf, 0.50), last.metrics.output_samples / 24000.0,
            last.chunks, rss_kib / 1024.0, peak_kib / 1024.0,
            last.peak,
            std::sqrt(last.sum_squares / last.metrics.output_samples),
            static_cast<unsigned long long>(last.hash));
        std::printf("run,conditioning_ms,ttfa_ms,total_ms,rtf,frames,output_samples,eos\n");
        for (std::size_t index = 0; index < runs.size(); ++index) {
            const auto& metrics = runs[index].metrics;
            std::printf("%zu,%.3f,%.3f,%.3f,%.4f,%d,%d,%d\n", index + 1,
                        metrics.conditioning_ms, metrics.first_audio_ms,
                        metrics.total_ms, rtf[index], metrics.frames_generated,
                        metrics.output_samples, metrics.stopped_on_eos ? 1 : 0);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Pocket TTS benchmark failed: %s\n", error.what());
        return 3;
    }
    return 0;
}
