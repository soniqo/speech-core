#include "speech_core/models/litert_kokoro_tts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path bundle;
    std::string text = "Hello world.";
    std::vector<int> threads = {1, 2, 4, 8};
    int warmup = 1;
    int runs = 5;
    float speed = 1.0f;
    std::filesystem::path output;
    std::string variant = "monolithic";
};

size_t count_phonemes(const std::filesystem::path& bundle,
                      const std::string& text) {
    speech_core::KokoroPhonemizer phonemizer;
    if (!phonemizer.load_vocab((bundle / "vocab_index.json").string()) ||
        !phonemizer.load_dictionaries(bundle.string())) {
        throw std::runtime_error("failed to load Kokoro phonemizer data");
    }
    phonemizer.set_language("en");
    return phonemizer.tokenize(text, 32768).size();
}

double read_proc_memory_mib(const std::string& field) {
#if defined(__linux__) || defined(__ANDROID__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind(field + ":", 0) != 0) continue;
        std::istringstream values(line.substr(field.size() + 1));
        double kib = 0.0;
        values >> kib;
        return kib / 1024.0;
    }
#else
    (void)field;
#endif
    return 0.0;
}

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) return 0.0;
    const size_t rank = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(fraction * sorted.size())));
    return sorted[std::min(rank - 1, sorted.size() - 1)];
}

uint64_t fnv1a64(const std::vector<float>& audio) {
    constexpr uint64_t offset = UINT64_C(14695981039346656037);
    constexpr uint64_t prime = UINT64_C(1099511628211);
    uint64_t hash = offset;
    const auto* bytes = reinterpret_cast<const unsigned char*>(audio.data());
    for (size_t i = 0; i < audio.size() * sizeof(float); ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

void write_audio(const std::filesystem::path& path,
                 const std::vector<float>& audio) {
    if (path.empty()) return;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(audio.data()),
                 static_cast<std::streamsize>(audio.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write output: " + path.string());
    }
}

std::vector<int> parse_threads(const std::string& value) {
    std::vector<int> result;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string item = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        const int threads = std::stoi(item);
        if (threads <= 0 || threads > 64) {
            throw std::invalid_argument("--threads values must be in [1, 64]");
        }
        result.push_back(threads);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (result.empty()) throw std::invalid_argument("--threads cannot be empty");
    return result;
}

Options parse_args(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: speech_kokoro_litert_bench BUNDLE_DIR "
            "[--threads 1,2,4,8] [--warmup N] [--runs N] [--text TEXT] "
            "[--speed FACTOR] [--output AUDIO_F32] "
            "[--variant monolithic|equivalent32]");
    }
    Options options;
    options.bundle = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--threads") {
            options.threads = parse_threads(value);
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value);
        } else if (arg == "--runs") {
            options.runs = std::stoi(value);
        } else if (arg == "--text") {
            options.text = value;
        } else if (arg == "--speed") {
            options.speed = std::stof(value);
        } else if (arg == "--output") {
            options.output = value;
        } else if (arg == "--variant") {
            options.variant = value;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.warmup < 0 || options.runs <= 0) {
        throw std::invalid_argument("--warmup must be >= 0 and --runs must be > 0");
    }
    if (!std::isfinite(options.speed) || options.speed <= 0.0f) {
        throw std::invalid_argument("--speed must be finite and positive");
    }
    if (options.variant != "monolithic" && options.variant != "equivalent32") {
        throw std::invalid_argument(
            "--variant must be monolithic or equivalent32");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto model = options.bundle / "kokoro-e2e-fp32.tflite";
        const auto encoder = options.bundle / "kokoro-encoder.tflite";
        const auto recurrent =
            options.bundle / "kokoro-recurrent-equivalent32.tflite";
        const auto vocoder = options.bundle / "kokoro-vocoder.tflite";
        const auto voices = options.bundle / "voices";
        const size_t phonemes = count_phonemes(options.bundle, options.text);

        std::cout << "Kokoro LiteRT integrated benchmark\n"
                  << "variant=" << options.variant << "\n"
                  << "model="
                  << (options.variant == "monolithic"
                          ? model.string()
                          : encoder.string() + ";" + recurrent.string() + ";" +
                                vocoder.string())
                  << "\n"
                  << "text=\"" << options.text << "\"\n"
                  << "phonemes=" << phonemes << " speed=" << options.speed << "\n"
                  << "warmup=" << options.warmup << " runs=" << options.runs << "\n"
                  << "note=run one thread count per process for isolated peak RSS\n\n"
                  << "threads,load_ms,cold_ms,cold_rtf,p50_ms,p90_ms,"
                     "audio_seconds,p50_rtf,p90_rtf,model_runs,current_rss_mib,"
                     "peak_rss_mib,encoder_p50_ms,recurrent_p50_ms,"
                     "vocoder_p50_ms,fnv1a64\n";

        for (int threads : options.threads) {
            const auto load_start = std::chrono::steady_clock::now();
            std::unique_ptr<speech_core::LiteRTKokoroTts> tts;
            if (options.variant == "equivalent32") {
                tts = std::make_unique<speech_core::LiteRTKokoroTts>(
                    encoder.string(), recurrent.string(), vocoder.string(),
                    voices.string(), options.bundle.string(), false, threads);
            } else {
                tts = std::make_unique<speech_core::LiteRTKokoroTts>(
                    model.string(), voices.string(), options.bundle.string(),
                    false, threads);
            }
            const auto load_stop = std::chrono::steady_clock::now();
            const double load_seconds =
                std::chrono::duration<double>(load_stop - load_start).count();
            tts->set_seed(1234);
            tts->set_speed(options.speed);

            std::vector<float> audio;
            const auto synthesize = [&] {
                audio.clear();
                tts->synthesize(options.text, "en",
                    [&](const float* samples, size_t length, bool) {
                        audio.insert(audio.end(), samples, samples + length);
                    });
            };

            const auto cold_start = std::chrono::steady_clock::now();
            synthesize();
            const auto cold_stop = std::chrono::steady_clock::now();
            const double cold_seconds =
                std::chrono::duration<double>(cold_stop - cold_start).count();
            for (int i = 0; i < options.warmup; ++i) synthesize();

            std::vector<double> elapsed;
            std::array<std::vector<double>, 3> stage_elapsed;
            elapsed.reserve(static_cast<size_t>(options.runs));
            for (int i = 0; i < options.runs; ++i) {
                const auto start = std::chrono::steady_clock::now();
                synthesize();
                const auto stop = std::chrono::steady_clock::now();
                elapsed.push_back(std::chrono::duration<double>(stop - start).count());
                if (options.variant == "equivalent32") {
                    const auto stages = tts->last_stage_ms();
                    for (size_t stage = 0; stage < stages.size(); ++stage) {
                        stage_elapsed[stage].push_back(stages[stage]);
                    }
                }
            }
            std::sort(elapsed.begin(), elapsed.end());
            const double p50 = percentile(elapsed, 0.50);
            const double p90 = percentile(elapsed, 0.90);
            const double audio_seconds =
                static_cast<double>(audio.size()) / tts->output_sample_rate();
            const double cold_rtf =
                audio_seconds > 0.0 ? cold_seconds / audio_seconds : 0.0;
            const double p50_rtf = audio_seconds > 0.0 ? p50 / audio_seconds : 0.0;
            const double p90_rtf = audio_seconds > 0.0 ? p90 / audio_seconds : 0.0;
            const uint64_t audio_hash = fnv1a64(audio);
            write_audio(options.output, audio);

            std::cout << threads << ',' << std::fixed << std::setprecision(3)
                      << load_seconds * 1000.0 << ',' << cold_seconds * 1000.0 << ','
                      << std::setprecision(4) << cold_rtf << ','
                      << std::setprecision(3) << p50 * 1000.0 << ','
                      << p90 * 1000.0 << ',' << audio_seconds << ','
                      << std::setprecision(4) << p50_rtf << ',' << p90_rtf << ','
                      << tts->model_runs_last_synthesis() << ','
                      << std::setprecision(1) << read_proc_memory_mib("VmRSS") << ','
                      << read_proc_memory_mib("VmHWM") << ',';
            for (auto& stage : stage_elapsed) {
                std::sort(stage.begin(), stage.end());
                std::cout << std::setprecision(3)
                          << percentile(stage, 0.50) << ',';
            }
            std::cout
                      << std::hex << std::setw(16) << std::setfill('0') << audio_hash
                      << std::dec << std::setfill(' ') << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
