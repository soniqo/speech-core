#include "speech_core/audio/resampler.h"
#include "speech_core/models/onnx_nemotron_streaming_stt.h"
#include "speech_core/models/onnx_pocket_tts.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kMaxAcceptedWer = 0.10;
constexpr double kMaxAcceptedCer = 0.05;

struct TestCase {
    std::string category;
    std::string text;
    std::vector<std::string> references;
};

struct Result {
    TestCase test;
    std::string chosen_reference;
    std::string transcript;
    int word_errors = 0;
    int reference_words = 0;
    int character_errors = 0;
    int reference_characters = 0;
    double wer = 0.0;
    double cer = 0.0;
    double stt_ms = 0.0;
    double audio_seconds = 0.0;
    float peak = 0.0f;
    float rms = 0.0f;
    speech_core::PocketTtsMetrics tts;
};

struct Aggregate {
    int cases = 0;
    int exact = 0;
    int word_errors = 0;
    int reference_words = 0;
    int character_errors = 0;
    int reference_characters = 0;
};

double milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> values;
    std::size_t start = 0;
    while (true) {
        const auto end = text.find(delimiter, start);
        values.push_back(text.substr(start, end - start));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return values;
}

std::vector<TestCase> read_corpus(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open round-trip corpus: " + path);

    std::vector<TestCase> cases;
    std::string line;
    int line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split(line, '\t');
        if (fields.size() != 3 || fields[0].empty() || fields[1].empty() ||
            fields[2].empty()) {
            throw std::runtime_error(
                "Invalid round-trip corpus line " + std::to_string(line_number));
        }
        auto references = split(fields[2], '|');
        for (const auto& reference : references) {
            if (reference.empty()) {
                throw std::runtime_error(
                    "Empty round-trip reference on line " +
                    std::to_string(line_number));
            }
        }
        cases.push_back({fields[0], fields[1], std::move(references)});
    }
    if (cases.empty()) throw std::runtime_error("Round-trip corpus is empty");
    return cases;
}

std::string normalize(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    bool pending_space = false;
    for (const unsigned char byte : input) {
        if (byte >= 128) {
            if (pending_space && !output.empty()) output.push_back(' ');
            pending_space = false;
            output.push_back(static_cast<char>(byte));
        } else if ((byte >= 'a' && byte <= 'z') ||
                   (byte >= '0' && byte <= '9')) {
            if (pending_space && !output.empty()) output.push_back(' ');
            pending_space = false;
            output.push_back(static_cast<char>(byte));
        } else if (byte >= 'A' && byte <= 'Z') {
            if (pending_space && !output.empty()) output.push_back(' ');
            pending_space = false;
            output.push_back(static_cast<char>(byte - 'A' + 'a'));
        } else if (byte == '\'' || byte == '`') {
            // Join apostrophe contractions so punctuation style is not scored.
        } else if (byte == '&') {
            if (!output.empty() && output.back() != ' ') output.push_back(' ');
            output.append("and");
            pending_space = true;
        } else {
            pending_space = true;
        }
    }
    return output;
}

std::vector<std::string> words(const std::string& normalized) {
    std::istringstream stream(normalized);
    std::vector<std::string> result;
    std::string word;
    while (stream >> word) result.push_back(word);
    return result;
}

std::vector<unsigned char> characters(const std::string& normalized) {
    std::vector<unsigned char> result;
    for (const unsigned char byte : normalized) {
        if (byte != ' ') result.push_back(byte);
    }
    return result;
}

template <typename T>
int edit_distance(const std::vector<T>& reference,
                  const std::vector<T>& hypothesis) {
    std::vector<int> previous(hypothesis.size() + 1);
    std::vector<int> current(hypothesis.size() + 1);
    for (std::size_t index = 0; index <= hypothesis.size(); ++index) {
        previous[index] = static_cast<int>(index);
    }
    for (std::size_t row = 1; row <= reference.size(); ++row) {
        current[0] = static_cast<int>(row);
        for (std::size_t column = 1; column <= hypothesis.size(); ++column) {
            const int substitution = previous[column - 1] +
                (reference[row - 1] == hypothesis[column - 1] ? 0 : 1);
            current[column] = std::min({
                previous[column] + 1, current[column - 1] + 1, substitution});
        }
        previous.swap(current);
    }
    return previous.back();
}

Result score(TestCase test,
             std::string transcript,
             const speech_core::PocketTtsMetrics& tts_metrics,
             double stt_ms,
             const std::vector<float>& audio) {
    Result result;
    result.test = std::move(test);
    result.transcript = std::move(transcript);
    result.tts = tts_metrics;
    result.stt_ms = stt_ms;
    result.audio_seconds = audio.size() / 24000.0;

    double sum_squares = 0.0;
    for (const float sample : audio) {
        result.peak = std::max(result.peak, std::abs(sample));
        sum_squares += static_cast<double>(sample) * sample;
    }
    result.rms = audio.empty()
        ? 0.0f
        : static_cast<float>(std::sqrt(sum_squares / audio.size()));

    const auto hypothesis_words = words(normalize(result.transcript));
    const auto hypothesis_characters = characters(normalize(result.transcript));
    double best_wer = std::numeric_limits<double>::infinity();
    double best_cer = std::numeric_limits<double>::infinity();
    for (const auto& reference : result.test.references) {
        const auto normalized_reference = normalize(reference);
        const auto reference_words = words(normalized_reference);
        const auto reference_characters = characters(normalized_reference);
        const int word_errors = edit_distance(reference_words, hypothesis_words);
        const int character_errors = edit_distance(
            reference_characters, hypothesis_characters);
        const double wer = reference_words.empty()
            ? 0.0
            : static_cast<double>(word_errors) / reference_words.size();
        const double cer = reference_characters.empty()
            ? 0.0
            : static_cast<double>(character_errors) / reference_characters.size();
        if (wer < best_wer || (wer == best_wer && cer < best_cer)) {
            best_wer = wer;
            best_cer = cer;
            result.chosen_reference = reference;
            result.word_errors = word_errors;
            result.reference_words = static_cast<int>(reference_words.size());
            result.character_errors = character_errors;
            result.reference_characters =
                static_cast<int>(reference_characters.size());
        }
    }
    result.wer = best_wer;
    result.cer = best_cer;
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

void write_wav(const std::filesystem::path& path,
               const std::vector<float>& samples) {
    std::FILE* stream = std::fopen(path.string().c_str(), "wb");
    if (!stream) throw std::runtime_error("Cannot write WAV: " + path.string());
    const auto bytes = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    std::fwrite("RIFF", 1, 4, stream);
    write_u32(stream, 36 + bytes);
    std::fwrite("WAVEfmt ", 1, 8, stream);
    write_u32(stream, 16);
    write_u16(stream, 1);
    write_u16(stream, 1);
    write_u32(stream, 24000);
    write_u32(stream, 48000);
    write_u16(stream, 2);
    write_u16(stream, 16);
    std::fwrite("data", 1, 4, stream);
    write_u32(stream, bytes);
    for (const float sample : samples) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const auto value = static_cast<std::int16_t>(std::lrint(clipped * 32767.0f));
        write_u16(stream, static_cast<std::uint16_t>(value));
    }
    std::fclose(stream);
}

std::string json_escape(const std::string& value) {
    std::string result;
    for (const unsigned char byte : value) {
        switch (byte) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (byte < 0x20) {
                    char escaped[7];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", byte);
                    result += escaped;
                } else {
                    result.push_back(static_cast<char>(byte));
                }
        }
    }
    return result;
}

double ratio(int errors, int reference_units) {
    return reference_units == 0
        ? 0.0
        : static_cast<double>(errors) / reference_units;
}

void add(Aggregate& aggregate, const Result& result) {
    ++aggregate.cases;
    if (result.word_errors == 0) ++aggregate.exact;
    aggregate.word_errors += result.word_errors;
    aggregate.reference_words += result.reference_words;
    aggregate.character_errors += result.character_errors;
    aggregate.reference_characters += result.reference_characters;
}

void write_report(const std::string& path,
                  const std::vector<Result>& results,
                  const Aggregate& aggregate,
                  double tts_load_ms,
                  double stt_load_ms,
                  bool passed) {
    if (path.empty()) return;
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("Cannot write JSON report: " + path);
    stream << std::fixed << std::setprecision(6);
    stream << "{\n  \"format_version\": 1,\n"
           << "  \"tts_load_ms\": " << tts_load_ms << ",\n"
           << "  \"stt_load_ms\": " << stt_load_ms << ",\n"
           << "  \"thresholds\": {\"max_wer\": " << kMaxAcceptedWer
           << ", \"max_cer\": " << kMaxAcceptedCer << "},\n"
           << "  \"summary\": {\"cases\": " << aggregate.cases
           << ", \"exact\": " << aggregate.exact
           << ", \"word_errors\": " << aggregate.word_errors
           << ", \"reference_words\": " << aggregate.reference_words
           << ", \"wer\": "
           << ratio(aggregate.word_errors, aggregate.reference_words)
           << ", \"character_errors\": " << aggregate.character_errors
           << ", \"reference_characters\": "
           << aggregate.reference_characters << ", \"cer\": "
           << ratio(aggregate.character_errors, aggregate.reference_characters)
           << ", \"passed\": " << (passed ? "true" : "false") << "},\n"
           << "  \"cases\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        stream << "    {\"index\": " << index + 1
               << ", \"category\": \"" << json_escape(result.test.category)
               << "\", \"text\": \"" << json_escape(result.test.text)
               << "\", \"reference\": \""
               << json_escape(result.chosen_reference)
               << "\", \"transcript\": \"" << json_escape(result.transcript)
               << "\", \"word_errors\": " << result.word_errors
               << ", \"reference_words\": " << result.reference_words
               << ", \"wer\": " << result.wer
               << ", \"character_errors\": " << result.character_errors
               << ", \"reference_characters\": "
               << result.reference_characters << ", \"cer\": " << result.cer
               << ", \"audio_seconds\": " << result.audio_seconds
               << ", \"tts_ttfa_ms\": " << result.tts.first_audio_ms
               << ", \"tts_total_ms\": " << result.tts.total_ms
               << ", \"stt_ms\": " << result.stt_ms
               << ", \"frames\": " << result.tts.frames_generated
               << ", \"eos\": "
               << (result.tts.stopped_on_eos ? "true" : "false")
               << ", \"peak\": " << result.peak << ", \"rms\": "
               << result.rms << "}";
        if (index + 1 != results.size()) stream << ',';
        stream << '\n';
    }
    stream << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "Usage: %s POCKET_BUNDLE STT_BUNDLE CORPUS_TSV [THREADS] [STEPS] "
            "[SEED] [REPORT_JSON] [FAILED_WAV_DIR]\n",
            argv[0]);
        return 2;
    }

    const std::string pocket_bundle = argv[1];
    const std::string stt_bundle = argv[2];
    const std::string corpus_path = argv[3];
    const int threads = argc > 4 ? std::atoi(argv[4]) : 2;
    const int steps = argc > 5 ? std::atoi(argv[5]) : 4;
    const int seed = argc > 6 ? std::atoi(argv[6]) : 42;
    const std::string report_path = argc > 7 ? argv[7] : "";
    const std::string failed_wav_dir = argc > 8 ? argv[8] : "";
    const char* write_all_value =
        std::getenv("SPEECH_POCKET_TTS_ROUNDTRIP_WRITE_ALL");
    const bool write_all_wavs =
        write_all_value && std::string(write_all_value) == "1";
    if (threads < 1 || steps < 1 || seed < 0) return 2;

    try {
        const auto cases = read_corpus(corpus_path);
        speech_core::PocketTtsConfig tts_config;
        tts_config.intra_threads = threads;
        tts_config.flow_steps = steps;
        tts_config.seed = seed;
        tts_config.max_frames = 200;

        auto started = Clock::now();
        speech_core::OnnxPocketTts tts(pocket_bundle, tts_config);
        const double tts_load_ms = milliseconds(started, Clock::now());

        started = Clock::now();
        speech_core::OnnxNemotronStreamingStt stt(
            stt_bundle + "/parakeet-eou-encoder.onnx",
            stt_bundle + "/parakeet-eou-decoder.onnx",
            stt_bundle + "/parakeet-eou-joint.onnx",
            stt_bundle + "/vocab.json",
            /*hw_accel=*/false);
        const double stt_load_ms = milliseconds(started, Clock::now());

        if (!failed_wav_dir.empty()) {
            std::filesystem::create_directories(failed_wav_dir);
        }

        std::printf("Pocket TTS -> Parakeet EOU round-trip intelligibility\n");
        std::printf("cases=%zu threads=%d steps=%d seed=%d "
                    "tts_load_ms=%.3f stt_load_ms=%.3f\n",
                    cases.size(), threads, steps, seed, tts_load_ms, stt_load_ms);

        std::vector<Result> results;
        results.reserve(cases.size());
        Aggregate aggregate;
        std::unordered_map<std::string, Aggregate> categories;
        int empty_audio = 0;
        int empty_transcript = 0;
        int eos_failures = 0;

        for (std::size_t index = 0; index < cases.size(); ++index) {
            std::vector<float> audio_24k;
            tts.synthesize(cases[index].text, "en",
                [&](const float* samples, std::size_t count, bool final) {
                    if (!final && samples && count) {
                        audio_24k.insert(audio_24k.end(), samples, samples + count);
                    }
                });
            const auto tts_metrics = tts.last_metrics();
            if (audio_24k.empty()) ++empty_audio;
            if (!tts_metrics.stopped_on_eos) ++eos_failures;

            auto speech_16k = speech_core::Resampler::resample(
                audio_24k.data(), audio_24k.size(), 24000, 16000);
            // Give the streaming recognizer stable boundaries without altering
            // the synthesized speech itself: 150 ms before and 300 ms after.
            std::vector<float> stt_audio(2400, 0.0f);
            stt_audio.insert(stt_audio.end(), speech_16k.begin(), speech_16k.end());
            stt_audio.insert(stt_audio.end(), 4800, 0.0f);

            const auto stt_started = Clock::now();
            const auto transcription = stt.transcribe(
                stt_audio.data(), stt_audio.size(), 16000);
            const double stt_ms = milliseconds(stt_started, Clock::now());
            if (normalize(transcription.text).empty()) ++empty_transcript;

            auto result = score(
                cases[index], transcription.text, tts_metrics, stt_ms, audio_24k);
            add(aggregate, result);
            add(categories[result.test.category], result);
            std::printf(
                "CASE %02zu [%s] WER=%5.1f%% CER=%5.1f%% audio=%.2fs "
                "TTFA=%.1fms STT=%.1fms eos=%d\n  REF: %s\n  HYP: %s\n",
                index + 1, result.test.category.c_str(), result.wer * 100.0,
                result.cer * 100.0, result.audio_seconds,
                result.tts.first_audio_ms, result.stt_ms,
                result.tts.stopped_on_eos ? 1 : 0,
                result.chosen_reference.c_str(), result.transcript.c_str());

            if (!failed_wav_dir.empty() &&
                (write_all_wavs || result.word_errors != 0 ||
                 !result.tts.stopped_on_eos ||
                 normalize(result.transcript).empty())) {
                char name[64];
                std::snprintf(name, sizeof(name), "case_%02zu.wav", index + 1);
                write_wav(std::filesystem::path(failed_wav_dir) / name, audio_24k);
            }
            results.push_back(std::move(result));
            std::fflush(stdout);
        }

        std::printf("\nCATEGORY SUMMARY\n");
        std::vector<std::string> category_names;
        category_names.reserve(categories.size());
        for (const auto& entry : categories) category_names.push_back(entry.first);
        std::sort(category_names.begin(), category_names.end());
        for (const auto& name : category_names) {
            const auto& category = categories.at(name);
            std::printf("%s cases=%d exact=%d WER=%.2f%% CER=%.2f%%\n",
                name.c_str(), category.cases, category.exact,
                ratio(category.word_errors, category.reference_words) * 100.0,
                ratio(category.character_errors,
                      category.reference_characters) * 100.0);
        }

        const double corpus_wer = ratio(aggregate.word_errors, aggregate.reference_words);
        const double corpus_cer = ratio(
            aggregate.character_errors, aggregate.reference_characters);
        const bool passed = empty_audio == 0 && empty_transcript == 0 &&
            eos_failures == 0 && corpus_wer <= kMaxAcceptedWer &&
            corpus_cer <= kMaxAcceptedCer;
        std::printf(
            "\nSUMMARY cases=%d exact=%d exact_rate=%.2f%% word_errors=%d/%d "
            "WER=%.2f%% char_errors=%d/%d CER=%.2f%% empty_audio=%d "
            "empty_transcript=%d eos_failures=%d gate=%s\n",
            aggregate.cases, aggregate.exact,
            aggregate.cases ? aggregate.exact * 100.0 / aggregate.cases : 0.0,
            aggregate.word_errors, aggregate.reference_words, corpus_wer * 100.0,
            aggregate.character_errors, aggregate.reference_characters,
            corpus_cer * 100.0, empty_audio, empty_transcript, eos_failures,
            passed ? "PASS" : "FAIL");

        write_report(report_path, results, aggregate, tts_load_ms, stt_load_ms, passed);
        return passed ? 0 : 4;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Pocket TTS round-trip failed: %s\n", error.what());
        return 3;
    }
}
