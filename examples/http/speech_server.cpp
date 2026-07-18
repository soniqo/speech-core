#include "openai_tts_server.h"

#include "speech_core/models/kokoro_tts.h"
#include "speech_core/models/parakeet_stt.h"

#include "default_model_dir.h"
#include "httplib.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_dir = speech_example_model_dir();
    std::string api_key;
    bool allow_unauthenticated_remote = false;
};

bool is_loopback_host(const std::string& host) {
    return host == "127.0.0.1" || host == "::1";
}

bool is_nonempty_regular_file(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) return false;
    return std::filesystem::file_size(path, error) > 0 && !error;
}

bool is_valid_voice_file(const std::filesystem::path& path) {
    constexpr uintmax_t kVoiceEmbeddingBytes = 256 * sizeof(float);
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) return false;
    return std::filesystem::file_size(path, error) == kVoiceEmbeddingBytes && !error;
}

void print_usage(const char* executable) {
    std::cout
        << "usage: " << executable << " [options]\n"
        << "\n"
        << "OpenAI-compatible local audio server backed by ONNX models.\n"
        << "Endpoints:\n"
        << "  POST /v1/audio/speech          Kokoro text-to-speech\n"
        << "  POST /v1/audio/transcriptions  Parakeet speech-to-text\n"
        << "\n"
        << "options:\n"
        << "  --host HOST       bind address (default: 127.0.0.1)\n"
        << "  --port PORT       listen port (default: 8080)\n"
        << "  --model-dir PATH  Kokoro and Parakeet ONNX bundle directory\n"
        << "                    (default: $SPEECH_MODEL_DIR or platform cache)\n"
        << "  --api-key KEY     require Authorization: Bearer KEY\n"
        << "                    (default: $SPEECH_SERVER_API_KEY or no auth)\n"
        << "  --allow-unauthenticated-remote\n"
        << "                    permit a non-loopback bind without an API key\n"
        << "  -h, --help        show this help\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    if (const char* key = std::getenv("SPEECH_SERVER_API_KEY")) {
        options.api_key = key;
    }
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (argument == "--allow-unauthenticated-remote") {
            options.allow_unauthenticated_remote = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++i];
        if (argument == "--host") {
            options.host = value;
        } else if (argument == "--port") {
            size_t parsed = 0;
            try {
                options.port = std::stoi(value, &parsed);
            } catch (const std::exception&) {
                throw std::invalid_argument("port must be in 1...65535");
            }
            if (parsed != value.size() || options.port <= 0 || options.port > 65535) {
                throw std::invalid_argument("port must be in 1...65535");
            }
        } else if (argument == "--model-dir") {
            options.model_dir = value;
        } else if (argument == "--api-key") {
            options.api_key = value;
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.host.empty()) throw std::invalid_argument("host must not be empty");
    if (options.model_dir.empty()) throw std::invalid_argument("model directory must not be empty");
    if (!is_loopback_host(options.host) && options.api_key.empty() &&
        !options.allow_unauthenticated_remote) {
        throw std::invalid_argument(
            "a non-loopback bind requires --api-key or SPEECH_SERVER_API_KEY; "
            "use --allow-unauthenticated-remote only on a trusted network");
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        namespace fs = std::filesystem;
        const fs::path model_dir(options.model_dir);
        const fs::path model_path = model_dir / "kokoro-e2e.onnx";
        const fs::path voices_dir = model_dir / "voices";
        const fs::path stt_encoder_path =
            model_dir / "parakeet-encoder-int8.onnx";
        const fs::path stt_decoder_path =
            model_dir / "parakeet-decoder-joint-int8.onnx";
        const fs::path stt_vocab_path = model_dir / "vocab.json";
        if (!is_nonempty_regular_file(model_path)) {
            throw std::runtime_error(
                "missing " + model_path.string() +
#ifdef _WIN32
                "; run .\\speech_download_models.ps1 first");
#else
                "; run speech download-models first");
#endif
        }
        if (!fs::is_directory(voices_dir)) {
            throw std::runtime_error("missing voice directory " + voices_dir.string());
        }
        for (const char* support_file : {
                 "kokoro-e2e.onnx.data", "vocab_index.json",
                 "us_gold.json", "us_silver.json"}) {
            const fs::path path = model_dir / support_file;
            if (!is_nonempty_regular_file(path)) {
                throw std::runtime_error("missing or empty " + path.string());
            }
        }
        if (!is_valid_voice_file(voices_dir / "af_heart.bin")) {
            throw std::runtime_error(
                "missing or invalid default voice " +
                (voices_dir / "af_heart.bin").string());
        }

        auto tts = std::make_unique<speech_core::KokoroTts>(
            model_path.string(),
            voices_dir.string(),
            model_dir.string(),
            /*hw_accel=*/false);
        std::unique_ptr<speech_core::ParakeetStt> stt;
        std::mutex inference_mutex;

        httplib::Server server;
        speech_core::http::register_openai_tts_routes(
            server,
            [&](const speech_core::http::OpenAITtsRequest& request) {
                std::lock_guard<std::mutex> lock(inference_mutex);
                const fs::path voice_path = voices_dir / (request.voice + ".bin");
                if (!is_valid_voice_file(voice_path)) {
                    throw speech_core::http::InvalidSpeechRequest(
                        "Unknown or invalid TTS voice: " + request.voice);
                }
                tts->set_voice(request.voice);
                tts->set_speed(request.speed);

                std::vector<float> samples;
                tts->synthesize(
                    request.input,
                    request.language,
                    [&](const float* chunk, size_t length, bool) {
                        samples.insert(samples.end(), chunk, chunk + length);
                    });
                return speech_core::http::TtsAudio{
                    std::move(samples), tts->output_sample_rate()};
            },
            options.api_key);
        speech_core::http::register_openai_transcription_routes(
            server,
            [&](const speech_core::http::OpenAITranscriptionRequest& request) {
                std::lock_guard<std::mutex> lock(inference_mutex);
                if (!stt) {
                    for (const fs::path& path : {
                             stt_encoder_path, stt_decoder_path, stt_vocab_path}) {
                        if (!is_nonempty_regular_file(path)) {
                            throw std::runtime_error(
                                "missing or empty " + path.string());
                        }
                    }
                    stt = std::make_unique<speech_core::ParakeetStt>(
                        stt_encoder_path.string(),
                        stt_decoder_path.string(),
                        stt_vocab_path.string(),
                        /*hw_accel=*/false);
                }
                return stt->transcribe(
                    request.samples.data(), request.samples.size(),
                    request.sample_rate);
            },
            options.api_key);

        if (!is_loopback_host(options.host) &&
            options.allow_unauthenticated_remote && options.api_key.empty()) {
            std::cerr
                << "warning: speech-server is listening beyond loopback without an API key\n";
        }
        if (!server.bind_to_port(options.host, options.port)) {
            throw std::runtime_error("could not bind the configured address");
        }
        std::cout << "speech-server listening on http://" << options.host << ':'
                  << options.port << "\n";
        std::cout << "POST /v1/audio/speech\n"
                  << "POST /v1/audio/transcriptions\n" << std::flush;
        if (!server.listen_after_bind()) {
            throw std::runtime_error("server stopped before accepting requests");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "speech-server: " << error.what() << '\n';
        return 1;
    }
}
