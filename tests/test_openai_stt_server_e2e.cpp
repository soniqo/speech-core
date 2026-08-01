#include "openai_tts_server.h"

#include "speech_core/models/parakeet_stt.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace {

#define REQUIRE(condition) do {                                                \
    if (!(condition)) {                                                        \
        std::fprintf(stderr, "REQUIRE failed at %s:%d: %s\n",                \
                     __FILE__, __LINE__, #condition);                          \
        std::abort();                                                          \
    }                                                                          \
} while (false)

bool is_nonempty_file(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && !error &&
        std::filesystem::file_size(path, error) > 0u && !error;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("could not open test audio");
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

}  // namespace

int main() {
    const char* configured_dir = std::getenv("SPEECH_MODEL_DIR");
    if (!configured_dir || !*configured_dir) {
        std::puts("OpenAI transcription E2E skipped: SPEECH_MODEL_DIR is not set");
        return 0;
    }

    const std::filesystem::path model_dir(configured_dir);
    const std::filesystem::path encoder =
        model_dir / "parakeet-encoder-int8.onnx";
    const std::filesystem::path decoder =
        model_dir / "parakeet-decoder-joint-int8.onnx";
    const std::filesystem::path vocabulary = model_dir / "vocab.json";
    if (!is_nonempty_file(encoder) || !is_nonempty_file(decoder) ||
        !is_nonempty_file(vocabulary)) {
        std::puts("OpenAI transcription E2E skipped: Parakeet bundle is incomplete");
        return 0;
    }

    speech_core::ParakeetStt stt(
        encoder.string(), decoder.string(), vocabulary.string(),
        /*hw_accel=*/false);
    httplib::Server server;
    speech_core::http::register_openai_transcription_routes(
        server,
        [&](const speech_core::http::OpenAITranscriptionRequest& request) {
            return stt.transcribe(
                request.samples.data(), request.samples.size(),
                request.sample_rate);
        });

    const int port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread thread([&] { server.listen_after_bind(); });
    while (!server.is_running()) std::this_thread::sleep_for(2ms);

    httplib::Client client("127.0.0.1", port);
    const std::string audio = read_file(
        std::filesystem::path(SPEECH_CORE_TEST_DATA_DIR) / "test_audio.wav");
    const httplib::UploadFormDataItems form = {
        {"file", audio, "test_audio.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
        {"response_format", "json", "", ""},
    };
    auto response = client.Post("/v1/audio/transcriptions", form);
    server.stop();
    thread.join();

    REQUIRE(response && response->status == 200);
    const auto body = nlohmann::json::parse(response->body);
    REQUIRE(body.contains("text") && body["text"].is_string());
    std::string transcript = body["text"].get<std::string>();
    std::transform(transcript.begin(), transcript.end(), transcript.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    int matched_words = 0;
    for (const char* word : {"guarantee", "replacement", "shipped", "tomorrow"}) {
        if (transcript.find(word) != std::string::npos) ++matched_words;
    }
    REQUIRE(matched_words >= 2);
    std::puts("OpenAI transcription E2E passed");
    return 0;
}
