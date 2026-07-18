#include "openai_tts_server.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using speech_core::http::AudioResponseFormat;
using speech_core::http::OpenAITtsRequest;
using speech_core::http::TtsAudio;
using namespace std::chrono_literals;

namespace {

#define REQUIRE(condition) do {                                                \
    if (!(condition)) {                                                        \
        std::fprintf(stderr, "REQUIRE failed at %s:%d: %s\n",                \
                     __FILE__, __LINE__, #condition);                          \
        std::abort();                                                          \
    }                                                                          \
} while (false)

uint32_t read_u32(const std::string& value, size_t offset) {
    return static_cast<uint32_t>(static_cast<unsigned char>(value[offset])) |
           (static_cast<uint32_t>(static_cast<unsigned char>(value[offset + 1])) << 8u) |
           (static_cast<uint32_t>(static_cast<unsigned char>(value[offset + 2])) << 16u) |
           (static_cast<uint32_t>(static_cast<unsigned char>(value[offset + 3])) << 24u);
}

class TestServer {
public:
    TestServer(speech_core::http::SynthesizeCallback callback,
               std::string api_key = {}) {
        speech_core::http::register_openai_tts_routes(
            server_, std::move(callback), std::move(api_key));
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) throw std::runtime_error("could not bind test server");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        while (!server_.is_running()) std::this_thread::sleep_for(2ms);
    }

    ~TestServer() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    int port() const { return port_; }

private:
    httplib::Server server_;
    int port_ = 0;
    std::thread thread_;
};

nlohmann::json error_body(const httplib::Result& response) {
    REQUIRE(response);
    return nlohmann::json::parse(response->body);
}

httplib::Result post(httplib::Client& client,
                     const nlohmann::json& body,
                     const httplib::Headers& headers = {}) {
    return client.Post("/v1/audio/speech", headers, body.dump(), "application/json");
}

void test_health_and_wav_contract() {
    OpenAITtsRequest captured;
    TestServer server([&](const OpenAITtsRequest& request) {
        captured = request;
        return TtsAudio{{-1.0f, 0.0f, 1.0f}, 24000};
    });
    httplib::Client client("127.0.0.1", server.port());

    auto health = client.Get("/health");
    REQUIRE(health && health->status == 200);
    REQUIRE(nlohmann::json::parse(health->body)["status"] == "ok");

    auto response = post(client, {
        {"model", "tts-1"},
        {"input", "Bonjour"},
        {"voice", "alloy"},
        {"language", "French"},
        {"stream_format", "audio"},
    });
    REQUIRE(response && response->status == 200);
    REQUIRE(response->get_header_value("Content-Type") == "audio/wav");
    REQUIRE(response->body.size() == 50);
    REQUIRE(response->body.compare(0, 4, "RIFF") == 0);
    REQUIRE(response->body.compare(8, 4, "WAVE") == 0);
    REQUIRE(read_u32(response->body, 24) == 24000);
    REQUIRE(read_u32(response->body, 40) == 6);
    REQUIRE(captured.model == "tts-1");
    REQUIRE(captured.voice == "af_alloy");
    REQUIRE(captured.language == "fr");
    REQUIRE(captured.speed == 1.0f);
    REQUIRE(captured.response_format == AudioResponseFormat::Wav);
}

void test_pcm_and_optional_fields() {
    OpenAITtsRequest captured;
    TestServer server([&](const OpenAITtsRequest& request) {
        captured = request;
        return TtsAudio{{-1.0f, 1.0f}, 24000};
    });
    httplib::Client client("127.0.0.1", server.port());
    auto response = post(client, {
        {"model", "kokoro"},
        {"input", "Hola"},
        {"voice", {{"id", "ef_dora"}}},
        {"response_format", "pcm"},
        {"speed", 1.25},
        {"language", "es-MX"},
    });
    REQUIRE(response && response->status == 200);
    REQUIRE(response->get_header_value("Content-Type") == "audio/pcm");
    REQUIRE(response->body.size() == 4);
    REQUIRE(captured.voice == "ef_dora");
    REQUIRE(captured.language == "es");
    REQUIRE(captured.speed == 1.25f);
    REQUIRE(captured.response_format == AudioResponseFormat::Pcm);
}

void assert_bad_request(const nlohmann::json& body,
                        const std::string& message_fragment) {
    TestServer server([](const OpenAITtsRequest&) {
        return TtsAudio{{0.0f}, 24000};
    });
    httplib::Client client("127.0.0.1", server.port());
    auto response = post(client, body);
    REQUIRE(response && response->status == 400);
    const auto parsed = error_body(response);
    REQUIRE(parsed["error"]["type"] == "invalid_request_error");
    REQUIRE(parsed["error"]["message"].get<std::string>().find(message_fragment) !=
            std::string::npos);
}

void test_request_validation() {
    assert_bad_request(
        {{"input", "hello"}, {"voice", "alloy"}}, "Missing 'model'");
    assert_bad_request(
        {{"model", "tts-1"}, {"voice", "alloy"}}, "Missing 'input'");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"}}, "Missing 'voice'");
    assert_bad_request(
        {{"model", "unknown"}, {"input", "hello"}, {"voice", "alloy"}},
        "Unknown TTS model");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"}, {"voice", "alloy"},
         {"response_format", "mp3"}},
        "Unsupported response format");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"}, {"voice", "alloy"},
         {"speed", 4.1}},
        "outside the supported range");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"}, {"voice", "alloy"},
         {"instructions", "Speak warmly"}},
        "not supported by the local Kokoro model");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"}, {"voice", "alloy"},
         {"stream_format", "sse"}},
        "use 'audio'");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"},
         {"voice", "../../private/voice"}},
        "ASCII letters");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", "hello"},
         {"voice", std::string(129, 'a')}},
        "1 to 128");
    assert_bad_request(
        {{"model", "tts-1"}, {"input", std::string(4097, 'a')},
         {"voice", "alloy"}},
        "4096 characters");
}

void test_authentication_and_error_envelopes() {
    TestServer authenticated([](const OpenAITtsRequest&) {
        return TtsAudio{{0.0f}, 24000};
    }, "secret");
    httplib::Client auth_client("127.0.0.1", authenticated.port());
    const nlohmann::json body = {
        {"model", "tts-1"}, {"input", "hello"}, {"voice", "alloy"}};

    auto unauthorized = post(auth_client, body);
    REQUIRE(unauthorized && unauthorized->status == 401);
    REQUIRE(unauthorized->get_header_value("WWW-Authenticate") == "Bearer");
    REQUIRE(error_body(unauthorized)["error"]["type"] == "authentication_error");

    auto authorized = post(
        auth_client, body, {{"Authorization", "Bearer secret"}});
    REQUIRE(authorized && authorized->status == 200);

    TestServer failing([](const OpenAITtsRequest&) -> TtsAudio {
        throw std::runtime_error("model unavailable");
    });
    httplib::Client fail_client("127.0.0.1", failing.port());
    auto failure = post(fail_client, body);
    REQUIRE(failure && failure->status == 500);
    const auto parsed = error_body(failure);
    REQUIRE(parsed["error"]["type"] == "server_error");
    REQUIRE(parsed["error"]["message"] == "Speech synthesis failed");
    REQUIRE(parsed["error"]["message"].get<std::string>().find("model unavailable") ==
            std::string::npos);

    TestServer invalid_voice([](const OpenAITtsRequest&) -> TtsAudio {
        throw speech_core::http::InvalidSpeechRequest("Unknown TTS voice");
    });
    httplib::Client invalid_voice_client("127.0.0.1", invalid_voice.port());
    auto bad_voice = post(invalid_voice_client, body);
    REQUIRE(bad_voice && bad_voice->status == 400);
    REQUIRE(error_body(bad_voice)["error"]["message"] == "Unknown TTS voice");

    TestServer non_finite([](const OpenAITtsRequest&) -> TtsAudio {
        return TtsAudio{{std::numeric_limits<float>::quiet_NaN()}, 24000};
    });
    httplib::Client non_finite_client("127.0.0.1", non_finite.port());
    auto invalid_audio = post(non_finite_client, body);
    REQUIRE(invalid_audio && invalid_audio->status == 500);
    REQUIRE(error_body(invalid_audio)["error"]["message"] ==
            "Speech synthesis failed");
}

}  // namespace

int main() {
    test_health_and_wav_contract();
    test_pcm_and_optional_fields();
    test_request_validation();
    test_authentication_and_error_envelopes();
    std::puts("OpenAI TTS server tests passed");
    return 0;
}
