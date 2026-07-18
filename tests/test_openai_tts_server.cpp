#include "openai_tts_server.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using speech_core::http::AudioResponseFormat;
using speech_core::http::OpenAITranscriptionRequest;
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

void append_u16(std::string& value, uint16_t number) {
    value.push_back(static_cast<char>(number & 0xffu));
    value.push_back(static_cast<char>((number >> 8u) & 0xffu));
}

void append_u24(std::string& value, uint32_t number) {
    value.push_back(static_cast<char>(number & 0xffu));
    value.push_back(static_cast<char>((number >> 8u) & 0xffu));
    value.push_back(static_cast<char>((number >> 16u) & 0xffu));
}

void append_u32(std::string& value, uint32_t number) {
    value.push_back(static_cast<char>(number & 0xffu));
    value.push_back(static_cast<char>((number >> 8u) & 0xffu));
    value.push_back(static_cast<char>((number >> 16u) & 0xffu));
    value.push_back(static_cast<char>((number >> 24u) & 0xffu));
}

std::string make_wav(uint16_t format,
                     uint16_t channels,
                     uint32_t sample_rate,
                     uint16_t bits_per_sample,
                     const std::string& payload) {
    std::string wav = "RIFF";
    append_u32(wav, static_cast<uint32_t>(36u + payload.size()));
    wav += "WAVEfmt ";
    append_u32(wav, 16u);
    append_u16(wav, format);
    append_u16(wav, channels);
    append_u32(wav, sample_rate);
    const uint16_t block_align = static_cast<uint16_t>(
        channels * (bits_per_sample / 8u));
    append_u32(wav, sample_rate * block_align);
    append_u16(wav, block_align);
    append_u16(wav, bits_per_sample);
    wav += "data";
    append_u32(wav, static_cast<uint32_t>(payload.size()));
    wav += payload;
    return wav;
}

class TestServer {
public:
    TestServer(speech_core::http::SynthesizeCallback callback,
               std::string api_key = {},
               speech_core::http::TranscribeCallback transcribe = {}) {
        speech_core::http::register_openai_tts_routes(
            server_, std::move(callback), api_key);
        if (transcribe) {
            speech_core::http::register_openai_transcription_routes(
                server_, std::move(transcribe), std::move(api_key));
        }
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

class TranscriptionTestServer {
public:
    TranscriptionTestServer(speech_core::http::TranscribeCallback callback,
                            std::string api_key = {}) {
        speech_core::http::register_openai_transcription_routes(
            server_, std::move(callback), std::move(api_key));
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) throw std::runtime_error("could not bind test server");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        while (!server_.is_running()) std::this_thread::sleep_for(2ms);
    }

    ~TranscriptionTestServer() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }

    int port() const { return port_; }

private:
    httplib::Server server_;
    int port_ = 0;
    std::thread thread_;
};

std::string wav_bytes(const std::vector<float>& samples, int sample_rate) {
    const std::vector<uint8_t> wav = speech_core::http::encode_wav_pcm16(
        samples.data(), samples.size(), sample_rate);
    return std::string(reinterpret_cast<const char*>(wav.data()), wav.size());
}

httplib::UploadFormDataItems transcription_form(
    const std::string& wav,
    std::string response_format = "json") {
    httplib::UploadFormDataItems items = {
        {"file", wav, "recording.wav", "audio/wav"},
        {"model", "whisper-1", "", ""},
    };
    if (!response_format.empty()) {
        items.push_back({"response_format", std::move(response_format), "", ""});
    }
    return items;
}

httplib::Result post_transcription(
    httplib::Client& client,
    const httplib::UploadFormDataItems& items,
    const httplib::Headers& headers = {}) {
    return client.Post("/v1/audio/transcriptions", headers, items);
}

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

void test_tts_payload_limit_remains_one_mibibyte() {
    TestServer server([](const OpenAITtsRequest&) {
        return TtsAudio{{0.0f}, 24000};
    }, {}, [](const OpenAITranscriptionRequest&) {
        return speech_core::TranscriptionResult{};
    });
    httplib::Client client("127.0.0.1", server.port());
    const std::string oversized(1024 * 1024 + 1, 'x');
    auto response = client.Post(
        "/v1/audio/speech", oversized, "application/json");
    REQUIRE(response && response->status == 413);
    REQUIRE(error_body(response)["error"]["message"] ==
            "The request body must not exceed 1 MiB");
}

void test_transcription_json_contract_and_resampling() {
    OpenAITranscriptionRequest captured;
    TranscriptionTestServer server([&](const OpenAITranscriptionRequest& request) {
        captured = request;
        return speech_core::TranscriptionResult{
            "hello world", "en", 0.9f, 0.0f, 0.1f};
    });
    httplib::Client client("127.0.0.1", server.port());

    std::vector<float> samples(800);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.25f;
    }
    auto form = transcription_form(wav_bytes(samples, 8000));
    form.push_back({"language", "en", "", ""});
    form.push_back({"prompt", "product names", "", ""});
    form.push_back({"temperature", "0.25", "", ""});
    auto response = post_transcription(client, form);

    REQUIRE(response && response->status == 200);
    REQUIRE(response->get_header_value("Content-Type") == "application/json");
    REQUIRE(nlohmann::json::parse(response->body)["text"] == "hello world");
    REQUIRE(captured.model == "whisper-1");
    REQUIRE(captured.filename == "recording.wav");
    REQUIRE(captured.language == "en");
    REQUIRE(captured.prompt == "product names");
    REQUIRE(captured.temperature == 0.25f);
    REQUIRE(captured.response_format == "json");
    REQUIRE(captured.sample_rate == 16000);
    REQUIRE(captured.samples.size() == 1600);
    REQUIRE(std::all_of(
        captured.samples.begin(), captured.samples.end(),
        [](float sample) { return std::isfinite(sample); }));
}

void assert_bad_transcription(
    const httplib::UploadFormDataItems& items,
    const std::string& message_fragment);

void test_transcription_response_formats() {
    TranscriptionTestServer server([](const OpenAITranscriptionRequest&) {
        return speech_core::TranscriptionResult{
            "format test", "fr", 0.8f, 0.0f, 0.25f};
    });
    httplib::Client client("127.0.0.1", server.port());
    const std::string wav = wav_bytes(std::vector<float>(4000, 0.1f), 16000);

    auto text_response = post_transcription(
        client, transcription_form(wav, "text"));
    REQUIRE(text_response && text_response->status == 200);
    REQUIRE(text_response->get_header_value("Content-Type") ==
            "text/plain; charset=utf-8");
    REQUIRE(text_response->body == "format test");

    auto verbose_response = post_transcription(
        client, transcription_form(wav, "verbose_json"));
    REQUIRE(verbose_response && verbose_response->status == 200);
    const auto verbose = nlohmann::json::parse(verbose_response->body);
    REQUIRE(verbose["task"] == "transcribe");
    REQUIRE(verbose["language"] == "fr");
    REQUIRE(verbose["duration"] == 0.25);
    REQUIRE(verbose["segments"].size() == 1);
    REQUIRE(verbose["segments"][0]["text"] == "format test");

    auto srt_response = post_transcription(
        client, transcription_form(wav, "srt"));
    REQUIRE(srt_response && srt_response->status == 200);
    REQUIRE(srt_response->get_header_value("Content-Type") ==
            "application/x-subrip; charset=utf-8");
    REQUIRE(srt_response->body.find(
        "00:00:00,000 --> 00:00:00,250") != std::string::npos);

    auto vtt_response = post_transcription(
        client, transcription_form(wav, "vtt"));
    REQUIRE(vtt_response && vtt_response->status == 200);
    REQUIRE(vtt_response->get_header_value("Content-Type") ==
            "text/vtt; charset=utf-8");
    REQUIRE(vtt_response->body.find(
        "WEBVTT\n\n00:00:00.000 --> 00:00:00.250") == 0);
}

void test_transcription_wav_decoding_and_downmix() {
    OpenAITranscriptionRequest captured;
    TranscriptionTestServer server([&](const OpenAITranscriptionRequest& request) {
        captured = request;
        return speech_core::TranscriptionResult{"decoded", "en"};
    });
    httplib::Client client("127.0.0.1", server.port());

    std::string stereo_pcm16;
    append_u16(stereo_pcm16, 32767u);
    append_u16(stereo_pcm16, 32768u);
    append_u16(stereo_pcm16, 16384u);
    append_u16(stereo_pcm16, 49152u);
    auto stereo_response = post_transcription(
        client, transcription_form(
            make_wav(1, 2, 16000, 16, stereo_pcm16)));
    REQUIRE(stereo_response && stereo_response->status == 200);
    REQUIRE(captured.samples.size() == 2);
    REQUIRE(std::abs(captured.samples[0]) < 0.00002f);
    REQUIRE(std::abs(captured.samples[1]) < 0.00002f);

    std::string pcm24;
    append_u24(pcm24, 0x400000u);
    auto pcm24_response = post_transcription(
        client, transcription_form(make_wav(1, 1, 16000, 24, pcm24)));
    REQUIRE(pcm24_response && pcm24_response->status == 200);
    REQUIRE(captured.samples.size() == 1);
    REQUIRE(std::abs(captured.samples[0] - 0.5f) < 0.000001f);

    std::string pcm32;
    append_u32(pcm32, 0x40000000u);
    auto pcm32_response = post_transcription(
        client, transcription_form(make_wav(1, 1, 16000, 32, pcm32)));
    REQUIRE(pcm32_response && pcm32_response->status == 200);
    REQUIRE(captured.samples.size() == 1);
    REQUIRE(std::abs(captured.samples[0] - 0.5f) < 0.000001f);

    std::string float32;
    float sample = 0.25f;
    uint32_t sample_bits = 0;
    std::memcpy(&sample_bits, &sample, sizeof(sample_bits));
    append_u32(float32, sample_bits);
    auto float32_response = post_transcription(
        client, transcription_form(make_wav(3, 1, 16000, 32, float32)));
    REQUIRE(float32_response && float32_response->status == 200);
    REQUIRE(captured.samples.size() == 1);
    REQUIRE(captured.samples[0] == 0.25f);

    std::string non_finite;
    sample = std::numeric_limits<float>::quiet_NaN();
    std::memcpy(&sample_bits, &sample, sizeof(sample_bits));
    append_u32(non_finite, sample_bits);
    assert_bad_transcription(
        transcription_form(make_wav(3, 1, 16000, 32, non_finite)),
        "non-finite audio");
}

void assert_bad_transcription(
    const httplib::UploadFormDataItems& items,
    const std::string& message_fragment) {
    TranscriptionTestServer server([](const OpenAITranscriptionRequest&) {
        return speech_core::TranscriptionResult{};
    });
    httplib::Client client("127.0.0.1", server.port());
    auto response = post_transcription(client, items);
    REQUIRE(response && response->status == 400);
    const auto parsed = error_body(response);
    REQUIRE(parsed["error"]["type"] == "invalid_request_error");
    REQUIRE(parsed["error"]["message"].get<std::string>().find(message_fragment) !=
            std::string::npos);
}

void test_transcription_request_validation() {
    const std::string wav = wav_bytes(std::vector<float>(160, 0.0f), 16000);
    assert_bad_transcription(
        {{"model", "whisper-1", "", ""}}, "Missing 'file'");
    assert_bad_transcription(
        {{"file", wav, "audio.wav", "audio/wav"}}, "Missing 'model'");
    assert_bad_transcription(
        {{"file", "not a wav", "audio.wav", "audio/wav"},
         {"model", "whisper-1", "", ""}},
        "valid RIFF/WAVE");

    auto unsupported = transcription_form(wav, "mp3");
    assert_bad_transcription(unsupported, "Unsupported response format");

    auto duplicate_file = transcription_form(wav);
    duplicate_file.push_back({"file", wav, "second.wav", "audio/wav"});
    assert_bad_transcription(duplicate_file, "Duplicate 'file'");

    auto invalid_temperature = transcription_form(wav);
    invalid_temperature.push_back({"temperature", "1.1", "", ""});
    assert_bad_transcription(invalid_temperature, "between 0 and 1");

    TranscriptionTestServer server([](const OpenAITranscriptionRequest&) {
        return speech_core::TranscriptionResult{};
    });
    httplib::Client client("127.0.0.1", server.port());
    auto wrong_content_type = client.Post(
        "/v1/audio/transcriptions", "{}", "application/json");
    REQUIRE(wrong_content_type && wrong_content_type->status == 400);
    REQUIRE(error_body(wrong_content_type)["error"]["message"] ==
            "The transcription endpoint requires multipart/form-data");
}

void test_transcription_file_size_limit() {
    bool called = false;
    TranscriptionTestServer server([&](const OpenAITranscriptionRequest&) {
        called = true;
        return speech_core::TranscriptionResult{};
    });
    httplib::Client client("127.0.0.1", server.port());
    const std::string oversized(25 * 1024 * 1024 + 1, 'x');
    auto response = post_transcription(
        client, transcription_form(oversized));
    REQUIRE(response && response->status == 413);
    REQUIRE(error_body(response)["error"]["message"] ==
            "The uploaded audio file must not exceed 25 MiB");
    REQUIRE(!called);
}

void test_transcription_duration_limit() {
    bool called = false;
    TranscriptionTestServer server([&](const OpenAITranscriptionRequest&) {
        called = true;
        return speech_core::TranscriptionResult{};
    });
    httplib::Client client("127.0.0.1", server.port());
    const std::string pcm((8000 * 10 * 60 + 1) * 2, '\0');
    auto response = post_transcription(
        client, transcription_form(make_wav(1, 1, 8000, 16, pcm)));
    REQUIRE(response && response->status == 400);
    REQUIRE(error_body(response)["error"]["message"] ==
            "The uploaded WAV file must not exceed 10 minutes");
    REQUIRE(!called);
}

void test_transcription_authentication_and_backend_errors() {
    const std::string wav = wav_bytes(std::vector<float>(160, 0.0f), 16000);
    TranscriptionTestServer authenticated(
        [](const OpenAITranscriptionRequest&) {
            return speech_core::TranscriptionResult{"ok", "en"};
        },
        "secret");
    httplib::Client auth_client("127.0.0.1", authenticated.port());
    const auto form = transcription_form(wav);

    auto unauthorized = post_transcription(auth_client, form);
    REQUIRE(unauthorized && unauthorized->status == 401);
    REQUIRE(unauthorized->get_header_value("WWW-Authenticate") == "Bearer");
    REQUIRE(error_body(unauthorized)["error"]["type"] == "authentication_error");

    auto authorized = post_transcription(
        auth_client, form, {{"Authorization", "Bearer secret"}});
    REQUIRE(authorized && authorized->status == 200);

    TranscriptionTestServer failing(
        [](const OpenAITranscriptionRequest&) -> speech_core::TranscriptionResult {
            throw std::runtime_error("/private/model/path");
        });
    httplib::Client fail_client("127.0.0.1", failing.port());
    auto failure = post_transcription(fail_client, form);
    REQUIRE(failure && failure->status == 500);
    REQUIRE(error_body(failure)["error"]["message"] ==
            "Speech transcription failed");
    REQUIRE(failure->body.find("/private/model/path") == std::string::npos);
}

}  // namespace

int main() {
    test_health_and_wav_contract();
    test_pcm_and_optional_fields();
    test_request_validation();
    test_authentication_and_error_envelopes();
    test_tts_payload_limit_remains_one_mibibyte();
    test_transcription_json_contract_and_resampling();
    test_transcription_response_formats();
    test_transcription_wav_decoding_and_downmix();
    test_transcription_request_validation();
    test_transcription_file_size_limit();
    test_transcription_duration_limit();
    test_transcription_authentication_and_backend_errors();
    std::puts("OpenAI audio server tests passed");
    return 0;
}
