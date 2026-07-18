#include "openai_tts_server.h"

#include "speech_core/audio/pcm_codec.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace speech_core::http {
namespace {

constexpr size_t kMaximumInputCharacters = 4096;
constexpr size_t kMaximumRequestBytes = 1024 * 1024;
constexpr size_t kMaximumVoiceIdBytes = 128;

const std::unordered_set<std::string> kModelAliases = {
    "kokoro",
    "kokoro-82m",
    "kokoro-82m-onnx",
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
    "gpt-4o-mini-tts-2025-12-15",
};

const std::unordered_set<std::string> kOpenAIVoices = {
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "juniper", "marin", "nova", "onyx", "sage", "shimmer", "verse",
};

std::string trim_ascii(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    const auto first = std::find_if(value.begin(), value.end(), not_space);
    const auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

size_t utf8_character_count(const std::string& value) {
    size_t count = 0;
    for (unsigned char c : value) {
        if ((c & 0xC0u) != 0x80u) ++count;
    }
    return count;
}

std::string map_language(std::string language) {
    language = ascii_lower(trim_ascii(std::move(language)));
    const size_t separator = language.find_first_of("-_");
    const std::string primary = language.substr(0, separator);
    if (primary.empty() || primary == "en" || language == "english") return "en";
    if (primary == "zh" || primary == "cmn" || language == "chinese" ||
        language == "mandarin") return "zh";
    if (primary == "ja" || language == "japanese") return "ja";
    if (primary == "ko" || language == "korean") return "ko";
    if (primary == "fr" || language == "french") return "fr";
    if (primary == "es" || language == "spanish") return "es";
    if (primary == "pt" || language == "portuguese") return "pt";
    if (primary == "it" || language == "italian") return "it";
    if (primary == "hi" || language == "hindi") return "hi";
    return "en";
}

std::string resolve_voice(std::string voice) {
    voice = trim_ascii(std::move(voice));
    const std::string normalized = ascii_lower(voice);
    if (normalized == "alloy") return "af_alloy";
    return kOpenAIVoices.count(normalized) ? "af_heart" : voice;
}

bool valid_voice_id(const std::string& voice) {
    return !voice.empty() && voice.size() <= kMaximumVoiceIdBytes &&
        std::all_of(voice.begin(), voice.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9') || c == '_' || c == '-';
        });
}

void append_u16(std::vector<uint8_t>& out, uint16_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
}

void append_u32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

void set_json(httplib::Response& response, int status, const nlohmann::json& body) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_header("X-Content-Type-Options", "nosniff");
    response.set_content(body.dump(), "application/json");
}

void set_openai_error(httplib::Response& response,
                      int status,
                      const std::string& message) {
    const char* type = status == 401 || status == 403
        ? "authentication_error"
        : (status >= 500 ? "server_error" : "invalid_request_error");
    set_json(response, status, {
        {"error", {
            {"message", message},
            {"type", type},
            {"code", nullptr},
            {"param", nullptr},
        }},
    });
}

bool constant_time_equal(const std::string& lhs, const std::string& rhs) {
    const size_t count = std::max(lhs.size(), rhs.size());
    size_t diff = lhs.size() ^ rhs.size();
    for (size_t i = 0; i < count; ++i) {
        const unsigned char a = i < lhs.size()
            ? static_cast<unsigned char>(lhs[i]) : 0;
        const unsigned char b = i < rhs.size()
            ? static_cast<unsigned char>(rhs[i]) : 0;
        diff |= static_cast<size_t>(a ^ b);
    }
    return diff == 0;
}

bool authorized(const httplib::Request& request, const std::string& api_key) {
    if (api_key.empty()) return true;
    constexpr const char* kBearer = "Bearer ";
    const std::string header = request.get_header_value("Authorization");
    if (header.compare(0, 7, kBearer) != 0) return false;
    return constant_time_equal(header.substr(7), api_key);
}

OpenAITtsRequest parse_request(const std::string& body) {
    nlohmann::json payload;
    try {
        payload = nlohmann::json::parse(body);
    } catch (const std::exception&) {
        throw InvalidSpeechRequest("Invalid JSON request body");
    }
    if (!payload.is_object()) {
        throw InvalidSpeechRequest("Invalid JSON request body");
    }

    const auto required_string = [&payload](const char* field) {
        if (!payload.contains(field) || !payload[field].is_string()) {
            throw InvalidSpeechRequest(std::string("Missing '") + field + "' field");
        }
        std::string value = payload[field].get<std::string>();
        if (trim_ascii(value).empty()) {
            throw InvalidSpeechRequest(std::string("Missing '") + field + "' field");
        }
        return value;
    };

    OpenAITtsRequest request;
    request.model = ascii_lower(trim_ascii(required_string("model")));
    request.input = required_string("input");
    if (!payload.contains("voice")) {
        throw InvalidSpeechRequest("Missing 'voice' field");
    }
    std::string voice;
    if (payload["voice"].is_string()) {
        voice = payload["voice"].get<std::string>();
    } else if (payload["voice"].is_object() &&
               payload["voice"].contains("id") &&
               payload["voice"]["id"].is_string()) {
        voice = payload["voice"]["id"].get<std::string>();
    } else {
        throw InvalidSpeechRequest(
            "The 'voice' field must be a string or an object containing 'id'");
    }
    if (trim_ascii(voice).empty()) {
        throw InvalidSpeechRequest("Missing 'voice' field");
    }
    request.voice = resolve_voice(std::move(voice));

    if (!kModelAliases.count(request.model)) {
        throw InvalidSpeechRequest("Unknown TTS model: " + request.model);
    }
    if (!valid_voice_id(request.voice)) {
        throw InvalidSpeechRequest(
            "The 'voice' field must contain 1 to " +
            std::to_string(kMaximumVoiceIdBytes) +
            " ASCII letters, numbers, '_' or '-'");
    }
    if (utf8_character_count(request.input) > kMaximumInputCharacters) {
        throw InvalidSpeechRequest(
            "The 'input' field must not exceed " +
            std::to_string(kMaximumInputCharacters) + " characters");
    }

    if (payload.contains("response_format")) {
        if (!payload["response_format"].is_string()) {
            throw InvalidSpeechRequest("Invalid JSON request body");
        }
        const std::string format = ascii_lower(
            trim_ascii(payload["response_format"].get<std::string>()));
        if (format == "wav") {
            request.response_format = AudioResponseFormat::Wav;
        } else if (format == "pcm") {
            request.response_format = AudioResponseFormat::Pcm;
        } else {
            throw InvalidSpeechRequest(
                "Unsupported response format '" + format + "'; use 'wav' or 'pcm'");
        }
    }

    if (payload.contains("speed")) {
        if (!payload["speed"].is_number()) {
            throw InvalidSpeechRequest("Invalid JSON request body");
        }
        const double speed = payload["speed"].get<double>();
        if (!std::isfinite(speed) || speed < 0.25 || speed > 4.0) {
            throw InvalidSpeechRequest(
                "Speed " + std::to_string(speed) +
                " is outside the supported range 0.25...4.0");
        }
        request.speed = static_cast<float>(speed);
    }

    if (payload.contains("language")) {
        if (!payload["language"].is_string()) {
            throw InvalidSpeechRequest("Invalid JSON request body");
        }
        request.language = map_language(payload["language"].get<std::string>());
    }

    if (payload.contains("instructions")) {
        if (!payload["instructions"].is_string()) {
            throw InvalidSpeechRequest("Invalid JSON request body");
        }
        if (!trim_ascii(payload["instructions"].get<std::string>()).empty()) {
            throw InvalidSpeechRequest(
                "The 'instructions' field is not supported by the local Kokoro model");
        }
    }

    if (payload.contains("stream_format")) {
        if (!payload["stream_format"].is_string()) {
            throw InvalidSpeechRequest("Invalid JSON request body");
        }
        const std::string stream_format = ascii_lower(
            trim_ascii(payload["stream_format"].get<std::string>()));
        if (stream_format != "audio") {
            throw InvalidSpeechRequest(
                "Unsupported stream format '" + stream_format +
                "'; use 'audio'");
        }
    }
    return request;
}

}  // namespace

std::vector<uint8_t> encode_wav_pcm16(
    const float* samples,
    size_t count,
    int sample_rate) {
    if ((!samples && count != 0) || sample_rate <= 0) {
        throw std::invalid_argument("invalid PCM buffer or sample rate");
    }
    constexpr size_t kHeaderBytes = 44;
    constexpr size_t kBytesPerSample = 2;
    const size_t maximum_samples =
        (std::numeric_limits<uint32_t>::max() - 36u) / kBytesPerSample;
    if (count > maximum_samples) {
        throw std::length_error("audio is too large for RIFF/WAVE");
    }

    std::vector<uint8_t> pcm = PCMCodec::float_to_pcm16(samples, count);
    std::vector<uint8_t> wav;
    wav.reserve(kHeaderBytes + pcm.size());
    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    append_u32(wav, static_cast<uint32_t>(36u + pcm.size()));
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    append_u32(wav, 16);
    append_u16(wav, 1);
    append_u16(wav, 1);
    append_u32(wav, static_cast<uint32_t>(sample_rate));
    append_u32(wav, static_cast<uint32_t>(
        static_cast<uint64_t>(sample_rate) * kBytesPerSample));
    append_u16(wav, 2);
    append_u16(wav, 16);
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(wav, static_cast<uint32_t>(pcm.size()));
    wav.insert(wav.end(), pcm.begin(), pcm.end());
    return wav;
}

void register_openai_tts_routes(
    httplib::Server& server,
    SynthesizeCallback synthesize,
    std::string api_key) {
    if (!synthesize) {
        throw std::invalid_argument("OpenAI TTS server requires a synthesis callback");
    }
    server.set_payload_max_length(kMaximumRequestBytes);

    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        set_json(response, 200, {{"status", "ok"}});
    });

    server.Post("/v1/audio/speech",
        [synthesize = std::move(synthesize), api_key = std::move(api_key)](
            const httplib::Request& http_request,
            httplib::Response& response) {
            if (!authorized(http_request, api_key)) {
                response.set_header("WWW-Authenticate", "Bearer");
                set_openai_error(response, 401, "Invalid API key");
                return;
            }

            try {
                const OpenAITtsRequest request = parse_request(http_request.body);
                TtsAudio audio = synthesize(request);
                if (audio.samples.empty() || audio.sample_rate <= 0) {
                    throw std::runtime_error("synthesis produced no audio");
                }
                if (!std::all_of(audio.samples.begin(), audio.samples.end(),
                                 [](float sample) { return std::isfinite(sample); })) {
                    throw std::runtime_error("synthesis produced non-finite audio");
                }

                std::vector<uint8_t> bytes;
                const char* content_type = nullptr;
                if (request.response_format == AudioResponseFormat::Wav) {
                    bytes = encode_wav_pcm16(
                        audio.samples.data(), audio.samples.size(), audio.sample_rate);
                    content_type = "audio/wav";
                } else {
                    bytes = PCMCodec::float_to_pcm16(
                        audio.samples.data(), audio.samples.size());
                    content_type = "audio/pcm";
                }
                response.status = 200;
                response.set_header("Cache-Control", "no-store");
                response.set_header("X-Content-Type-Options", "nosniff");
                response.set_content(
                    std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
                    content_type);
            } catch (const InvalidSpeechRequest& error) {
                set_openai_error(response, 400, error.what());
            } catch (const std::exception&) {
                // Do not expose backend exception strings: they can contain
                // local paths, runtime details, or future model diagnostics.
                set_openai_error(response, 500, "Speech synthesis failed");
            }
        });
}

}  // namespace speech_core::http
