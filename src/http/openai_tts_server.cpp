#include "openai_tts_server.h"

#include "speech_core/audio/pcm_codec.h"
#include "speech_core/audio/resampler.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace speech_core::http {
namespace {

constexpr size_t kMaximumInputCharacters = 4096;
constexpr size_t kMaximumTtsRequestBytes = 1024 * 1024;
constexpr size_t kMaximumAudioFileBytes = 25 * 1024 * 1024;
constexpr size_t kMaximumServerRequestBytes = 26 * 1024 * 1024;
constexpr size_t kMaximumVoiceIdBytes = 128;
constexpr size_t kMaximumModelBytes = 128;
constexpr size_t kMaximumFilenameBytes = 255;
constexpr size_t kMaximumLanguageBytes = 64;
constexpr int kTranscriptionSampleRate = 16000;
constexpr size_t kMaximumAudioSeconds = 10 * 60;

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

class RequestTooLarge : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
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

void set_text(httplib::Response& response,
              int status,
              std::string body,
              const char* content_type) {
    response.status = status;
    response.set_header("Cache-Control", "no-store");
    response.set_header("X-Content-Type-Options", "nosniff");
    response.set_content(std::move(body), content_type);
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

uint16_t read_u16(const std::string& bytes, size_t offset) {
    return static_cast<uint16_t>(
        static_cast<unsigned char>(bytes[offset]) |
        (static_cast<uint16_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8u));
}

uint32_t read_u32(const std::string& bytes, size_t offset) {
    return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 1])) << 8u) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 2])) << 16u) |
        (static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset + 3])) << 24u);
}

std::vector<float> decode_wav_to_mono_16k(const std::string& bytes) {
    if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 ||
        bytes.compare(8, 4, "WAVE") != 0) {
        throw InvalidSpeechRequest("The uploaded file must be a valid RIFF/WAVE file");
    }

    const uint64_t declared_end = static_cast<uint64_t>(read_u32(bytes, 4)) + 8u;
    if (declared_end < 12u || declared_end > bytes.size()) {
        throw InvalidSpeechRequest("The uploaded WAV file is truncated");
    }
    const size_t riff_end = static_cast<size_t>(declared_end);

    bool have_format = false;
    bool have_data = false;
    uint16_t format = 0;
    uint16_t channels = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 0;
    uint32_t sample_rate = 0;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while (offset + 8u <= riff_end) {
        const uint32_t chunk_size_u32 = read_u32(bytes, offset + 4u);
        const size_t chunk_size = static_cast<size_t>(chunk_size_u32);
        const size_t content_offset = offset + 8u;
        if (chunk_size > riff_end - content_offset) {
            throw InvalidSpeechRequest("The uploaded WAV file contains a truncated chunk");
        }
        if (bytes.compare(offset, 4, "fmt ") == 0 && !have_format) {
            if (chunk_size < 16u) {
                throw InvalidSpeechRequest("The uploaded WAV file has an invalid format chunk");
            }
            format = read_u16(bytes, content_offset);
            channels = read_u16(bytes, content_offset + 2u);
            sample_rate = read_u32(bytes, content_offset + 4u);
            block_align = read_u16(bytes, content_offset + 12u);
            bits_per_sample = read_u16(bytes, content_offset + 14u);
            have_format = true;
        } else if (bytes.compare(offset, 4, "data") == 0 && !have_data) {
            data_offset = content_offset;
            data_size = chunk_size;
            have_data = true;
        }

        const size_t padded_size = chunk_size + (chunk_size & 1u);
        if (padded_size > riff_end - content_offset) {
            if (chunk_size == riff_end - content_offset) break;
            throw InvalidSpeechRequest("The uploaded WAV file contains an invalid chunk size");
        }
        offset = content_offset + padded_size;
    }

    if (!have_format || !have_data || data_size == 0u) {
        throw InvalidSpeechRequest("The uploaded WAV file must contain format and audio data");
    }
    if (channels == 0u || channels > 32u) {
        throw InvalidSpeechRequest("The uploaded WAV file has an unsupported channel count");
    }
    if (sample_rate < 8000u || sample_rate > 384000u) {
        throw InvalidSpeechRequest("The uploaded WAV file has an unsupported sample rate");
    }

    const bool is_pcm = format == 1u &&
        (bits_per_sample == 16u || bits_per_sample == 24u || bits_per_sample == 32u);
    const bool is_float = format == 3u && bits_per_sample == 32u;
    if (!is_pcm && !is_float) {
        throw InvalidSpeechRequest(
            "Unsupported WAV encoding; use PCM16, PCM24, PCM32, or Float32");
    }
    const size_t bytes_per_sample = bits_per_sample / 8u;
    const size_t expected_block_align = static_cast<size_t>(channels) * bytes_per_sample;
    if (block_align != expected_block_align || data_size % expected_block_align != 0u) {
        throw InvalidSpeechRequest("The uploaded WAV file has an invalid sample layout");
    }

    const size_t frame_count = data_size / expected_block_align;
    if (static_cast<uint64_t>(frame_count) >
        static_cast<uint64_t>(sample_rate) * kMaximumAudioSeconds) {
        throw InvalidSpeechRequest(
            "The uploaded WAV file must not exceed 10 minutes");
    }
    std::vector<float> mono;
    mono.reserve(frame_count);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        double sum = 0.0;
        for (size_t channel = 0; channel < channels; ++channel) {
            const size_t sample_offset = data_offset +
                frame * expected_block_align + channel * bytes_per_sample;
            float sample = 0.0f;
            if (is_float) {
                const uint32_t raw = read_u32(bytes, sample_offset);
                static_assert(sizeof(float) == sizeof(raw), "Float32 must be 32 bits");
                std::memcpy(&sample, &raw, sizeof(sample));
            } else if (bits_per_sample == 16u) {
                int32_t value = read_u16(bytes, sample_offset);
                if (value >= (1 << 15)) value -= (1 << 16);
                sample = static_cast<float>(value) / 32768.0f;
            } else if (bits_per_sample == 24u) {
                int32_t value =
                    static_cast<unsigned char>(bytes[sample_offset]) |
                    (static_cast<int32_t>(
                         static_cast<unsigned char>(bytes[sample_offset + 1u])) << 8u) |
                    (static_cast<int32_t>(
                         static_cast<unsigned char>(bytes[sample_offset + 2u])) << 16u);
                if (value >= (1 << 23)) value -= (1 << 24);
                sample = static_cast<float>(value) / 8388608.0f;
            } else {
                int64_t value = read_u32(bytes, sample_offset);
                if (value >= (int64_t{1} << 31)) value -= (int64_t{1} << 32);
                sample = static_cast<float>(
                    static_cast<double>(value) / 2147483648.0);
            }
            if (!std::isfinite(sample)) {
                throw InvalidSpeechRequest("The uploaded WAV file contains non-finite audio");
            }
            sample = std::clamp(sample, -1.0f, 1.0f);
            sum += sample;
        }
        mono.push_back(static_cast<float>(sum / channels));
    }

    if (sample_rate == kTranscriptionSampleRate) return mono;
    std::vector<float> resampled = speech_core::Resampler::resample(
        mono.data(), mono.size(), static_cast<int>(sample_rate),
        kTranscriptionSampleRate);
    if (resampled.empty()) {
        throw InvalidSpeechRequest("The uploaded WAV file contains too little audio");
    }
    if (!std::all_of(resampled.begin(), resampled.end(),
                     [](float sample) { return std::isfinite(sample); })) {
        throw InvalidSpeechRequest("The uploaded WAV file produced invalid audio");
    }
    return resampled;
}

std::string single_form_field(const httplib::Request& request,
                              const char* name,
                              bool required,
                              size_t maximum_bytes) {
    const size_t count = request.form.get_field_count(name);
    if (count > 1u) {
        throw InvalidSpeechRequest(std::string("Duplicate '") + name + "' field");
    }
    std::string value = count == 1u ? request.form.get_field(name) : std::string{};
    value = trim_ascii(std::move(value));
    if (required && value.empty()) {
        throw InvalidSpeechRequest(std::string("Missing '") + name + "' field");
    }
    if (value.size() > maximum_bytes) {
        throw InvalidSpeechRequest(
            std::string("The '") + name + "' field is too long");
    }
    return value;
}

OpenAITranscriptionRequest parse_transcription_request(
    const httplib::Request& request) {
    if (!request.is_multipart_form_data()) {
        throw InvalidSpeechRequest(
            "The transcription endpoint requires multipart/form-data");
    }
    if (request.form.get_file_count("file") == 0u) {
        throw InvalidSpeechRequest("Missing 'file' field");
    }
    if (request.form.get_file_count("file") != 1u) {
        throw InvalidSpeechRequest("Duplicate 'file' field");
    }

    const httplib::FormData& file = request.form.files.find("file")->second;
    if (file.content.empty()) {
        throw InvalidSpeechRequest("The uploaded audio file is empty");
    }
    if (file.content.size() > kMaximumAudioFileBytes) {
        throw RequestTooLarge("The uploaded audio file must not exceed 25 MiB");
    }

    OpenAITranscriptionRequest parsed;
    parsed.model = single_form_field(request, "model", true, kMaximumModelBytes);
    parsed.filename = file.filename.empty() ? "audio.wav" : file.filename;
    if (parsed.filename.size() > kMaximumFilenameBytes) {
        throw InvalidSpeechRequest("The uploaded filename is too long");
    }
    parsed.language = single_form_field(
        request, "language", false, kMaximumLanguageBytes);
    parsed.prompt = single_form_field(
        request, "prompt", false, kMaximumInputCharacters * 4u);
    if (utf8_character_count(parsed.prompt) > kMaximumInputCharacters) {
        throw InvalidSpeechRequest("The 'prompt' field must not exceed 4096 characters");
    }

    parsed.response_format = ascii_lower(single_form_field(
        request, "response_format", false, 32u));
    if (parsed.response_format.empty()) parsed.response_format = "json";
    static const std::unordered_set<std::string> formats = {
        "json", "text", "verbose_json", "srt", "vtt"};
    if (!formats.count(parsed.response_format)) {
        throw InvalidSpeechRequest(
            "Unsupported response format '" + parsed.response_format +
            "'; use 'json', 'text', 'verbose_json', 'srt', or 'vtt'");
    }

    const std::string temperature = single_form_field(
        request, "temperature", false, 32u);
    if (!temperature.empty()) {
        size_t consumed = 0;
        double value = 0.0;
        try {
            value = std::stod(temperature, &consumed);
        } catch (const std::exception&) {
            throw InvalidSpeechRequest("The 'temperature' field must be a number");
        }
        if (consumed != temperature.size() || !std::isfinite(value) ||
            value < 0.0 || value > 1.0) {
            throw InvalidSpeechRequest(
                "The 'temperature' field must be between 0 and 1");
        }
        parsed.temperature = static_cast<float>(value);
    }
    parsed.samples = decode_wav_to_mono_16k(file.content);
    parsed.sample_rate = kTranscriptionSampleRate;
    return parsed;
}

std::string timestamp(double seconds, char separator) {
    if (!std::isfinite(seconds) || seconds < 0.0) seconds = 0.0;
    const uint64_t milliseconds = static_cast<uint64_t>(std::llround(seconds * 1000.0));
    const uint64_t hours = milliseconds / 3600000u;
    const uint64_t minutes = (milliseconds / 60000u) % 60u;
    const uint64_t whole_seconds = (milliseconds / 1000u) % 60u;
    const uint64_t remainder = milliseconds % 1000u;
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%02llu:%02llu:%02llu%c%03llu",
                  static_cast<unsigned long long>(hours),
                  static_cast<unsigned long long>(minutes),
                  static_cast<unsigned long long>(whole_seconds), separator,
                  static_cast<unsigned long long>(remainder));
    return buffer;
}

void set_transcription_response(
    httplib::Response& response,
    const OpenAITranscriptionRequest& request,
    const speech_core::TranscriptionResult& result) {
    const double duration = static_cast<double>(request.samples.size()) /
        static_cast<double>(request.sample_rate);
    const double start = std::isfinite(result.start_time)
        ? std::clamp(static_cast<double>(result.start_time), 0.0, duration) : 0.0;
    const double requested_end = result.end_time > result.start_time
        ? static_cast<double>(result.end_time) : duration;
    const double end = std::isfinite(requested_end)
        ? std::clamp(requested_end, start, duration) : duration;
    if (request.response_format == "text") {
        set_text(response, 200, result.text, "text/plain; charset=utf-8");
    } else if (request.response_format == "srt") {
        set_text(response, 200,
                 "1\n" + timestamp(start, ',') + " --> " + timestamp(end, ',') +
                     "\n" + result.text + "\n",
                 "application/x-subrip; charset=utf-8");
    } else if (request.response_format == "vtt") {
        set_text(response, 200,
                 "WEBVTT\n\n" + timestamp(start, '.') + " --> " +
                     timestamp(end, '.') + "\n" + result.text + "\n",
                 "text/vtt; charset=utf-8");
    } else if (request.response_format == "verbose_json") {
        const std::string language = !result.language.empty()
            ? result.language : (!request.language.empty() ? request.language : "unknown");
        set_json(response, 200, {
            {"task", "transcribe"},
            {"language", language},
            {"duration", duration},
            {"text", result.text},
            {"segments", nlohmann::json::array({{
                {"id", 0},
                {"seek", 0},
                {"start", start},
                {"end", end},
                {"text", result.text},
                {"tokens", nlohmann::json::array()},
                {"temperature", request.temperature},
                {"avg_logprob", 0.0},
                {"compression_ratio", 1.0},
                {"no_speech_prob", 0.0},
            }})},
        });
    } else {
        set_json(response, 200, {{"text", result.text}});
    }
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
    // Preserve the original transport cap when this is the only registered
    // audio route. The transcription registrar raises the shared server limit;
    // the handler-level check below still protects TTS in a combined server.
    server.set_payload_max_length(kMaximumTtsRequestBytes);

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
                if (http_request.body.size() > kMaximumTtsRequestBytes) {
                    set_openai_error(
                        response, 413, "The request body must not exceed 1 MiB");
                    return;
                }
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

void register_openai_transcription_routes(
    httplib::Server& server,
    TranscribeCallback transcribe,
    std::string api_key) {
    if (!transcribe) {
        throw std::invalid_argument(
            "OpenAI transcription server requires a transcription callback");
    }
    server.set_payload_max_length(kMaximumServerRequestBytes);
    auto request_mutex = std::make_shared<std::mutex>();

    server.Post("/v1/audio/transcriptions",
        [transcribe = std::move(transcribe), api_key = std::move(api_key),
         request_mutex = std::move(request_mutex)](
            const httplib::Request& http_request,
            httplib::Response& response) {
            if (!authorized(http_request, api_key)) {
                response.set_header("WWW-Authenticate", "Bearer");
                set_openai_error(response, 401, "Invalid API key");
                return;
            }

            // Bound decoded-audio and model memory even when cpp-httplib
            // dispatches several large multipart requests concurrently.
            std::lock_guard<std::mutex> request_lock(*request_mutex);
            try {
                OpenAITranscriptionRequest request =
                    parse_transcription_request(http_request);
                const speech_core::TranscriptionResult result = transcribe(request);
                set_transcription_response(response, request, result);
            } catch (const RequestTooLarge& error) {
                set_openai_error(response, 413, error.what());
            } catch (const InvalidSpeechRequest& error) {
                set_openai_error(response, 400, error.what());
            } catch (const std::exception&) {
                // Do not expose backend exception strings: they can contain
                // local paths, runtime details, or model diagnostics.
                set_openai_error(response, 500, "Speech transcription failed");
            }
        });
}

}  // namespace speech_core::http
