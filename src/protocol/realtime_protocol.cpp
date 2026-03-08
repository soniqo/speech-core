#include "speech_core/protocol/realtime_protocol.h"
#include "speech_core/audio/pcm_codec.h"

// Minimal JSON handling — no external dependency.
// For production, consider nlohmann/json or simdjson.

#include <sstream>

namespace speech_core {

namespace {

// Simple JSON value extraction (handles flat objects only)
std::string extract_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    auto end = json.find('"', pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

std::string escape_json(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"':  result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:   result += c; break;
        }
    }
    return result;
}

}  // namespace

std::optional<PipelineEvent> RealtimeProtocol::parse(const std::string& json) {
    std::string type = extract_string(json, "type");
    if (type.empty()) return std::nullopt;

    PipelineEvent event;
    event.session_id = extract_string(json, "session_id");
    event.item_id = extract_string(json, "item_id");
    event.response_id = extract_string(json, "response_id");

    if (type == "session.update") {
        event.type = EventType::SessionUpdated;
    } else if (type == "input_audio_buffer.append") {
        event.type = EventType::InputAudioBufferAppend;
        std::string audio_b64 = extract_string(json, "audio");
        if (!audio_b64.empty()) {
            event.audio_data = PCMCodec::from_base64(audio_b64);
        }
    } else if (type == "input_audio_buffer.commit") {
        event.type = EventType::InputAudioBufferCommit;
    } else if (type == "input_audio_buffer.clear") {
        event.type = EventType::InputAudioBufferClear;
    } else if (type == "response.create") {
        event.type = EventType::ResponseCreated;
        event.text = extract_string(json, "instructions");
    } else {
        return std::nullopt;
    }

    return event;
}

std::string RealtimeProtocol::serialize(const PipelineEvent& event) {
    // Delegate to specific serializers based on type
    switch (event.type) {
    case EventType::SessionCreated:
        return session_created(event.session_id);
    case EventType::TranscriptionCompleted:
        return transcription_completed(event.item_id, event.text);
    case EventType::Error:
        return error(event.text);
    default:
        break;
    }

    // Generic fallback
    std::ostringstream ss;
    ss << "{\"type\":\"unknown\"}";
    return ss.str();
}

std::string RealtimeProtocol::session_created(const std::string& session_id) {
    std::ostringstream ss;
    ss << "{\"type\":\"session.created\","
       << "\"session\":{\"id\":\"" << escape_json(session_id) << "\","
       << "\"model\":\"speech-core\","
       << "\"modalities\":[\"audio\",\"text\"],"
       << "\"input_audio_format\":\"pcm16\","
       << "\"output_audio_format\":\"pcm16\"}}";
    return ss.str();
}

std::string RealtimeProtocol::audio_delta(
    const std::string& response_id,
    const uint8_t* pcm16_data, size_t byte_count) {
    std::ostringstream ss;
    ss << "{\"type\":\"response.audio.delta\","
       << "\"response_id\":\"" << escape_json(response_id) << "\","
       << "\"delta\":\"" << PCMCodec::to_base64(pcm16_data, byte_count) << "\"}";
    return ss.str();
}

std::string RealtimeProtocol::transcription_completed(
    const std::string& item_id,
    const std::string& transcript) {
    std::ostringstream ss;
    ss << "{\"type\":\"conversation.item.input_audio_transcription.completed\","
       << "\"item_id\":\"" << escape_json(item_id) << "\","
       << "\"transcript\":\"" << escape_json(transcript) << "\"}";
    return ss.str();
}

std::string RealtimeProtocol::error(const std::string& message) {
    std::ostringstream ss;
    ss << "{\"type\":\"error\","
       << "\"error\":{\"type\":\"invalid_request_error\","
       << "\"message\":\"" << escape_json(message) << "\"}}";
    return ss.str();
}

}  // namespace speech_core
