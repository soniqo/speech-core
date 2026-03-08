#pragma once

#include <string>
#include <optional>

#include "speech_core/protocol/events.h"

namespace speech_core {

/// Parser and serializer for the OpenAI Realtime API JSON protocol.
///
/// Handles conversion between JSON strings (as sent over WebSocket)
/// and typed PipelineEvent structs. Does NOT own the transport —
/// the platform layer manages WebSocket connections.
class RealtimeProtocol {
public:
    /// Parse a JSON message string into a PipelineEvent.
    /// @param json  Raw JSON string from WebSocket
    /// @return Parsed event, or nullopt if the message is malformed
    static std::optional<PipelineEvent> parse(const std::string& json);

    /// Serialize a PipelineEvent to a JSON string.
    /// @param event  Event to serialize
    /// @return JSON string suitable for sending over WebSocket
    static std::string serialize(const PipelineEvent& event);

    /// Create a session.created event.
    static std::string session_created(const std::string& session_id);

    /// Create a response.audio.delta event with base64-encoded PCM16 audio.
    static std::string audio_delta(
        const std::string& response_id,
        const uint8_t* pcm16_data, size_t byte_count);

    /// Create a transcription completed event.
    static std::string transcription_completed(
        const std::string& item_id,
        const std::string& transcript);

    /// Create an error event.
    static std::string error(const std::string& message);
};

}  // namespace speech_core
