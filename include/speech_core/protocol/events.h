#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// Event types for the voice agent pipeline.
/// Compatible with OpenAI Realtime API event naming.
enum class EventType {
    // Session
    SessionCreated,
    SessionUpdated,

    // Input audio
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    InputAudioBufferClear,
    InputAudioBufferCleared,
    InputAudioBufferCommitted,

    // VAD / turn detection
    SpeechStarted,
    SpeechEnded,

    // Transcription
    TranscriptionPartial,
    TranscriptionCompleted,

    // Response lifecycle
    ResponseCreated,
    ResponseDone,

    // Audio output
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseAudioTranscriptDelta,
    ResponseAudioTranscriptDone,

    // Conversation
    ConversationItemCreated,

    // Error
    Error
};

/// A pipeline event with associated data.
struct PipelineEvent {
    EventType type;
    std::string session_id;
    std::string item_id;
    std::string response_id;

    // Payload — only relevant fields populated per event type
    std::string text;                   // transcript, error message, token
    std::vector<uint8_t> audio_data;    // PCM16 bytes (for audio delta)
    float start_time = 0.0f;
    float end_time = 0.0f;
    float confidence = 0.0f;
};

}  // namespace speech_core
