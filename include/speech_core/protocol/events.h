#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// Event types emitted by the voice agent pipeline.
enum class EventType {
    // Session
    SessionCreated,

    // VAD / turn detection
    SpeechStarted,
    SpeechEnded,

    // Transcription
    TranscriptionCompleted,

    // Response lifecycle
    ResponseCreated,
    ResponseInterrupted,
    ResponseDone,

    // Audio output
    ResponseAudioDelta,

    // Tool calling
    ToolCallStarted,
    ToolCallCompleted,

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

    // Per-stage latency in milliseconds (populated where applicable)
    float stt_duration_ms = 0.0f;   // TranscriptionCompleted, ResponseDone
    float llm_duration_ms = 0.0f;   // ResponseCreated, ResponseDone
    float tts_duration_ms = 0.0f;   // ResponseDone
};

}  // namespace speech_core
