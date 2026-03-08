#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace speech_core {

struct ToolDefinition;  // forward declaration, defined in tools/tool_types.h

// ---------------------------------------------------------------------------
// Message types for conversation context
// ---------------------------------------------------------------------------

enum class MessageRole { System, User, Assistant, Tool };

struct Message {
    MessageRole role;
    std::string content;
    double timestamp = 0.0;  // seconds since session start
};

// ---------------------------------------------------------------------------
// STT — Speech-to-Text
// ---------------------------------------------------------------------------

struct TranscriptionResult {
    std::string text;
    float confidence = 1.0f;
    float start_time = 0.0f;  // seconds
    float end_time = 0.0f;
};

class STTInterface {
public:
    virtual ~STTInterface() = default;

    /// Transcribe audio buffer to text.
    /// @param audio  PCM Float32 samples
    /// @param length Number of samples
    /// @param sample_rate Sample rate in Hz
    /// @return Transcription result
    virtual TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) = 0;

    /// Expected input sample rate in Hz.
    virtual int input_sample_rate() const = 0;
};

// ---------------------------------------------------------------------------
// TTS — Text-to-Speech
// ---------------------------------------------------------------------------

/// Called for each audio chunk during streaming synthesis.
/// @param samples  PCM Float32 samples
/// @param length   Number of samples
/// @param is_final True if this is the last chunk
using TTSChunkCallback = std::function<void(const float* samples, size_t length, bool is_final)>;

class TTSInterface {
public:
    virtual ~TTSInterface() = default;

    /// Synthesize text to audio, streaming chunks via callback.
    virtual void synthesize(
        const std::string& text,
        const std::string& language,
        TTSChunkCallback on_chunk) = 0;

    /// Output sample rate in Hz.
    virtual int output_sample_rate() const = 0;

    /// Cancel any in-progress synthesis. Thread-safe.
    virtual void cancel() {}
};

// ---------------------------------------------------------------------------
// LLM — Language Model
// ---------------------------------------------------------------------------

/// A tool call request from the LLM.
struct ToolCall {
    std::string name;       // tool name
    std::string arguments;  // JSON arguments string
};

/// LLM response — either text or a tool call.
struct LLMResponse {
    std::string text;                // accumulated text response
    std::vector<ToolCall> tool_calls;  // tool calls requested by the LLM
};

/// Called for each token during streaming generation.
/// @param token  Text token
/// @param is_final True if generation is complete
using LLMTokenCallback = std::function<void(const std::string& token, bool is_final)>;

class LLMInterface {
public:
    virtual ~LLMInterface() = default;

    /// Generate a response given conversation history.
    /// The implementation should populate tool_calls if the LLM decides
    /// to call tools, or stream text tokens via on_token.
    /// @return LLM response with text and/or tool calls
    virtual LLMResponse chat(
        const std::vector<Message>& messages,
        LLMTokenCallback on_token) = 0;

    /// Provide tool definitions to the LLM.
    /// Called once when tools are registered. Default: no-op.
    virtual void set_tools(const std::vector<ToolDefinition>& /*tools*/) {}

    /// Cancel any in-progress generation. Thread-safe.
    virtual void cancel() {}
};

// ---------------------------------------------------------------------------
// VAD — Voice Activity Detection (chunk-level)
// ---------------------------------------------------------------------------

class VADInterface {
public:
    virtual ~VADInterface() = default;

    /// Process a chunk of audio and return speech probability [0, 1].
    /// @param samples  PCM Float32 samples at input_sample_rate()
    /// @param length   Number of samples (typically 512 for Silero)
    virtual float process_chunk(const float* samples, size_t length) = 0;

    /// Reset internal state (LSTM hidden state, etc.).
    virtual void reset() = 0;

    /// Expected input sample rate in Hz (typically 16000).
    virtual int input_sample_rate() const = 0;

    /// Expected chunk size in samples.
    virtual size_t chunk_size() const = 0;
};

// ---------------------------------------------------------------------------
// Speech Enhancement
// ---------------------------------------------------------------------------

class EnhancerInterface {
public:
    virtual ~EnhancerInterface() = default;

    /// Enhance audio by removing noise.
    /// @param audio  Input PCM Float32 samples
    /// @param length Number of samples
    /// @param sample_rate Sample rate in Hz
    /// @param output Pre-allocated output buffer (same length)
    virtual void enhance(
        const float* audio, size_t length, int sample_rate,
        float* output) = 0;

    virtual int input_sample_rate() const = 0;
};

}  // namespace speech_core
