#pragma once

#include "speech_core/tts_synthesis_options.h"

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
    std::string language;  // detected language (e.g. "russian", "english"), empty if unknown
    float confidence = 1.0f;
    float start_time = 0.0f;  // seconds
    float end_time = 0.0f;
};

/// Partial transcription result from streaming STT.
struct PartialResult {
    std::string text;
    std::string language;
    float confidence = 0.0f;
};

class STTInterface {
public:
    virtual ~STTInterface() = default;

    /// Transcribe audio buffer to text (batch mode).
    /// @param audio  PCM Float32 samples
    /// @param length Number of samples
    /// @param sample_rate Sample rate in Hz
    /// @return Transcription result
    virtual TranscriptionResult transcribe(
        const float* audio, size_t length, int sample_rate) = 0;

    /// Transcribe N audio buffers as one batch. Default implementation loops
    /// over transcribe() — backends that can share an encoder pass across
    /// inputs (e.g. ParakeetStt under CUDA) override this for throughput.
    ///
    /// @param audios     Pointers to N PCM Float32 buffers
    /// @param lengths    Length of each buffer in samples
    /// @param n          Number of items in the batch
    /// @param sample_rate Sample rate in Hz (uniform across the batch)
    /// @return N TranscriptionResults in input order
    virtual std::vector<TranscriptionResult> transcribe_batch(
        const float* const* audios, const size_t* lengths,
        size_t n, int sample_rate)
    {
        std::vector<TranscriptionResult> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(transcribe(audios[i], lengths[i], sample_rate));
        }
        return out;
    }

    /// Expected input sample rate in Hz.
    virtual int input_sample_rate() const = 0;

    /// Cancel any in-progress transcription. Thread-safe.
    virtual void cancel() {}

    // --- Optional streaming interface ---
    // Override these to enable real-time partial transcription.
    // When supports_streaming() returns true, the pipeline feeds audio
    // chunks during speech via begin/push/end instead of batch transcribe.

    /// Whether this STT model supports streaming transcription.
    virtual bool supports_streaming() const { return false; }

    /// Begin a new streaming transcription session.
    virtual void begin_stream(int /*sample_rate*/) {}

    /// Feed an audio chunk during speech. Returns structured partial result.
    virtual PartialResult push_chunk(const float* /*audio*/, size_t /*length*/) { return {}; }

    /// Mark a segment boundary without ending the stream.
    /// Useful for force-split utterances in multi-utterance sessions.
    virtual void flush_stream() {}

    /// Finalize the stream and return the authoritative transcription.
    virtual TranscriptionResult end_stream() { return {}; }

    /// Cancel the current stream (e.g. on interruption).
    virtual void cancel_stream() {}
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

    /// Synthesize with explicit delivery and post-processing options.
    ///
    /// Default implementation preserves Streaming + no post-processing by
    /// delegating to synthesize(). Buffered mode accumulates every emitted
    /// chunk for this text input, applies the requested offline processing,
    /// and invokes on_chunk once with is_final=true.
    virtual void synthesize_with_options(
        const std::string& text,
        const std::string& language,
        const TtsSynthesisOptions& options,
        TTSChunkCallback on_chunk);

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

// ---------------------------------------------------------------------------
// Echo Cancellation
// ---------------------------------------------------------------------------

/// Acoustic echo cancellation interface.
///
/// The pipeline feeds TTS output as far-end reference via feed_reference(),
/// then runs cancel_echo() on mic input before VAD:
///
///   TTS output ──► feed_reference()
///   Mic input  ──► cancel_echo() ──► clean signal ──► enhance ──► VAD ──► STT
///
/// Implementations may use SpeexDSP, WebRTC AEC, or platform-native AEC.
/// The interface is intentionally simple — frame alignment, sample rate
/// conversion, and internal buffering are the implementation's responsibility.
class EchoCancellerInterface {
public:
    virtual ~EchoCancellerInterface() = default;

    /// Feed far-end (TTS playback) reference samples.
    /// Called from the TTS synthesis callback on the worker thread.
    /// Thread-safe: may be called concurrently with cancel_echo().
    /// @param samples  PCM Float32 at input_sample_rate()
    /// @param length   Number of samples
    virtual void feed_reference(const float* samples, size_t length) = 0;

    /// Remove echo from near-end (mic) audio.
    /// Called from push_audio() on the audio thread.
    /// @param input   Mic input PCM Float32
    /// @param length  Number of samples
    /// @param output  Pre-allocated output buffer (same length)
    virtual void cancel_echo(const float* input, size_t length, float* output) = 0;

    /// Expected sample rate in Hz (must match both mic and reference).
    virtual int input_sample_rate() const = 0;

    /// Reset internal state (e.g. on pipeline start/stop).
    virtual void reset() = 0;
};

// ---------------------------------------------------------------------------
// Diarization — speaker segmentation, embedding, and clustering
// ---------------------------------------------------------------------------
// These support multi-speaker meeting transcription (server-side). Unlike the
// real-time conversational interfaces above, they operate on a whole audio
// buffer. A Diarizer typically composes a SegmentationInterface (local speaker
// activity per window) with an EmbeddingInterface (speaker vectors for
// cross-window clustering).

/// One speaker-labelled time span.
struct DiarizedSegment {
    float start   = 0.0f;  // seconds
    float end     = 0.0f;  // seconds
    int   speaker = -1;    // local speaker id; -1 = unassigned
};

/// One segmentation window's raw output.
struct SegmentationWindow {
    float start_time = 0.0f;
    float end_time   = 0.0f;
    std::vector<float> posteriors;        // [frames × powerset_classes]
    std::vector<float> speaker_activity;  // [frames × max_local_speakers]
};

/// Tunables for end-to-end diarization.
struct DiarizerConfig {
    float onset                = 0.5f;
    float offset               = 0.3f;
    float min_speech_duration  = 0.3f;  // seconds
    float clustering_threshold = 0.715f;
    int   min_speakers         = 0;     // 0 = auto
    int   max_speakers         = 0;     // 0 = auto
};

/// Local speaker-activity segmentation over fixed windows (e.g. Pyannote).
class SegmentationInterface {
public:
    virtual ~SegmentationInterface() = default;

    /// Run segmentation over the whole buffer, returning per-window posteriors.
    virtual std::vector<SegmentationWindow> segment(
        const float* audio, size_t length, int sample_rate) = 0;

    /// Expected input sample rate in Hz.
    virtual int input_sample_rate() const = 0;

    /// Maximum simultaneous local speakers the model resolves per window.
    virtual int max_local_speakers() const = 0;
};

/// Fixed-dimension speaker embedding (e.g. WeSpeaker ResNet34).
class EmbeddingInterface {
public:
    virtual ~EmbeddingInterface() = default;

    /// Embed an audio span into a speaker vector.
    virtual std::vector<float> embed(
        const float* audio, size_t length, int sample_rate) = 0;

    /// Output embedding dimensionality.
    virtual int embedding_dim() const = 0;

    /// Expected input sample rate in Hz.
    virtual int input_sample_rate() const = 0;
};

/// End-to-end diarization: audio in, speaker-labelled segments out.
/// Typically composes a SegmentationInterface + an EmbeddingInterface.
class DiarizerInterface {
public:
    virtual ~DiarizerInterface() = default;

    virtual std::vector<DiarizedSegment> diarize(
        const float* audio, size_t length, int sample_rate,
        const DiarizerConfig& config) = 0;
};

// ---------------------------------------------------------------------------
// Full-duplex speech-to-speech (PersonaPlex / Moshi-style)
// ---------------------------------------------------------------------------

/// One ~80 ms (12.5 Hz) output chunk emitted by a FullDuplexSpeechInterface.
/// Mirrors the streaming contract speech-swift exposes via
/// PersonaPlexModel.respondStream — agent audio plus the text tokens that were
/// generated during the same frame window.
struct FullDuplexChunk {
    const float* samples;       ///< 24 kHz Float32 mono PCM (callee-owned, valid until next call)
    size_t length;              ///< Number of samples
    int sample_rate;            ///< Typically 24000
    std::vector<int> text_tokens;  ///< SentencePiece text tokens generated this chunk
    bool is_final;              ///< True on the last chunk of the response
};

/// Called for each agent-audio chunk emitted during streaming generation.
using FullDuplexChunkCallback = std::function<void(const FullDuplexChunk&)>;

/// Full-duplex speech-to-speech: user audio in, agent audio + text out,
/// running simultaneously at the model's frame rate (12.5 Hz for PersonaPlex).
///
/// Used for models that listen and speak at the same time, conditioned on a
/// voice preset and an optional text system prompt. PersonaPlex / Moshi are
/// the canonical example; speech-swift's `PersonaPlexModel` satisfies the same
/// conceptual contract via its MLX backend.
///
/// Lifecycle:
///   1. Construct (loads weights, voices, tokenizer).
///   2. `set_voice("NATM0")`, optionally `set_system_prompt(...)`.
///   3. `respond_stream(user_pcm, length, sample_rate, on_chunk)` — blocking
///      streaming call. `on_chunk` is invoked for each ~2 s of generated
///      agent audio plus the text tokens accumulated in that window.
///   4. The session's KV cache persists across `respond_stream` calls so
///      successive turns share conversation history; call `reset_session()`
///      to clear it.
class FullDuplexSpeechInterface {
public:
    virtual ~FullDuplexSpeechInterface() = default;

    /// Stream agent audio + text in response to user audio. Blocks until the
    /// model emits its end-of-response signal or hits its max-frame budget.
    /// @param user_audio    PCM Float32 samples
    /// @param length        Number of samples
    /// @param sample_rate   Sample rate in Hz (24 kHz expected; resampled if not)
    /// @param on_chunk      Called for each emitted audio + text chunk
    virtual void respond_stream(
        const float* user_audio, size_t length, int sample_rate,
        FullDuplexChunkCallback on_chunk) = 0;

    /// Select the voice preset (e.g. "NATM0", "VARF2"). Implementations
    /// document the available names; PersonaPlex ships 18 presets.
    virtual void set_voice(const std::string& voice_name) = 0;

    /// Set a text system prompt that steers behaviour. Tokenised with the
    /// model's SentencePiece tokenizer and prepended to the conversation.
    virtual void set_system_prompt(const std::string& /*prompt*/) {}

    /// Cap the number of generation frames per respond_stream call. Default
    /// is the model's training-time max (typically 2000 frames = ~2.7 min).
    virtual void set_max_frames(int /*max_frames*/) {}

    /// Reset the conversation KV cache and the generation state. After this,
    /// the next respond_stream starts fresh as if no history existed.
    virtual void reset_session() = 0;

    /// Cancel any in-progress generation. Thread-safe.
    virtual void cancel() {}

    /// Output sample rate (typically 24000 for PersonaPlex / Moshi).
    virtual int output_sample_rate() const = 0;
};

}  // namespace speech_core
