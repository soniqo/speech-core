#pragma once

#include "speech_core/interfaces.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace httplib {
class Server;
}

namespace speech_core::http {

enum class AudioResponseFormat {
    Wav,
    Pcm,
};

struct OpenAITtsRequest {
    std::string model;
    std::string input;
    std::string voice;
    AudioResponseFormat response_format = AudioResponseFormat::Wav;
    float speed = 1.0f;
    std::string language = "en";
};

struct TtsAudio {
    std::vector<float> samples;
    int sample_rate = 0;
};

struct OpenAITranscriptionRequest {
    std::string model;
    std::string filename;
    std::string language;
    std::string prompt;
    std::string response_format = "json";
    float temperature = 0.0f;
    std::vector<float> samples;
    int sample_rate = 16000;
};

/// Throw from a backend callback when a request is syntactically valid but
/// cannot be served (for example, an unknown native voice). The HTTP adapter
/// maps this to an OpenAI-shaped 400 response instead of a server 500.
class InvalidSpeechRequest : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using SynthesizeCallback =
    std::function<TtsAudio(const OpenAITtsRequest& request)>;

using TranscribeCallback =
    std::function<speech_core::TranscriptionResult(
        const OpenAITranscriptionRequest& request)>;

/// Register the local health route and OpenAI-compatible speech endpoint on
/// an existing cpp-httplib server. When api_key is non-empty, requests must
/// include `Authorization: Bearer <api_key>`. cpp-httplib may invoke the
/// synthesis callback concurrently; backend implementations must synchronize
/// mutable model state.
void register_openai_tts_routes(
    httplib::Server& server,
    SynthesizeCallback synthesize,
    std::string api_key = {});

/// Register the OpenAI-compatible multipart transcription endpoint on an
/// existing cpp-httplib server. Uploaded WAV audio is decoded, downmixed, and
/// resampled to 16 kHz mono Float32 before the callback runs. Transcription
/// requests through this registrar are serialized to bound decoded-audio and
/// model memory. The callback must still coordinate with other routes that
/// share its mutable model state.
void register_openai_transcription_routes(
    httplib::Server& server,
    TranscribeCallback transcribe,
    std::string api_key = {});

/// Encode mono Float32 PCM as a RIFF/WAVE byte buffer containing PCM16-LE.
std::vector<uint8_t> encode_wav_pcm16(
    const float* samples,
    size_t count,
    int sample_rate);

}  // namespace speech_core::http
