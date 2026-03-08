#pragma once

#include <string>

#include "speech_core/vad/vad_config.h"

namespace speech_core {

/// Configuration for the voice agent pipeline.
struct AgentConfig {
    /// VAD configuration.
    VADConfig vad = VADConfig::silero_default();

    /// Whether the user can interrupt agent speech.
    bool allow_interruptions = true;

    /// Seconds of silence after interruption before resuming playback
    /// (false-interruption recovery). 0 = disabled.
    float interruption_recovery_timeout = 0.4f;

    /// Minimum delay between consecutive agent speech outputs (seconds).
    float min_speech_gap = 0.1f;

    /// Maximum user utterance duration before force-splitting (seconds).
    float max_utterance_duration = 15.0f;

    /// Language hint for STT/TTS (e.g. "en", "zh", "de"). Empty = auto-detect.
    std::string language;

    /// Pipeline mode.
    enum class Mode {
        /// STT -> LLM -> TTS (standard voice agent)
        Pipeline,
        /// STT only (transcription)
        TranscribeOnly,
        /// Echo mode (STT result spoken back via TTS, for testing)
        Echo
    };

    Mode mode = Mode::Pipeline;
};

}  // namespace speech_core
