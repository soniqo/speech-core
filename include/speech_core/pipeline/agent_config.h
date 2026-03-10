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

    /// Minimum speech duration during agent playback before triggering
    /// interruption (seconds). Filters AEC residual echo.
    /// 0 = use min_speech_duration. LiveKit default: 0.5s.
    float min_interruption_duration = 1.0f;

    /// Seconds of silence after interruption before resuming playback
    /// (false-interruption recovery). 0 = disabled.
    float interruption_recovery_timeout = 0.4f;

    /// Minimum delay between consecutive agent speech outputs (seconds).
    float min_speech_gap = 0.1f;

    /// Maximum user utterance duration before force-splitting (seconds).
    float max_utterance_duration = 15.0f;

    /// Maximum TTS response duration (seconds). Prevents hallucination loops
    /// where the model generates endless audio. 0 = unlimited.
    float max_response_duration = 10.0f;

    /// Post-playback guard duration (seconds). After platform calls
    /// resume_listening(), delay before actually resuming to let AEC settle.
    float post_playback_guard = 0.3f;

    /// Start STT as soon as silence begins (PendingSilence) instead of waiting
    /// for silence to confirm (min_silence_duration). Saves ~0.5s of latency
    /// at the cost of occasional split utterances on mid-sentence pauses.
    bool eager_stt = true;

    /// Run a dummy STT transcription at pipeline start to warm up the model.
    /// First inference on CoreML/Neural Engine is slow due to cold start;
    /// this brings subsequent latency from ~3s to <1s.
    bool warmup_stt = true;

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
