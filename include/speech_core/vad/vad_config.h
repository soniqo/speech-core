#pragma once

namespace speech_core {

/// Configuration for VAD hysteresis and duration filtering.
struct VADConfig {
    /// Onset threshold — speech starts when probability exceeds this.
    float onset = 0.5f;

    /// Offset threshold — speech ends when probability drops below this.
    float offset = 0.35f;

    /// Minimum speech duration in seconds before confirming.
    float min_speech_duration = 0.25f;

    /// Minimum silence duration in seconds before ending speech.
    float min_silence_duration = 0.1f;

    /// Default configuration for Silero VAD v5 streaming.
    static constexpr VADConfig silero_default() {
        return {0.5f, 0.35f, 0.25f, 0.1f};
    }

    /// Default configuration for Pyannote segmentation.
    static constexpr VADConfig pyannote_default() {
        return {0.767f, 0.377f, 0.136f, 0.067f};
    }
};

}  // namespace speech_core
