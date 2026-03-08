#pragma once

#include <cstddef>
#include <vector>

#include "speech_core/vad/vad_config.h"

namespace speech_core {

/// Events emitted by the streaming VAD processor.
struct VADEvent {
    enum Type {
        SpeechStarted,  ///< Speech detected and confirmed (duration >= min_speech_duration)
        SpeechEnded     ///< Speech ended (silence >= min_silence_duration)
    };

    Type type;
    float start_time;  ///< Speech start time in seconds
    float end_time;    ///< Speech end time in seconds (only valid for SpeechEnded)
};

/// Event-driven streaming VAD processor.
///
/// Accepts speech probabilities from a VADInterface and applies hysteresis
/// with duration filtering via a four-state machine. Platform-independent —
/// does not run any ML model itself.
///
/// Port of StreamingVADProcessor from speech-swift.
class StreamingVAD {
public:
    /// @param config  VAD thresholds and duration constraints
    /// @param chunk_duration  Duration of each chunk in seconds (e.g. 0.032 for 512 samples at 16kHz)
    explicit StreamingVAD(VADConfig config = VADConfig::silero_default(),
                          float chunk_duration = 0.032f);

    /// Feed a speech probability and get events back.
    /// @param probability  Speech probability [0, 1] from VADInterface
    /// @return Zero or more VAD events
    std::vector<VADEvent> process(float probability);

    /// Flush any pending speech segment at end of stream.
    /// @return Zero or more final events
    std::vector<VADEvent> flush();

    /// Reset all state. Call between processing different audio streams.
    void reset();

    /// Current time position in seconds.
    float current_time() const;

private:
    enum class State {
        Silence,
        PendingSpeech,
        Speech,
        PendingSilence
    };

    VADConfig config_;
    float chunk_duration_;
    int chunk_count_ = 0;

    State state_ = State::Silence;
    float speech_start_ = 0.0f;
    float silence_start_ = 0.0f;

    std::vector<VADEvent> process_prob(float prob, float time);
};

}  // namespace speech_core
