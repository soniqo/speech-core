#include "speech_core/vad/streaming_vad.h"

namespace speech_core {

StreamingVAD::StreamingVAD(VADConfig config, float chunk_duration)
    : config_(config), chunk_duration_(chunk_duration) {}

std::vector<VADEvent> StreamingVAD::process(float probability) {
    float time = static_cast<float>(chunk_count_) * chunk_duration_;
    chunk_count_++;
    return process_prob(probability, time);
}

std::vector<VADEvent> StreamingVAD::flush() {
    std::vector<VADEvent> events;
    float end_time = static_cast<float>(chunk_count_) * chunk_duration_;

    switch (state_) {
    case State::Silence:
        break;

    case State::PendingSpeech:
        if (end_time - speech_start_ >= config_.min_speech_duration) {
            events.push_back({VADEvent::SpeechStarted, speech_start_, speech_start_});
            events.push_back({VADEvent::SpeechEnded, speech_start_, end_time});
        }
        break;

    case State::Speech:
        events.push_back({VADEvent::SpeechEnded, speech_start_, end_time});
        break;

    case State::PendingSilence:
        events.push_back({VADEvent::SpeechEnded, speech_start_, silence_start_});
        break;
    }

    state_ = State::Silence;
    return events;
}

void StreamingVAD::reset() {
    chunk_count_ = 0;
    state_ = State::Silence;
    speech_start_ = 0.0f;
    silence_start_ = 0.0f;
}

float StreamingVAD::current_time() const {
    return static_cast<float>(chunk_count_) * chunk_duration_;
}

std::vector<VADEvent> StreamingVAD::process_prob(float prob, float time) {
    std::vector<VADEvent> events;
    float next_time = time + chunk_duration_;

    switch (state_) {
    case State::Silence:
        if (prob >= config_.onset) {
            speech_start_ = time;
            state_ = State::PendingSpeech;
        }
        break;

    case State::PendingSpeech:
        if (prob < config_.offset) {
            // False alarm
            state_ = State::Silence;
        } else if (next_time - speech_start_ >= config_.min_speech_duration) {
            // Speech confirmed
            events.push_back({VADEvent::SpeechStarted, speech_start_, speech_start_});
            state_ = State::Speech;
        }
        break;

    case State::Speech:
        if (prob < config_.offset) {
            silence_start_ = time;
            state_ = State::PendingSilence;
            events.push_back({VADEvent::SpeechPaused, speech_start_, time});
        }
        break;

    case State::PendingSilence:
        if (prob >= config_.onset) {
            // Speech resumed
            state_ = State::Speech;
            events.push_back({VADEvent::SpeechResumed, speech_start_, time});
        } else if (next_time - silence_start_ >= config_.min_silence_duration) {
            // Silence confirmed
            events.push_back({VADEvent::SpeechEnded, speech_start_, silence_start_});
            // Check if new speech starting
            if (prob >= config_.onset) {
                speech_start_ = time;
                state_ = State::PendingSpeech;
            } else {
                state_ = State::Silence;
            }
        }
        break;
    }

    return events;
}

}  // namespace speech_core
