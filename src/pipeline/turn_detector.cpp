#include "speech_core/pipeline/turn_detector.h"

#include <cstring>

namespace speech_core {

TurnDetector::TurnDetector(
    VADInterface& vad, const AgentConfig& config, EventCallback on_event)
    : vad_(vad),
      config_(config),
      streaming_vad_(config.vad,
                     static_cast<float>(vad.chunk_size()) /
                     static_cast<float>(vad.input_sample_rate())),
      on_event_(std::move(on_event)) {
    // Pre-speech ring buffer: holds last N seconds of audio
    if (config.vad.pre_speech_buffer_duration > 0.0f) {
        pre_speech_capacity_ = static_cast<size_t>(
            config.vad.pre_speech_buffer_duration *
            static_cast<float>(vad.input_sample_rate()));
    }
}

void TurnDetector::push_audio(const float* samples, size_t count) {
    size_t chunk_size = vad_.chunk_size();
    size_t offset = 0;

    while (offset + chunk_size <= count) {
        float prob = vad_.process_chunk(samples + offset, chunk_size);
        auto events = streaming_vad_.process(prob);

        for (const auto& vad_event : events) {
            if (vad_event.type == VADEvent::SpeechStarted) {
                // Prepend pre-speech audio to capture onset
                utterance_buffer_ = pre_speech_ring_;
                pre_speech_ring_.clear();
                utterance_start_ = vad_event.start_time;
                in_speech_ = true;

                // If agent is speaking, this is an interruption
                if (agent_speaking_ && config_.allow_interruptions) {
                    interruption_time_ = vad_event.start_time;
                    TurnEvent interrupt;
                    interrupt.type = TurnEvent::Interruption;
                    interrupt.time = vad_event.start_time;
                    on_event_(interrupt);
                }

                TurnEvent started;
                started.type = TurnEvent::UserSpeechStarted;
                started.time = vad_event.start_time;
                on_event_(started);

            } else if (vad_event.type == VADEvent::SpeechEnded) {
                in_speech_ = false;

                // Check for interruption recovery
                if (interruption_time_ >= 0.0f &&
                    config_.interruption_recovery_timeout > 0.0f &&
                    (vad_event.end_time - interruption_time_) <
                        config_.interruption_recovery_timeout) {
                    // Brief interruption — recover
                    TurnEvent recovered;
                    recovered.type = TurnEvent::InterruptionRecovered;
                    recovered.time = vad_event.end_time;
                    on_event_(recovered);
                    interruption_time_ = -1.0f;
                    utterance_buffer_.clear();
                } else {
                    // Normal speech ended
                    interruption_time_ = -1.0f;
                    TurnEvent ended;
                    ended.type = TurnEvent::UserSpeechEnded;
                    ended.time = vad_event.end_time;
                    ended.audio = utterance_buffer_;
                    on_event_(ended);
                    utterance_buffer_.clear();
                }
            }
        }

        if (in_speech_) {
            // Buffer audio during speech
            utterance_buffer_.insert(utterance_buffer_.end(),
                                     samples + offset,
                                     samples + offset + chunk_size);

            // Force-split if utterance exceeds max duration
            float elapsed = streaming_vad_.current_time() - utterance_start_;
            if (config_.max_utterance_duration > 0.0f &&
                elapsed >= config_.max_utterance_duration) {
                force_end_utterance(streaming_vad_.current_time());
            }
        } else if (pre_speech_capacity_ > 0) {
            // Maintain rolling pre-speech buffer
            pre_speech_ring_.insert(pre_speech_ring_.end(),
                                    samples + offset,
                                    samples + offset + chunk_size);
            if (pre_speech_ring_.size() > pre_speech_capacity_) {
                pre_speech_ring_.erase(
                    pre_speech_ring_.begin(),
                    pre_speech_ring_.begin() +
                        static_cast<long>(pre_speech_ring_.size() - pre_speech_capacity_));
            }
        }

        offset += chunk_size;
    }
}

void TurnDetector::force_end_utterance(float time) {
    in_speech_ = false;

    TurnEvent ended;
    ended.type = TurnEvent::UserSpeechEnded;
    ended.time = time;
    ended.audio = utterance_buffer_;
    on_event_(ended);
    utterance_buffer_.clear();

    // Reset VAD state so it can detect new speech
    streaming_vad_.reset();
    vad_.reset();
}

void TurnDetector::set_agent_speaking(bool speaking) {
    agent_speaking_ = speaking;
}

void TurnDetector::flush() {
    auto events = streaming_vad_.flush();
    for (const auto& vad_event : events) {
        if (vad_event.type == VADEvent::SpeechEnded) {
            in_speech_ = false;
            TurnEvent ended;
            ended.type = TurnEvent::UserSpeechEnded;
            ended.time = vad_event.end_time;
            ended.audio = utterance_buffer_;
            on_event_(ended);
        }
    }
    utterance_buffer_.clear();
}

void TurnDetector::reset() {
    streaming_vad_.reset();
    vad_.reset();
    utterance_buffer_.clear();
    pre_speech_ring_.clear();
    utterance_start_ = 0.0f;
    agent_speaking_ = false;
    in_speech_ = false;
    interruption_time_ = -1.0f;
}

}  // namespace speech_core
