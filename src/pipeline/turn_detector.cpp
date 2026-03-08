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
      on_event_(std::move(on_event)) {}

void TurnDetector::push_audio(const float* samples, size_t count) {
    // Accumulate into utterance buffer when speech is active
    size_t chunk_size = vad_.chunk_size();
    size_t offset = 0;

    while (offset + chunk_size <= count) {
        float prob = vad_.process_chunk(samples + offset, chunk_size);
        auto events = streaming_vad_.process(prob);

        for (const auto& vad_event : events) {
            if (vad_event.type == VADEvent::SpeechStarted) {
                utterance_buffer_.clear();
                utterance_start_ = vad_event.start_time;

                // If agent is speaking, this is an interruption
                if (agent_speaking_ && config_.allow_interruptions) {
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
                TurnEvent ended;
                ended.type = TurnEvent::UserSpeechEnded;
                ended.time = vad_event.end_time;
                ended.audio = utterance_buffer_;
                on_event_(ended);
                utterance_buffer_.clear();
            }
        }

        // Buffer audio during speech
        if (!utterance_buffer_.empty() ||
            streaming_vad_.current_time() - utterance_start_ >= 0) {
            // Check if we're in a speech state by seeing if buffer is accumulating
            // Simple heuristic: buffer if we got a SpeechStarted but no SpeechEnded yet
        }

        // Always buffer — TurnDetector trims on SpeechEnded
        utterance_buffer_.insert(utterance_buffer_.end(),
                                 samples + offset,
                                 samples + offset + chunk_size);

        offset += chunk_size;
    }
}

void TurnDetector::set_agent_speaking(bool speaking) {
    agent_speaking_ = speaking;
}

void TurnDetector::flush() {
    auto events = streaming_vad_.flush();
    for (const auto& vad_event : events) {
        if (vad_event.type == VADEvent::SpeechEnded) {
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
    utterance_start_ = 0.0f;
    agent_speaking_ = false;
}

}  // namespace speech_core
