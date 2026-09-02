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

float TurnDetector::classify_turn() {
    if (!turn_completion_) return -1.0f;
    return turn_completion_->turn_complete_probability(
        utterance_buffer_.data(), utterance_buffer_.size(),
        vad_.input_sample_rate());
}

void TurnDetector::end_turn(float time, bool eager, float probability) {
    in_speech_ = false;
    turn_hold_ = false;
    interruption_time_ = -1.0f;

    TurnEvent ended;
    ended.type = TurnEvent::UserSpeechEnded;
    ended.time = time;
    ended.eager = eager;
    ended.turn_completion_probability = probability;
    ended.audio = utterance_buffer_;
    on_event_(ended);
    utterance_buffer_.clear();
}

void TurnDetector::hold_turn(float time) {
    // The classifier says the user is not done: keep the turn open. Audio
    // keeps accumulating into utterance_buffer_ (in_speech_ stays true), a
    // later SpeechStarted continues the same turn, and the silence cap in
    // push_audio ends it if nothing more is said.
    turn_hold_ = true;
    turn_hold_since_ = time;
    eager_pending_ = false;
}

void TurnDetector::push_audio(const float* samples, size_t count) {
    size_t chunk_size = vad_.chunk_size();
    size_t offset = 0;

    while (offset + chunk_size <= count) {
        float prob = vad_.process_chunk(samples + offset, chunk_size);

        // Post-playback guard: keep the VAD model warm but suppress events.
        // The audio itself must keep flowing into the pre-speech ring —
        // a user who starts talking inside the guard window (common right
        // after the agent's reply) loses the utterance head otherwise, and
        // the onset prepend has nothing to rescue it with.
        if (guard_remaining_samples_ > 0) {
            size_t dec = std::min(guard_remaining_samples_, chunk_size);
            guard_remaining_samples_ -= dec;
            buffer_pre_speech(samples + offset, chunk_size);
            offset += chunk_size;
            continue;
        }

        auto events = streaming_vad_.process(prob);

        for (const auto& vad_event : events) {
            if (vad_event.type == VADEvent::SpeechStarted) {
                // The user resuming after a pause the classifier vetoed is
                // the same turn: keep the buffer and start time, and don't
                // re-announce the turn to the pipeline.
                const bool continuing_held_turn = turn_hold_;
                if (continuing_held_turn) {
                    turn_hold_ = false;
                } else {
                    // Prepend pre-speech audio to capture onset
                    utterance_buffer_ = pre_speech_ring_;
                    pre_speech_ring_.clear();
                    utterance_start_ = vad_event.start_time;
                    in_speech_ = true;
                }
                interruption_confirmed_ = false;

                // If agent is speaking, defer interruption until min duration met
                if (agent_speaking_ && config_.allow_interruptions) {
                    interruption_time_ = vad_event.start_time;
                    // If min_interruption_duration <= 0, fire immediately
                    if (config_.min_interruption_duration <= 0.0f) {
                        interruption_confirmed_ = true;
                        TurnEvent interrupt;
                        interrupt.type = TurnEvent::Interruption;
                        interrupt.time = vad_event.start_time;
                        on_event_(interrupt);
                    }
                }

                if (!continuing_held_turn) {
                    TurnEvent started;
                    started.type = TurnEvent::UserSpeechStarted;
                    started.time = vad_event.start_time;
                    on_event_(started);
                }

            } else if (vad_event.type == VADEvent::SpeechPaused) {
                // Eager STT: start timer when silence begins. After
                // eager_stt_delay elapses, fire UserSpeechEnded early.
                if (config_.eager_stt && in_speech_ && !agent_speaking_
                    && !eager_utterance_sent_ && !turn_hold_) {
                    if (config_.eager_stt_delay <= 0.0f) {
                        // No delay — fire immediately (original behavior),
                        // unless the classifier says the user is mid-sentence.
                        const float p = classify_turn();
                        if (p >= 0.0f && p < config_.turn_completion_threshold) {
                            hold_turn(vad_event.end_time);
                        } else {
                            eager_utterance_sent_ = true;
                            end_turn(vad_event.end_time, /*eager=*/true, p);
                        }
                    } else {
                        // Start deferred eager timer
                        eager_pending_ = true;
                        eager_pending_time_ = streaming_vad_.current_time();
                    }
                }

            } else if (vad_event.type == VADEvent::SpeechResumed) {
                // Cancel deferred eager — speech resumed before delay elapsed
                eager_pending_ = false;
                // A classifier hold ends the same way: the turn continues.
                turn_hold_ = false;

                // Speech resumed after a pause — if we sent an eager utterance,
                // that's already being processed. Treat resumed speech as a
                // new utterance (prepend pre-speech ring for context).
                if (eager_utterance_sent_) {
                    eager_utterance_sent_ = false;
                    in_speech_ = true;
                    utterance_buffer_ = pre_speech_ring_;
                    pre_speech_ring_.clear();
                    utterance_start_ = vad_event.end_time;

                    TurnEvent started;
                    started.type = TurnEvent::UserSpeechStarted;
                    started.time = vad_event.end_time;
                    started.eager_resumed = true;
                    on_event_(started);
                }

            } else if (vad_event.type == VADEvent::SpeechEnded) {
                eager_pending_ = false;  // silence confirmed — normal path handles it

                // If eager STT already fired, silence just confirmed it — skip
                if (eager_utterance_sent_) {
                    in_speech_ = false;
                    eager_utterance_sent_ = false;
                    utterance_buffer_.clear();
                    interruption_time_ = -1.0f;
                }
                // During agent speaking: discard speech that was too short
                // to confirm interruption (AEC residual echo)
                else if (agent_speaking_ && !interruption_confirmed_) {
                    in_speech_ = false;
                    // False trigger (AEC residual echo) — restore audio to
                    // pre-speech ring so onset context is preserved for the
                    // next real utterance.
                    pre_speech_ring_ = std::move(utterance_buffer_);
                    if (pre_speech_ring_.size() > pre_speech_capacity_) {
                        pre_speech_ring_.erase(
                            pre_speech_ring_.begin(),
                            pre_speech_ring_.begin() +
                                static_cast<long>(pre_speech_ring_.size() - pre_speech_capacity_));
                    }
                    interruption_time_ = -1.0f;
                }
                // Check for interruption recovery
                else if (interruption_time_ >= 0.0f &&
                    config_.interruption_recovery_timeout > 0.0f &&
                    (vad_event.end_time - interruption_time_) <
                        config_.interruption_recovery_timeout) {
                    in_speech_ = false;
                    // Brief interruption — recover
                    TurnEvent recovered;
                    recovered.type = TurnEvent::InterruptionRecovered;
                    recovered.time = vad_event.end_time;
                    on_event_(recovered);
                    interruption_time_ = -1.0f;
                    utterance_buffer_.clear();
                }
                // The classifier already vetoed this pause (at the eager
                // moment); silence confirming it changes nothing.
                else if (turn_hold_) {
                    // keep holding
                }
                // Normal speech ended — ask the classifier, if any
                else {
                    const float p = classify_turn();
                    if (p >= 0.0f && p < config_.turn_completion_threshold) {
                        hold_turn(vad_event.end_time);
                    } else {
                        end_turn(vad_event.end_time, /*eager=*/false, p);
                    }
                }
            }
        }

        // Deferred eager STT: fire after delay elapses in PendingSilence
        if (eager_pending_ && !eager_utterance_sent_) {
            float elapsed = streaming_vad_.current_time() - eager_pending_time_;
            if (elapsed >= config_.eager_stt_delay) {
                eager_pending_ = false;
                const float p = classify_turn();
                if (p >= 0.0f && p < config_.turn_completion_threshold) {
                    hold_turn(eager_pending_time_);
                } else {
                    eager_utterance_sent_ = true;
                    end_turn(eager_pending_time_, /*eager=*/true, p);
                }
            }
        }

        // Classifier hold: the user never resumed — end the turn on the cap.
        if (turn_hold_ && config_.turn_completion_max_silence > 0.0f) {
            float held = streaming_vad_.current_time() - turn_hold_since_;
            if (held >= config_.turn_completion_max_silence) {
                end_turn(turn_hold_since_, /*eager=*/false, -1.0f);
            }
        }

        if (in_speech_) {
            // Buffer audio during speech
            utterance_buffer_.insert(utterance_buffer_.end(),
                                     samples + offset,
                                     samples + offset + chunk_size);

            // Check deferred interruption — fire once speech exceeds min duration
            if (agent_speaking_ && config_.allow_interruptions &&
                !interruption_confirmed_ && interruption_time_ >= 0.0f) {
                float speech_duration = streaming_vad_.current_time() - interruption_time_;
                if (speech_duration >= config_.min_interruption_duration) {
                    interruption_confirmed_ = true;
                    TurnEvent interrupt;
                    interrupt.type = TurnEvent::Interruption;
                    interrupt.time = interruption_time_;
                    on_event_(interrupt);
                }
            }

            // Force-split if utterance exceeds max duration
            float elapsed = streaming_vad_.current_time() - utterance_start_;
            if (config_.max_utterance_duration > 0.0f &&
                elapsed >= config_.max_utterance_duration) {
                force_end_utterance(streaming_vad_.current_time());
            }
        } else if (pre_speech_capacity_ > 0) {
            buffer_pre_speech(samples + offset, chunk_size);
        }

        offset += chunk_size;
    }
}

void TurnDetector::buffer_pre_speech(const float* samples, size_t count) {
    if (pre_speech_capacity_ == 0) return;
    pre_speech_ring_.insert(pre_speech_ring_.end(), samples, samples + count);
    if (pre_speech_ring_.size() > pre_speech_capacity_) {
        pre_speech_ring_.erase(
            pre_speech_ring_.begin(),
            pre_speech_ring_.begin() +
                static_cast<long>(pre_speech_ring_.size() - pre_speech_capacity_));
    }
}

void TurnDetector::force_end_utterance(float time) {
    end_turn(time, /*eager=*/false, -1.0f);

    // Reset VAD state so it can detect new speech
    streaming_vad_.reset();
    vad_.reset();
}

void TurnDetector::set_agent_speaking(bool speaking) {
    agent_speaking_ = speaking;

    // Retroactive interruption: if user is already speaking when the agent
    // starts (e.g. eager STT dispatched, user spoke during STT processing),
    // begin the deferred interruption timer so barge-in can still trigger.
    if (speaking && in_speech_ && config_.allow_interruptions
        && !interruption_confirmed_ && interruption_time_ < 0.0f) {
        interruption_time_ = streaming_vad_.current_time();
        if (config_.min_interruption_duration <= 0.0f) {
            interruption_confirmed_ = true;
            TurnEvent interrupt;
            interrupt.type = TurnEvent::Interruption;
            interrupt.time = interruption_time_;
            on_event_(interrupt);
        }
    }
}

void TurnDetector::set_post_playback_guard(float seconds) {
    guard_remaining_samples_ = static_cast<size_t>(
        seconds * static_cast<float>(vad_.input_sample_rate()));
}

void TurnDetector::flush() {
    if (turn_hold_) {
        // The VAD is idle; the classifier was still waiting for more speech.
        // End of stream settles it, with whatever was buffered meanwhile.
        end_turn(turn_hold_since_, /*eager=*/false, -1.0f);
        streaming_vad_.flush();
        return;
    }
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
    // Do NOT reset vad_ model state — its internal LSTM needs continuous
    // context to detect speech/silence transitions accurately.
    utterance_buffer_.clear();
    // Keep pre_speech_ring_ — it holds recent audio that may overlap with
    // the onset of the next utterance. Clearing it after resume_listening()
    // discards context that the STT model needs for accurate transcription.
    utterance_start_ = 0.0f;
    agent_speaking_ = false;
    in_speech_ = false;
    interruption_time_ = -1.0f;
    interruption_confirmed_ = false;
    eager_utterance_sent_ = false;
    eager_pending_ = false;
    guard_remaining_samples_ = 0;
    turn_hold_ = false;
    turn_hold_since_ = 0.0f;
}

void TurnDetector::reset_for_new_stream() {
    reset();
    pre_speech_ring_.clear();
    vad_.reset();
}

}  // namespace speech_core
