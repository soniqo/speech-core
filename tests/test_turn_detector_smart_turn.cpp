// TurnDetector with an end-of-turn classifier attached (Smart Turn semantics).
//
// The classifier is scripted so the test is pure orchestration: no ORT, no
// model file. It pins down the contract that TurnCompletionInterface adds on
// top of the VAD hysteresis:
//
//   * a pause the classifier accepts ends the turn exactly as before;
//   * a pause it vetoes keeps the turn open — audio keeps accumulating, the
//     user resuming continues the same turn (no second UserSpeechStarted),
//     and the next pause asks the classifier again on the whole turn;
//   * a vetoed pause that stays silent ends on turn_completion_max_silence;
//   * eager STT respects the veto too (no early utterance on a mid-sentence
//     pause) and still fires when the classifier agrees;
//   * flush() settles a held turn; the agent-speaking discard path never
//     consults the classifier.

#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/pipeline/turn_detector.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

constexpr size_t kChunk = 512;
constexpr float kChunkSeconds = 512.0f / 16000.0f;

class ScriptedVAD : public VADInterface {
public:
    std::vector<float> probs;
    size_t idx = 0;
    float process_chunk(const float*, size_t) override {
        size_t i = idx++;
        return (i < probs.size()) ? probs[i] : 0.0f;
    }
    void reset() override { idx = 0; }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return kChunk; }
};

class ScriptedTurnModel : public TurnCompletionInterface {
public:
    std::vector<float> answers;   // consumed in order; last one repeats
    std::vector<size_t> lengths;  // audio length seen per call
    std::vector<int> rates;
    float turn_complete_probability(
        const float* samples, size_t length, int sample_rate) override {
        assert(samples != nullptr || length == 0);
        lengths.push_back(length);
        rates.push_back(sample_rate);
        if (answers.empty()) return 1.0f;
        const size_t i = std::min(lengths.size() - 1, answers.size() - 1);
        return answers[i];
    }
};

struct Harness {
    ScriptedVAD vad;
    ScriptedTurnModel model;
    AgentConfig config;
    std::vector<TurnEvent> events;
    std::unique_ptr<TurnDetector> detector;

    Harness() {
        config.vad.min_speech_duration = 0.25f;
        config.vad.min_silence_duration = 0.1f;
        config.vad.pre_speech_buffer_duration = 0.0f;
        config.eager_stt = false;
        config.turn_completion_threshold = 0.5f;
        config.turn_completion_max_silence = 1.0f;
        config.max_utterance_duration = 0.0f;
    }

    void build(bool attach_model) {
        detector = std::make_unique<TurnDetector>(
            vad, config, [this](const TurnEvent& e) { events.push_back(e); });
        if (attach_model) detector->set_turn_completion(&model);
    }

    /// Append `seconds` of chunks with the given VAD probability.
    void script(float prob, float seconds) {
        const size_t n = static_cast<size_t>(std::lround(seconds / kChunkSeconds));
        for (size_t i = 0; i < n; ++i) vad.probs.push_back(prob);
    }

    void run_all() {
        std::vector<float> audio(vad.probs.size() * kChunk, 0.01f);
        detector->push_audio(audio.data(), audio.size());
    }

    size_t count(TurnEvent::Type type) const {
        size_t n = 0;
        for (const auto& e : events) if (e.type == type) ++n;
        return n;
    }

    const TurnEvent& last(TurnEvent::Type type) const {
        for (auto it = events.rbegin(); it != events.rend(); ++it) {
            if (it->type == type) return *it;
        }
        assert(false && "event not found");
        return events.front();
    }
};

void test_without_model_unchanged() {
    Harness h;
    h.build(false);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.5f);
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechStarted) == 1);
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(h.last(TurnEvent::UserSpeechEnded).turn_completion_probability < 0.0f);
    assert(h.model.lengths.empty());
    std::printf("  no model: unchanged\n");
}

void test_complete_pause_ends_turn() {
    Harness h;
    h.model.answers = {0.93f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.5f);
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechStarted) == 1);
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(h.model.lengths.size() == 1);
    assert(h.model.rates[0] == 16000);
    const auto& ended = h.last(TurnEvent::UserSpeechEnded);
    assert(std::fabs(ended.turn_completion_probability - 0.93f) < 1e-6f);
    assert(!ended.eager);
    // The classifier saw the same audio the event carries: the speech after
    // onset confirmation (no pre-speech ring in this harness) plus the
    // pending-silence chunks before confirmation — about 0.85 s.
    assert(ended.audio.size() == h.model.lengths[0]);
    assert(ended.audio.size() >= 12000);
    std::printf("  complete pause: ended on silence confirmation\n");
}

void test_incomplete_then_resume_is_one_turn() {
    Harness h;
    h.model.answers = {0.2f, 0.9f};
    h.build(true);
    h.script(0.9f, 1.0f);   // "I'd like to ..."
    h.script(0.0f, 0.4f);   // mid-sentence pause (vetoed)
    h.script(0.9f, 1.0f);   // "... book a table"
    h.script(0.0f, 0.5f);   // real end (accepted)
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechStarted) == 1);
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(h.model.lengths.size() == 2);
    assert(h.model.lengths[1] > h.model.lengths[0]);
    const auto& ended = h.last(TurnEvent::UserSpeechEnded);
    // Whole turn: both speech segments plus the pause in between (~2.25 s).
    assert(ended.audio.size() >= static_cast<size_t>(2.0f * 16000));
    assert(std::fabs(ended.turn_completion_probability - 0.9f) < 1e-6f);
    assert(!h.detector->turn_held());
    std::printf("  vetoed pause + resume: one turn, two classifier calls\n");
}

void test_incomplete_then_silence_cap() {
    Harness h;
    h.model.answers = {0.1f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.3f);   // pause vetoed at ~1.1 s (silence start ~1.0 s)
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 0);
    assert(h.detector->turn_held());
    assert(h.detector->in_speech());
    h.vad.probs.clear();
    h.vad.idx = 0;
    h.script(0.0f, 1.0f);   // silence continues past the 1 s cap
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(h.model.lengths.size() == 1);   // no re-run on the cap
    const auto& ended = h.last(TurnEvent::UserSpeechEnded);
    assert(ended.turn_completion_probability < 0.0f);
    assert(std::fabs(ended.time - 1.0f) < 2 * kChunkSeconds);  // pause start, not cap time
    assert(!h.detector->turn_held());
    assert(!h.detector->in_speech());
    std::printf("  vetoed pause + silence: ended on the cap\n");
}

void test_eager_respects_veto() {
    Harness h;
    h.config.eager_stt = true;
    h.config.eager_stt_delay = 0.2f;
    h.config.vad.min_silence_duration = 0.6f;
    h.config.turn_completion_max_silence = 1.5f;
    h.model.answers = {0.2f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.8f);   // eager would fire at 0.2 s, silence confirms at 0.6 s
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 0);
    assert(h.model.lengths.size() == 1);   // asked once, at the eager moment
    assert(h.detector->turn_held());
    h.vad.probs.clear();
    h.vad.idx = 0;
    h.script(0.0f, 1.0f);
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(!h.last(TurnEvent::UserSpeechEnded).eager);
    std::printf("  eager STT: veto suppresses the early utterance\n");
}

void test_eager_fires_when_complete() {
    Harness h;
    h.config.eager_stt = true;
    h.config.eager_stt_delay = 0.2f;
    h.config.vad.min_silence_duration = 0.6f;
    h.model.answers = {0.8f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.8f);
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    const auto& ended = h.last(TurnEvent::UserSpeechEnded);
    assert(ended.eager);
    assert(std::fabs(ended.turn_completion_probability - 0.8f) < 1e-6f);
    assert(h.model.lengths.size() == 1);
    std::printf("  eager STT: fires when the classifier agrees\n");
}

void test_flush_settles_held_turn() {
    Harness h;
    h.model.answers = {0.1f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.3f);
    h.run_all();
    assert(h.detector->turn_held());
    h.detector->flush();
    assert(h.count(TurnEvent::UserSpeechEnded) == 1);
    assert(!h.detector->turn_held());
    assert(h.last(TurnEvent::UserSpeechEnded).audio.size() >= 12000);
    std::printf("  flush: settles a held turn\n");
}

void test_agent_speaking_discard_skips_model() {
    Harness h;
    h.config.min_interruption_duration = 1.0f;
    h.model.answers = {0.9f};
    h.build(true);
    h.detector->set_agent_speaking(true);
    h.script(0.9f, 0.4f);   // too short to confirm an interruption
    h.script(0.0f, 0.5f);
    h.run_all();
    assert(h.count(TurnEvent::UserSpeechEnded) == 0);
    assert(h.count(TurnEvent::Interruption) == 0);
    assert(h.model.lengths.empty());
    std::printf("  agent speaking: echo discard never asks the classifier\n");
}

void test_reset_clears_hold() {
    Harness h;
    h.model.answers = {0.1f};
    h.build(true);
    h.script(0.9f, 1.0f);
    h.script(0.0f, 0.3f);
    h.run_all();
    assert(h.detector->turn_held());
    h.detector->reset();
    assert(!h.detector->turn_held());
    assert(!h.detector->in_speech());
    std::printf("  reset: clears the hold\n");
}

}  // namespace

int main() {
    std::printf("TurnDetector + turn completion classifier\n");
    test_without_model_unchanged();
    test_complete_pause_ends_turn();
    test_incomplete_then_resume_is_one_turn();
    test_incomplete_then_silence_cap();
    test_eager_respects_veto();
    test_eager_fires_when_complete();
    test_flush_settles_held_turn();
    test_agent_speaking_discard_skips_model();
    test_reset_clears_hold();
    std::printf("all turn completion tests passed\n");
    return 0;
}
