#include "speech_core/models/transducer_beam.h"

#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>

using namespace speech_core::transducer;

// State = number of text tokens emitted so far. Aux = unused (int).
using Hyp = BeamHyp<int, int>;

// No-op biasing.
static std::pair<float, int> no_bonus(int /*token*/, int aux) { return {0.0f, aux}; }

// ---------------------------------------------------------------------------
// Toy transducer: emit tokens [1..target] one per state step. Blank is more
// probable than the correct token (blank_logit > token_logit), which is what
// makes plain RNN-T beam under-emit — the exact bug we are fixing. vocab:
// 0 = blank, 1..3 = text tokens.
// ---------------------------------------------------------------------------
struct EmitToy {
    int   target;
    float token_logit;
    float blank_logit;
    std::vector<float> buf;
    const std::vector<float>& joint(int /*frame*/, int count) {
        buf.assign(4, -10.0f);
        buf[0] = blank_logit;
        if (count < target) buf[count + 1] = token_logit;
        else buf[0] = 10.0f;  // done — blank dominates so it just advances
        return buf;
    }
    static int predict(int /*token*/, int count) { return count + 1; }
};

static std::vector<int> run_emit(float emit_bonus) {
    EmitToy toy{/*target=*/3, /*token_logit=*/1.0f, /*blank_logit=*/2.0f, {}};
    BeamParams p;
    p.beam_size = 4;
    p.max_symbols = 8;
    p.blank_id = 0;
    p.emit_bonus = emit_bonus;

    std::vector<Hyp> beams;
    beams.push_back(Hyp{{}, 0.0, /*state=*/0, /*aux=*/0, false});
    beam_decode_frames(beams, /*frame_begin=*/0, /*frame_count=*/3, p,
                       [&](int f, int s) -> const std::vector<float>& { return toy.joint(f, s); },
                       EmitToy::predict, no_bonus);
    const Hyp* best = best_hyp(beams);
    return best ? best->tokens : std::vector<int>{};
}

void test_blank_bias_under_emits_without_reward() {
    // Blank is preferred, so with no emission reward the beam emits (almost)
    // nothing — the truncation bug.
    const auto out = run_emit(/*emit_bonus=*/0.0f);
    assert(out.size() < 3);
    std::printf("  PASS: blank_bias_under_emits_without_reward (emitted %zu of 3)\n", out.size());
}

void test_emit_reward_recovers_full_sequence() {
    const auto out = run_emit(/*emit_bonus=*/3.0f);
    assert((out == std::vector<int>{1, 2, 3}));
    std::printf("  PASS: emit_reward_recovers_full_sequence\n");
}

void test_reward_threshold_is_monotone() {
    // More reward never emits fewer tokens.
    const size_t a = run_emit(0.0f).size();
    const size_t b = run_emit(1.5f).size();
    const size_t c = run_emit(3.0f).size();
    assert(a <= b && b <= c);
    std::printf("  PASS: reward_threshold_is_monotone (%zu <= %zu <= %zu)\n", a, b, c);
}

// ---------------------------------------------------------------------------
// Toy for contextual biasing: two text tokens (1, 2) with equal acoustics at
// the first step. The bonus callback boosts token 2, so the beam should emit 2.
// ---------------------------------------------------------------------------
struct ChoiceToy {
    std::vector<float> buf;
    const std::vector<float>& joint(int /*frame*/, int count) {
        buf.assign(3, -10.0f);
        buf[0] = 0.5f;  // blank
        if (count == 0) { buf[1] = 1.0f; buf[2] = 1.0f; }  // tokens 1, 2 tie
        else buf[0] = 10.0f;
        return buf;
    }
    static int predict(int /*token*/, int count) { return count + 1; }
};

void test_bias_breaks_ties() {
    ChoiceToy toy;
    BeamParams p;
    p.beam_size = 4; p.max_symbols = 4; p.blank_id = 0; p.emit_bonus = 2.0f;

    auto boost_two = [](int token, int aux) -> std::pair<float, int> {
        return {token == 2 ? 5.0f : 0.0f, aux};
    };

    std::vector<Hyp> beams;
    beams.push_back(Hyp{{}, 0.0, 0, 0, false});
    beam_decode_frames(beams, 0, 2, p,
                       [&](int f, int s) -> const std::vector<float>& { return toy.joint(f, s); },
                       ChoiceToy::predict, boost_two);
    const Hyp* best = best_hyp(beams);
    assert(best && best->tokens == std::vector<int>{2});
    std::printf("  PASS: bias_breaks_ties (chose token 2)\n");
}

// ---------------------------------------------------------------------------
// EOU: emitting the terminal token ends the hypothesis and is not written into
// the token output. vocab: 0 = blank, 1 = text, 2 = eou.
// ---------------------------------------------------------------------------
struct EouToy {
    std::vector<float> buf;
    const std::vector<float>& joint(int /*frame*/, int count) {
        buf.assign(3, -10.0f);
        buf[0] = 0.5f;
        if (count == 0) buf[1] = 5.0f;        // emit text
        else if (count == 1) buf[2] = 5.0f;   // then EOU
        else buf[0] = 10.0f;
        return buf;
    }
    static int predict(int /*token*/, int count) { return count + 1; }
};

void test_eou_terminates_and_is_not_emitted() {
    EouToy toy;
    BeamParams p;
    p.beam_size = 4; p.max_symbols = 4; p.blank_id = 0; p.eou_id = 2; p.emit_bonus = 2.0f;

    std::vector<Hyp> beams;
    beams.push_back(Hyp{{}, 0.0, 0, 0, false});
    beam_decode_frames(beams, 0, 3, p,
                       [&](int f, int s) -> const std::vector<float>& { return toy.joint(f, s); },
                       EouToy::predict, no_bonus);
    const Hyp* best = best_hyp(beams);
    assert(best && best->terminal);
    assert(best->tokens == std::vector<int>{1});  // EOU (2) not in the transcript
    std::printf("  PASS: eou_terminates_and_is_not_emitted\n");
}

int main() {
    std::printf("test_transducer_beam:\n");
    test_blank_bias_under_emits_without_reward();
    test_emit_reward_recovers_full_sequence();
    test_reward_threshold_is_monotone();
    test_bias_breaks_ties();
    test_eou_terminates_and_is_not_emitted();
    std::printf("All transducer beam tests passed.\n");
    return 0;
}
