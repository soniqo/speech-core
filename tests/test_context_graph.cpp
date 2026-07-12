#include "speech_core/models/context_graph.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace speech_core;

// Feed a sequence of token surface strings through the graph, summing bonuses.
static float run(const ContextGraph& g, const std::vector<std::string>& tokens) {
    float total = 0.0f;
    ContextGraph::State s = g.start();
    for (const auto& t : tokens) {
        auto step = g.advance(s, t);
        total += step.bonus;
        s = step.state;
    }
    return total;
}

static const char* SP = "\xE2\x96\x81";  // U+2581 word marker
static std::string w(const std::string& s) { return std::string(SP) + s; }

void test_normalize() {
    assert(ContextGraph::normalize(w("Set")) == " set");
    assert(ContextGraph::normalize("Qo!") == "qo");
    assert(ContextGraph::normalize(w("8")) == " 8");
    std::printf("  PASS: normalize\n");
}

void test_empty_graph_is_noop() {
    ContextGraph g;
    assert(g.empty());
    assert(run(g, {w("anything"), "goes"}) == 0.0f);
    std::printf("  PASS: empty_graph_is_noop\n");
}

void test_completed_phrase_scores_positive() {
    ContextGraph g({"soniqo"}, /*per_char=*/1.5f, /*completion=*/3.0f);
    // Tokens that spell "soniqo" at a word boundary.
    const float hit = run(g, {w("so"), "ni", "qo"});
    // 6 matched chars after the anchor space (" soniqo") + one completion.
    // Space(+1.5) s o n i q o = 7 forward chars * 1.5 + 3.0 completion.
    assert(hit > 10.0f);
    std::printf("  PASS: completed_phrase_scores_positive (bonus=%.2f)\n", hit);
}

void test_unrelated_text_nets_near_zero() {
    ContextGraph g({"soniqo"}, 1.5f, 3.0f);
    // "so" starts matching " so" then "up" breaks it — the fail arc refunds.
    const float miss = run(g, {w("so"), "up"});
    assert(std::fabs(miss) < 1e-3f);
    std::printf("  PASS: unrelated_text_nets_near_zero (net=%.4f)\n", miss);
}

void test_partial_prefix_does_not_beat_full_match() {
    ContextGraph g({"soniqo"}, 1.5f, 3.0f);
    const float full    = run(g, {w("soniqo")});
    const float partial = run(g, {w("son"), "der"});  // "sonder" breaks the match
    assert(full > partial);
    assert(partial < 1.0f);  // refunded to ~0 once the match breaks
    std::printf("  PASS: partial_prefix_does_not_beat_full_match (full=%.2f partial=%.2f)\n",
                full, partial);
}

void test_multiword_phrase() {
    ContextGraph g({"set volume"}, 1.5f, 3.0f);
    const float hit  = run(g, {w("set"), w("volume")});
    const float near = run(g, {w("said"), w("william")});
    assert(hit > 10.0f);
    assert(hit > near);
    std::printf("  PASS: multiword_phrase (hit=%.2f near=%.2f)\n", hit, near);
}

void test_word_boundary_anchor() {
    // "son" must not match inside "person" (no word boundary before "son").
    ContextGraph g({"son"}, 1.5f, 3.0f);
    const float inside = run(g, {w("person")});
    const float atword = run(g, {w("son")});
    assert(atword > inside);
    std::printf("  PASS: word_boundary_anchor (inside=%.2f atword=%.2f)\n",
                inside, atword);
}

void test_multiple_phrases() {
    ContextGraph g({"soniqo", "set volume", "play"}, 1.5f, 3.0f);
    assert(run(g, {w("play")}) > 5.0f);
    assert(run(g, {w("soniqo")}) > 5.0f);
    assert(std::fabs(run(g, {w("random"), "junk"})) < 1e-3f);
    std::printf("  PASS: multiple_phrases\n");
}

int main() {
    std::printf("test_context_graph:\n");
    test_normalize();
    test_empty_graph_is_noop();
    test_completed_phrase_scores_positive();
    test_unrelated_text_nets_near_zero();
    test_partial_prefix_does_not_beat_full_match();
    test_multiword_phrase();
    test_word_boundary_anchor();
    test_multiple_phrases();
    std::printf("All context graph tests passed.\n");
    return 0;
}
