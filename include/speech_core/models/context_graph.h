#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Aho-Corasick character automaton for shallow-fusion ASR context biasing.
///
/// Built from a list of surface phrases (command words, a brand name, live
/// contact/track names). During beam search each hypothesis carries an opaque
/// [State]; [advance] appends the decoded surface text of one emitted token and
/// returns a log-domain score bonus that nudges the beam toward hypotheses that
/// spell out a phrase.
///
/// Matching is on decoded text, not token ids, so it is tokenizer-agnostic — it
/// works even when the model's exact sub-word segmentation is unknown (our
/// Parakeet-EOU export ships only a decode vocabulary, no encoder tokenizer).
/// Phrases are anchored to word starts (the SentencePiece word marker U+2581
/// normalizes to a leading space), so "son" biases the word "Soniqo" but not
/// the "son" inside "person".
///
/// The per-character score telescopes: matching root->...->node accrues
/// `per_char` per depth, and a broken partial match follows a fail arc back
/// toward the root, refunding the bonus it earned. Only sustained matches keep
/// their boost; unrelated text nets ~0.
class ContextGraph {
public:
    ContextGraph() = default;

    /// @param phrases    surface phrases to bias toward (case-insensitive).
    /// @param per_char   log-domain bonus per matched character inside a phrase.
    /// @param completion extra bonus applied when a whole phrase completes.
    /// @param max_bonus  ceiling on the per-character accumulation a single
    ///                   phrase match can reach (completion is added on top).
    ///                   0 = uncapped (default) — the same behavior as
    ///                   sherpa-onnx / k2, where longer phrases accumulate a
    ///                   larger boost and the score must be tuned by hand. A
    ///                   positive cap bounds each phrase's contribution, so an
    ///                   over-wide beam or long bias list can't let a long
    ///                   phrase override clear audio; set it low enough and
    ///                   sufficiently long phrases all reach the same ceiling.
    explicit ContextGraph(const std::vector<std::string>& phrases,
                          float per_char = 1.5f,
                          float completion = 3.0f,
                          float max_bonus = 0.0f);

    /// True when no phrase produced any matchable content (biasing is a no-op).
    bool empty() const { return nodes_.size() <= 1; }

    /// Opaque per-hypothesis match state — an automaton node index. 0 = root.
    using State = int;
    State start() const { return 0; }

    struct Step {
        float bonus;
        State state;
    };
    /// Append one emitted token's surface text (as it appears in the vocab,
    /// including any U+2581 markers); returns the bonus and the next state.
    Step advance(State s, const std::string& token_text) const;

    /// Normalize a piece/phrase to the matching alphabet: U+2581 -> space,
    /// ASCII upper -> lower, keep [a-z0-9 ], drop the rest. Exposed for tests.
    static std::string normalize(const std::string& text);

private:
    struct Node {
        std::unordered_map<char, int> children;
        int   fail  = 0;
        int   depth = 0;
        bool  end   = false;   // a phrase terminates here
        bool  output = false;  // this node, or one on its fail chain, is an end
    };

    int  go(int node, char c) const;  // Aho-Corasick goto via fail links

    std::vector<Node>  nodes_{Node{}};  // nodes_[0] == root
    std::vector<float> score_{0.0f};    // per_char * depth, indexed by node
    float per_char_   = 1.5f;
    float completion_ = 3.0f;
};

}  // namespace speech_core
