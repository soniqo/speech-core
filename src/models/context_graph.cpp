#include "speech_core/models/context_graph.h"

#include <algorithm>
#include <cctype>
#include <deque>

namespace speech_core {

std::string ContextGraph::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char b = static_cast<unsigned char>(text[i]);
        // SentencePiece word marker U+2581 (E2 96 81) -> word-boundary space.
        if (b == 0xE2 && i + 2 < text.size() &&
            static_cast<unsigned char>(text[i + 1]) == 0x96 &&
            static_cast<unsigned char>(text[i + 2]) == 0x81) {
            out.push_back(' ');
            i += 3;
            continue;
        }
        if (b < 0x80) {
            const char c = static_cast<char>(b);
            if (c >= 'A' && c <= 'Z') {
                out.push_back(static_cast<char>(c - 'A' + 'a'));
            } else if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == ' ') {
                out.push_back(c);
            }
            // else: punctuation — dropped
            i += 1;
        } else {
            // Other multi-byte UTF-8 (accented Latin, etc.): skip the lead byte;
            // continuation bytes fall through and are dropped as >= 0x80.
            i += 1;
        }
    }
    return out;
}

ContextGraph::ContextGraph(const std::vector<std::string>& phrases,
                           float per_char, float completion, float max_bonus)
    : per_char_(per_char), completion_(completion) {
    // Build the trie. Each phrase is anchored with a leading space so it only
    // matches at a word boundary (the emitted text has a space from U+2581 at
    // every word start, including the first).
    for (const auto& raw : phrases) {
        const std::string norm = normalize(" " + raw);
        if (norm.size() <= 1) continue;  // empty or just the anchor space
        int node = 0;
        for (char c : norm) {
            auto it = nodes_[node].children.find(c);
            if (it != nodes_[node].children.end()) {
                node = it->second;
            } else {
                const int child = static_cast<int>(nodes_.size());
                nodes_.push_back(Node{});
                nodes_.back().depth = nodes_[node].depth + 1;
                nodes_[node].children[c] = child;
                node = child;
            }
        }
        nodes_[node].end = true;
    }

    // Aho-Corasick fail links + output flags via BFS over trie depth.
    std::deque<int> queue;
    for (auto& [c, child] : nodes_[0].children) {
        nodes_[child].fail = 0;
        queue.push_back(child);
    }
    while (!queue.empty()) {
        const int u = queue.front();
        queue.pop_front();
        nodes_[u].output = nodes_[u].end || nodes_[nodes_[u].fail].output;
        // Snapshot children keys/values: building fail links reads other nodes.
        std::vector<std::pair<char, int>> kids(nodes_[u].children.begin(),
                                               nodes_[u].children.end());
        for (auto& [c, v] : kids) {
            int f = nodes_[u].fail;
            while (f != 0 && nodes_[f].children.find(c) == nodes_[f].children.end()) {
                f = nodes_[f].fail;
            }
            auto it = nodes_[f].children.find(c);
            nodes_[v].fail = (it != nodes_[f].children.end() && it->second != v)
                                 ? it->second
                                 : 0;
            queue.push_back(v);
        }
    }

    // Node score = per_char * depth, optionally capped. The cap flattens the
    // deep region of every phrase, so no single match can accumulate an
    // unbounded per-character boost (telescoping bonuses derive from these
    // scores, so the cap needs no special-casing in advance()). Fail-arc
    // refunds still work: a shallower fail target always has a lower score.
    score_.assign(nodes_.size(), 0.0f);
    for (size_t n = 0; n < nodes_.size(); ++n) {
        float s = per_char_ * static_cast<float>(nodes_[n].depth);
        if (max_bonus > 0.0f) s = std::min(s, max_bonus);
        score_[n] = s;
    }
}

int ContextGraph::go(int node, char c) const {
    while (node != 0 && nodes_[node].children.find(c) == nodes_[node].children.end()) {
        node = nodes_[node].fail;
    }
    auto it = nodes_[node].children.find(c);
    return it != nodes_[node].children.end() ? it->second : 0;
}

ContextGraph::Step ContextGraph::advance(State s, const std::string& token_text) const {
    if (empty()) return {0.0f, 0};
    const std::string norm = normalize(token_text);
    float bonus = 0.0f;
    int n = s;
    for (char c : norm) {
        const int d = go(n, c);
        // Telescoping: forward arcs add per_char; fail arcs (broken match)
        // land on a shallower node and refund the difference.
        bonus += score_[d] - score_[n];
        if (nodes_[d].output) bonus += completion_;
        n = d;
    }
    return {bonus, n};
}

}  // namespace speech_core
