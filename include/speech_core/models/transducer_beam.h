#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace speech_core::transducer {

/// Modified RNN-T beam search, extracted from any specific model so the
/// algorithm can be unit-tested with a toy transducer (no ML runtime).
///
/// Frame-synchronous: within one committed encoder frame each hypothesis may
/// emit up to `max_symbols` tokens; a blank ends its emission for that frame
/// and advances it. Streaming callers keep the `beams` vector across windows,
/// exactly like the greedy predictor state.
///
/// **Blank-bias correction.** Plain RNN-T beam search under-emits: blank is
/// usually high-probability, so a short, blank-heavy hypothesis out-scores the
/// complete one and the transcript gets truncated. `emit_bonus` adds a fixed
/// reward per emitted text token (a shallow-fusion blank penalty), which
/// counteracts the bias. It is the simplest of the standard fixes (the others
/// being hypothesis recombination and length normalization).

struct BeamParams {
    int   beam_size   = 4;
    int   max_symbols = 8;
    int   blank_id    = 0;
    int   eou_id      = -1;   ///< terminal token (ends the hypothesis); -1 = none
    int   eob_id      = -1;   ///< soft boundary: advances predictor, emits no text
    float emit_bonus  = 0.0f; ///< per-emitted-text-token reward; corrects blank bias
};

template <class State, class Aux>
struct BeamHyp {
    std::vector<int> tokens;      ///< emitted text token ids (no blank/eou/eob)
    double           score = 0.0;
    State            state;       ///< predictor state
    Aux              aux;         ///< biasing state (e.g. ContextGraph::State)
    bool             terminal = false;
};

namespace detail {
inline void log_softmax(const std::vector<float>& logits, std::vector<float>& out) {
    out.resize(logits.size());
    float mx = logits.empty() ? 0.0f : logits[0];
    for (float v : logits) mx = std::max(mx, v);
    double sum = 0.0;
    for (float v : logits) sum += std::exp(static_cast<double>(v) - mx);
    const double lse = static_cast<double>(mx) + std::log(sum);
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(logits[i]) - lse);
    }
}
}  // namespace detail

/// Advance `beams` (in/out, persistent across streaming windows) over
/// `frame_count` committed frames starting at window-local index `frame_begin`.
///
///   joint(int frame, const State&) -> const std::vector<float>&   (logits)
///   predict(int token, const State&) -> State                     (advance predictor)
///   bonus(int token, const Aux&) -> std::pair<float, Aux>         (biasing delta + next aux)
///
/// `joint` must return a reference valid until the next `joint` call (a reused
/// buffer is fine — the result is consumed immediately).
template <class State, class Aux, class Joint, class Predict, class Bonus>
void beam_decode_frames(std::vector<BeamHyp<State, Aux>>& beams,
                        int frame_begin, int frame_count,
                        const BeamParams& p,
                        Joint joint, Predict predict, Bonus bonus) {
    using Hyp = BeamHyp<State, Aux>;
    const int beam_size = std::max(1, p.beam_size);

    struct Cand {
        int    base;    // index into `active`
        int    token;
        double score;
        Aux    aux;     // resulting biasing state for a text token
    };

    std::vector<float> lp;
    std::vector<std::pair<float, int>> ranked;

    for (int f = 0; f < frame_count; ++f) {
        const int frame = frame_begin + f;

        std::vector<Hyp> active, frozen;
        for (auto& h : beams) (h.terminal ? frozen : active).push_back(std::move(h));
        beams.clear();

        for (int step = 0; step < p.max_symbols && !active.empty(); ++step) {
            std::vector<Cand> cands;
            cands.reserve(active.size() * (static_cast<size_t>(beam_size) + 1));

            for (size_t i = 0; i < active.size(); ++i) {
                const std::vector<float>& logits = joint(frame, active[i].state);
                detail::log_softmax(logits, lp);
                const int n = static_cast<int>(lp.size());

                // Blank: advance a frame, predictor + aux unchanged.
                cands.push_back({static_cast<int>(i), p.blank_id,
                                 active[i].score + lp[p.blank_id], active[i].aux});

                // Top `beam_size` non-blank tokens (logit rank == logprob rank).
                ranked.clear();
                for (int t = 0; t < n; ++t) if (t != p.blank_id) ranked.emplace_back(lp[t], t);
                const size_t keep = std::min(static_cast<size_t>(beam_size), ranked.size());
                std::partial_sort(ranked.begin(), ranked.begin() + keep, ranked.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                for (size_t k = 0; k < keep; ++k) {
                    const int t = ranked[k].second;
                    double sc = active[i].score + lp[t];
                    Aux    na = active[i].aux;
                    if (t != p.eou_id && t != p.eob_id) {
                        auto pr = bonus(t, active[i].aux);  // {delta, next aux}
                        sc += pr.first + p.emit_bonus;      // reward emitting text
                        na = std::move(pr.second);
                    }
                    cands.push_back({static_cast<int>(i), t, sc, std::move(na)});
                }
            }

            const size_t keepc = std::min(static_cast<size_t>(beam_size), cands.size());
            std::partial_sort(cands.begin(), cands.begin() + keepc, cands.end(),
                              [](const Cand& a, const Cand& b) { return a.score > b.score; });
            cands.resize(keepc);

            std::vector<Hyp> next_active;
            for (auto& c : cands) {
                Hyp h = active[c.base];  // copy: several cands can share a base
                h.score = c.score;
                if (c.token == p.blank_id) {
                    frozen.push_back(std::move(h));
                } else if (c.token == p.eou_id) {
                    h.state = predict(c.token, active[c.base].state);
                    h.terminal = true;
                    frozen.push_back(std::move(h));
                } else if (c.token == p.eob_id) {
                    h.state = predict(c.token, active[c.base].state);
                    next_active.push_back(std::move(h));
                } else {
                    h.tokens.push_back(c.token);
                    h.aux   = std::move(c.aux);
                    h.state = predict(c.token, active[c.base].state);
                    next_active.push_back(std::move(h));
                }
            }
            active = std::move(next_active);
        }

        beams = std::move(frozen);
        for (auto& h : active) beams.push_back(std::move(h));
        if (static_cast<int>(beams.size()) > beam_size) {
            std::partial_sort(beams.begin(), beams.begin() + beam_size, beams.end(),
                              [](const Hyp& a, const Hyp& b) { return a.score > b.score; });
            beams.resize(beam_size);
        }
    }
}

template <class State, class Aux>
const BeamHyp<State, Aux>* best_hyp(const std::vector<BeamHyp<State, Aux>>& beams) {
    const BeamHyp<State, Aux>* best = nullptr;
    for (const auto& h : beams) if (!best || h.score > best->score) best = &h;
    return best;
}

}  // namespace speech_core::transducer
