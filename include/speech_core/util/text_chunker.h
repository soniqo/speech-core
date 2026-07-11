#pragma once

#include <functional>
#include <string>
#include <vector>

namespace speech_core {

/// Counts synthesis tokens for a piece of text (e.g. phonemizer tokens,
/// including any per-call BOS/EOS the tokenizer adds).
using TokenCounter = std::function<size_t(const std::string&)>;

/// Split text into chunks suitable for fixed-capacity TTS synthesis.
///
/// Chunks are packed greedily to at most `max_tokens` each, preferring
/// sentence boundaries, then clause boundaries (,;:), then word boundaries,
/// then UTF-8 character boundaries for a single oversized word. A trailing
/// chunk smaller than `min_tail_tokens` is merged into the previous chunk
/// when the combined count stays within `hard_cap_tokens` (the model's
/// absolute input capacity, typically larger than the packing budget), so
/// tiny tail utterances — which some models synthesize unreliably — are
/// avoided. Whitespace-only input yields no chunks.
std::vector<std::string> chunk_text_for_synthesis(
    const std::string& text,
    const TokenCounter& count_tokens,
    size_t max_tokens,
    size_t hard_cap_tokens,
    size_t min_tail_tokens);

/// Split one unsafe synthesis chunk into two balanced, strictly smaller
/// pieces. Semantic boundary class (sentence, clause, then word) takes
/// priority over balance; UTF-8 character boundaries are the last resort.
/// Candidates inside a punctuation/closing-quote run are skipped so the run
/// stays on the left. Returns empty when no split can keep both children
/// within [`min_tokens`, `max_tokens`].
std::vector<std::string> split_text_for_synthesis_retry(
    const std::string& text,
    const TokenCounter& count_tokens,
    size_t min_tokens,
    size_t max_tokens);

}  // namespace speech_core
