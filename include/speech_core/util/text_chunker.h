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

}  // namespace speech_core
