#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core::parakeet {

/// Convert a locale or language tag ("en-US", "fr_FR") to the lower-case
/// ISO 639 language token used by Parakeet ("en", "fr").
std::string normalize_language_code(const std::string& language);

/// Resolve language tags to Parakeet token IDs. Unknown languages are skipped.
std::vector<int> resolve_language_tokens(
    const std::unordered_map<int, std::string>& token_to_language,
    const std::vector<std::string>& languages);

/// If greedy_token is a Parakeet language token and guidance is configured,
/// replace it with the highest-scoring guided language token.
int apply_language_guidance(
    const float* token_logits,
    int greedy_token,
    float* greedy_score,
    const std::unordered_map<int, std::string>& token_to_language,
    const std::vector<int>& guided_language_tokens);

}  // namespace speech_core::parakeet
