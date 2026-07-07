#include "speech_core/models/parakeet_language_guidance.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <unordered_set>

namespace speech_core::parakeet {

std::string normalize_language_code(const std::string& language) {
    std::string out;
    out.reserve(language.size());
    for (char ch : language) {
        if (ch == '-' || ch == '_') break;
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

std::vector<int> resolve_language_tokens(
    const std::unordered_map<int, std::string>& token_to_language,
    const std::vector<std::string>& languages)
{
    std::unordered_map<std::string, int> language_to_token;
    language_to_token.reserve(token_to_language.size());
    for (const auto& [token, language] : token_to_language) {
        language_to_token[normalize_language_code(language)] = token;
    }

    std::vector<int> resolved;
    std::unordered_set<int> seen;
    for (const auto& language : languages) {
        const std::string code = normalize_language_code(language);
        if (code.empty() || code == "auto") continue;
        auto it = language_to_token.find(code);
        if (it == language_to_token.end()) continue;
        if (seen.insert(it->second).second) {
            resolved.push_back(it->second);
        }
    }
    return resolved;
}

int apply_language_guidance(
    const float* token_logits,
    int greedy_token,
    float* greedy_score,
    const std::unordered_map<int, std::string>& token_to_language,
    const std::vector<int>& guided_language_tokens)
{
    if (token_logits == nullptr || guided_language_tokens.empty()) {
        return greedy_token;
    }
    if (token_to_language.find(greedy_token) == token_to_language.end()) {
        return greedy_token;
    }

    int best_token = greedy_token;
    float best_score = -std::numeric_limits<float>::infinity();
    for (int token : guided_language_tokens) {
        const float score = token_logits[token];
        if (score > best_score) {
            best_score = score;
            best_token = token;
        }
    }
    if (greedy_score != nullptr) {
        *greedy_score = best_score;
    }
    return best_token;
}

}  // namespace speech_core::parakeet
