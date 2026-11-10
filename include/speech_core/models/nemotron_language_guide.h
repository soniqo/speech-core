#pragma once

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace speech_core {

/// Experimental fractional prompt conditioning, matching speech-swift.
/// Empty/zero returns no override, preserving the ordinary one-hot path.
/// This does not mask vocabulary or represent language probabilities.
inline std::vector<float> nemotron_language_guide_mask(
    const std::vector<std::string>& expected, float weight, int prompt_count,
    int auto_slot, int selected_slot,
    const std::unordered_map<std::string, int>& languages,
    const std::vector<std::string>& vocabulary) {
    if (!std::isfinite(weight) || weight < 0 || weight > .1f)
        throw std::invalid_argument("Language-guide weight must be between 0 and 0.1");
    if (expected.empty()) return {};
    if (auto_slot < 0 || auto_slot >= prompt_count || selected_slot != auto_slot)
        throw std::invalid_argument("A language guide requires the automatic-language prompt");
    std::set<int> slots;
    for (const auto& locale : expected) {
        const auto it = languages.find(locale);
        const auto tag = "<" + locale + ">";
        if (it == languages.end() || it->second < 0 || it->second >= prompt_count
            || it->second == auto_slot
            || std::find(vocabulary.begin(), vocabulary.end(), tag) == vocabulary.end())
            throw std::invalid_argument("Unsupported exact language-guide locale: " + locale);
        slots.insert(it->second);
    }
    if (weight == 0) return {};
    std::vector<float> mask(static_cast<size_t>(prompt_count), 0.f);
    mask[auto_slot] = 1.f - weight;
    for (const int slot : slots) mask[slot] = weight / static_cast<float>(slots.size());
    return mask;
}

}  // namespace speech_core
