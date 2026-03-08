#include "speech_core/tools/intent_matcher.h"

#include <algorithm>
#include <regex>

namespace speech_core {

IntentMatcher::IntentMatcher(const ToolRegistry& registry)
    : registry_(registry) {}

std::string IntentMatcher::match(const std::string& transcript) const {
    // Convert transcript to lowercase for case-insensitive matching
    std::string lower = transcript;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    for (const auto& tool : registry_.tools()) {
        for (const auto& trigger : tool.triggers) {
            try {
                std::regex pattern(trigger,
                    std::regex_constants::icase |
                    std::regex_constants::ECMAScript);
                if (std::regex_search(lower, pattern)) {
                    return tool.name;
                }
            } catch (const std::regex_error&) {
                // Invalid regex — try as plain substring
                std::string lower_trigger = trigger;
                std::transform(lower_trigger.begin(), lower_trigger.end(),
                               lower_trigger.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (lower.find(lower_trigger) != std::string::npos) {
                    return tool.name;
                }
            }
        }
    }
    return "";
}

}  // namespace speech_core
