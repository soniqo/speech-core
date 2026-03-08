#pragma once

#include <string>

#include "speech_core/tools/tool_registry.h"

namespace speech_core {

/// Matches transcripts against tool trigger patterns.
///
/// Uses regex matching (case-insensitive) against the triggers
/// defined in each tool. Returns the first matching tool name.
class IntentMatcher {
public:
    explicit IntentMatcher(const ToolRegistry& registry);

    /// Match a transcript against all registered tool triggers.
    /// @param transcript  The STT output text
    /// @return Matching tool name, or empty string if no match.
    std::string match(const std::string& transcript) const;

private:
    const ToolRegistry& registry_;
};

}  // namespace speech_core
