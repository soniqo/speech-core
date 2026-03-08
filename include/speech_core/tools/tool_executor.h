#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include "speech_core/tools/tool_types.h"

namespace speech_core {

/// Executes tool commands and enforces cooldowns.
class ToolExecutor {
public:
    /// Execute a tool's shell command.
    /// Returns early with on_cooldown=true if cooldown hasn't elapsed.
    ToolResult execute(const ToolDefinition& tool);

    /// Check if a tool is on cooldown.
    bool is_on_cooldown(const std::string& tool_name, int cooldown_seconds) const;

    /// Reset all cooldowns.
    void reset_cooldowns() { last_execution_.clear(); }

private:
    using Clock = std::chrono::steady_clock;
    std::unordered_map<std::string, Clock::time_point> last_execution_;
};

}  // namespace speech_core
