#pragma once

#include <string>
#include <vector>

namespace speech_core {

/// Definition of a tool that can be triggered by voice commands.
struct ToolDefinition {
    /// Unique tool name (e.g. "tell_time").
    std::string name;

    /// Human-readable description.
    std::string description;

    /// Regex patterns that trigger this tool (matched against transcript).
    std::vector<std::string> triggers;

    /// Shell command to execute.
    std::string command;

    /// Maximum execution time in seconds (0 = no limit).
    int timeout = 5;

    /// Minimum seconds between invocations (0 = no cooldown).
    int cooldown = 30;
};

/// Result of executing a tool.
struct ToolResult {
    std::string tool_name;
    std::string output;
    bool success = false;
    bool on_cooldown = false;
};

}  // namespace speech_core
