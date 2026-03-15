#pragma once

#include <functional>
#include <string>
#include <vector>

namespace speech_core {

/// Callback-based tool handler. Takes (tool_name, arguments) and returns output.
using ToolHandler = std::function<std::string(const std::string& name, const std::string& args)>;

/// Definition of a tool that can be triggered by voice commands.
struct ToolDefinition {
    /// Unique tool name (e.g. "tell_time").
    std::string name;

    /// Human-readable description.
    std::string description;

    /// Regex patterns that trigger this tool (matched against transcript).
    std::vector<std::string> triggers;

    /// Shell command to execute (used when handler is not set).
    std::string command;

    /// Platform callback handler (takes priority over shell command).
    ToolHandler handler;

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
