#pragma once

#include <string>
#include <vector>

#include "speech_core/tools/tool_types.h"

namespace speech_core {

/// Registry of available tools.
class ToolRegistry {
public:
    /// Add a tool definition.
    void add(ToolDefinition tool);

    /// Load tools from a JSON string.
    /// Expected format:
    /// ```json
    /// [
    ///   {
    ///     "name": "tell_time",
    ///     "description": "Tell the current time",
    ///     "triggers": ["what time", "current time"],
    ///     "command": "date '+%I:%M %p'",
    ///     "timeout": 5,
    ///     "cooldown": 30
    ///   }
    /// ]
    /// ```
    /// @return Number of tools loaded, or -1 on parse error.
    int load_json(const std::string& json);

    /// Find a tool by name. Returns nullptr if not found.
    const ToolDefinition* find(const std::string& name) const;

    /// All registered tools.
    const std::vector<ToolDefinition>& tools() const { return tools_; }

    /// Number of registered tools.
    size_t size() const { return tools_.size(); }

    /// Remove all tools.
    void clear() { tools_.clear(); }

private:
    std::vector<ToolDefinition> tools_;
};

}  // namespace speech_core
