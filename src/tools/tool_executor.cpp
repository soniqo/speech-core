#include "speech_core/tools/tool_executor.h"

#include <array>
#include <cstdio>

namespace speech_core {

ToolResult ToolExecutor::execute(const ToolDefinition& tool) {
    ToolResult result;
    result.tool_name = tool.name;

    // Check cooldown
    if (is_on_cooldown(tool.name, tool.cooldown)) {
        result.on_cooldown = true;
        result.output = "Tool is on cooldown";
        return result;
    }

    // Use callback handler if available, otherwise fall back to popen
    if (tool.handler) {
        try {
            result.output = tool.handler(tool.name, "");
            result.success = true;
        } catch (const std::exception& ex) {
            result.output = std::string("Tool handler failed: ") + ex.what();
            result.success = false;
        }
    } else {
        // Execute the shell command via popen
        FILE* pipe = popen(tool.command.c_str(), "r");
        if (!pipe) {
            result.output = "Failed to execute command";
            return result;
        }

        std::string output;
        std::array<char, 256> buffer;
        while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
            output += buffer.data();
        }

        int status = pclose(pipe);
        result.success = (status == 0);
        result.output = output;
    }

    // Trim trailing newline
    while (!result.output.empty() && result.output.back() == '\n') {
        result.output.pop_back();
    }

    // Record execution time for cooldown
    last_execution_[tool.name] = Clock::now();

    return result;
}

bool ToolExecutor::is_on_cooldown(const std::string& tool_name,
                                   int cooldown_seconds) const
{
    if (cooldown_seconds <= 0) return false;

    auto it = last_execution_.find(tool_name);
    if (it == last_execution_.end()) return false;

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        Clock::now() - it->second).count();
    return elapsed < cooldown_seconds;
}

}  // namespace speech_core
