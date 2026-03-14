#pragma once

#include <functional>
#include <string>
#include <vector>

#include "speech_core/interfaces.h"

namespace speech_core {

/// Callback to count tokens in a text string.
using TokenCounter = std::function<int(const std::string& text)>;

/// Tracks conversation history for multi-turn LLM interactions.
class ConversationContext {
public:
    /// @param system_prompt  Initial system prompt for the LLM
    /// @param max_messages   Maximum messages to retain (0 = unlimited)
    /// @param max_tokens     Maximum total tokens (0 = disabled, requires token_counter)
    explicit ConversationContext(
        const std::string& system_prompt = "",
        size_t max_messages = 50,
        size_t max_tokens = 0);

    /// Set a token counting function for token-based trimming.
    void set_token_counter(TokenCounter counter);

    /// Add a user message (typically from STT).
    void add_user_message(const std::string& text, double timestamp = 0.0);

    /// Add an assistant message (typically from LLM).
    void add_assistant_message(const std::string& text, double timestamp = 0.0);

    /// Add a tool result message.
    void add_tool_message(const std::string& tool_name, const std::string& output,
                          double timestamp = 0.0);

    /// Get all messages for LLM input (including system prompt).
    const std::vector<Message>& messages() const { return messages_; }

    /// Number of messages (excluding system prompt).
    size_t turn_count() const;

    /// Clear all messages except system prompt.
    void clear();

private:
    std::vector<Message> messages_;
    size_t max_messages_;
    size_t max_tokens_;
    TokenCounter token_counter_;

    void trim();
};

}  // namespace speech_core
