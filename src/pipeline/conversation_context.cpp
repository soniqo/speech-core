#include "speech_core/pipeline/conversation_context.h"

namespace speech_core {

ConversationContext::ConversationContext(
    const std::string& system_prompt, size_t max_messages, size_t max_tokens)
    : max_messages_(max_messages), max_tokens_(max_tokens) {
    if (!system_prompt.empty()) {
        messages_.push_back({MessageRole::System, system_prompt, 0.0});
    }
}

void ConversationContext::set_token_counter(TokenCounter counter) {
    token_counter_ = std::move(counter);
}

void ConversationContext::add_user_message(const std::string& text, double timestamp) {
    messages_.push_back({MessageRole::User, text, timestamp});
    trim();
}

void ConversationContext::add_assistant_message(const std::string& text, double timestamp) {
    messages_.push_back({MessageRole::Assistant, text, timestamp});
    trim();
}

void ConversationContext::add_tool_message(
    const std::string& tool_name, const std::string& output, double timestamp)
{
    std::string content = "[" + tool_name + "] " + output;
    messages_.push_back({MessageRole::Tool, content, timestamp});
}

void ConversationContext::trim() {
    // Index of first non-system message
    size_t first = (!messages_.empty() && messages_[0].role == MessageRole::System) ? 1 : 0;

    // Message-count trimming: keep system prompt + last max_messages_
    if (max_messages_ > 0 && messages_.size() > max_messages_ + first) {
        auto start = messages_.begin() + static_cast<ptrdiff_t>(first);
        auto end = messages_.end() - static_cast<ptrdiff_t>(max_messages_);
        if (start < end) {
            messages_.erase(start, end);
        }
    }

    // Token-based trimming: remove oldest non-system messages until within budget
    if (max_tokens_ > 0 && token_counter_ && messages_.size() > first) {
        while (messages_.size() > first + 1) {
            int total = 0;
            for (const auto& m : messages_) {
                total += token_counter_(m.content);
            }
            if (total <= static_cast<int>(max_tokens_)) break;
            messages_.erase(messages_.begin() + static_cast<ptrdiff_t>(first));
        }
    }
}

size_t ConversationContext::turn_count() const {
    size_t count = 0;
    for (const auto& m : messages_) {
        if (m.role != MessageRole::System) count++;
    }
    return count;
}

void ConversationContext::clear() {
    if (!messages_.empty() && messages_[0].role == MessageRole::System) {
        messages_.resize(1);
    } else {
        messages_.clear();
    }
}

}  // namespace speech_core
