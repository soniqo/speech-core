#include "speech_core/pipeline/conversation_context.h"

namespace speech_core {

ConversationContext::ConversationContext(
    const std::string& system_prompt, size_t max_messages)
    : max_messages_(max_messages) {
    if (!system_prompt.empty()) {
        messages_.push_back({MessageRole::System, system_prompt, 0.0});
    }
}

void ConversationContext::add_user_message(const std::string& text, double timestamp) {
    messages_.push_back({MessageRole::User, text, timestamp});

    // Trim old messages (keep system prompt + recent turns)
    if (max_messages_ > 0 && messages_.size() > max_messages_ + 1) {
        // Keep first message (system prompt) and last max_messages_
        auto start = messages_.begin() + 1;
        auto end = messages_.end() - static_cast<ptrdiff_t>(max_messages_);
        messages_.erase(start, end);
    }
}

void ConversationContext::add_assistant_message(const std::string& text, double timestamp) {
    messages_.push_back({MessageRole::Assistant, text, timestamp});

    if (max_messages_ > 0 && messages_.size() > max_messages_ + 1) {
        auto start = messages_.begin() + 1;
        auto end = messages_.end() - static_cast<ptrdiff_t>(max_messages_);
        messages_.erase(start, end);
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
    // Keep system prompt if present
    if (!messages_.empty() && messages_[0].role == MessageRole::System) {
        messages_.resize(1);
    } else {
        messages_.clear();
    }
}

}  // namespace speech_core
