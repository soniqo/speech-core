#include "speech_core/pipeline/conversation_context.h"

#include <cassert>
#include <cstdio>

using namespace speech_core;

void test_construction_with_system_prompt() {
    ConversationContext ctx("You are a helpful assistant.");
    assert(ctx.messages().size() == 1);
    assert(ctx.messages()[0].role == MessageRole::System);
    assert(ctx.messages()[0].content == "You are a helpful assistant.");
    assert(ctx.turn_count() == 0);
    printf("  PASS: construction_with_system_prompt\n");
}

void test_construction_without_system_prompt() {
    ConversationContext ctx("");
    assert(ctx.messages().empty());
    assert(ctx.turn_count() == 0);
    printf("  PASS: construction_without_system_prompt\n");
}

void test_add_messages() {
    ConversationContext ctx("system");
    ctx.add_user_message("hello");
    ctx.add_assistant_message("hi there");

    assert(ctx.messages().size() == 3);
    assert(ctx.messages()[1].role == MessageRole::User);
    assert(ctx.messages()[1].content == "hello");
    assert(ctx.messages()[2].role == MessageRole::Assistant);
    assert(ctx.messages()[2].content == "hi there");
    printf("  PASS: add_messages\n");
}

void test_add_tool_message() {
    ConversationContext ctx("system");
    ctx.add_tool_message("tell_time", "3:14 PM");

    assert(ctx.messages().size() == 2);
    assert(ctx.messages()[1].role == MessageRole::Tool);
    assert(ctx.messages()[1].content == "[tell_time] 3:14 PM");
    printf("  PASS: add_tool_message\n");
}

void test_turn_count() {
    ConversationContext ctx("system");
    assert(ctx.turn_count() == 0);
    ctx.add_user_message("a");
    assert(ctx.turn_count() == 1);
    ctx.add_assistant_message("b");
    assert(ctx.turn_count() == 2);
    ctx.add_tool_message("t", "r");
    assert(ctx.turn_count() == 3);
    printf("  PASS: turn_count\n");
}

void test_clear_with_system_prompt() {
    ConversationContext ctx("system");
    ctx.add_user_message("a");
    ctx.add_assistant_message("b");
    ctx.clear();

    assert(ctx.messages().size() == 1);
    assert(ctx.messages()[0].role == MessageRole::System);
    assert(ctx.turn_count() == 0);
    printf("  PASS: clear_with_system_prompt\n");
}

void test_clear_without_system_prompt() {
    ConversationContext ctx("");
    ctx.add_user_message("a");
    ctx.add_assistant_message("b");
    ctx.clear();

    assert(ctx.messages().empty());
    printf("  PASS: clear_without_system_prompt\n");
}

void test_message_count_trimming_with_system() {
    ConversationContext ctx("system", 4);
    for (int i = 0; i < 10; i++) {
        ctx.add_user_message("u" + std::to_string(i));
        ctx.add_assistant_message("a" + std::to_string(i));
    }
    // System prompt + max 4 messages
    assert(ctx.messages().size() <= 5);
    assert(ctx.messages()[0].role == MessageRole::System);
    assert(ctx.messages()[0].content == "system");
    printf("  PASS: message_count_trimming_with_system\n");
}

void test_message_count_trimming_without_system() {
    ConversationContext ctx("", 4);
    for (int i = 0; i < 10; i++) {
        ctx.add_user_message("u" + std::to_string(i));
        ctx.add_assistant_message("a" + std::to_string(i));
    }
    assert(ctx.messages().size() <= 4);
    // Most recent messages preserved
    assert(ctx.messages().back().content == "a9");
    printf("  PASS: message_count_trimming_without_system\n");
}

void test_token_trimming() {
    ConversationContext ctx("sys", 0, 20);  // unlimited messages, 20 token limit
    // Word counter: 1 token per word
    ctx.set_token_counter([](const std::string& text) -> int {
        if (text.empty()) return 0;
        int n = 1;
        for (char c : text) if (c == ' ') n++;
        return n;
    });

    // "sys" = 1 token
    // Each message ~3-4 tokens
    for (int i = 0; i < 10; i++) {
        ctx.add_user_message("hello world msg " + std::to_string(i));
        ctx.add_assistant_message("ok got it " + std::to_string(i));
    }

    // Count total tokens
    int total = 0;
    for (const auto& m : ctx.messages()) {
        int n = m.content.empty() ? 0 : 1;
        for (char c : m.content) if (c == ' ') n++;
        total += n;
    }
    assert(total <= 20);
    assert(ctx.messages()[0].role == MessageRole::System);
    printf("  PASS: token_trimming\n");
}

void test_token_trimming_without_counter() {
    ConversationContext ctx("sys", 0, 10);  // max 10 tokens but no counter set
    for (int i = 0; i < 20; i++) {
        ctx.add_user_message("long message with many words number " + std::to_string(i));
    }
    // No counter → no token trimming → all messages kept
    assert(ctx.messages().size() == 21);  // system + 20
    printf("  PASS: token_trimming_without_counter\n");
}

void test_mask_tool_results_enabled() {
    ConversationContext ctx("system", 6, 0, true);

    ctx.add_user_message("what time");
    ctx.add_assistant_message("let me check");
    ctx.add_tool_message("tell_time", "3:14 PM");
    ctx.add_assistant_message("it is 3:14 PM");
    ctx.add_user_message("thanks");
    ctx.add_assistant_message("welcome");
    // 6 messages at limit — add one more to trigger trim
    ctx.add_user_message("another question");

    // Tool message should have been dropped first
    bool has_tool = false;
    for (const auto& m : ctx.messages()) {
        if (m.role == MessageRole::Tool) has_tool = true;
    }
    assert(!has_tool);
    assert(ctx.messages()[0].role == MessageRole::System);
    printf("  PASS: mask_tool_results_enabled\n");
}

void test_mask_tool_results_disabled() {
    ConversationContext ctx("system", 4, 0, false);

    ctx.add_user_message("u1");
    ctx.add_tool_message("tool", "result");
    ctx.add_assistant_message("a1");
    ctx.add_user_message("u2");
    ctx.add_assistant_message("a2");

    // With masking disabled, tool messages trimmed same as others
    // 5 non-system messages, max 4 → oldest trimmed
    assert(ctx.messages().size() <= 5);  // system + 4
    printf("  PASS: mask_tool_results_disabled\n");
}

void test_tool_tail_protection() {
    // Tool messages in the last 2 positions should not be masked
    ConversationContext ctx("system", 4, 0, true);

    ctx.add_user_message("u1");
    ctx.add_assistant_message("a1");
    ctx.add_user_message("call tool");
    ctx.add_assistant_message("calling...");
    ctx.add_tool_message("tool", "result");  // position -1 from end after next add
    ctx.add_user_message("trigger trim");    // tool is now at -2, protected

    // The tool message should still be present (in last 2)
    bool has_tool = false;
    for (const auto& m : ctx.messages()) {
        if (m.role == MessageRole::Tool) has_tool = true;
    }
    assert(has_tool);
    printf("  PASS: tool_tail_protection\n");
}

void test_combined_message_and_token_trimming() {
    ConversationContext ctx("s", 10, 30, true);
    ctx.set_token_counter([](const std::string& text) -> int {
        if (text.empty()) return 0;
        int n = 1;
        for (char c : text) if (c == ' ') n++;
        return n;
    });

    for (int i = 0; i < 8; i++) {
        ctx.add_user_message("hello world " + std::to_string(i));
        ctx.add_assistant_message("ok " + std::to_string(i));
    }

    // Should satisfy both: <= 10 messages AND <= 30 tokens
    assert(ctx.messages().size() <= 11);  // system + 10
    int total = 0;
    for (const auto& m : ctx.messages()) {
        int n = m.content.empty() ? 0 : 1;
        for (char c : m.content) if (c == ' ') n++;
        total += n;
    }
    assert(total <= 30);
    printf("  PASS: combined_message_and_token_trimming\n");
}

void test_unlimited_messages() {
    ConversationContext ctx("system", 0);  // 0 = unlimited
    for (int i = 0; i < 100; i++) {
        ctx.add_user_message("msg" + std::to_string(i));
    }
    assert(ctx.messages().size() == 101);  // system + 100
    printf("  PASS: unlimited_messages\n");
}

int main() {
    printf("test_conversation_context:\n");
    test_construction_with_system_prompt();
    test_construction_without_system_prompt();
    test_add_messages();
    test_add_tool_message();
    test_turn_count();
    test_clear_with_system_prompt();
    test_clear_without_system_prompt();
    test_message_count_trimming_with_system();
    test_message_count_trimming_without_system();
    test_token_trimming();
    test_token_trimming_without_counter();
    test_mask_tool_results_enabled();
    test_mask_tool_results_disabled();
    test_tool_tail_protection();
    test_combined_message_and_token_trimming();
    test_unlimited_messages();
    printf("All conversation context tests passed.\n");
    return 0;
}
