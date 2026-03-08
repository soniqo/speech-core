#include "speech_core/tools/tool_registry.h"
#include "speech_core/tools/intent_matcher.h"
#include "speech_core/tools/tool_executor.h"

#include <cassert>
#include <cstdio>
#include <thread>

using namespace speech_core;

// ---------------------------------------------------------------------------
// ToolRegistry tests
// ---------------------------------------------------------------------------

void test_registry_add_and_find() {
    ToolRegistry reg;
    ToolDefinition tool;
    tool.name = "greet";
    tool.description = "Say hello";
    tool.triggers = {"hello", "hi there"};
    tool.command = "echo Hello!";
    reg.add(tool);

    assert(reg.size() == 1);
    assert(reg.find("greet") != nullptr);
    assert(reg.find("greet")->description == "Say hello");
    assert(reg.find("nonexistent") == nullptr);
    printf("  PASS: registry_add_and_find\n");
}

void test_registry_load_json() {
    ToolRegistry reg;
    std::string json = R"([
        {
            "name": "tell_time",
            "description": "Tell the current time",
            "triggers": ["what time", "current time"],
            "command": "date '+%I:%M %p'",
            "timeout": 5,
            "cooldown": 30
        },
        {
            "name": "greet",
            "description": "Say hello",
            "triggers": ["hello", "hi"],
            "command": "echo Hello!",
            "timeout": 3,
            "cooldown": 0
        }
    ])";

    int count = reg.load_json(json);
    assert(count == 2);
    assert(reg.size() == 2);

    auto* time_tool = reg.find("tell_time");
    assert(time_tool != nullptr);
    assert(time_tool->triggers.size() == 2);
    assert(time_tool->triggers[0] == "what time");
    assert(time_tool->timeout == 5);
    assert(time_tool->cooldown == 30);

    auto* greet_tool = reg.find("greet");
    assert(greet_tool != nullptr);
    assert(greet_tool->cooldown == 0);

    printf("  PASS: registry_load_json\n");
}

void test_registry_json_malformed() {
    ToolRegistry reg;
    assert(reg.load_json("not json") == -1);
    assert(reg.load_json("") == -1);
    assert(reg.load_json("{}") == -1);  // not an array
    printf("  PASS: registry_json_malformed\n");
}

void test_registry_json_empty_array() {
    ToolRegistry reg;
    assert(reg.load_json("[]") == 0);
    assert(reg.size() == 0);
    printf("  PASS: registry_json_empty_array\n");
}

// ---------------------------------------------------------------------------
// IntentMatcher tests
// ---------------------------------------------------------------------------

void test_matcher_basic() {
    ToolRegistry reg;
    ToolDefinition tool;
    tool.name = "tell_time";
    tool.triggers = {"what time", "current time"};
    reg.add(tool);

    IntentMatcher matcher(reg);
    assert(matcher.match("What time is it?") == "tell_time");
    assert(matcher.match("Tell me the current time please") == "tell_time");
    assert(matcher.match("How are you?") == "");
    printf("  PASS: matcher_basic\n");
}

void test_matcher_case_insensitive() {
    ToolRegistry reg;
    ToolDefinition tool;
    tool.name = "greet";
    tool.triggers = {"hello", "hi there"};
    reg.add(tool);

    IntentMatcher matcher(reg);
    assert(matcher.match("HELLO world") == "greet");
    assert(matcher.match("Hi There buddy") == "greet");
    assert(matcher.match("hey") == "");
    printf("  PASS: matcher_case_insensitive\n");
}

void test_matcher_regex() {
    ToolRegistry reg;
    ToolDefinition tool;
    tool.name = "weather";
    tool.triggers = {"weather|forecast", "how.*outside"};
    reg.add(tool);

    IntentMatcher matcher(reg);
    assert(matcher.match("What's the weather?") == "weather");
    assert(matcher.match("Show me the forecast") == "weather");
    assert(matcher.match("How is it outside?") == "weather");
    assert(matcher.match("Tell me a joke") == "");
    printf("  PASS: matcher_regex\n");
}

void test_matcher_first_match_wins() {
    ToolRegistry reg;

    ToolDefinition t1;
    t1.name = "tool_a";
    t1.triggers = {"hello"};
    reg.add(t1);

    ToolDefinition t2;
    t2.name = "tool_b";
    t2.triggers = {"hello"};  // same trigger
    reg.add(t2);

    IntentMatcher matcher(reg);
    assert(matcher.match("hello") == "tool_a");  // first registered wins
    printf("  PASS: matcher_first_match_wins\n");
}

void test_matcher_no_tools() {
    ToolRegistry reg;
    IntentMatcher matcher(reg);
    assert(matcher.match("anything") == "");
    printf("  PASS: matcher_no_tools\n");
}

// ---------------------------------------------------------------------------
// ToolExecutor tests
// ---------------------------------------------------------------------------

void test_executor_echo() {
    ToolDefinition tool;
    tool.name = "echo_test";
    tool.command = "echo 'hello from tool'";
    tool.cooldown = 0;

    ToolExecutor executor;
    auto result = executor.execute(tool);
    assert(result.success);
    assert(result.tool_name == "echo_test");
    assert(result.output == "hello from tool");
    assert(!result.on_cooldown);
    printf("  PASS: executor_echo\n");
}

void test_executor_cooldown() {
    ToolDefinition tool;
    tool.name = "cooldown_test";
    tool.command = "echo ok";
    tool.cooldown = 60;  // 60 second cooldown

    ToolExecutor executor;

    // First execution should work
    auto r1 = executor.execute(tool);
    assert(r1.success);
    assert(!r1.on_cooldown);

    // Second execution should be on cooldown
    auto r2 = executor.execute(tool);
    assert(r2.on_cooldown);
    assert(r2.output == "Tool is on cooldown");

    printf("  PASS: executor_cooldown\n");
}

void test_executor_no_cooldown() {
    ToolDefinition tool;
    tool.name = "no_cooldown";
    tool.command = "echo ok";
    tool.cooldown = 0;  // no cooldown

    ToolExecutor executor;
    auto r1 = executor.execute(tool);
    auto r2 = executor.execute(tool);
    assert(r1.success);
    assert(r2.success);
    assert(!r2.on_cooldown);
    printf("  PASS: executor_no_cooldown\n");
}

void test_executor_failed_command() {
    ToolDefinition tool;
    tool.name = "fail_test";
    tool.command = "false";  // exits with code 1
    tool.cooldown = 0;

    ToolExecutor executor;
    auto result = executor.execute(tool);
    assert(!result.success);
    printf("  PASS: executor_failed_command\n");
}

void test_executor_reset_cooldowns() {
    ToolDefinition tool;
    tool.name = "reset_test";
    tool.command = "echo ok";
    tool.cooldown = 60;

    ToolExecutor executor;
    executor.execute(tool);

    // On cooldown
    assert(executor.is_on_cooldown("reset_test", 60));

    // Reset
    executor.reset_cooldowns();
    assert(!executor.is_on_cooldown("reset_test", 60));

    // Can execute again
    auto result = executor.execute(tool);
    assert(result.success);
    assert(!result.on_cooldown);
    printf("  PASS: executor_reset_cooldowns\n");
}

int main() {
    printf("test_tools:\n");

    // Registry
    test_registry_add_and_find();
    test_registry_load_json();
    test_registry_json_malformed();
    test_registry_json_empty_array();

    // Matcher
    test_matcher_basic();
    test_matcher_case_insensitive();
    test_matcher_regex();
    test_matcher_first_match_wins();
    test_matcher_no_tools();

    // Executor
    test_executor_echo();
    test_executor_cooldown();
    test_executor_no_cooldown();
    test_executor_failed_command();
    test_executor_reset_cooldowns();

    printf("All tools tests passed.\n");
    return 0;
}
