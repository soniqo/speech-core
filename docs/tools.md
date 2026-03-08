# Tool Calling

The tool engine adds function-calling capabilities to the voice pipeline. When a user says something that matches a tool trigger, the pipeline executes the tool and uses the result to generate a response.

## Flow

```
transcript → IntentMatcher → match? ─yes─→ ToolExecutor → result
                                │                            │
                                no                    inject into context
                                │                            │
                                ▼                            ▼
                          normal LLM path              LLM with tool result
                                │                            │
                                ▼                            ▼
                              TTS                          TTS
```

1. After STT produces a transcript, `IntentMatcher` checks all registered tool triggers
2. If a trigger matches, `ToolExecutor` runs the shell command
3. The tool result is injected as a `Tool` message in `ConversationContext`
4. The LLM receives the full conversation including the tool result
5. The LLM generates a natural response incorporating the tool output

If no LLM is configured, the tool output is spoken directly.

## Tool Definition

```cpp
ToolDefinition tool;
tool.name = "tell_time";
tool.description = "Tell the current time";
tool.triggers = {"what time", "current time"};  // regex patterns
tool.command = "date '+%I:%M %p'";
tool.timeout = 5;   // seconds (advisory)
tool.cooldown = 30;  // seconds between invocations
```

### JSON format

Tools can be loaded from JSON:

```json
[
  {
    "name": "tell_time",
    "description": "Tell the current time",
    "triggers": ["what time", "current time"],
    "command": "date '+%I:%M %p'",
    "timeout": 5,
    "cooldown": 30
  },
  {
    "name": "weather",
    "description": "Current weather",
    "triggers": ["weather|forecast", "how.*outside"],
    "command": "curl -s wttr.in/?format='%t+%C'",
    "timeout": 10,
    "cooldown": 60
  }
]
```

```cpp
ToolRegistry registry;
registry.load_json(json_string);
```

## Trigger Patterns

Triggers are matched as regex patterns (case-insensitive) against the full transcript. If a regex is invalid, it falls back to plain substring matching.

Examples:
- `"what time"` — matches "What time is it?", "Do you know what time it is?"
- `"weather|forecast"` — matches "What's the weather?" or "Show me the forecast"
- `"how.*outside"` — matches "How is it outside?"

First matching tool wins. Tools are checked in registration order.

## Cooldown

Each tool has a `cooldown` period (in seconds). After execution, the same tool cannot be triggered again until the cooldown expires. Set `cooldown = 0` to disable.

When a tool is on cooldown, `try_tool_call()` returns false and the pipeline falls through to normal LLM processing.

## Events

| Event | When | Payload |
|---|---|---|
| `ToolCallStarted` | Trigger matched, execution beginning | `text` = tool name |
| `ToolCallCompleted` | Execution finished | `text` = command output |

## Usage

```cpp
VoicePipeline pipeline(stt, tts, &llm, vad, config, on_event);

// Register tools
pipeline.tool_registry().add({
    "tell_time", "Tell the current time",
    {"what time", "current time"},
    "date '+%I:%M %p'", 5, 30
});

// Or load from JSON
pipeline.tool_registry().load_json(tools_json);

pipeline.start();
```
