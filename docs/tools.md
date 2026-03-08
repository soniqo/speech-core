# Tool Calling

The tool engine adds function-calling capabilities to the voice pipeline. The **LLM decides** when to call tools — the pipeline executes them and feeds results back.

## Flow

```
transcript → LLM (with tool definitions)
                │
                ├─ text response → TTS → audio
                │
                └─ tool_call(name, args) → ToolExecutor → result
                                                            │
                                                    inject into context
                                                            │
                                                    LLM (with result) → TTS → audio
```

1. Tool definitions are passed to the LLM via `set_tools()`
2. The LLM receives the conversation and decides whether to call a tool
3. If the LLM returns `ToolCall` requests, the pipeline executes the matching commands
4. Tool results are injected as `Tool` messages in the conversation
5. The LLM is called again with the updated conversation
6. The final text response is spoken via TTS

## Tool Definition

```cpp
ToolDefinition tool;
tool.name = "tell_time";
tool.description = "Tell the current time";
tool.triggers = {};  // optional, for standalone IntentMatcher use
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
    "triggers": [],
    "command": "date '+%I:%M %p'",
    "timeout": 5,
    "cooldown": 30
  },
  {
    "name": "weather",
    "description": "Current weather conditions",
    "triggers": [],
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

## LLM Interface

The `LLMInterface::chat()` returns an `LLMResponse` containing text and/or tool calls:

```cpp
struct ToolCall {
    std::string name;       // tool name
    std::string arguments;  // JSON arguments
};

struct LLMResponse {
    std::string text;
    std::vector<ToolCall> tool_calls;
};
```

The LLM implementation decides when to call tools based on the conversation and tool definitions provided via `set_tools()`.

## Cooldown

Each tool has a `cooldown` period (in seconds). After execution, the same tool cannot be triggered again until the cooldown expires. Set `cooldown = 0` to disable.

## IntentMatcher (standalone)

`IntentMatcher` is available as a standalone component for regex-based pattern matching — useful outside the pipeline for quick intent classification. The pipeline itself does **not** use it; the LLM decides tool calls.

## Events

| Event | When | Payload |
|---|---|---|
| `ToolCallStarted` | LLM requested a tool call | `text` = tool name |
| `ToolCallCompleted` | Tool execution finished | `text` = command output |

## Usage

```cpp
VoicePipeline pipeline(stt, tts, &llm, vad, config, on_event);

// Register tools
pipeline.tool_registry().add({
    "tell_time", "Tell the current time",
    {}, "date '+%I:%M %p'", 5, 30
});

pipeline.start();
```
