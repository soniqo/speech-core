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
3. If the LLM returns `ToolCall` requests, the pipeline executes the matching tools
4. Tool results are injected as `Tool` messages in the conversation
5. The LLM is called again with the updated conversation
6. The final text response is spoken via TTS

## Tool Definition

Tools can use either a **callback handler** (for platform consumers like Swift/Kotlin) or a **shell command** (for CLI usage). The handler takes priority when both are set.

### Callback handler (C++)

```cpp
ToolDefinition tool;
tool.name = "tell_time";
tool.description = "Tell the current time";
tool.cooldown = 0;
tool.handler = [](const std::string& name, const std::string& args) -> std::string {
    return "3:14 PM";
};
pipeline.tool_registry().add(tool);
```

### Shell command

```cpp
ToolDefinition tool;
tool.name = "tell_time";
tool.description = "Tell the current time";
tool.triggers = {};  // optional, for standalone IntentMatcher use
tool.command = "date '+%I:%M %p'";
tool.timeout = 5;   // seconds (advisory)
tool.cooldown = 30;  // seconds between invocations
```

### C API

```c
// Callback-based tool
const char* my_handler(const char* name, const char* args, void* ctx) {
    return "3:14 PM";
}

const char* triggers[] = {"what time", NULL};
sc_tool_definition_t tool = {
    .name = "tell_time",
    .description = "Tell the current time",
    .triggers = triggers,
    .handler = my_handler,
    .handler_context = NULL,
    .timeout = 5,
    .cooldown = 30
};
sc_pipeline_add_tool(pipeline, tool);
```

### JSON format

Tools can be loaded from JSON (shell-command tools only):

```json
[
  {
    "name": "tell_time",
    "description": "Tell the current time",
    "triggers": [],
    "command": "date '+%I:%M %p'",
    "timeout": 5,
    "cooldown": 30
  }
]
```

```cpp
// C++ API
pipeline.tool_registry().load_json(json_string);

// C API
sc_pipeline_load_tools_json(pipeline, json_string);
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

## History trimming

When `mask_tool_results` is enabled (default), tool result messages are dropped before conversation messages during history trimming. Tool outputs are self-contained — the LLM already acted on them, so the raw output can be dropped while keeping user/assistant turns. The last 2 messages are always protected to avoid dropping the current turn's tool result before the LLM sees it.

## IntentMatcher (standalone)

`IntentMatcher` is available as a standalone component for regex-based pattern matching — useful outside the pipeline for quick intent classification. The pipeline itself does **not** use it; the LLM decides tool calls.

## Events

| Event | When | Payload |
|---|---|---|
| `ToolCallStarted` | LLM requested a tool call | `text` = tool name |
| `ToolCallCompleted` | Tool execution finished | `text` = tool output |

## Usage

```cpp
VoicePipeline pipeline(stt, tts, &llm, vad, config, on_event);

// Callback-based tool
ToolDefinition tool;
tool.name = "tell_time";
tool.description = "Tell the current time";
tool.cooldown = 0;
tool.handler = [](const std::string& name, const std::string& args) {
    return "3:14 PM";
};
pipeline.tool_registry().add(tool);

pipeline.start();
```
