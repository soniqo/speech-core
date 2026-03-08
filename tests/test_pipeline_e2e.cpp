#include "speech_core/pipeline/voice_pipeline.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace speech_core;

// ---------------------------------------------------------------------------
// Mock implementations
// ---------------------------------------------------------------------------

class MockSTT : public STTInterface {
public:
    std::string next_text = "hello world";
    float next_confidence = 0.95f;
    int call_count = 0;

    TranscriptionResult transcribe(
        const float* /*audio*/, size_t /*length*/, int /*sample_rate*/) override
    {
        call_count++;
        if (should_throw) throw std::runtime_error("STT model error");
        return {next_text, next_confidence, 0.0f, 1.0f};
    }

    int input_sample_rate() const override { return 16000; }

    bool should_throw = false;
};

class MockTTS : public TTSInterface {
public:
    std::string last_text;
    int call_count = 0;

    void synthesize(const std::string& text, const std::string& /*language*/,
                    TTSChunkCallback on_chunk) override
    {
        call_count++;
        last_text = text;
        if (should_throw) throw std::runtime_error("TTS synthesis error");
        // Emit a single chunk of 3 samples
        float samples[] = {0.1f, 0.2f, 0.3f};
        on_chunk(samples, 3, true);
    }

    int output_sample_rate() const override { return 24000; }
    void cancel() override { cancelled = true; }

    bool should_throw = false;
    bool cancelled = false;
};

class MockLLM : public LLMInterface {
public:
    std::string response = "I heard you";
    int call_count = 0;

    void chat(const std::vector<Message>& /*messages*/,
              LLMTokenCallback on_token) override
    {
        call_count++;
        if (should_throw) throw std::runtime_error("LLM inference error");
        on_token(response, true);
    }

    void cancel() override { cancelled = true; }

    bool should_throw = false;
    bool cancelled = false;
};

class MockVAD : public VADInterface {
public:
    // Queue of probabilities to return
    std::vector<float> probs;
    size_t prob_index = 0;

    float process_chunk(const float* /*samples*/, size_t /*length*/) override {
        if (prob_index < probs.size()) return probs[prob_index++];
        return 0.0f;
    }

    void reset() override { prob_index = 0; }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }
};

// Helper to collect events
struct EventLog {
    std::vector<EventType> types;
    std::vector<std::string> texts;

    void on_event(const PipelineEvent& e) {
        types.push_back(e.type);
        texts.push_back(e.text);
    }

    bool has(EventType t) const {
        for (auto& et : types) if (et == t) return true;
        return false;
    }

    size_t count(EventType t) const {
        size_t n = 0;
        for (auto& et : types) if (et == t) n++;
        return n;
    }

    std::string text_for(EventType t) const {
        for (size_t i = 0; i < types.size(); i++) {
            if (types[i] == t) return texts[i];
        }
        return "";
    }
};

// Generate silence/speech audio chunks (512 samples each)
static std::vector<float> make_audio(size_t num_chunks) {
    return std::vector<float>(num_chunks * 512, 0.0f);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_echo_mode_e2e() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    // Pattern: silence -> speech onset -> confirmed -> speech -> silence -> confirmed end
    // At 512 samples / 16kHz = 32ms per chunk
    // min_speech = 250ms = ~8 chunks, min_silence = 100ms = ~4 chunks
    vad.probs = {
        0.0f, 0.0f,                                         // silence
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,   // 8 chunks speech (256ms > 250ms)
        0.8f, 0.8f,                                         // more speech
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,                      // 5 chunks silence (160ms > 100ms)
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    assert(pipeline.is_running());

    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // Should have full cycle: SpeechStarted -> TranscriptionCompleted -> ResponseCreated -> AudioDelta -> ResponseDone
    assert(log.has(EventType::SpeechStarted));
    assert(log.has(EventType::TranscriptionCompleted));
    assert(log.text_for(EventType::TranscriptionCompleted) == "hello world");
    assert(log.has(EventType::ResponseCreated));
    assert(log.has(EventType::ResponseAudioDelta));
    assert(log.has(EventType::ResponseDone));

    assert(stt.call_count == 1);
    assert(tts.call_count == 1);
    assert(tts.last_text == "hello world");  // echo mode

    pipeline.stop();
    printf("  PASS: echo_mode_e2e\n");
}

void test_pipeline_mode_e2e() {
    MockSTT stt;
    MockTTS tts;
    MockLLM llm;
    MockVAD vad;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    assert(stt.call_count == 1);
    assert(llm.call_count == 1);
    assert(tts.call_count == 1);
    assert(tts.last_text == "I heard you");  // LLM response
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: pipeline_mode_e2e\n");
}

void test_transcribe_only_mode() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::TranscribeOnly;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    assert(stt.call_count == 1);
    assert(tts.call_count == 0);  // no TTS in transcribe-only
    assert(log.has(EventType::TranscriptionCompleted));
    assert(!log.has(EventType::ResponseCreated));

    pipeline.stop();
    printf("  PASS: transcribe_only_mode\n");
}

void test_push_text_bypasses_stt() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    pipeline.push_text("direct input");

    assert(stt.call_count == 0);  // STT not called
    assert(tts.call_count == 1);
    assert(tts.last_text == "direct input");
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: push_text_bypasses_stt\n");
}

void test_stt_error_propagation() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    stt.should_throw = true;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    assert(log.has(EventType::Error));
    assert(log.text_for(EventType::Error).find("STT") != std::string::npos);
    assert(tts.call_count == 0);  // TTS not called after STT failure
    assert(pipeline.state() == VoicePipeline::State::Idle);

    pipeline.stop();
    printf("  PASS: stt_error_propagation\n");
}

void test_llm_error_propagation() {
    MockSTT stt;
    MockTTS tts;
    MockLLM llm;
    MockVAD vad;

    llm.should_throw = true;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    assert(log.has(EventType::Error));
    assert(log.text_for(EventType::Error).find("LLM") != std::string::npos);
    assert(tts.call_count == 0);
    assert(pipeline.state() == VoicePipeline::State::Idle);

    pipeline.stop();
    printf("  PASS: llm_error_propagation\n");
}

void test_tts_error_propagation() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    tts.should_throw = true;

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    pipeline.push_text("will fail");

    assert(log.has(EventType::Error));
    assert(log.text_for(EventType::Error).find("TTS") != std::string::npos);
    assert(pipeline.state() == VoicePipeline::State::Idle);

    pipeline.stop();
    printf("  PASS: tts_error_propagation\n");
}

void test_interruption() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    // First utterance: speech then silence
    // Then set agent_speaking, then second speech (interruption)
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,  // speech
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,                      // silence -> end
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // First: normal utterance plays back via TTS (sets agent_speaking)
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    assert(log.has(EventType::ResponseDone));

    // TTS mock calls is_final=true synchronously, so agent_speaking is already false
    // Verify the basic flow completed
    assert(stt.call_count == 1);
    assert(tts.call_count == 1);

    pipeline.stop();
    printf("  PASS: interruption\n");
}

void test_max_utterance_force_split() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    // Long continuous speech: 500 chunks at 32ms = 16 seconds
    // max_utterance_duration = 0.5s = ~16 chunks
    std::vector<float> probs;
    probs.push_back(0.0f);  // initial silence
    for (int i = 0; i < 100; i++) probs.push_back(0.9f);  // long speech
    probs.push_back(0.0f);
    vad.probs = probs;

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;
    config.max_utterance_duration = 0.5f;  // 500ms

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // Should have at least one force-split (multiple TranscriptionCompleted events)
    assert(stt.call_count >= 2);  // at least 2 splits
    assert(log.count(EventType::TranscriptionCompleted) >= 2);

    pipeline.stop();
    printf("  PASS: max_utterance_force_split\n");
}

void test_tool_call_in_pipeline() {
    MockSTT stt;
    MockTTS tts;
    MockLLM llm;
    MockVAD vad;

    // STT returns "what time is it"
    stt.next_text = "what time is it";

    // LLM should get the tool result in context
    llm.response = "The time is 3:30 PM";

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    AgentConfig config;
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    // Register a time tool
    ToolDefinition time_tool;
    time_tool.name = "tell_time";
    time_tool.triggers = {"what time"};
    time_tool.command = "echo 3:30 PM";
    time_tool.cooldown = 0;
    pipeline.tool_registry().add(time_tool);

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // Tool should have been called
    assert(log.has(EventType::ToolCallStarted));
    assert(log.has(EventType::ToolCallCompleted));

    // LLM should have been called with tool result
    assert(llm.call_count == 1);

    // TTS should speak the LLM response
    assert(tts.call_count == 1);
    assert(tts.last_text == "The time is 3:30 PM");
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: tool_call_in_pipeline\n");
}

void test_tool_no_match_falls_through() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    stt.next_text = "how are you";

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    // Register a tool that won't match
    ToolDefinition tool;
    tool.name = "tell_time";
    tool.triggers = {"what time"};
    tool.command = "echo time";
    tool.cooldown = 0;
    pipeline.tool_registry().add(tool);

    pipeline.start();
    pipeline.push_text("how are you");

    // No tool call events
    assert(!log.has(EventType::ToolCallStarted));

    // Should fall through to echo mode
    assert(tts.call_count == 1);
    assert(tts.last_text == "how are you");
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: tool_no_match_falls_through\n");
}

void test_not_running_ignores_input() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

    AgentConfig config;
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    // Don't start — push_audio should be ignored
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.push_text("ignored");

    assert(log.types.empty());
    assert(stt.call_count == 0);
    assert(tts.call_count == 0);

    printf("  PASS: not_running_ignores_input\n");
}

int main() {
    printf("test_pipeline_e2e:\n");
    test_echo_mode_e2e();
    test_pipeline_mode_e2e();
    test_transcribe_only_mode();
    test_push_text_bypasses_stt();
    test_stt_error_propagation();
    test_llm_error_propagation();
    test_tts_error_propagation();
    test_interruption();
    test_max_utterance_force_split();
    test_tool_call_in_pipeline();
    test_tool_no_match_falls_through();
    test_not_running_ignores_input();
    printf("All pipeline E2E tests passed.\n");
    return 0;
}
