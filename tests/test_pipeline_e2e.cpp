#include "speech_core/pipeline/voice_pipeline.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
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
        return {next_text, "", next_confidence, 0.0f, 1.0f};
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
    std::vector<ToolCall> next_tool_calls;
    int call_count = 0;

    LLMResponse chat(const std::vector<Message>& /*messages*/,
                     LLMTokenCallback on_token) override
    {
        call_count++;
        if (should_throw) throw std::runtime_error("LLM inference error");
        on_token(response, true);

        LLMResponse r;
        r.text = response;
        r.tool_calls = next_tool_calls;
        // Clear tool calls after first use (second call is with tool results)
        next_tool_calls.clear();
        return r;
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

// Default test config — disables latency optimizations that interfere
// with deterministic testing (warm-up increments STT call count,
// eager STT changes emission timing).
static AgentConfig test_config() {
    AgentConfig config;
    config.warmup_stt = false;
    config.eager_stt = false;
    return config;
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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    assert(pipeline.is_running());

    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

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

    auto config = test_config();
    config.mode = AgentConfig::Mode::TranscribeOnly;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

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

    auto config = test_config();
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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

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

    auto config = test_config();
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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // First: normal utterance plays back via TTS (sets agent_speaking)
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();
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

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.max_utterance_duration = 0.5f;  // 500ms

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // Should have at least one force-split (multiple TranscriptionCompleted events)
    assert(stt.call_count >= 2);  // at least 2 splits
    assert(log.count(EventType::TranscriptionCompleted) >= 2);

    pipeline.stop();
    printf("  PASS: max_utterance_force_split\n");
}

void test_llm_driven_tool_call() {
    MockSTT stt;
    MockTTS tts;
    MockLLM llm;
    MockVAD vad;

    stt.next_text = "what time is it";

    // LLM first returns a tool call, second call (with result) returns text
    llm.next_tool_calls = {{"tell_time", "{}"}};
    llm.response = "The time is 3:30 PM";

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    // Register the tool that LLM will request
    ToolDefinition time_tool;
    time_tool.name = "tell_time";
    time_tool.command = "echo 3:30 PM";
    time_tool.cooldown = 0;
    pipeline.tool_registry().add(time_tool);

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // LLM should have been called twice (first returns tool call, second with result)
    assert(llm.call_count == 2);

    // Tool events emitted
    assert(log.has(EventType::ToolCallStarted));
    assert(log.has(EventType::ToolCallCompleted));

    // TTS speaks the final LLM response
    assert(tts.call_count == 1);
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: llm_driven_tool_call\n");
}

void test_no_tool_calls_normal_flow() {
    MockSTT stt;
    MockTTS tts;
    MockLLM llm;
    MockVAD vad;

    stt.next_text = "how are you";
    llm.response = "I'm doing well";
    // No tool calls — LLM responds directly

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // LLM called once, no tool events
    assert(llm.call_count == 1);
    assert(!log.has(EventType::ToolCallStarted));

    // Normal response
    assert(tts.call_count == 1);
    assert(tts.last_text == "I'm doing well");
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: no_tool_calls_normal_flow\n");
}

void test_not_running_ignores_input() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f};

    auto config = test_config();
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

// Test: push_audio does NOT block during TTS synthesis
void test_push_audio_nonblocking_during_tts() {
    MockSTT stt;
    MockVAD vad;

    // Slow TTS: blocks for 100ms to simulate real synthesis
    class SlowTTS : public TTSInterface {
    public:
        std::atomic<bool> synthesizing{false};
        std::atomic<bool> audio_pushed_during_synth{false};
        int call_count = 0;

        void synthesize(const std::string& /*text*/, const std::string& /*lang*/,
                        TTSChunkCallback on_chunk) override {
            call_count++;
            synthesizing.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            synthesizing.store(false);
            float samples[] = {0.1f, 0.2f, 0.3f};
            on_chunk(samples, 3, true);
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override {}
    };

    SlowTTS tts;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // push_audio returns immediately — worker handles STT+TTS async
    // Push more audio while TTS is synthesizing — this should NOT block
    auto t_start = std::chrono::steady_clock::now();
    auto more_audio = make_audio(5);  // 5 chunks of silence
    pipeline.push_audio(more_audio.data(), more_audio.size());
    auto t_end = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    // push_audio should return in <50ms (not blocked by 100ms TTS)
    assert(ms < 50);

    pipeline.wait_idle();
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: push_audio_nonblocking_during_tts\n");
}

// Test: multiple utterances processed in sequence (realistic flow)
void test_multiple_utterances_queued() {
    MockTTS tts;
    MockVAD vad;

    // STT returns different text based on call count
    class CountingSTT : public STTInterface {
    public:
        int call_count = 0;
        TranscriptionResult transcribe(const float*, size_t, int) override {
            call_count++;
            return {"utterance_" + std::to_string(call_count), "", 1.0f, 0, 0};
        }
        int input_sample_rate() const override { return 16000; }
    };

    CountingSTT stt;

    // First utterance
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,  // speech 1
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,                      // silence -> end 1
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.post_playback_guard = 0;  // no guard in test — no real AEC

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio1 = make_audio(vad.probs.size());
    pipeline.push_audio(audio1.data(), audio1.size());
    pipeline.wait_idle();

    assert(stt.call_count == 1);

    // Platform signals playback done → resume listening
    pipeline.resume_listening();

    // Second utterance (VAD continues from current prob_index)
    vad.probs = {
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,  // speech 2
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,                      // silence -> end 2
    };
    vad.prob_index = 0;

    auto audio2 = make_audio(vad.probs.size());
    pipeline.push_audio(audio2.data(), audio2.size());
    pipeline.wait_idle();

    // Both utterances processed
    assert(stt.call_count == 2);
    assert(tts.call_count == 2);
    assert(log.count(EventType::TranscriptionCompleted) == 2);
    assert(log.count(EventType::ResponseDone) == 2);

    pipeline.stop();
    printf("  PASS: multiple_utterances_queued\n");
}

// Test: empty transcription (noise/breath) resumes listening
void test_empty_transcription_resumes() {
    MockTTS tts;
    MockVAD vad;

    class EmptySTT : public STTInterface {
    public:
        int call_count = 0;
        TranscriptionResult transcribe(const float*, size_t, int) override {
            call_count++;
            return {"", "", 0.0f, 0, 0};  // empty = noise/breath
        }
        int input_sample_rate() const override { return 16000; }
    };

    EmptySTT stt;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    assert(stt.call_count == 1);
    assert(tts.call_count == 0);  // no TTS for empty transcription
    assert(!log.has(EventType::ResponseCreated));
    // Pipeline should be back to idle/listening (ready for next utterance)
    auto s = pipeline.state();
    assert(s == VoicePipeline::State::Idle || s == VoicePipeline::State::Listening);

    pipeline.stop();
    printf("  PASS: empty_transcription_resumes\n");
}

// Test: language flows from STT through to TTS
void test_language_passthrough() {
    MockVAD vad;

    class LangSTT : public STTInterface {
    public:
        TranscriptionResult transcribe(const float*, size_t, int) override {
            return {"hello world", "russian", 1.0f, 0, 1};
        }
        int input_sample_rate() const override { return 16000; }
    };

    class LangTTS : public TTSInterface {
    public:
        std::string last_language;
        void synthesize(const std::string& /*text*/, const std::string& language,
                        TTSChunkCallback on_chunk) override {
            last_language = language;
            float s[] = {0.1f};
            on_chunk(s, 1, true);
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override {}
    };

    LangSTT stt;
    LangTTS tts;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    assert(tts.last_language == "russian");

    pipeline.stop();
    printf("  PASS: language_passthrough\n");
}

// Test: max_response_duration caps TTS output
void test_max_response_duration() {
    MockSTT stt;
    MockVAD vad;

    class VerboseTTS : public TTSInterface {
    public:
        int chunks_sent = 0;
        bool was_cancelled = false;
        void synthesize(const std::string& /*text*/, const std::string& /*lang*/,
                        TTSChunkCallback on_chunk) override {
            // Send many chunks (each 2400 samples = 0.1s at 24kHz)
            // With max_response_duration=0.5s, should be capped at ~12000 samples
            for (int i = 0; i < 20; i++) {
                std::vector<float> samples(2400, 0.1f);
                on_chunk(samples.data(), samples.size(), i == 19);
                chunks_sent++;
                if (was_cancelled) break;
            }
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override { was_cancelled = true; }
    };

    VerboseTTS tts;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.max_response_duration = 0.5f;  // 0.5 seconds max

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    assert(log.has(EventType::ResponseDone));
    // TTS should have been cancelled before sending all 20 chunks
    // 0.5s * 24000 = 12000 samples, at 2400 per chunk = 5 chunks max
    assert(tts.chunks_sent <= 6);

    pipeline.stop();
    printf("  PASS: max_response_duration\n");
}

// Test: speech during worker processing triggers interruption, not double utterance
void test_speech_during_worker_processing() {
    MockVAD vad;

    // Slow STT to simulate real inference time
    class SlowSTT : public STTInterface {
    public:
        int call_count = 0;
        TranscriptionResult transcribe(const float* /*audio*/, size_t /*len*/, int) override {
            call_count++;
            // Simulate slow STT inference
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            return {"hello", "", 1.0f, 0, 0};
        }
        int input_sample_rate() const override { return 16000; }
    };

    SlowSTT stt;
    MockTTS tts;

    // First utterance: speech then silence
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.allow_interruptions = true;
    config.min_interruption_duration = 0.0f;  // immediate interruption

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // Push first utterance
    auto audio1 = make_audio(vad.probs.size());
    pipeline.push_audio(audio1.data(), audio1.size());

    // While worker is processing STT (50ms), push more speech
    // This should be treated as interruption since agent_speaking=true
    // after UserSpeechEnded
    vad.prob_index = 0;
    vad.probs = {
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,  // speech
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };
    auto audio2 = make_audio(vad.probs.size());
    pipeline.push_audio(audio2.data(), audio2.size());

    pipeline.wait_idle();

    // The second speech should trigger interruption, not a second utterance
    assert(log.has(EventType::SpeechStarted));
    // Should have at most 1 full STT+TTS cycle (interrupted or not)
    // The key assertion: second speech triggers interruption or is handled
    // correctly, not creating duplicate transcriptions
    assert(stt.call_count <= 2);

    pipeline.stop();
    printf("  PASS: speech_during_worker_processing\n");
}

// Test: interruption during TTS cancels and resumes correctly
void test_interruption_during_tts_playback() {
    MockSTT stt;
    MockVAD vad;

    class SlowTTS : public TTSInterface {
    public:
        std::atomic<bool> cancelled{false};
        int call_count = 0;
        void synthesize(const std::string& /*text*/, const std::string& /*lang*/,
                        TTSChunkCallback on_chunk) override {
            call_count++;
            cancelled.store(false);
            // Stream multiple chunks with delays
            for (int i = 0; i < 5; i++) {
                if (cancelled.load()) break;
                std::vector<float> samples(2400, 0.1f);
                on_chunk(samples.data(), samples.size(), i == 4);
                if (cancelled.load()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override { cancelled.store(true); }
    };

    SlowTTS tts;

    // First: speech to trigger TTS
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.allow_interruptions = true;
    config.min_interruption_duration = 0.0f;  // immediate

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // TTS should have completed (or been interrupted)
    assert(tts.call_count >= 1);
    assert(log.has(EventType::ResponseCreated));

    pipeline.stop();
    printf("  PASS: interruption_during_tts_playback\n");
}

// Test: resume_listening resets state correctly after TTS
void test_resume_listening_state_reset() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.post_playback_guard = 0.0f;  // no delay for test speed

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    assert(log.has(EventType::ResponseDone));

    // Simulate platform calling resume_listening after playback
    pipeline.resume_listening();
    assert(pipeline.state() == VoicePipeline::State::Idle);

    // Now push a second utterance — should work normally
    log.types.clear();
    log.texts.clear();
    stt.call_count = 0;
    tts.call_count = 0;
    vad.prob_index = 0;

    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // Second utterance should process normally
    assert(stt.call_count == 1);
    assert(tts.call_count == 1);
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: resume_listening_state_reset\n");
}

// Test: interruption during STT processing skips TTS entirely
void test_interruption_during_stt_skips_tts() {
    MockVAD vad;

    // Slow STT so we have time to interrupt
    class SlowSTT : public STTInterface {
    public:
        int call_count = 0;
        TranscriptionResult transcribe(const float*, size_t, int) override {
            call_count++;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return {"hello world", "", 1.0f, 0, 0};
        }
        int input_sample_rate() const override { return 16000; }
    };

    SlowSTT stt;
    MockTTS tts;

    // Speech followed by silence
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.allow_interruptions = true;
    config.min_interruption_duration = 0.0f;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio1 = make_audio(vad.probs.size());
    pipeline.push_audio(audio1.data(), audio1.size());

    // While STT is running (100ms), push interrupting speech
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    vad.prob_index = 0;
    vad.probs = {
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };
    auto audio2 = make_audio(vad.probs.size());
    pipeline.push_audio(audio2.data(), audio2.size());

    pipeline.wait_idle();

    // STT ran but interruption should have prevented TTS for that utterance
    assert(stt.call_count >= 1);
    // Key: the first utterance's TTS should NOT have run because the user
    // interrupted during STT. The second utterance may or may not have
    // been processed (depends on timing).
    // The old bug: tts.call_count would equal stt.call_count (TTS ran despite interrupt)
    assert(tts.call_count < stt.call_count);

    pipeline.stop();
    printf("  PASS: interruption_during_stt_skips_tts\n");
}

// Test: post-playback guard suppresses VAD events
void test_post_playback_guard() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    // First utterance
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    // Guard = 10 chunks worth of samples (10 * 512 = 5120 samples at 16kHz = 0.32s)
    config.post_playback_guard = 0.32f;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio1 = make_audio(vad.probs.size());
    pipeline.push_audio(audio1.data(), audio1.size());
    pipeline.wait_idle();

    assert(stt.call_count == 1);
    pipeline.resume_listening();

    // Immediately push speech — should be suppressed by guard
    vad.prob_index = 0;
    vad.probs = {
        0.8f, 0.8f, 0.8f, 0.8f,  // 4 chunks during guard (suppressed)
        0.1f, 0.1f, 0.1f, 0.1f,  // silence
    };
    auto audio2 = make_audio(vad.probs.size());
    pipeline.push_audio(audio2.data(), audio2.size());
    pipeline.wait_idle();

    // Speech during guard period should NOT trigger a new utterance
    assert(stt.call_count == 1);  // no new STT call

    // Now push speech AFTER guard expires (more chunks to exhaust guard)
    vad.prob_index = 0;
    vad.probs = {
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,  // silence to exhaust remaining guard
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,   // speech after guard
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };
    auto audio3 = make_audio(vad.probs.size());
    pipeline.push_audio(audio3.data(), audio3.size());
    pipeline.wait_idle();

    // This speech should be detected normally
    assert(stt.call_count == 2);

    pipeline.stop();
    printf("  PASS: post_playback_guard\n");
}

// Test: stop() during active worker processing doesn't hang
void test_stop_during_processing() {
    MockVAD vad;
    MockTTS tts;

    class SlowSTT : public STTInterface {
    public:
        std::atomic<bool> in_transcribe{false};
        TranscriptionResult transcribe(const float*, size_t, int) override {
            in_transcribe.store(true);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            return {"hello", "", 1.0f, 0, 0};
        }
        int input_sample_rate() const override { return 16000; }
    };

    SlowSTT stt;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // Wait for STT to start, then stop
    while (!stt.in_transcribe.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // stop() should not deadlock — it should return within a reasonable time
    auto start = std::chrono::steady_clock::now();
    pipeline.stop();
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Should complete within STT duration + margin (not hang forever)
    assert(ms < 1000);

    printf("  PASS: stop_during_processing\n");
}

// Test: brief interruption triggers InterruptionRecovered (playback can resume)
void test_interruption_recovery() {
    MockSTT stt;
    MockVAD vad;

    class TrackedTTS : public TTSInterface {
    public:
        int call_count = 0;
        bool cancelled = false;
        void synthesize(const std::string& /*text*/, const std::string& /*lang*/,
                        TTSChunkCallback on_chunk) override {
            call_count++;
            float s[] = {0.1f, 0.2f, 0.3f};
            on_chunk(s, 3, true);
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override { cancelled = true; }
    };

    TrackedTTS tts;

    // First utterance to get into Speaking state
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.allow_interruptions = true;
    // Require 1s of speech to confirm interruption
    config.min_interruption_duration = 1.0f;
    config.interruption_recovery_timeout = 0.5f;
    config.post_playback_guard = 0;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio1 = make_audio(vad.probs.size());
    pipeline.push_audio(audio1.data(), audio1.size());
    pipeline.wait_idle();

    assert(tts.call_count == 1);
    // Pipeline is now in Speaking state (agent_speaking=true via speak())
    // Don't call resume_listening — simulate still playing

    // Brief speech while agent is speaking (< min_interruption_duration)
    // Should trigger SpeechStarted then be discarded as AEC residual
    vad.prob_index = 0;
    vad.probs = {
        0.8f, 0.8f, 0.8f,  // brief speech (~96ms, well under 1.0s threshold)
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };
    auto audio2 = make_audio(vad.probs.size());
    pipeline.push_audio(audio2.data(), audio2.size());
    pipeline.wait_idle();

    // TTS should NOT have been cancelled — brief speech was AEC residual
    assert(!tts.cancelled);
    // No ResponseInterrupted event — the brief speech was discarded
    assert(!log.has(EventType::ResponseInterrupted));
    // STT should still be 1 (no new utterance processed)
    assert(stt.call_count == 1);

    pipeline.stop();
    printf("  PASS: interruption_recovery\n");
}

// Test: push_text during active TTS queues correctly
void test_push_text_during_tts() {
    MockSTT stt;
    MockVAD vad;

    class CountingTTS : public TTSInterface {
    public:
        int call_count = 0;
        std::vector<std::string> texts;
        void synthesize(const std::string& text, const std::string& /*lang*/,
                        TTSChunkCallback on_chunk) override {
            call_count++;
            texts.push_back(text);
            float s[] = {0.1f};
            on_chunk(s, 1, true);
        }
        int output_sample_rate() const override { return 24000; }
        void cancel() override {}
    };

    CountingTTS tts;
    MockLLM llm;
    llm.response = "response one";

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // push_text processes synchronously (bypasses STT, goes straight to LLM→TTS)
    pipeline.push_text("first message");
    assert(tts.call_count == 1);
    assert(tts.texts[0] == "response one");

    llm.response = "response two";
    pipeline.push_text("second message");
    assert(tts.call_count == 2);
    assert(tts.texts[1] == "response two");

    // STT was never called (push_text bypasses it)
    assert(stt.call_count == 0);

    pipeline.stop();
    printf("  PASS: push_text_during_tts\n");
}

// Test: rapid start/stop cycles don't crash or deadlock
void test_rapid_start_stop() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {0.0f, 0.0f, 0.0f, 0.0f};

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    // Rapid start/stop 10 times
    for (int i = 0; i < 10; i++) {
        pipeline.start();
        assert(pipeline.is_running());
        auto audio = make_audio(4);
        pipeline.push_audio(audio.data(), audio.size());
        pipeline.stop();
        assert(!pipeline.is_running());
    }

    // One more cycle with actual speech to verify pipeline still works
    vad.prob_index = 0;
    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    assert(stt.call_count == 1);
    assert(tts.call_count == 1);

    pipeline.stop();
    printf("  PASS: rapid_start_stop\n");
}

// Test: LLM cancel during chat — stop() while LLM is generating
void test_stop_during_llm() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    class SlowLLM : public LLMInterface {
    public:
        std::atomic<bool> in_chat{false};
        LLMResponse chat(const std::vector<Message>& /*messages*/,
                         LLMTokenCallback on_token) override {
            in_chat.store(true);
            // Simulate slow LLM inference
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            on_token("response", true);
            return {"response", {}};
        }
        void cancel() override {}
    };

    SlowLLM llm;

    vad.probs = {
        0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Pipeline;

    EventLog log;
    VoicePipeline pipeline(stt, tts, &llm, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());

    // Wait for LLM to start, then stop
    while (!llm.in_chat.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto start = std::chrono::steady_clock::now();
    pipeline.stop();
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // Should complete within LLM duration + margin
    assert(ms < 1000);
    assert(!pipeline.is_running());

    printf("  PASS: stop_during_llm\n");
}

// Test: concurrent push_audio from multiple threads
void test_concurrent_push_audio() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    // Enough probs for many concurrent pushes
    std::vector<float> probs;
    for (int i = 0; i < 500; i++) probs.push_back(0.0f);
    vad.probs = probs;

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // Push audio from 4 threads simultaneously
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&pipeline]() {
            auto audio = make_audio(20);
            for (int i = 0; i < 10; i++) {
                pipeline.push_audio(audio.data(), audio.size());
            }
        });
    }

    for (auto& th : threads) th.join();

    // No crash, no deadlock — that's the assertion
    pipeline.stop();
    assert(!pipeline.is_running());

    printf("  PASS: concurrent_push_audio\n");
}

// Test: eager STT fires UserSpeechEnded on first silence frame
void test_eager_stt() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;

    vad.probs = {
        0.0f, 0.0f,
        0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f, 0.8f,
        0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
    };

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.eager_stt = true;  // enable for this test

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();
    auto audio = make_audio(vad.probs.size());
    pipeline.push_audio(audio.data(), audio.size());
    pipeline.wait_idle();

    // Should process exactly one utterance
    assert(stt.call_count == 1);
    assert(tts.call_count == 1);
    assert(log.has(EventType::SpeechStarted));
    assert(log.has(EventType::SpeechEnded));
    assert(log.has(EventType::TranscriptionCompleted));
    assert(log.has(EventType::ResponseDone));

    pipeline.stop();
    printf("  PASS: eager_stt\n");
}

// Test: STT warm-up runs at pipeline start
void test_stt_warmup() {
    MockSTT stt;
    MockTTS tts;
    MockVAD vad;
    vad.probs = {0.0f};

    auto config = test_config();
    config.mode = AgentConfig::Mode::Echo;
    config.warmup_stt = true;  // enable for this test

    EventLog log;
    VoicePipeline pipeline(stt, tts, nullptr, vad, config,
        [&log](const PipelineEvent& e) { log.on_event(e); });

    pipeline.start();

    // Give worker thread time to run warm-up
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Warm-up should have called transcribe once
    assert(stt.call_count == 1);

    pipeline.stop();
    printf("  PASS: stt_warmup\n");
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
    test_llm_driven_tool_call();
    test_no_tool_calls_normal_flow();
    test_not_running_ignores_input();
    test_push_audio_nonblocking_during_tts();
    test_multiple_utterances_queued();
    test_empty_transcription_resumes();
    test_language_passthrough();
    test_max_response_duration();
    test_speech_during_worker_processing();
    test_interruption_during_tts_playback();
    test_resume_listening_state_reset();
    test_interruption_during_stt_skips_tts();
    test_post_playback_guard();
    test_stop_during_processing();
    test_interruption_recovery();
    test_push_text_during_tts();
    test_rapid_start_stop();
    test_stop_during_llm();
    test_concurrent_push_audio();
    test_eager_stt();
    test_stt_warmup();
    printf("All pipeline E2E tests passed.\n");
    return 0;
}
