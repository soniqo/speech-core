#include "speech_core/vad/streaming_vad.h"

#include <cassert>
#include <cstdio>

using namespace speech_core;

void test_silence_only() {
    StreamingVAD vad(VADConfig::silero_default(), 0.032f);
    for (int i = 0; i < 100; i++) {
        auto events = vad.process(0.1f);
        assert(events.empty());
    }
    auto final_events = vad.flush();
    assert(final_events.empty());
    printf("  PASS: silence_only\n");
}

void test_clear_speech() {
    StreamingVAD vad(VADConfig::silero_default(), 0.032f);

    // 10 chunks of silence
    for (int i = 0; i < 10; i++) vad.process(0.1f);

    // Speech onset — need enough chunks to exceed min_speech_duration (0.25s)
    bool got_started = false;
    for (int i = 0; i < 15; i++) {
        auto events = vad.process(0.9f);
        for (auto& e : events) {
            if (e.type == VADEvent::SpeechStarted) got_started = true;
        }
    }
    assert(got_started);

    // Speech offset — need enough silence for min_silence_duration (0.1s)
    bool got_ended = false;
    for (int i = 0; i < 10; i++) {
        auto events = vad.process(0.1f);
        for (auto& e : events) {
            if (e.type == VADEvent::SpeechEnded) got_ended = true;
        }
    }
    assert(got_ended);
    printf("  PASS: clear_speech\n");
}

void test_false_alarm() {
    StreamingVAD vad(VADConfig::silero_default(), 0.032f);

    // Very brief onset (1 chunk) then drop — should NOT trigger speech
    vad.process(0.9f);
    auto events = vad.process(0.1f);
    // Should return to silence without emitting SpeechStarted
    for (auto& e : events) {
        assert(e.type != VADEvent::SpeechStarted);
    }
    printf("  PASS: false_alarm\n");
}

void test_flush_active_speech() {
    StreamingVAD vad(VADConfig::silero_default(), 0.032f);

    // Start confirmed speech
    for (int i = 0; i < 15; i++) vad.process(0.9f);

    // Flush while speaking
    auto events = vad.flush();
    bool got_ended = false;
    for (auto& e : events) {
        if (e.type == VADEvent::SpeechEnded) got_ended = true;
    }
    assert(got_ended);
    printf("  PASS: flush_active_speech\n");
}

void test_reset() {
    StreamingVAD vad(VADConfig::silero_default(), 0.032f);
    for (int i = 0; i < 15; i++) vad.process(0.9f);
    vad.reset();
    assert(vad.current_time() == 0.0f);
    printf("  PASS: reset\n");
}

int main() {
    printf("test_streaming_vad:\n");
    test_silence_only();
    test_clear_speech();
    test_false_alarm();
    test_flush_active_speech();
    test_reset();
    printf("All streaming VAD tests passed.\n");
    return 0;
}
