#include "speech_core/protocol/realtime_protocol.h"

#include <cassert>
#include <cstdio>

using namespace speech_core;

void test_session_created() {
    auto json = RealtimeProtocol::session_created("sess-123");
    assert(json.find("session.created") != std::string::npos);
    assert(json.find("sess-123") != std::string::npos);
    assert(json.find("pcm16") != std::string::npos);
    printf("  PASS: session_created\n");
}

void test_transcription_completed() {
    auto json = RealtimeProtocol::transcription_completed(
        "item-1", "Hello world");
    assert(json.find("transcription.completed") != std::string::npos);
    assert(json.find("Hello world") != std::string::npos);
    assert(json.find("item-1") != std::string::npos);
    printf("  PASS: transcription_completed\n");
}

void test_error() {
    auto json = RealtimeProtocol::error("Something went wrong");
    assert(json.find("error") != std::string::npos);
    assert(json.find("Something went wrong") != std::string::npos);
    printf("  PASS: error\n");
}

void test_parse_audio_append() {
    std::string json = R"({"type":"input_audio_buffer.append","audio":"AQID"})";
    auto event = RealtimeProtocol::parse(json);
    assert(event.has_value());
    assert(event->type == EventType::InputAudioBufferAppend);
    assert(!event->audio_data.empty());
    printf("  PASS: parse_audio_append\n");
}

void test_parse_commit() {
    std::string json = R"({"type":"input_audio_buffer.commit"})";
    auto event = RealtimeProtocol::parse(json);
    assert(event.has_value());
    assert(event->type == EventType::InputAudioBufferCommit);
    printf("  PASS: parse_commit\n");
}

void test_parse_unknown() {
    std::string json = R"({"type":"unknown.event"})";
    auto event = RealtimeProtocol::parse(json);
    assert(!event.has_value());
    printf("  PASS: parse_unknown\n");
}

void test_parse_malformed() {
    auto event = RealtimeProtocol::parse("not json at all");
    assert(!event.has_value());
    printf("  PASS: parse_malformed\n");
}

void test_escape_json() {
    auto json = RealtimeProtocol::error("quote\"and\\backslash");
    assert(json.find("\\\"") != std::string::npos);
    assert(json.find("\\\\") != std::string::npos);
    printf("  PASS: escape_json\n");
}

int main() {
    printf("test_realtime_protocol:\n");
    test_session_created();
    test_transcription_completed();
    test_error();
    test_parse_audio_append();
    test_parse_commit();
    test_parse_unknown();
    test_parse_malformed();
    test_escape_json();
    printf("All realtime protocol tests passed.\n");
    return 0;
}
