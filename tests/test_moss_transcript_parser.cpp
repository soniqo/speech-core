#include "speech_core/transcription/moss_transcript_parser.h"

#include <cassert>
#include <iostream>

using speech_core::contains_moss_wire_marker;
using speech_core::parse_moss_transcript;
using speech_core::sanitize_moss_segment_text;

int main() {
    {
        const auto result = parse_moss_transcript(
            "[0.93][S01] Hello there.[2.10]"
            "[1.75][S02] General Kenobi.[3.25]");
        assert(result.text == "Hello there. General Kenobi.");
        assert(result.segments.size() == 2);
        assert(result.segments[0].speaker == "S01");
        assert(result.segments[0].start_time == 0.93f);
        assert(result.segments[0].end_time == 2.10f);
        assert(result.segments[1].speaker == "S02");
    }
    {
        const std::string malformed = "[2.0][S01] backwards[1.0]";
        const auto result = parse_moss_transcript(malformed);
        assert(result.segments.empty());
        assert(result.text == malformed);
    }
    {
        const auto result = parse_moss_transcript(
            "[0][S01]   [1]"
            "[4][S02] backwards[3]"
            "[5][S03] valid[6]");
        assert(result.segments.size() == 1);
        assert(result.segments[0].speaker == "S03");
        assert(result.segments[0].text == "valid");
    }
    {
        const auto text = sanitize_moss_segment_text(
            "[S01] When we were banned,[0.63][S01] it left a vacuum.");
        assert(text == "When we were banned, it left a vacuum.");
        assert(contains_moss_wire_marker("[0.63]"));
        assert(contains_moss_wire_marker("[S01]"));
        assert(!contains_moss_wire_marker("[important]"));
    }
    {
        const auto result = parse_moss_transcript(
            "noise [0.0][S01] Привет, как дела?[1.0] tail");
        assert(result.segments.size() == 1);
        assert(result.text == "Привет, как дела?");
    }

    std::cout << "MOSS transcript parser tests passed\n";
    return 0;
}
