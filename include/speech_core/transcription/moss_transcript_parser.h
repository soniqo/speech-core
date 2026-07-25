#pragma once

#include <string>
#include <vector>

#include "speech_core/interfaces.h"

namespace speech_core {

/// Parse MOSS's compact `[start][Sxx]text[end]` wire format.
///
/// Malformed or backwards segments are ignored. If no valid structured
/// segment exists, `plain_text` preserves the trimmed raw output so the caller
/// can apply source-specific fail-closed policy.
DiarizedTranscriptionResult parse_moss_transcript(const std::string& raw_text);

/// Consume control tokens accidentally repeated inside an otherwise valid
/// MOSS segment. Spoken words and punctuation are preserved.
std::string sanitize_moss_segment_text(const std::string& text);

/// True when text contains a MOSS timestamp or speaker control token.
bool contains_moss_wire_marker(const std::string& text);

}  // namespace speech_core
