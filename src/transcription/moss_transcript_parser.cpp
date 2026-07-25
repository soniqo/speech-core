#include "speech_core/transcription/moss_transcript_parser.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace speech_core {
namespace {

bool ascii_space(char value) {
    return value == ' ' || value == '\t' || value == '\n' ||
           value == '\r' || value == '\f' || value == '\v';
}

std::string trim_ascii(const std::string& text) {
    std::size_t first = 0;
    while (first < text.size() && ascii_space(text[first])) ++first;
    std::size_t last = text.size();
    while (last > first && ascii_space(text[last - 1])) --last;
    return text.substr(first, last - first);
}

bool parse_timestamp_token(
    const std::string& text, std::size_t open, std::size_t& after,
    double& value) {
    if (open >= text.size() || text[open] != '[') return false;
    const std::size_t close = text.find(']', open + 1);
    if (close == std::string::npos || close == open + 1) return false;
    bool saw_digit = false;
    bool saw_dot = false;
    for (std::size_t index = open + 1; index < close; ++index) {
        const char character = text[index];
        if (character >= '0' && character <= '9') {
            saw_digit = true;
        } else if (character == '.' && !saw_dot) {
            saw_dot = true;
        } else {
            return false;
        }
    }
    if (!saw_digit) return false;
    const std::string token = text.substr(open + 1, close - open - 1);
    char* parsed_end = nullptr;
    value = std::strtod(token.c_str(), &parsed_end);
    if (!parsed_end || *parsed_end != '\0' ||
        !std::isfinite(value)) {
        return false;
    }
    after = close + 1;
    return true;
}

bool parse_speaker_token(
    const std::string& text, std::size_t open, std::size_t& after,
    std::string& speaker) {
    if (open >= text.size() || text[open] != '[') return false;
    const std::size_t close = text.find(']', open + 1);
    if (close == std::string::npos || close <= open + 2 ||
        text[open + 1] != 'S') {
        return false;
    }
    for (std::size_t index = open + 2; index < close; ++index) {
        if (text[index] < '0' || text[index] > '9') return false;
    }
    speaker = text.substr(open + 1, close - open - 1);
    after = close + 1;
    return true;
}

bool marker_at(
    const std::string& text, std::size_t open, std::size_t& after) {
    double timestamp = 0.0;
    std::string speaker;
    return parse_timestamp_token(text, open, after, timestamp) ||
           parse_speaker_token(text, open, after, speaker);
}

std::string collapse_ascii_whitespace(const std::string& text) {
    std::ostringstream result;
    bool needs_space = false;
    for (char character : text) {
        if (ascii_space(character)) {
            needs_space = result.tellp() > 0;
            continue;
        }
        if (needs_space) {
            result << ' ';
            needs_space = false;
        }
        result << character;
    }
    return result.str();
}

bool has_ascii_or_utf8_lexical_content(const std::string& text) {
    for (unsigned char character : text) {
        if (character >= 0x80 || std::isalnum(character)) return true;
    }
    return false;
}

}  // namespace

bool contains_moss_wire_marker(const std::string& text) {
    std::size_t position = 0;
    while ((position = text.find('[', position)) != std::string::npos) {
        std::size_t after = position;
        if (marker_at(text, position, after)) return true;
        ++position;
    }
    return false;
}

std::string sanitize_moss_segment_text(const std::string& text) {
    std::string stripped;
    stripped.reserve(text.size());
    std::size_t position = 0;
    while (position < text.size()) {
        if (text[position] == '[') {
            std::size_t after = position;
            if (marker_at(text, position, after)) {
                stripped.push_back(' ');
                position = after;
                continue;
            }
        }
        stripped.push_back(text[position++]);
    }
    std::string collapsed = collapse_ascii_whitespace(stripped);
    if (!has_ascii_or_utf8_lexical_content(collapsed)) return {};
    return collapsed;
}

DiarizedTranscriptionResult parse_moss_transcript(
    const std::string& raw_text) {
    DiarizedTranscriptionResult result;
    result.raw_text = raw_text;
    const std::string trimmed = trim_ascii(raw_text);

    std::size_t search = 0;
    while (search < trimmed.size()) {
        const std::size_t start_open = trimmed.find('[', search);
        if (start_open == std::string::npos) break;

        std::size_t after_start = start_open;
        double start = 0.0;
        if (!parse_timestamp_token(
                trimmed, start_open, after_start, start)) {
            search = start_open + 1;
            continue;
        }

        std::size_t after_speaker = after_start;
        std::string speaker;
        if (!parse_speaker_token(
                trimmed, after_start, after_speaker, speaker)) {
            search = start_open + 1;
            continue;
        }

        std::size_t end_open = trimmed.find('[', after_speaker);
        bool emitted = false;
        while (end_open != std::string::npos) {
            std::size_t after_end = end_open;
            double end = 0.0;
            if (parse_timestamp_token(
                    trimmed, end_open, after_end, end)) {
                const std::string text = sanitize_moss_segment_text(
                    trimmed.substr(after_speaker, end_open - after_speaker));
                if (end >= start && !text.empty()) {
                    result.segments.push_back({
                        static_cast<float>(start),
                        static_cast<float>(end),
                        speaker,
                        text,
                    });
                    search = after_end;
                    emitted = true;
                }
                break;
            }
            end_open = trimmed.find('[', end_open + 1);
        }
        if (!emitted) search = start_open + 1;
    }

    if (result.segments.empty()) {
        result.text = trimmed;
    } else {
        std::ostringstream plain;
        for (const auto& segment : result.segments) {
            if (plain.tellp() > 0) plain << ' ';
            plain << segment.text;
        }
        result.text = plain.str();
    }
    return result;
}

}  // namespace speech_core
