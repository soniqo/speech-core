#pragma once

#include <cctype>
#include <cstddef>
#include <string>
#include <utility>

namespace speech_core {

inline std::string trim_ascii_whitespace(std::string s) {
    std::size_t begin = 0;
    while (begin < s.size() &&
           std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    std::size_t end = s.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

inline std::string sanitize_voxcpm2_instruction(
    const std::string& instruction) {
    std::string out;
    out.reserve(instruction.size());
    for (std::size_t i = 0; i < instruction.size();) {
        const unsigned char c = static_cast<unsigned char>(instruction[i]);
        if (c == '(' || c == ')') {
            ++i;
            continue;
        }
        if (i + 3 <= instruction.size()
            && static_cast<unsigned char>(instruction[i]) == 0xEF
            && static_cast<unsigned char>(instruction[i + 1]) == 0xBC
            && (static_cast<unsigned char>(instruction[i + 2]) == 0x88
                || static_cast<unsigned char>(instruction[i + 2]) == 0x89)) {
            i += 3;
            continue;
        }
        out.push_back(instruction[i]);
        ++i;
    }
    return trim_ascii_whitespace(std::move(out));
}

inline std::string format_voxcpm2_prompt(const std::string& text,
                                         const std::string& instruction) {
    const std::string control = sanitize_voxcpm2_instruction(instruction);
    if (control.empty()) return text;
    return "(" + control + ")" + text;
}

}  // namespace speech_core
