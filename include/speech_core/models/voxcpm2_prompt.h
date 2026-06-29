#pragma once

#include <string>

namespace speech_core {

inline std::string format_voxcpm2_prompt(const std::string& text,
                                         const std::string& instruction) {
    if (instruction.empty()) return text;
    return "(" + instruction + ")" + text;
}

}  // namespace speech_core
