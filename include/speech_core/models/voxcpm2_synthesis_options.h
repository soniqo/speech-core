#pragma once

#include <cstdint>

namespace speech_core {

enum class VoxCPM2SynthesisMode : std::uint32_t {
    Streaming = 0,
    Buffered = 1,
};

using VoxCPM2PostProcessFlags = std::uint32_t;

constexpr VoxCPM2PostProcessFlags kVoxCPM2PostProcessNone = 0u;
constexpr VoxCPM2PostProcessFlags kVoxCPM2PostProcessDeEsser = 1u << 0;

struct VoxCPM2SynthesisOptions {
    VoxCPM2SynthesisMode mode = VoxCPM2SynthesisMode::Streaming;
    VoxCPM2PostProcessFlags postprocess_flags = kVoxCPM2PostProcessNone;
};

}  // namespace speech_core
