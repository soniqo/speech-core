#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace speech_core {

enum class TtsSynthesisMode : std::uint32_t {
    Streaming = 0,
    Buffered = 1,
};

using TtsPostProcessFlags = std::uint32_t;

constexpr TtsPostProcessFlags kTtsPostProcessNone = 0u;
constexpr TtsPostProcessFlags kTtsPostProcessDeEsser = 1u << 0;

struct TtsSynthesisOptions {
    TtsSynthesisMode mode = TtsSynthesisMode::Streaming;
    TtsPostProcessFlags postprocess_flags = kTtsPostProcessNone;
};

void validate_tts_synthesis_options(const TtsSynthesisOptions& options,
                                    const char* owner);

std::vector<float> apply_tts_postprocess(const float* samples,
                                         size_t length,
                                         int sample_rate,
                                         TtsPostProcessFlags flags);

std::vector<float> apply_tts_postprocess(std::vector<float> samples,
                                         int sample_rate,
                                         TtsPostProcessFlags flags);

}  // namespace speech_core
