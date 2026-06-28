#pragma once

#include "speech_core/tts_synthesis_options.h"

namespace speech_core {

using VoxCPM2SynthesisMode = TtsSynthesisMode;
using VoxCPM2PostProcessFlags = TtsPostProcessFlags;
using VoxCPM2SynthesisOptions = TtsSynthesisOptions;

constexpr VoxCPM2PostProcessFlags kVoxCPM2PostProcessNone = kTtsPostProcessNone;
constexpr VoxCPM2PostProcessFlags kVoxCPM2PostProcessDeEsser = kTtsPostProcessDeEsser;

}  // namespace speech_core
