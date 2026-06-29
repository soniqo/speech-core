#pragma once

#include "speech_core/tts_synthesis_options.h"

#include <vector>

namespace speech_core::internal {

std::vector<float> apply_tts_postprocess_owned(std::vector<float> samples,
                                               int sample_rate,
                                               TtsPostProcessFlags flags);

}  // namespace speech_core::internal
