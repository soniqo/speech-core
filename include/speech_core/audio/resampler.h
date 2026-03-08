#pragma once

#include <cstddef>
#include <vector>

namespace speech_core {

/// Simple linear resampler for sample rate conversion.
///
/// Suitable for speech (not music). For high-quality resampling,
/// platform-specific implementations (vDSP, Oboe) should be used.
class Resampler {
public:
    /// Resample audio from one sample rate to another.
    /// @param input        Source samples
    /// @param input_length Number of source samples
    /// @param from_rate    Source sample rate in Hz
    /// @param to_rate      Target sample rate in Hz
    /// @return Resampled audio
    static std::vector<float> resample(
        const float* input, size_t input_length,
        int from_rate, int to_rate);
};

}  // namespace speech_core
