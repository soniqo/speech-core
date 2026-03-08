#include "speech_core/audio/resampler.h"

#include <algorithm>
#include <cmath>

namespace speech_core {

std::vector<float> Resampler::resample(
    const float* input, size_t input_length,
    int from_rate, int to_rate) {

    if (from_rate == to_rate || input_length == 0) {
        return {input, input + input_length};
    }

    double ratio = static_cast<double>(from_rate) / static_cast<double>(to_rate);
    size_t output_length = static_cast<size_t>(
        static_cast<double>(input_length) / ratio);

    std::vector<float> output(output_length);

    for (size_t i = 0; i < output_length; i++) {
        double src_idx = static_cast<double>(i) * ratio;
        size_t idx0 = static_cast<size_t>(src_idx);
        float frac = static_cast<float>(src_idx - static_cast<double>(idx0));
        size_t idx1 = std::min(idx0 + 1, input_length - 1);
        output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
    }

    return output;
}

}  // namespace speech_core
