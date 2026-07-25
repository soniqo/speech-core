#include "speech_core/models/onnx_redimnet_speaker_embedding.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using speech_core::OnnxReDimNetSpeakerEmbedding;

    std::vector<float> two_seconds(32000);
    for (std::size_t index = 0; index < two_seconds.size(); ++index) {
        two_seconds[index] = static_cast<float>(index);
    }
    const auto repeated =
        OnnxReDimNetSpeakerEmbedding::prepare_audio(
            two_seconds.data(), two_seconds.size());
    assert(repeated.size() == 96000);
    assert(repeated[0] == 0.0f);
    assert(repeated[31999] == 31999.0f);
    assert(repeated[32000] == 0.0f);
    assert(repeated[95999] == 31999.0f);

    std::vector<float> long_audio(100000);
    for (std::size_t index = 0; index < long_audio.size(); ++index) {
        long_audio[index] = static_cast<float>(index);
    }
    const auto cropped =
        OnnxReDimNetSpeakerEmbedding::prepare_audio(
            long_audio.data(), long_audio.size());
    assert(cropped.front() == 2000.0f);
    assert(cropped.back() == 97999.0f);

    bool rejected = false;
    try {
        std::vector<float> short_audio(31999);
        (void)OnnxReDimNetSpeakerEmbedding::prepare_audio(
            short_audio.data(), short_audio.size());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        (void)OnnxReDimNetSpeakerEmbedding::prepare_audio(
            nullptr, 0, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "ReDimNet audio preparation tests passed\n";
    return 0;
}
