#include "speech_core/models/moss_whisper_features.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using speech_core::MossWhisperFeatureExtractor;

    assert(MossWhisperFeatureExtractor::audio_token_count(0) == 0);
    assert(MossWhisperFeatureExtractor::audio_token_count(1) == 1);
    assert(MossWhisperFeatureExtractor::audio_token_count(16000) == 13);
    assert(MossWhisperFeatureExtractor::audio_token_count(480000) == 375);

    std::vector<float> audio(16000);
    for (std::size_t index = 0; index < audio.size(); ++index) {
        audio[index] = static_cast<float>(
            static_cast<int>(index % 257) - 128) / 1024.0f;
    }
    const auto features =
        MossWhisperFeatureExtractor().extract_padded_chunk(
            audio.data(), audio.size());
    assert(features.mel_bins == 80);
    assert(features.time_frames == 3000);
    assert(features.data.size() == 240000);

    struct Fixture {
        int mel;
        int frame;
        float value;
    };
    const std::vector<Fixture> fixtures = {
        {0, 0, 1.1125666f},
        {0, 1, 1.0164478f},
        {0, 10, 0.99898136f},
        {5, 0, 0.31692964f},
        {5, 50, 0.8176489f},
        {10, 99, 0.5653128f},
        {20, 10, 0.5086993f},
        {20, 50, 0.54108286f},
        {40, 20, 0.18729377f},
        {40, 99, 0.25527418f},
        {79, 0, -0.8874334f},
        {79, 50, 0.1336894f},
        {0, 100, 0.91144305f},
        {20, 100, 0.49058753f},
        {40, 100, 0.29211235f},
        {79, 100, 0.09761804f},
        {0, 101, 0.06924677f},
        {20, 250, -0.8874334f},
        {79, 2999, -0.8874334f},
    };
    for (const auto& fixture : fixtures) {
        const float actual = features.data[
            static_cast<std::size_t>(
                fixture.mel * features.time_frames + fixture.frame)];
        assert(std::fabs(actual - fixture.value) <= 5e-4f);
    }

    std::cout << "MOSS Whisper frontend tests passed\n";
    return 0;
}
