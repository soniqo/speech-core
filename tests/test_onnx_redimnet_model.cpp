#include "speech_core/models/onnx_redimnet_speaker_embedding.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    const char* configured = std::getenv("SPEECH_REDIMNET_ONNX");
    if (!configured || !*configured) {
        std::cout
            << "SPEECH_REDIMNET_ONNX not set — skipping ReDimNet model test\n";
        return 0;
    }

    speech_core::OnnxReDimNetSpeakerEmbedding model(
        configured, /*hardware_acceleration=*/false);
    std::vector<float> audio(32000);
    for (std::size_t index = 0; index < audio.size(); ++index) {
        const float time = static_cast<float>(index) / 16000.0f;
        audio[index] =
            0.12f * std::sin(2.0f * 3.14159265358979323846f * 173.0f * time)
            + 0.06f
                * std::sin(
                    2.0f * 3.14159265358979323846f * 271.0f * time);
    }

    const auto first = model.embed(audio.data(), audio.size(), 16000);
    const auto second = model.embed(audio.data(), audio.size(), 16000);
    assert(first.size() == 192);
    assert(second.size() == first.size());
    float norm = 0.0f;
    float dot = 0.0f;
    for (std::size_t index = 0; index < first.size(); ++index) {
        assert(std::isfinite(first[index]));
        norm += first[index] * first[index];
        dot += first[index] * second[index];
    }
    assert(std::fabs(std::sqrt(norm) - 1.0f) < 1e-5f);
    assert(dot > 0.99999f);

    const auto short_probe = model.embed_short_utterance(
        audio.data(), 9600, 16000);
    assert(short_probe.size() == 192);

    std::cout << "ReDimNet ONNX model test passed\n";
    return 0;
}
