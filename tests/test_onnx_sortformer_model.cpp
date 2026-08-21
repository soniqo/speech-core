#include "speech_core/models/onnx_sortformer_diarizer.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << "Sortformer ONNX model test failed: " << message << '\n';
    return 1;
}

}  // namespace

int main() {
    const char* configured = std::getenv("SPEECH_SORTFORMER_ONNX");
    if (!configured || !*configured) {
        std::cout
            << "SPEECH_SORTFORMER_ONNX not set — skipping Sortformer model test\n";
        return 0;
    }

    speech_core::OnnxSortformerDiarizer model(
        configured, /*hardware_acceleration=*/false);
    if (model.speakers() != 4) return fail("unexpected speaker count");
    if (model.chunk_frames() <= 0) return fail("invalid chunk geometry");
    if (!(model.frame_seconds() > 0.0f)) return fail("invalid frame duration");

    std::vector<float> audio(2 * 16000);
    for (std::size_t index = 0; index < audio.size(); ++index) {
        const float time = static_cast<float>(index) / 16000.0f;
        audio[index] =
            0.08f * std::sin(2.0f * 3.14159265358979323846f * 173.0f * time);
    }

    std::vector<float> activity = model.push_audio(audio.data(), audio.size());
    const std::vector<float> tail = model.end_stream();
    activity.insert(activity.end(), tail.begin(), tail.end());
    if (activity.empty()) return fail("flush produced no activity frames");
    if (activity.size() % static_cast<std::size_t>(model.speakers()) != 0) {
        return fail("activity tensor is not frame-major by speaker");
    }
    for (float probability : activity) {
        if (!std::isfinite(probability)) return fail("non-finite probability");
        if (probability < 0.0f || probability > 1.0f) {
            return fail("probability outside [0, 1]");
        }
    }
    if (model.frames_emitted()
        != activity.size() / static_cast<std::size_t>(model.speakers())) {
        return fail("reported frame count differs from output");
    }
    if (!model.end_stream().empty()) return fail("second flush was not empty");

    std::cout << "Sortformer ONNX model test passed\n";
    return 0;
}
