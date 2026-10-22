// Smart Turn v3.2 ONNX wrapper against the published graph.
//
// Gated on SPEECH_SMART_TURN_ONNX (path to smart-turn-v3.2-int8.onnx or the
// fp32 graph) so CI without models skips cleanly. Checks the I/O contract, the
// 8 s window preparation, determinism, and that the finished sentence in
// tests/data/test_audio.wav (speech from 5 s to 9 s, silence elsewhere) is
// scored as complete: the Python reference on its first 12 s is 0.97 for the
// fp32 graph and 0.97 for the dynamic int8 graph.

#include "speech_core/models/onnx_smart_turn.h"
#include "speech_core/audio/wav_io.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

}  // namespace

int main() {
    const char* configured = std::getenv("SPEECH_SMART_TURN_ONNX");
    if (!configured || !*configured) {
        std::cout << "SPEECH_SMART_TURN_ONNX not set — skipping Smart Turn model test\n";
        return 0;
    }

    // Window preparation is pure and cheap; pin it before touching ORT.
    {
        std::vector<float> ramp(1000);
        for (size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<float>(i);
        const auto window = speech_core::OnnxSmartTurn::prepare_window(ramp.data(), ramp.size());
        if (window.size() != speech_core::OnnxSmartTurn::kWindowSamples) return fail("window size");
        if (window[0] != 0.0f || window[window.size() - 1001] != 0.0f) return fail("front padding");
        if (window[window.size() - 1000] != 0.0f || window.back() != 999.0f) return fail("tail placement");

        std::vector<float> longer(speech_core::OnnxSmartTurn::kWindowSamples + 5000);
        for (size_t i = 0; i < longer.size(); ++i) longer[i] = static_cast<float>(i);
        const auto tail = speech_core::OnnxSmartTurn::prepare_window(longer.data(), longer.size());
        if (tail.front() != 5000.0f || tail.back() != static_cast<float>(longer.size() - 1)) {
            return fail("long turns keep the last 8 s");
        }
    }

    speech_core::OnnxSmartTurn model(configured, /*hardware_acceleration=*/false);

    // Synthetic harmonic burst: finite, in range, deterministic.
    std::vector<float> tone(48000);
    for (size_t i = 0; i < tone.size(); ++i) {
        const float t = static_cast<float>(i) / 16000.0f;
        tone[i] = 0.2f * std::sin(2.0f * 3.14159265f * 140.0f * t)
                + 0.1f * std::sin(2.0f * 3.14159265f * 280.0f * t);
    }
    const float p1 = model.turn_complete_probability(tone.data(), tone.size(), 16000);
    const float p2 = model.turn_complete_probability(tone.data(), tone.size(), 16000);
    if (!std::isfinite(p1) || p1 < 0.0f || p1 > 1.0f) return fail("probability out of range");
    if (std::fabs(p1 - p2) > 1e-5f) return fail("not deterministic");

    // Resampling path: the same tone at 48 kHz must land close to 16 kHz.
    std::vector<float> tone48(tone.size() * 3);
    for (size_t i = 0; i < tone48.size(); ++i) {
        const float t = static_cast<float>(i) / 48000.0f;
        tone48[i] = 0.2f * std::sin(2.0f * 3.14159265f * 140.0f * t)
                  + 0.1f * std::sin(2.0f * 3.14159265f * 280.0f * t);
    }
    const float p48 = model.turn_complete_probability(tone48.data(), tone48.size(), 48000);
    if (std::fabs(p48 - p1) > 0.1f) return fail("48 kHz input diverges from 16 kHz");

    // Digital silence must not produce NaN (the front-end normalisation guards var=0).
    std::vector<float> silence(16000, 0.0f);
    const float ps = model.turn_complete_probability(silence.data(), silence.size(), 16000);
    if (!std::isfinite(ps)) return fail("silence produced a non-finite probability");

    // Real utterance: a finished sentence from the shared fixture.
    const char* fixture = std::getenv("SPEECH_CORE_TEST_AUDIO");
    std::string wav = fixture && *fixture ? fixture : "tests/data/test_audio.wav";
    speech_core::WavData in;
    if (in.load_mono(wav)) {
        // The fixture trails off into 11 s of silence; keep the sentence plus
        // three seconds of pause, which is what a VAD hand-off looks like.
        in.samples.resize(std::min(in.samples.size(), static_cast<size_t>(12) * static_cast<size_t>(in.sample_rate)));
        const float p = model.turn_complete_probability(in.samples.data(), in.samples.size(), in.sample_rate);
        std::cout << "  " << wav << ": p(complete) = " << p << "\n";
        if (p < 0.85f) return fail("finished fixture sentence scored as incomplete");
    } else {
        std::cout << "  (fixture " << wav << " not found; skipping the reference check)\n";
    }

    std::cout << "Smart Turn ONNX model test passed (p_tone=" << p1 << ")\n";
    return 0;
}
