#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/models/onnx_pocket_tts.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::string bundle_path() {
    const char* value = std::getenv("SPEECH_POCKET_TTS_BUNDLE");
    return value ? value : "";
}

void test_streaming_frame_contract(speech_core::OnnxPocketTts& tts) {
    int audio_callbacks = 0;
    int final_callbacks = 0;
    std::size_t sample_count = 0;
    float peak = 0.0f;
    tts.synthesize("Hello world.", "en", [&](const float* samples,
                                               std::size_t length,
                                               bool final) {
        if (final) {
            ++final_callbacks;
            assert(samples == nullptr);
            assert(length == 0);
            return;
        }
        ++audio_callbacks;
        assert(samples != nullptr);
        assert(length == 1920);
        sample_count += length;
        for (std::size_t index = 0; index < length; ++index) {
            assert(std::isfinite(samples[index]));
            peak = std::max(peak, std::abs(samples[index]));
        }
    });

    const auto metrics = tts.last_metrics();
    assert(audio_callbacks > 0);
    assert(final_callbacks == 1);
    assert(sample_count == static_cast<std::size_t>(audio_callbacks) * 1920);
    assert(metrics.frames_generated == audio_callbacks);
    assert(metrics.output_samples == static_cast<int>(sample_count));
    assert(metrics.first_audio_ms > metrics.conditioning_ms);
    assert(metrics.total_ms >= metrics.first_audio_ms);
    assert(metrics.stopped_on_eos);
    assert(!metrics.cancelled);
    assert(peak > 0.01f && peak <= 1.5f);
    std::printf("  PASS: streaming_frame_contract (%d frames)\n", audio_callbacks);
}

void test_callback_can_cancel(speech_core::OnnxPocketTts& tts) {
    int audio_callbacks = 0;
    int final_callbacks = 0;
    tts.synthesize("This sentence would normally generate several frames.", "en",
        [&](const float*, std::size_t length, bool final) {
            if (final) {
                ++final_callbacks;
                return;
            }
            assert(length == 1920);
            ++audio_callbacks;
            tts.cancel();
        });
    const auto metrics = tts.last_metrics();
    assert(audio_callbacks == 1);
    assert(final_callbacks == 1);
    assert(metrics.frames_generated == 1);
    assert(metrics.cancelled);
    std::printf("  PASS: callback_can_cancel\n");
}

}  // namespace

int main() {
    const std::string bundle = bundle_path();
    if (bundle.empty()) {
        std::printf("SKIP: SPEECH_POCKET_TTS_BUNDLE is unset\n");
        return 0;
    }

    speech_core::PocketTtsConfig config;
    config.seed = 42;
    config.max_frames = 100;
    config.intra_threads = 2;
    speech_core::OnnxPocketTts tts(bundle, config);
    test_streaming_frame_contract(tts);
    test_callback_can_cancel(tts);
    std::printf("All Pocket TTS integration tests passed.\n");
    return 0;
}
