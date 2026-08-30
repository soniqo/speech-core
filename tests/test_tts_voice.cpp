// TTSInterface::set_voice() contract: the base default is a no-op, an
// override receives the id verbatim, and the empty id means "restore the
// backend default". Pure interface test — no model runtime involved.

// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/interfaces.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace speech_core;

namespace {

// A fixed-voice backend: overrides nothing beyond the pure virtuals.
class FixedVoiceTts : public TTSInterface {
public:
    void synthesize(const std::string& text,
                    const std::string& /*language*/,
                    TTSChunkCallback on_chunk) override {
        std::vector<float> pcm(text.size(), 0.25f);
        on_chunk(pcm.data(), pcm.size(), true);
    }
    int output_sample_rate() const override { return 24000; }
};

// A preset backend: records what the host asked for.
class PresetTts : public TTSInterface {
public:
    std::string voice = "F1";
    int set_calls = 0;

    void set_voice(const std::string& voice_id) override {
        ++set_calls;
        voice = voice_id.empty() ? "F1" : voice_id;
    }
    void synthesize(const std::string& /*text*/,
                    const std::string& /*language*/,
                    TTSChunkCallback on_chunk) override {
        const float sample = 0.0f;
        on_chunk(&sample, 1, true);
    }
    int output_sample_rate() const override { return 44100; }
};

size_t render(TTSInterface& tts) {
    size_t samples = 0;
    tts.synthesize("hello", "en", [&](const float*, size_t n, bool) { samples += n; });
    return samples;
}

}  // namespace

void test_default_set_voice_is_a_noop() {
    FixedVoiceTts tts;
    TTSInterface& iface = tts;
    iface.set_voice("M1");
    iface.set_voice("");
    assert(render(iface) == 5);
    std::printf("  PASS: default_set_voice_is_a_noop\n");
}

void test_override_receives_id_and_empty_restores_default() {
    PresetTts tts;
    TTSInterface& iface = tts;

    iface.set_voice("M3");
    assert(tts.voice == "M3");
    assert(render(iface) == 1);

    iface.set_voice("");
    assert(tts.voice == "F1");
    assert(tts.set_calls == 2);
    std::printf("  PASS: override_receives_id_and_empty_restores_default\n");
}

// The per-call pattern a host uses to offer a voice on one synthesize() call
// without leaking it into later calls.
void test_scoped_voice_round_trip() {
    PresetTts tts;
    TTSInterface& iface = tts;

    const std::string requested = "F4";
    if (!requested.empty()) iface.set_voice(requested);
    render(iface);
    if (!requested.empty()) iface.set_voice("");

    assert(tts.voice == "F1");
    assert(tts.set_calls == 2);
    std::printf("  PASS: scoped_voice_round_trip\n");
}

int main() {
    std::printf("test_tts_voice:\n");
    test_default_set_voice_is_a_noop();
    test_override_receives_id_and_empty_restores_default();
    test_scoped_voice_round_trip();
    std::printf("All TTS voice tests passed.\n");
    return 0;
}
