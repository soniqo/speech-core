// Regression test: speech that begins inside the post-playback guard window
// must keep its head.
//
// The guard (set_post_playback_guard) suppresses VAD events for a short
// window after agent playback so residual echo cannot trigger a turn. The
// bug: guarded chunks were skipped entirely — including the pre-speech ring
// fill — so when a user answered immediately after the agent's reply (the
// natural rhythm), the first ~0.3 s of their utterance was unrecoverable and
// STT saw "…playing music" instead of "stop playing music". The fix keeps
// the ring rolling during the guard while still suppressing events.
//
// The audio pushed here encodes the chunk index in every sample, so the
// assertion can check exactly which chunk the captured utterance begins at:
// with the fix it must include audio from inside the guard window.

#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/pipeline/turn_detector.h"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

class ScriptedVAD : public VADInterface {
public:
    std::vector<float> probs;
    size_t idx = 0;
    float process_chunk(const float*, size_t) override {
        size_t i = idx++;
        return (i < probs.size()) ? probs[i] : 0.0f;
    }
    void reset() override { idx = 0; }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }
};

}  // namespace

int main() {
    printf("test_turn_detector_guard_preroll:\n");

    const size_t chunk = 512;                       // 32 ms at 16 kHz
    const float guard_seconds = 0.3f;
    const size_t guard_chunks = 10;                 // ceil(0.3 s / 32 ms)
    const size_t speech_chunks = 47;                // ~1.5 s of speech
    const size_t silence_chunks = 12;

    ScriptedVAD vad;
    // Speech probability high from the very first chunk: the user starts
    // talking the instant the guard arms (right after the agent's reply).
    for (size_t i = 0; i < speech_chunks; i++) vad.probs.push_back(0.85f);
    for (size_t i = 0; i < silence_chunks; i++) vad.probs.push_back(0.05f);

    AgentConfig cfg;
    cfg.warmup_stt = false;
    cfg.eager_stt = false;
    cfg.max_utterance_duration = 0.0f;

    size_t started = 0;
    size_t ended = 0;
    std::vector<float> utterance;
    TurnDetector detector(vad, cfg, [&](const TurnEvent& e) {
        if (e.type == TurnEvent::UserSpeechStarted) started++;
        if (e.type == TurnEvent::UserSpeechEnded) {
            ended++;
            utterance = e.audio;
        }
    });

    detector.set_post_playback_guard(guard_seconds);

    // Every sample in chunk i carries the value i+1, so the utterance's
    // first sample identifies the earliest chunk that survived.
    std::vector<float> audio(chunk);
    for (size_t i = 0; i < speech_chunks + silence_chunks; i++) {
        for (auto& s : audio) s = static_cast<float>(i + 1);
        detector.push_audio(audio.data(), audio.size());
    }

    assert(started == 1);
    assert(ended == 1);
    assert(!utterance.empty());

    const float first_chunk_id = utterance.front();
    printf("  utterance begins at chunk %.0f (guard covers 1..%zu)\n",
           first_chunk_id, guard_chunks);
    // The head of the utterance must reach back inside the guard window.
    // Without the ring fill during the guard, the earliest surviving chunk
    // sits well past the guard boundary and the first word is gone.
    assert(first_chunk_id <= static_cast<float>(guard_chunks));
    printf("  PASS: guarded speech onset survives into the utterance\n");

    printf("All turn_detector_guard_preroll tests passed.\n");
    return 0;
}
