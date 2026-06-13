// Regression test for fdb_bench's interaction with TurnDetector's
// max_utterance_duration force-split path. Promoted from /tmp/vad_repro.cpp
// (the investigation artifact for the spurious-second-UserSpeechStarted
// bug surfaced when running fdb_bench against real-speech samples >15s).
//
// What this proves
// ----------------
// 1) Default AgentConfig sets max_utterance_duration = 15.0f. fdb_bench's
//    make_vad_script_for_audio() emits ~20 s of prob=0.85, which causes
//    TurnDetector::force_end_utterance() to fire at t ~ 15 s.
// 2) force_end_utterance() calls streaming_vad_.reset(): chunk_count_=0,
//    state_=Silence. Audio (still scripted at prob=0.85) keeps flowing, so
//    StreamingVAD immediately walks Silence -> PendingSpeech -> Speech and
//    emits a SECOND VADEvent::SpeechStarted, which TurnDetector forwards as
//    a second UserSpeechStarted.
// 3) That second UserSpeechStarted is the smoking gun. In the real pipeline
//    it arrives *while STT is still running on the worker thread*. When STT
//    finishes and process_utterance() calls
//    turn_detector_.set_agent_speaking(true), TurnDetector sees in_speech_=
//    true and arms the retroactive interruption path (turn_detector.cpp
//    lines 245-254). With min_interruption_duration=0 the Interruption
//    fires immediately and cancels the just-generated response.
//
// This is by-design behaviour for live interactive agents (force-split lets
// the agent respond mid-monologue, retroactive interrupt yields if the user
// is still talking). For benchmark replay with bounded input length, both
// are wrong defaults. fdb_bench sets max_utterance_duration = 0.0f to
// disable the force-split (see PR #72).
//
// What this DOES NOT need
// -----------------------
// No worker thread, no STT, no LLM, no TTS, no VoicePipeline. The bug is
// fully visible at the TurnDetector layer using the exact same prob script
// fdb_bench feeds in. That isolates the diagnosis.

// Force-enable asserts even under RelWithDebInfo / sanitizer builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/agent_config.h"
#include "speech_core/pipeline/turn_detector.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using namespace speech_core;

namespace {

// Verbatim copy of fdb_bench.cpp's ScriptedVAD — using a plain size_t
// instead of std::atomic since this test is single-threaded.
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

// Verbatim copy of fdb_bench.cpp's make_vad_script_for_audio.
std::vector<float> make_vad_script_for_audio(size_t samples_16k,
                                              size_t chunk_samples) {
    const size_t speech_chunks = samples_16k / chunk_samples + 1;
    const size_t silence_tail  = 10;
    std::vector<float> out;
    out.reserve(speech_chunks + silence_tail + 2);
    out.push_back(0.0f);
    for (size_t i = 0; i < speech_chunks; ++i) out.push_back(0.85f);
    for (size_t i = 0; i < silence_tail;  ++i) out.push_back(0.05f);
    return out;
}

struct Recorded {
    TurnEvent::Type type;
    float time;
    size_t audio_len;
};

struct Counts {
    size_t started = 0, ended = 0, interrupt = 0, recovered = 0;
};

Counts run_one(float max_utterance_duration) {
    // 20.032 s of 16 kHz audio = 320512 samples, exactly 626 chunks of 512.
    // Matches the FDB samples that triggered the production bug.
    const size_t samples_16k = 320512;
    const size_t chunk = 512;

    ScriptedVAD vad;
    vad.probs = make_vad_script_for_audio(samples_16k, chunk);

    AgentConfig cfg;
    cfg.warmup_stt = false;
    cfg.eager_stt = false;
    cfg.max_utterance_duration = max_utterance_duration;

    std::vector<Recorded> log;
    auto on_event = [&](const TurnEvent& e) {
        log.push_back({e.type, e.time, e.audio.size()});
    };

    TurnDetector td(vad, cfg, on_event);

    std::vector<float> silence(chunk, 0.0f);
    const size_t audio_chunks = samples_16k / chunk;
    for (size_t i = 0; i < audio_chunks; ++i) {
        td.push_audio(silence.data(), silence.size());
    }
    for (int i = 0; i < 12; ++i) {
        td.push_audio(silence.data(), silence.size());
    }

    Counts c;
    for (const auto& r : log) {
        if (r.type == TurnEvent::UserSpeechStarted)     ++c.started;
        if (r.type == TurnEvent::UserSpeechEnded)       ++c.ended;
        if (r.type == TurnEvent::Interruption)          ++c.interrupt;
        if (r.type == TurnEvent::InterruptionRecovered) ++c.recovered;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

void test_default_15s_force_split_emits_two_started() {
    // Documents the by-design behaviour. With the default 15 s cap on a
    // ~20 s monologue:
    //   - SpeechStarted #1 fires at ~0.288 s (real start)
    //   - UserSpeechEnded fires at 15 s via force_end_utterance, which
    //     resets streaming_vad AND vad_ (so the prob script restarts).
    //   - Continuing 0.85 probs from the restarted script walk Silence ->
    //     PendingSpeech -> Speech, emitting SpeechStarted #2 with a
    //     stale post-reset chunk_count_=1 timestamp (~0.288 s again).
    //   - The remaining ~5 s of pushes never reach the silence tail of
    //     the restarted script, so the second utterance has NO matching
    //     UserSpeechEnded — the user is left "stuck" in_speech_=true.
    // This is the subtle interaction that bites a benchmark replay: the
    // pipeline's worker then sees in_speech_=true when it calls
    // set_agent_speaking(true), and fires a retroactive Interruption.
    // If TurnDetector ever stops force-splitting, or stops resetting the
    // VADInterface inside force_end_utterance, this test will fail and
    // force a documented decision.
    Counts c = run_one(/*max_utterance_duration=*/15.0f);
    assert(c.started == 2 &&
        "default force-split is expected to emit 2 UserSpeechStarted");
    assert(c.ended == 1 &&
        "default force-split emits exactly 1 UserSpeechEnded — the second "
        "utterance never reaches its silence tail because vad_.reset() "
        "restarted the prob script and the remaining audio is too short");
    assert(c.interrupt == 0);
    std::printf("  PASS: default_15s_force_split_emits_two_started "
                "(started=%zu ended=%zu)\n", c.started, c.ended);
}

void test_fdb_bench_config_emits_single_clean_utterance() {
    // The production fix in fdb_bench: max_utterance_duration = 0 disables
    // force-split, so a 20 s monologue produces exactly one UserSpeech-
    // Started + UserSpeechEnded pair. If fdb_bench (or any other consumer
    // of replay-style bounded input) reverts this config, the cascade will
    // silently lose responses again — this assertion catches that.
    Counts c = run_one(/*max_utterance_duration=*/0.0f);
    assert(c.started == 1 &&
        "fdb_bench-style replay (no force-split) must emit exactly one "
        "UserSpeechStarted");
    assert(c.ended == 1 &&
        "fdb_bench-style replay must emit exactly one UserSpeechEnded");
    assert(c.interrupt == 0);
    std::printf("  PASS: fdb_bench_config_emits_single_clean_utterance\n");
}

void test_high_max_utterance_duration_also_works() {
    // Belt-and-suspenders: any value comfortably above the input duration
    // should produce a single clean utterance just like the 0.0f disable.
    Counts c = run_one(/*max_utterance_duration=*/60.0f);
    assert(c.started == 1);
    assert(c.ended == 1);
    assert(c.interrupt == 0);
    std::printf("  PASS: high_max_utterance_duration_also_works "
                "(60 s cap, 20 s input)\n");
}

}  // namespace

int main() {
    std::printf("test_turn_detector_force_split:\n");
    test_default_15s_force_split_emits_two_started();
    test_fdb_bench_config_emits_single_clean_utterance();
    test_high_max_utterance_duration_also_works();
    std::printf("All TurnDetector force-split regression tests passed.\n");
    return 0;
}
