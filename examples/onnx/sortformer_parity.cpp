// C++ gate for the streaming Sortformer wrapper.
//
// The Python parity in the consuming app's docs proves the exported GRAPH
// matches PyTorch. This proves the C++ WRAPPER matches the reference driver —
// a different claim, and the one that ships. Three things can only break here,
// and each is silent:
//
//   1. The mel front-end. It is NeMo's `AudioToMelSpectrogramPreprocessor`,
//      whose window is centred inside a wider FFT and whose padding is constant
//      rather than reflect. Neither is visible in the config. Features that are
//      subtly off produce confident, wrong speaker probabilities.
//
//   2. The windowing. A call owns 340 encoder frames and sees one frame of past
//      and forty of future. Off by a frame and every label slides; off by the
//      context and neighbouring calls report the same audio twice.
//
//   3. The prediction layout handed to the arrival-order cache. The graph emits
//      a fixed-width FIFO block; the reference driver's own update derives its
//      offsets from the FIFO's actual length. Reading one as the other takes the
//      chunk up to forty frames early on the first call of every recording, and
//      seeds the cache's scoring from that slice at the first compression —
//      which outlives the step that caused it.
//
// The audio must be long enough to overflow the speaker cache. A recording that
// fits one call never exercises the arrival-order update, and agreement then
// means only that one forward pass agreed.
//
// Usage: speech_sortformer_parity <sortformer.onnx> <audio.f32> <reference.f32>
//        (raw f32 mono @ 16 kHz; the reference is [frames x speakers] f32)

#include "speech_core/models/onnx_sortformer_diarizer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    const std::streamsize bytes = input.tellg();
    input.seekg(0);
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    return values;
}

/// Audio arrives from a capture callback in small blocks, never in one piece,
/// so it is fed that way here. A wrapper that only works when handed the whole
/// recording would pass a kinder test and fail in the product.
constexpr std::size_t kBlockSamples = 1600;  // 100 ms

constexpr float kActive = 0.5f;
constexpr double kMaxMeanError = 1e-4;
constexpr double kMinAgreement = 0.999;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(
            stderr,
            "usage: speech_sortformer_parity <model.onnx> <audio.f32> "
            "<reference.f32>\n");
        return 2;
    }
    const std::vector<float> audio = read_f32(argv[2]);
    const std::vector<float> reference = read_f32(argv[3]);

    // A refusal is a result here, and it was invisible: the wrapper rejects a
    // bundle whose config and graph describe different variants, and without
    // this the throw reached `std::terminate` and the process died at
    // 0xC0000409 with the reason it printed nowhere. The one thing a gate must
    // never do is fail silently.
    std::unique_ptr<speech_core::OnnxSortformerDiarizer> loaded;
    try {
        loaded = std::make_unique<speech_core::OnnxSortformerDiarizer>(
            argv[1], false);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    speech_core::OnnxSortformerDiarizer& diarizer = *loaded;
    const std::size_t speakers =
        static_cast<std::size_t>(diarizer.speakers());
    std::printf(
        "audio %.1f s, %d speakers, %d frames per call, %.3f s per frame\n",
        static_cast<double>(audio.size()) / 16000.0, diarizer.speakers(),
        diarizer.chunk_frames(), static_cast<double>(diarizer.frame_seconds()));

    std::vector<float> activity;
    for (std::size_t offset = 0; offset < audio.size();
         offset += kBlockSamples) {
        const std::size_t count =
            std::min(kBlockSamples, audio.size() - offset);
        const std::vector<float> produced =
            diarizer.push_audio(audio.data() + offset, count);
        activity.insert(activity.end(), produced.begin(), produced.end());
    }
    const std::vector<float> tail = diarizer.end_stream();
    activity.insert(activity.end(), tail.begin(), tail.end());

    const std::size_t frames = activity.size() / speakers;
    const std::size_t reference_frames = reference.size() / speakers;
    std::printf(
        "frames: wrapper %zu, reference %zu\n", frames, reference_frames);
    if (frames != reference_frames) {
        std::fprintf(
            stderr,
            "FAIL: frame counts differ. The windowing does not match the "
            "reference driver, so nothing below is comparable.\n");
        return 1;
    }
    if (frames == 0) {
        std::fprintf(stderr, "FAIL: no frames produced\n");
        return 1;
    }

    double total_error = 0.0;
    double worst = 0.0;
    std::size_t agreed = 0;
    for (std::size_t index = 0; index < activity.size(); ++index) {
        const double error =
            std::fabs(static_cast<double>(activity[index])
                      - static_cast<double>(reference[index]));
        total_error += error;
        if (error > worst) worst = error;
        if ((activity[index] > kActive) == (reference[index] > kActive)) {
            ++agreed;
        }
    }
    const double mean_error =
        total_error / static_cast<double>(activity.size());
    const double agreement =
        static_cast<double>(agreed) / static_cast<double>(activity.size());
    std::printf(
        "mean abs error %.8f  max %.6f  decision agreement %.6f\n",
        mean_error, worst, agreement);

    // A cache that never overflowed would agree trivially. Say how much of the
    // recording ran, so a shortened fixture cannot quietly weaken the gate.
    std::printf(
        "calls %zu (the arrival-order update runs from the first onwards)\n",
        frames / static_cast<std::size_t>(diarizer.chunk_frames()));

    // Per-call agreement, because where a run diverges says what is wrong. A
    // first call that already disagrees is the front-end or the windowing; one
    // that agrees and then drifts is the arrival-order cache.
    const std::size_t per_call =
        static_cast<std::size_t>(diarizer.chunk_frames()) * speakers;
    for (std::size_t start = 0, call = 0; start < activity.size();
         start += per_call, ++call) {
        const std::size_t stop = std::min(start + per_call, activity.size());
        double error = 0.0;
        std::size_t same = 0;
        for (std::size_t index = start; index < stop; ++index) {
            error += std::fabs(static_cast<double>(activity[index])
                               - static_cast<double>(reference[index]));
            if ((activity[index] > kActive) == (reference[index] > kActive)) {
                ++same;
            }
        }
        const double count = static_cast<double>(stop - start);
        std::printf(
            "  call %zu: mean %.6f agreement %.4f\n",
            call, error / count, static_cast<double>(same) / count);
    }
    if (argc > 4) {
        std::ofstream dump(argv[4], std::ios::binary);
        dump.write(reinterpret_cast<const char*>(activity.data()),
                   static_cast<std::streamsize>(activity.size()
                                                * sizeof(float)));
    }

    if (mean_error > kMaxMeanError || agreement < kMinAgreement) {
        std::fprintf(
            stderr,
            "FAIL: the wrapper does not reproduce the reference driver\n");
        return 1;
    }
    std::printf("sortformer wrapper parity: passed\n");
    return 0;
}
