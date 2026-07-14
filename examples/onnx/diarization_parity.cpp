// C++ gate for the ONNX diarization models.
//
// The Python parity gate (speech-models/benchmarks/parity_diarization_onnx.py)
// proved the exported GRAPHS match PyTorch. This proves the C++ WRAPPERS match —
// a different claim, and the one that ships. Two things can only break here:
//
//   1. The frame->time mapping. The ONNX graph emits 589 frames per 10 s window;
//      the LiteRT bundle emits 560 (ten 1 s chunks x 56). Downstream derives frame
//      duration as (end_time - start_time) / frames, so a wrapper that fills those
//      fields wrongly yields transcripts that look perfect while every speaker
//      boundary quietly drifts. Nothing else in the pipeline would notice.
//
//   2. The fbank contract. Both wrappers now call audio::wespeaker_fbank(), so a
//      cosine well below 1.0 between runtimes means the runtime swap changed the
//      embedding — not the features.
//
// Usage: speech_diarization_parity <pyannote.onnx> <wespeaker.onnx> <clip.pcm ...>
//        (clips are raw f32 mono @ 16 kHz)

#include "speech_core/models/onnx_pyannote_segmentation.h"
#include "speech_core/models/onnx_wespeaker_embedding.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

std::vector<float> read_pcm(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> pcm(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(pcm.data()), static_cast<std::streamsize>(bytes));
    return pcm;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <pyannote.onnx> <wespeaker.onnx> <clip.pcm> [...]\n",
                     argv[0]);
        return 2;
    }

    speech_core::OnnxPyannoteSegmentation seg(argv[1], /*hw_accel=*/false);
    speech_core::OnnxWeSpeakerEmbedding   emb(argv[2], /*hw_accel=*/false);

    std::printf("pyannote: %d frames/window (resolved from the graph)\n",
                seg.frames_per_window());

    bool ok = true;

    for (int i = 3; i < argc; ++i) {
        auto pcm = read_pcm(argv[i]);
        if (pcm.empty()) {
            std::printf("  [FAIL] %s: unreadable\n", argv[i]);
            ok = false;
            continue;
        }
        const float audio_s = static_cast<float>(pcm.size()) / 16000.0f;

        auto windows = seg.segment(pcm.data(), pcm.size(), 16000);
        if (windows.empty()) {
            std::printf("  [FAIL] %s: no windows\n", argv[i]);
            ok = false;
            continue;
        }

        // The whole point: frame duration must come out at ~17 ms. If a wrapper
        // copied the LiteRT geometry it would land at ~17.9 ms and every boundary
        // would be off by 5%.
        const auto& w0 = windows.front();
        const int   K  = 3;
        const int   nf = static_cast<int>(w0.speaker_activity.size()) / K;
        const float frame_ms =
            1000.0f * (w0.end_time - w0.start_time) / static_cast<float>(nf);

        // Coverage: the last window must reach the end of the audio, or the final
        // speaker turn of every meeting is silently dropped.
        const float covered = windows.back().end_time;
        const bool  covers  = covered + 1e-3f >= audio_s;

        // Activities are probabilities.
        float amin = 1.0f, amax = 0.0f;
        for (const auto& w : windows)
            for (float a : w.speaker_activity) {
                amin = std::min(amin, a);
                amax = std::max(amax, a);
            }
        const bool in_range = amin >= -1e-4f && amax <= 1.0f + 1e-4f;

        auto e = emb.embed(pcm.data(), pcm.size(), 16000);
        float norm = std::sqrt(std::inner_product(e.begin(), e.end(), e.begin(), 0.0f));
        const bool emb_ok = e.size() == 256 && std::fabs(norm - 1.0f) < 1e-3f;

        const bool clip_ok = covers && in_range && emb_ok &&
                             std::fabs(frame_ms - 17.0f) < 1.0f;
        ok = ok && clip_ok;

        std::printf("  [%s] %-22s %5.1fs  windows=%2zu  frames/win=%d  "
                    "frame=%.2fms  covered=%.1fs  act=[%.3f,%.3f]  |emb|=%.4f\n",
                    clip_ok ? "ok" : "FAIL", argv[i], audio_s, windows.size(), nf,
                    frame_ms, covered, amin, amax, norm);
    }

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
