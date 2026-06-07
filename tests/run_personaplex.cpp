// Standalone CLI: load the PersonaPlex wrapper against an on-disk bundle and
// run respond_stream with N frames of silence. Reports load time, frame
// generation rate (frames/sec), per-chunk callback timing, and final agent
// audio size. Use this to smoke-test end-to-end on CPU + CUDA EPs without
// the full ctest harness.
//
// Build (with ONNX): -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=...
// Run:               run_personaplex <bundle_dir> [num_user_frames]

#include "speech_core/models/onnx_personaplex.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: run_personaplex <bundle_dir> [num_user_frames=4]\n"
                     "  bundle expects: mimi_encoder.onnx, mimi_decoder.onnx,\n"
                     "                   temporal_step.onnx, depformer_step.onnx,\n"
                     "                   tokenizer_spm_32k_3.model,\n"
                     "                   voices/ (optional, set_voice is no-op if absent)\n");
        return 2;
    }
    const std::string dir = argv[1];
    int num_user_frames = (argc >= 3) ? std::atoi(argv[2]) : 4;

    const std::string enc = dir + "/mimi_encoder.onnx";
    const std::string dec = dir + "/mimi_decoder.onnx";
    const std::string tmp = dir + "/temporal_step.onnx";
    const std::string dep = dir + "/depformer_step.onnx";
    const std::string spm = dir + "/tokenizer_spm_32k_3.model";

    for (const auto& p : {enc, dec, tmp, dep, spm}) {
        if (!file_exists(p)) {
            std::fprintf(stderr, "Missing bundle file: %s\n", p.c_str());
            return 1;
        }
    }

    const bool hw_accel = (std::getenv("SPEECH_CORE_DISABLE_HW") == nullptr);
    std::printf("PersonaPlex bundle: %s  (hw_accel=%d)\n", dir.c_str(), (int)hw_accel);

    auto t0 = std::chrono::steady_clock::now();
    speech_core::OnnxPersonaPlex pp(enc, dec, tmp, dep, spm, dir + "/voices", hw_accel);
    auto t1 = std::chrono::steady_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("Load: %lld ms\n", static_cast<long long>(load_ms));

    pp.set_voice("NATM0");
    pp.set_system_prompt("You are a helpful assistant.");
    pp.set_max_frames(num_user_frames);
    pp.reset_session();

    // 24 kHz mono silence — exercises encode + the autoregressive loop without
    // needing a real WAV file in this smoke harness. Replace with a WAV load
    // for the validation gate (synth speech -> Parakeet STT round trip).
    const size_t samples_per_frame = 1920;  // 24000 / 12.5
    std::vector<float> user_pcm(samples_per_frame * num_user_frames, 0.0f);

    int chunks = 0;
    size_t total_emitted = 0;
    bool got_final = false;
    auto t_run0 = std::chrono::steady_clock::now();
    pp.respond_stream(user_pcm.data(), user_pcm.size(), 24000,
        [&](const speech_core::FullDuplexChunk& c) {
            ++chunks;
            total_emitted += c.length;
            if (c.is_final) got_final = true;
            std::printf("  chunk #%d: %zu samples (%.2fs), text_tokens=%zu, final=%d\n",
                        chunks, c.length,
                        static_cast<double>(c.length) / c.sample_rate,
                        c.text_tokens.size(), (int)c.is_final);
        });
    auto t_run1 = std::chrono::steady_clock::now();
    auto run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_run1 - t_run0).count();

    std::printf("\nrespond_stream: %lld ms\n", static_cast<long long>(run_ms));
    std::printf("  frames_generated: %d\n", pp.frames_generated());
    std::printf("  chunks:           %d\n", chunks);
    std::printf("  total samples:    %zu (%.2fs of agent audio)\n",
                total_emitted, static_cast<double>(total_emitted) / 24000.0);
    std::printf("  got_final:        %d\n", (int)got_final);
    if (pp.frames_generated() > 0) {
        double frame_ms = static_cast<double>(run_ms) / pp.frames_generated();
        std::printf("  per-frame:        %.1f ms\n", frame_ms);
        std::printf("  RTF (12.5 Hz):    %.3f (1.0 = realtime)\n", frame_ms / 80.0);
    }

    return got_final ? 0 : 3;
}
