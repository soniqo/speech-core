// Standalone CLI: load the PersonaPlex wrapper against an on-disk bundle and
// run respond_stream with N frames of silence. Reports load time, frame
// generation rate (frames/sec), per-chunk callback timing, and final agent
// audio size. Use this to smoke-test end-to-end on CPU + CUDA EPs without
// the full ctest harness.
//
// Build (with ONNX): -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=...
// Run:               run_personaplex <bundle_dir> [num_user_frames]

#include "speech_core/models/onnx_personaplex.h"
#include "speech_core/models/parakeet_stt.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) { std::fclose(f); return true; }
    return false;
}

// Minimal mono PCM16 WAV writer. Used to dump the agent audio so we can
// (a) listen to verify the model is producing speech-like signal and
// (b) feed it through Parakeet STT for the roundtrip transcription gate.
bool write_wav_mono_pcm16(const std::string& path,
                          const std::vector<float>& samples,
                          int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n = static_cast<uint32_t>(samples.size());
    const uint32_t byte_rate = sample_rate * 2;
    const uint32_t data_bytes = n * 2;
    const uint32_t riff_size = 36 + data_bytes;
    auto w32 = [&](uint32_t v){ f.write(reinterpret_cast<char*>(&v), 4); };
    auto w16 = [&](uint16_t v){ f.write(reinterpret_cast<char*>(&v), 2); };
    f.write("RIFF", 4); w32(riff_size); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1);
    w32(static_cast<uint32_t>(sample_rate)); w32(byte_rate); w16(2); w16(16);
    f.write("data", 4); w32(data_bytes);
    for (float s : samples) {
        if (s > 1.0f) s = 1.0f; else if (s < -1.0f) s = -1.0f;
        int16_t pcm = static_cast<int16_t>(s * 32767.0f);
        w16(static_cast<uint16_t>(pcm));
    }
    return f.good();
}

// Resample naive linear (for src->16k Parakeet input). Good enough for
// the roundtrip sanity gate; speech-swift's resampler has higher quality.
std::vector<float> resample_linear(const std::vector<float>& src, int src_rate, int dst_rate) {
    if (src_rate == dst_rate) return src;
    const double ratio = static_cast<double>(src_rate) / dst_rate;
    const size_t n_dst = static_cast<size_t>(src.size() / ratio);
    std::vector<float> out(n_dst);
    for (size_t i = 0; i < n_dst; ++i) {
        double sp = i * ratio;
        size_t i0 = static_cast<size_t>(sp);
        double f = sp - i0;
        float v0 = src[i0];
        float v1 = (i0 + 1 < src.size()) ? src[i0 + 1] : v0;
        out[i] = static_cast<float>(v0 * (1.0 - f) + v1 * f);
    }
    return out;
}

// Basic acoustic metrics on the agent audio. Speech should have non-trivial
// variance and non-trivial dynamic range; silence has neither.
struct AudioStats {
    double rms;
    float  peak;
    size_t n_nonzero;
};
AudioStats audio_stats(const std::vector<float>& s) {
    AudioStats st{0.0, 0.0f, 0};
    if (s.empty()) return st;
    double ss = 0.0;
    for (float v : s) {
        ss += static_cast<double>(v) * v;
        if (std::fabs(v) > st.peak) st.peak = std::fabs(v);
        if (v != 0.0f) ++st.n_nonzero;
    }
    st.rms = std::sqrt(ss / s.size());
    return st;
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
    std::vector<float> all_audio;
    auto t_run0 = std::chrono::steady_clock::now();
    pp.respond_stream(user_pcm.data(), user_pcm.size(), 24000,
        [&](const speech_core::FullDuplexChunk& c) {
            ++chunks;
            total_emitted += c.length;
            all_audio.insert(all_audio.end(), c.samples, c.samples + c.length);
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

    // Audio integrity stats — silence has rms<1e-4; speech is rms>=0.01.
    auto st = audio_stats(all_audio);
    std::printf("\nAgent audio integrity:\n");
    std::printf("  RMS:       %.6f\n", st.rms);
    std::printf("  peak |x|:  %.4f\n", st.peak);
    std::printf("  non-zero:  %zu / %zu (%.2f%%)\n", st.n_nonzero, all_audio.size(),
                100.0 * st.n_nonzero / std::max<size_t>(1, all_audio.size()));

    // Write WAV
    const std::string wav_out = dir + "/personaplex_out.wav";
    if (write_wav_mono_pcm16(wav_out, all_audio, 24000)) {
        std::printf("\nWrote %s (%zu samples @ 24 kHz)\n", wav_out.c_str(), all_audio.size());
    } else {
        std::fprintf(stderr, "WARN: failed to write %s\n", wav_out.c_str());
    }

    // Optional Parakeet roundtrip: when SPEECH_CORE_PARAKEET_DIR points at a
    // Parakeet ONNX bundle, transcribe the agent audio and print the result.
    // Lets us verify the model is producing speech-shaped acoustic content
    // (even nonsense transcripts mean Mimi+temporal+depformer are at least
    // generating audio in the right manifold).
    const char* pq_env = std::getenv("SPEECH_CORE_PARAKEET_DIR");
    if (pq_env) {
        std::string pq_dir = pq_env;
        std::string pq_enc = pq_dir + "/parakeet-encoder-int8.onnx";
        std::string pq_dec = pq_dir + "/parakeet-decoder-joint-int8.onnx";
        std::string pq_vocab = pq_dir + "/vocab.json";
        if (file_exists(pq_enc) && file_exists(pq_dec) && file_exists(pq_vocab)) {
            std::printf("\nParakeet roundtrip:\n");
            speech_core::ParakeetStt stt(pq_enc, pq_dec, pq_vocab, false);
            auto audio_16k = resample_linear(all_audio, 24000, 16000);
            auto result = stt.transcribe(audio_16k.data(), audio_16k.size(), 16000);
            std::printf("  transcript: \"%.200s\"\n", result.text.c_str());
            std::printf("  language:   %s\n", result.language.c_str());
            std::printf("  confidence: %.3f\n", result.confidence);
        } else {
            std::printf("\nSPEECH_CORE_PARAKEET_DIR set but bundle missing required files\n");
        }
    }

    return got_final ? 0 : 3;
}
