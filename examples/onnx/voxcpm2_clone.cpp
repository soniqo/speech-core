// Standalone VoxCPM2 voice-cloning CLI (ONNX Runtime backend).
//
// The ONNX counterpart of examples/litert/voxcpm2_clone.cpp: loads a VoxCPM2
// ONNX bundle, conditions on a reference clip, and synthesizes the given text
// in that voice — writing a 48 kHz mono WAV. The full 4-graph AR loop runs
// through OnnxVoxCPM2Tts, so this is the in-repo runner for validating the
// ONNX path end-to-end (perceptually, not just tensor cosines) on CPU and GPU.
//
// Execution provider is chosen by the SPEECH_CORE_ORT_PROVIDER env var
// (cpu | cuda | tensorrt). Requires a SPEECH_CORE_WITH_ONNX build; CUDA/TRT
// additionally need SPEECH_CORE_WITH_CUDA and a GPU onnxruntime (ORT_DIR).
//
// Usage:
//   speech_voxcpm2_clone_onnx <bundle_dir> <ref.wav> "<text>" <out.wav> \
//       [instruction] [max_steps] [seed]
//
//   bundle_dir : directory with voxcpm2-{decoder,audio-encoder,audio-decoder}.onnx
//                (+ external *.onnx.data) + tokenizer.json. voxcpm2-decoder.onnx
//                is the unified prefill+token-step graph (merged export).
//   ref.wav    : reference speaker clip (16-bit PCM WAV; "none" for plain TTS)
//   seed       : optional RNG seed for bit-reproducible renders

#include <speech_core/models/onnx_voxcpm2_tts.h>

#include "../common/utf8_args.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOutSampleRate = 48000;

// Minimal mono-float loader for canonical PCM-16 RIFF/WAVE. Multi-channel input
// is down-mixed by averaging. u8path: argv is UTF-8 (see utf8_args.h).
bool load_wav_mono(const std::string& path, std::vector<float>& out, int& sample_rate) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    char riff[4], wave[4];
    uint32_t file_size = 0;
    f.read(riff, 4);
    f.read(reinterpret_cast<char*>(&file_size), 4);
    f.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fprintf(stderr, "%s is not a RIFF/WAVE file\n", path.c_str());
        return false;
    }

    char chunk_id[4];
    uint32_t chunk_size = 0;
    uint16_t audio_format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    bool have_fmt = false;

    while (f.read(chunk_id, 4)) {
        f.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&channels), 2);
            f.read(reinterpret_cast<char*>(&rate), 4);
            f.seekg(6, std::ios::cur);
            f.read(reinterpret_cast<char*>(&bits), 2);
            if (chunk_size > 16) f.seekg(chunk_size - 16, std::ios::cur);
            have_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            if (!have_fmt || audio_format != 1 || bits != 16 || channels == 0) {
                std::fprintf(stderr, "%s: only 16-bit PCM WAV is supported\n", path.c_str());
                return false;
            }
            const size_t n = chunk_size / 2;
            std::vector<int16_t> pcm(n);
            f.read(reinterpret_cast<char*>(pcm.data()), chunk_size);
            const size_t frames = n / channels;
            out.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                int acc = 0;
                for (uint16_t c = 0; c < channels; ++c) acc += pcm[i * channels + c];
                out[i] = static_cast<float>(acc) / (channels * 32768.0f);
            }
            sample_rate = static_cast<int>(rate);
            return true;
        } else {
            f.seekg(chunk_size, std::ios::cur);
        }
    }
    std::fprintf(stderr, "%s: no data chunk\n", path.c_str());
    return false;
}

bool write_wav(const std::string& path, const float* samples, size_t count, int rate) {
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) return false;
    auto put32 = [&](uint32_t v) {
        char b[4] = {char(v & 0xFF), char((v >> 8) & 0xFF),
                     char((v >> 16) & 0xFF), char((v >> 24) & 0xFF)};
        f.write(b, 4);
    };
    auto put16 = [&](uint16_t v) {
        char b[2] = {char(v & 0xFF), char((v >> 8) & 0xFF)};
        f.write(b, 2);
    };
    const uint32_t data_bytes = static_cast<uint32_t>(count) * 2;
    f.write("RIFF", 4); put32(36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4); put32(16);
    put16(1);                                          // PCM
    put16(1);                                          // mono
    put32(static_cast<uint32_t>(rate));
    put32(static_cast<uint32_t>(rate) * 2);            // byte rate
    put16(2);                                          // block align
    put16(16);                                         // bits/sample
    f.write("data", 4); put32(data_bytes);
    for (size_t i = 0; i < count; ++i) {
        float s = samples[i];
        if (s < -1.0f) s = -1.0f;
        if (s >  1.0f) s =  1.0f;
        put16(static_cast<uint16_t>(static_cast<int16_t>(s * 32767.0f)));
    }
    return f.good();
}

}  // namespace

int main(int argc, char** argv) {
    // UTF-8 argv: on Windows the default char** conversion goes through the
    // active code page and turns non-Latin text into '?'.
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    if (args.size() < 5) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <ref.wav> \"<text>\" <out.wav> "
            "[instruction] [max_steps] [seed]\n"
            "  provider via SPEECH_CORE_ORT_PROVIDER=cpu|cuda|tensorrt\n",
            args.empty() ? "speech_voxcpm2_clone_onnx" : args[0].c_str());
        return 2;
    }
    const std::string bundle      = args[1];
    const std::string ref_wav     = args[2];
    const std::string text        = args[3];
    const std::string out_wav     = args[4];
    const std::string instruction = (args.size() >= 6) ? args[5] : "";
    const int         max_steps   = (args.size() >= 7) ? std::atoi(args[6].c_str()) : 256;
    const long        seed        = (args.size() >= 8) ? std::atol(args[7].c_str()) : 0;

    const char* prov = std::getenv("SPEECH_CORE_ORT_PROVIDER");
    std::fprintf(stderr, "ORT provider request: %s\n", (prov && *prov) ? prov : "cuda (build default)");

    const bool no_ref = (ref_wav == "none");
    std::vector<float> ref;
    int ref_rate = 0;
    if (!no_ref) {
        if (!load_wav_mono(ref_wav, ref, ref_rate)) return 1;
        std::fprintf(stderr, "reference: %zu samples @ %d Hz (%.2fs)\n",
                     ref.size(), ref_rate, ref.empty() ? 0.0 : double(ref.size()) / ref_rate);
    } else {
        std::fprintf(stderr, "reference: none (uncloned baseline)\n");
    }

    try {
        speech_core::OnnxVoxCPM2Tts tts(
            bundle + "/voxcpm2-decoder.onnx",
            bundle + "/voxcpm2-audio-encoder.onnx",
            bundle + "/voxcpm2-audio-decoder.onnx",
            bundle + "/tokenizer.json",
            /*hw_accel=*/true);

        if (!instruction.empty()) tts.set_instruction(instruction);
        tts.set_max_steps(max_steps);
        // Floor under the model's stop signal, scaled with word count — mirrors
        // the LiteRT CLI (VoxCPM2 false-stops on long non-Latin lines; a flat
        // floor pins short texts past their natural end into babble).
        int words = 0;
        bool in_word = false;
        for (const char c : text) {
            const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
            if (!ws && !in_word) { ++words; in_word = true; }
            if (ws) in_word = false;
        }
        int min_stop = words * 5 / 2;
        if (min_stop < 8) min_stop = 8;
        if (min_stop > max_steps - 16) min_stop = max_steps - 16;
        tts.set_min_steps_before_stop(min_stop);
        if (seed > 0) tts.set_seed(static_cast<uint32_t>(seed));
        if (!no_ref) tts.set_reference(ref.data(), ref.size(), ref_rate);

        std::vector<float> audio;
        bool got_final = false;
        tts.synthesize(text, "auto",
            [&](const float* chunk, size_t length, bool is_final) {
                if (chunk && length) audio.insert(audio.end(), chunk, chunk + length);
                if (is_final) got_final = true;
            });

        if (!got_final || audio.empty()) {
            std::fprintf(stderr, "synthesis produced no audio\n");
            return 1;
        }
        if (!write_wav(out_wav, audio.data(), audio.size(), kOutSampleRate)) {
            std::fprintf(stderr, "could not write %s\n", out_wav.c_str());
            return 1;
        }
        std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s (seed=%u)\n",
                     audio.size(), double(audio.size()) / kOutSampleRate,
                     kOutSampleRate, out_wav.c_str(), tts.seed_used());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
