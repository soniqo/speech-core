// Standalone VoxCPM2 voice-cloning CLI (ONNX Runtime backend).
//
// The ONNX counterpart of examples/litert/voxcpm2_clone.cpp: loads a VoxCPM2
// ONNX bundle, conditions on a reference clip, and synthesizes the given text
// in that voice — writing a 48 kHz mono WAV. The full 4-graph AR loop runs
// through OnnxVoxCPM2Tts, so this is the in-repo runner for validating the
// ONNX path end-to-end (perceptually, not just tensor cosines) on CPU and GPU.
//
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
#include <speech_core/audio/wav_io.h>

#include "../common/utf8_args.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

constexpr int kOutSampleRate = 48000;

int main(int argc, char** argv) {
    // UTF-8 argv: on Windows the default char** conversion goes through the
    // active code page and turns non-Latin text into '?'.
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    if (args.size() < 5) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <ref.wav> \"<text>\" <out.wav> "
            "[instruction] [max_steps] [seed]\n",
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

    const bool no_ref = (ref_wav == "none");
    speech_core::WavData in;
    if (!no_ref) {
        if (!in.load_mono(ref_wav)) {
            std::fprintf(stderr, "could not read reference WAV: %s\n", ref_wav.c_str());
            return 1;
        }
        std::fprintf(stderr, "reference: %zu samples @ %d Hz (%.2fs)\n",
                     in.samples.size(), in.sample_rate, in.duration());
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
        if (!no_ref) tts.set_reference(in.samples.data(), in.samples.size(), in.sample_rate);

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
        if (!speech_core::write_wav_mono_pcm16(out_wav, audio, kOutSampleRate)) {
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
