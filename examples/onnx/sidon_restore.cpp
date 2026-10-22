// Standalone Sidon speech-restoration CLI (ONNX Runtime backend).
//
// Loads a Sidon ONNX bundle (predictor + DAC vocoder), runs combined denoise +
// dereverb on an input clip, and writes a 48 kHz mono WAV. The C++ SeamlessM4T
// log-mel front-end turns the 16 kHz input into input_features[1,T,160], which
// feeds the predictor → vocoder pipeline (see OnnxSidonRestorer).
//
// Primary use case: clean a reverberant voice-cloning reference before a TTS
// voice-cloner. Offline / whole-clip; this is not a streaming tool.
//
//
// Usage:
//   speech_sidon_restore <bundle_dir> <in.wav> <out.wav>
//
//   bundle_dir : directory with sidon-predictor.onnx + sidon-vocoder.onnx
//   in.wav     : input clip (16-bit PCM WAV, any sample rate — resampled to
//                16 kHz internally)
//   out.wav    : restored output, 48 kHz mono 16-bit PCM

#include <speech_core/models/onnx_sidon_restorer.h>
#include <speech_core/audio/wav_io.h>

#include "../common/utf8_args.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOutSampleRate = 48000;
}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <in.wav> <out.wav>\n"
            "  bundle_dir : dir with sidon-predictor.onnx + sidon-vocoder.onnx\n",
            args.empty() ? "speech_sidon_restore" : args[0].c_str());
        return 2;
    }
    const std::string bundle  = args[1];
    const std::string in_wav  = args[2];
    const std::string out_wav = args[3];

    speech_core::WavData in;
    if (!in.load_mono(in_wav)) {
        std::fprintf(stderr, "could not read WAV: %s\n", in_wav.c_str());
        return 1;
    }
    std::fprintf(stderr, "input: %zu samples @ %d Hz (%.2fs)\n",
                 in.samples.size(), in.sample_rate, in.duration());

    try {
        speech_core::OnnxSidonRestorer restorer(
            bundle + "/sidon-predictor.onnx",
            bundle + "/sidon-vocoder.onnx",
            /*hw_accel=*/true);

        std::vector<float> restored = restorer.restore(in.samples.data(), in.samples.size(), in.sample_rate);
        if (restored.empty()) {
            std::fprintf(stderr, "restoration produced no audio (clip too short?)\n");
            return 1;
        }
        if (!speech_core::write_wav_mono_pcm16(out_wav, restored, kOutSampleRate)) {
            std::fprintf(stderr, "could not write %s\n", out_wav.c_str());
            return 1;
        }
        std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s\n",
                     restored.size(), double(restored.size()) / kOutSampleRate,
                     kOutSampleRate, out_wav.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
