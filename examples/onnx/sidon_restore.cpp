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
// Execution provider is chosen by SPEECH_CORE_ORT_PROVIDER (cpu | cuda |
// tensorrt). Requires a SPEECH_CORE_WITH_ONNX build; CUDA/TRT additionally need
// SPEECH_CORE_WITH_CUDA and a GPU onnxruntime (ORT_DIR).
//
// Usage:
//   speech_sidon_restore <bundle_dir> <in.wav> <out.wav>
//
//   bundle_dir : directory with sidon-predictor.onnx + sidon-vocoder.onnx
//   in.wav     : input clip (16-bit PCM WAV, any sample rate — resampled to
//                16 kHz internally)
//   out.wav    : restored output, 48 kHz mono 16-bit PCM

#include <speech_core/models/onnx_sidon_restorer.h>

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

// Minimal mono-float loader for canonical PCM-16 RIFF/WAVE. Multi-channel input
// is down-mixed by averaging. Mirrors examples/onnx/voxcpm2_clone.cpp.
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
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <in.wav> <out.wav>\n"
            "  bundle_dir : dir with sidon-predictor.onnx + sidon-vocoder.onnx\n"
            "  provider via SPEECH_CORE_ORT_PROVIDER=cpu|cuda|tensorrt\n",
            args.empty() ? "speech_sidon_restore" : args[0].c_str());
        return 2;
    }
    const std::string bundle  = args[1];
    const std::string in_wav  = args[2];
    const std::string out_wav = args[3];

    const char* prov = std::getenv("SPEECH_CORE_ORT_PROVIDER");
    std::fprintf(stderr, "ORT provider request: %s\n",
                 (prov && *prov) ? prov : "cpu/cuda (build default)");

    std::vector<float> in;
    int in_rate = 0;
    if (!load_wav_mono(in_wav, in, in_rate)) return 1;
    std::fprintf(stderr, "input: %zu samples @ %d Hz (%.2fs)\n",
                 in.size(), in_rate, in.empty() ? 0.0 : double(in.size()) / in_rate);

    try {
        speech_core::OnnxSidonRestorer restorer(
            bundle + "/sidon-predictor.onnx",
            bundle + "/sidon-vocoder.onnx",
            /*hw_accel=*/true);

        std::vector<float> restored = restorer.restore(in.data(), in.size(), in_rate);
        if (restored.empty()) {
            std::fprintf(stderr, "restoration produced no audio (clip too short?)\n");
            return 1;
        }
        if (!write_wav(out_wav, restored.data(), restored.size(), kOutSampleRate)) {
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
