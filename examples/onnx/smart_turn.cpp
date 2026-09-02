// Smart Turn v3.2 end-of-turn probability for a recorded utterance (ONNX backend).
//
// Feeds the last 8 s of a WAV file to OnnxSmartTurn and prints the probability
// that the speaker has finished their turn — the same call TurnDetector makes
// when a VAD pause is detected. Useful for tuning
// AgentConfig::turn_completion_threshold on your own recordings.
//
// Usage:
//   speech_smart_turn <in.wav> [--model path.onnx] [--threshold 0.5] [--json]
//
//   in.wav   : 16-bit PCM WAV, any sample rate (resampled to 16 kHz)
//   --model  : defaults to $SPEECH_MODEL_DIR/smart-turn-v3.2-int8.onnx, falling
//              back to smart-turn-v3.2.onnx (see scripts/download_models.sh)

#include <speech_core/models/onnx_smart_turn.h>

#include "../common/default_model_dir.h"
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

// Minimal mono-float loader for canonical PCM-16 RIFF/WAVE. Multi-channel input
// is down-mixed by averaging. Mirrors examples/onnx/sidon_restore.cpp.
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

std::string default_model_path() {
    const std::string dir = speech_example_model_dir();
    const std::string int8 = dir + "/smart-turn-v3.2-int8.onnx";
    if (std::filesystem::exists(std::filesystem::u8path(int8))) return int8;
    return dir + "/smart-turn-v3.2.onnx";
}

void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <in.wav> [--model path.onnx] [--threshold 0.5] [--json]\n"
        "  Prints the Smart Turn v3.2 probability that the speaker finished\n"
        "  their turn, judged from the last 8 s of the recording.\n",
        argv0);
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    const char* argv0 = args.empty() ? "speech_smart_turn" : args[0].c_str();

    std::string in_wav;
    std::string model_path;
    float threshold = 0.5f;
    bool json = false;
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--model" && i + 1 < args.size()) {
            model_path = args[++i];
        } else if (a == "--threshold" && i + 1 < args.size()) {
            threshold = std::strtof(args[++i].c_str(), nullptr);
        } else if (a == "--json") {
            json = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv0);
            return 0;
        } else if (in_wav.empty() && !a.empty() && a[0] != '-') {
            in_wav = a;
        } else {
            usage(argv0);
            return 2;
        }
    }
    if (in_wav.empty()) {
        usage(argv0);
        return 2;
    }
    if (model_path.empty()) model_path = default_model_path();

    std::vector<float> audio;
    int rate = 0;
    if (!load_wav_mono(in_wav, audio, rate)) return 1;

    try {
        speech_core::OnnxSmartTurn model(model_path, /*hardware_acceleration=*/false);
        const float probability =
            model.turn_complete_probability(audio.data(), audio.size(), rate);
        const bool complete = probability >= threshold;
        if (json) {
            std::printf("{\"probability\": %.4f, \"threshold\": %.2f, \"complete\": %s}\n",
                        probability, threshold, complete ? "true" : "false");
        } else {
            std::printf("turn complete probability: %.3f (%s, threshold %.2f)\n",
                        probability, complete ? "complete" : "incomplete", threshold);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
