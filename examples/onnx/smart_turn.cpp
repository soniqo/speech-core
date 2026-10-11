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
#include <speech_core/audio/wav_io.h>

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

    speech_core::WavData in;
    if (!in.load_mono(in_wav)) {
        std::fprintf(stderr, "could not read WAV: %s\n", in_wav.c_str());
        return 1;
    }

    try {
        speech_core::OnnxSmartTurn model(model_path, /*hardware_acceleration=*/false);
        const float probability =
            model.turn_complete_probability(in.samples.data(), in.samples.size(), in.sample_rate);
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
