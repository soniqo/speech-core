// Tiny CLI that runs Kokoro TTS on a piece of text and writes the audio to a WAV.
//
// Usage: speech_synthesize <model_dir> <output.wav> "<text>" [language]
//
// Pairs with speech_transcribe — round-trip a known prompt through synthesis
// and back through STT to surface phonemizer / tokenizer / decoder bugs
// without bouncing through Android.
//
// Calls KokoroTts directly (skipping the speech-core pipeline) so we can
// inspect the raw audio buffer the model emits.

#include <speech_core/models/kokoro_tts.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 24000;

static bool write_wav(const std::string& path,
                      const float* samples, size_t count, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

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
    put16(1);                               // PCM
    put16(1);                               // mono
    put32(static_cast<uint32_t>(sample_rate));
    put32(static_cast<uint32_t>(sample_rate) * 2);
    put16(2);                               // block align
    put16(16);                              // bits/sample
    f.write("data", 4); put32(data_bytes);

    for (size_t i = 0; i < count; i++) {
        float clamped = samples[i];
        if (clamped < -1.0f) clamped = -1.0f;
        if (clamped >  1.0f) clamped =  1.0f;
        int16_t v = static_cast<int16_t>(clamped * 32767.0f);
        put16(static_cast<uint16_t>(v));
    }
    return f.good();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <model_dir> <output.wav> \"<text>\" [language]\n"
            "  model_dir : directory holding kokoro-e2e.onnx + voices/*.bin\n"
            "  language  : BCP-47 tag (default: en). Auto-switches voice.\n",
            argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string out_wav   = argv[2];
    const std::string text      = argv[3];
    const std::string language  = (argc >= 5) ? argv[4] : "en";

    speech_core::KokoroTts tts(model_dir + "/kokoro-e2e.onnx",
                               model_dir + "/voices",
                               model_dir,
                               /*hw_accel=*/false);

    std::vector<float> samples;
    tts.synthesize(text, language,
        [&](const float* chunk, size_t length, bool /*is_final*/) {
            samples.insert(samples.end(), chunk, chunk + length);
        });

    if (samples.empty()) {
        std::fprintf(stderr, "synthesis produced no audio\n");
        return 1;
    }

    if (!write_wav(out_wav, samples.data(), samples.size(), kSampleRate)) {
        std::fprintf(stderr, "could not write %s\n", out_wav.c_str());
        return 1;
    }
    std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s\n",
                 samples.size(),
                 double(samples.size()) / double(kSampleRate),
                 kSampleRate, out_wav.c_str());
    return 0;
}
