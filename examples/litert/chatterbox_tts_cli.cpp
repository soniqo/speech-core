// Chatterbox TTS CLI: synthesize text -> 24 kHz mono float32 raw via LiteRTChatterboxTts.
//   chatterbox_tts_cli <bundle_dir> <lang> <out_f32.bin> <text...>
#include "speech_core/models/litert_chatterbox_tts.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace speech_core;

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: %s <bundle_dir> <lang> <out.bin> <text|@textfile.utf8>\n", argv[0]); return 2; }
    std::string dir = argv[1], lang = argv[2], out = argv[3], text;
    // "@path" reads UTF-8 text from a file (Windows argv mangles non-ASCII to '?').
    if (argv[4][0] == '@') {
        std::ifstream tf(argv[4] + 1, std::ios::binary);
        std::string s((std::istreambuf_iterator<char>(tf)), std::istreambuf_iterator<char>());
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        text = s;
    } else {
        for (int i = 4; i < argc; ++i) { if (i > 4) text += " "; text += argv[i]; }
    }

    LiteRTChatterboxTts tts(dir);
    tts.set_seed(1234);
    std::vector<float> wav;
    tts.synthesize(text, lang, [&](const float* s, size_t n, bool) { if (s && n) wav.assign(s, s + n); });

    std::ofstream f(out, std::ios::binary);
    f.write((const char*)wav.data(), wav.size() * sizeof(float));
    std::printf("synthesized: tokens=%d wav_samples=%zu (%.2fs) seed=%u -> %s\n",
                tts.tokens_generated(), wav.size(), wav.size() / 24000.0, tts.seed_used(), out.c_str());
    return wav.empty() ? 1 : 0;
}
