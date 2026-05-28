// Standalone VoxCPM2 voice-cloning CLI (LiteRT backend).
//
// Loads a VoxCPM2 LiteRT bundle, conditions on a reference speaker clip, and
// synthesizes the given text in that voice — writing a 48 kHz mono WAV.
//
// Usage:
//   speech_voxcpm2_clone <bundle_dir> <ref.wav> "<text>" <out.wav> \
//       [instruction] [max_steps]
//
//   bundle_dir  : directory holding voxcpm2-{text-prefill,token-step,
//                 audio-encoder,audio-decoder}.tflite + tokenizer.json
//   ref.wav     : reference speaker clip (mono/stereo PCM-16 WAV, any rate;
//                 resampled to 16 kHz and trimmed/padded to 6.4 s internally)
//   instruction : optional style prefix, e.g. "calm, clear delivery"
//   max_steps   : optional AR step cap (default 256, ~40 s ceiling)
//
// Pairs with speech_transcribe for a clone → ASR round-trip sanity check.

#include <speech_core/models/litert_voxcpm2_tts.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOutSampleRate = 48000;

// Minimal mono-float loader for a canonical PCM-16 RIFF/WAVE file. Multi-channel
// input is down-mixed by averaging. Returns false on any parse failure.
bool load_wav_mono(const std::string& path, std::vector<float>& out, int& sample_rate) {
    std::ifstream f(path, std::ios::binary);
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
            f.seekg(6, std::ios::cur);                 // byte_rate + block_align
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
    std::ofstream f(path, std::ios::binary);
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
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <bundle_dir> <ref.wav> \"<text>\" <out.wav> "
            "[instruction] [max_steps]\n",
            argv[0]);
        return 2;
    }
    const std::string bundle      = argv[1];
    const std::string ref_wav     = argv[2];
    const std::string text        = argv[3];
    const std::string out_wav     = argv[4];
    const std::string instruction = (argc >= 6) ? argv[5] : "clear, natural delivery";
    const int         max_steps   = (argc >= 7) ? std::atoi(argv[6]) : 256;

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
        speech_core::LiteRTVoxCPM2Tts tts(
            bundle + "/voxcpm2-text-prefill.tflite",
            bundle + "/voxcpm2-token-step.tflite",
            bundle + "/voxcpm2-audio-encoder.tflite",
            bundle + "/voxcpm2-audio-decoder.tflite",
            bundle + "/tokenizer.json",
            /*hw_accel=*/false);

        if (!instruction.empty()) tts.set_instruction(instruction);
        tts.set_max_steps(max_steps);
        tts.set_min_steps_before_stop(32);
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
        std::fprintf(stderr, "wrote %zu samples (%.2fs @ %d Hz) to %s\n",
                     audio.size(), double(audio.size()) / kOutSampleRate,
                     kOutSampleRate, out_wav.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
