// Standalone VoxCPM2 voice-cloning CLI (LiteRT backend).
//
// Loads a VoxCPM2 LiteRT bundle, conditions on a reference speaker clip, and
// synthesizes the given text in that voice — writing a 48 kHz mono WAV.
//
// Usage:
//   speech_voxcpm2_clone [bundle_dir] <ref.wav> "<text>" <out.wav> \
//       [instruction] [max_steps]
//
//   bundle_dir  : directory holding voxcpm2-{text-prefill,token-step,
//                 audio-encoder,audio-decoder}.tflite + tokenizer.json
//                 (default: $SPEECH_LITERT_MODEL_DIR, else the per-user cache
//                 dir where sc_voxcpm2_create_from_pretrained downloads the
//                 bundle — ~/.cache/speech-core/soniqo__VoxCPM2-LiteRT)
//   ref.wav     : reference speaker clip (mono/stereo PCM-16 WAV, any rate;
//                 resampled to 16 kHz and trimmed/padded to 6.4 s internally)
//   instruction : optional style prefix, e.g. "calm, clear delivery"
//                 (default: none — style conditioning shifts the clone away
//                 from the reference delivery)
//   max_steps   : optional AR step cap (default 256, ~40 s ceiling)
//   seed        : optional RNG seed for bit-reproducible renders (default:
//                 fresh random seed; the value used is printed either way,
//                 so any render can be reproduced after the fact)
//
// Pairs with speech_transcribe for a clone → ASR round-trip sanity check.

#include <speech_core/models/litert_voxcpm2_tts.h>

#include "../common/utf8_args.h"
#include "../common/default_model_dir.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kOutSampleRate = 48000;

// Minimal mono-float loader for a canonical PCM-16 RIFF/WAVE file. Multi-channel
// input is down-mixed by averaging. Returns false on any parse failure.
bool load_wav_mono(const std::string& path, std::vector<float>& out, int& sample_rate) {
    // u8path: argv is UTF-8 (see utf8_args.h); MSVC's char* ifstream overload
    // would reinterpret it through the active code page.
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
    // active code page and turns non-Latin text into '?' — the field report
    // behind this was a Hindi clone where the model was literally asked to
    // speak fourteen question marks.
    const std::vector<std::string> args = speech_examples::utf8_args(argc, argv);
    if (args.size() < 4) {
        std::fprintf(stderr,
            "usage: %s [bundle_dir] <ref.wav> \"<text>\" <out.wav> "
            "[instruction] [max_steps] [seed]\n"
            "  bundle_dir : VoxCPM2 LiteRT bundle (default: $SPEECH_LITERT_MODEL_DIR,\n"
            "               else %s)\n",
            args.empty() ? "speech_voxcpm2_clone" : args[0].c_str(),
            speech_example_voxcpm2_dir().c_str());
        return 2;
    }
    // bundle_dir is optional. Old form starts with the bundle directory; the
    // new form starts with ref.wav (a file, or the literal "none") —
    // disambiguate by whether args[1] is an existing directory. (The old
    // argc-threshold heuristic is gone: the optional [seed] arg makes a full
    // new-form call reach the same count as a short old-form one.)
    const bool has_dir = std::filesystem::is_directory(args[1]);
    const int base = has_dir ? 2 : 1;
    if (static_cast<int>(args.size()) < base + 3) {
        std::fprintf(stderr,
            "usage: %s [bundle_dir] <ref.wav> \"<text>\" <out.wav> "
            "[instruction] [max_steps] [seed]\n",
            args.empty() ? "speech_voxcpm2_clone" : args[0].c_str());
        return 2;
    }
    const std::string bundle      = has_dir ? args[1] : speech_example_voxcpm2_dir();
    const std::string ref_wav     = args[base];
    std::string       text        = args[base + 1];
    const std::string out_wav     = args[base + 2];
    // No default style instruction: conditioning on an (English) style line
    // measurably shifts a non-English clone away from the reference voice.
    const std::string instruction = (static_cast<int>(args.size()) >= base + 4) ? args[base + 3] : "";
    int               max_steps   = (static_cast<int>(args.size()) >= base + 5) ? std::atoi(args[base + 4].c_str()) : 256;
    long              seed        = (static_cast<int>(args.size()) >= base + 6) ? std::atol(args[base + 5].c_str()) : 0;

    // [verify] @file text source: read the prompt from a UTF-8 file when the
    // text arg is "@<path>", sidestepping Windows argv quote-mangling on phrases
    // with embedded double-quotes (which silently shift the later positional
    // args — e.g. max_steps parsed as 0, yielding a 0-step "no audio" render).
    if (!text.empty() && text[0] == '@') {
        std::ifstream tf(std::filesystem::u8path(text.substr(1)),
                         std::ios::binary | std::ios::ate);
        if (tf) {
            const std::streamsize sz = tf.tellg();
            tf.seekg(0);
            std::string ft(static_cast<size_t>(sz), '\0');
            tf.read(&ft[0], sz);
            while (!ft.empty() && (ft.back() == '\n' || ft.back() == '\r'
                                   || ft.back() == ' ' || ft.back() == '\t'))
                ft.pop_back();
            text = ft;
        }
    }
    // [verify] env overrides for deterministic A/B without the positional empty
    // instruction arg (PowerShell drops "" and shifts max_steps/seed left).
    if (const char* e = std::getenv("VOXCPM2_MAX_STEPS")) { if (*e) max_steps = std::atoi(e); }
    if (const char* e = std::getenv("VOXCPM2_SEED"))      { if (*e) seed     = std::atol(e); }
    std::fprintf(stderr, "PARSED text_len=%zu out=%s max_steps=%d seed=%ld\n",
                 text.size(), out_wav.c_str(), max_steps, seed);

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
        // Floor under the model's stop signal, scaled with word count.
        // VoxCPM2 fires its stop token prematurely on long non-Latin-script
        // lines (~2.1 steps/word measured), but a flat floor pins short
        // texts past their natural end and the AR fills the gap with babble
        // (a 4-word Hindi line at the old flat 32-step floor rendered ~6 s of
        // mostly noise). ~2.5 steps/word sits above the false-stop rate and
        // below a natural full read (~2.5-3 steps/word). Whitespace word
        // count: an undelimited line counts as one word and keeps the
        // 8-step minimum.
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
        // [verify-instrumentation] exact stop metadata for the over-generation
        // check: tokens_generated == AR steps emitted; stopped_on_stop_token
        // distinguishes a model-driven stop from hitting the max_steps cap.
        std::fprintf(stderr,
                     "METADATA tokens_generated=%d stopped_on_stop_token=%s "
                     "min_stop=%d max_steps=%d duration_s=%.2f\n",
                     tts.tokens_generated(),
                     tts.stopped_on_stop_token() ? "true" : "false",
                     min_stop, max_steps,
                     double(audio.size()) / kOutSampleRate);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
