// Tiny CLI that runs Parakeet STT on a WAV file and prints what it heard.
//
// Usage: speech_transcribe <model_dir> <input.wav>
//
// Reads PCM Float32 / Int16 / Int24 mono or stereo at any sample rate, then
// resamples + downmixes to 16 kHz mono Float32 and feeds it through the
// pipeline. Useful for diagnosing TTS round-trip quality (synthesise speech,
// transcribe it back, compare to the original prompt).
//
// No external deps beyond libspeech.

#include "speech.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kTargetSampleRate = 16000;
constexpr size_t kChunkSamples = 512;  // 32 ms at 16 kHz

// ---------------------------------------------------------------------------
// WAV reader
// ---------------------------------------------------------------------------

struct WavData {
    std::vector<float> samples;  // mono, target sample rate
    int sample_rate = 0;
    int original_sample_rate = 0;
    int original_channels = 0;
    int original_bits = 0;
};

static uint32_t read_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8)
         | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
static uint16_t read_u16(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

static bool load_wav(const std::string& path, WavData& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "cannot open " + path; return false; }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    if (bytes.size() < 44) { err = "file too small to be a WAV"; return false; }
    if (std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
        std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
        err = "not a RIFF/WAVE file";
        return false;
    }

    // Walk chunks looking for fmt + data.
    size_t pos = 12;
    uint16_t fmt_format = 0, fmt_channels = 0, fmt_bits = 0;
    uint32_t fmt_rate = 0;
    const uint8_t* data_ptr = nullptr;
    uint32_t data_len = 0;
    while (pos + 8 <= bytes.size()) {
        const uint8_t* hdr = bytes.data() + pos;
        const char tag[5] = {char(hdr[0]), char(hdr[1]), char(hdr[2]), char(hdr[3]), 0};
        uint32_t chunk_len = read_u32(hdr + 4);
        if (pos + 8 + chunk_len > bytes.size()) break;
        if (std::strcmp(tag, "fmt ") == 0 && chunk_len >= 16) {
            fmt_format = read_u16(hdr + 8);
            fmt_channels = read_u16(hdr + 10);
            fmt_rate = read_u32(hdr + 12);
            fmt_bits = read_u16(hdr + 22);
        } else if (std::strcmp(tag, "data") == 0) {
            data_ptr = hdr + 8;
            data_len = chunk_len;
            break;
        }
        pos += 8 + chunk_len + (chunk_len & 1);  // pad to even
    }
    if (!data_ptr || fmt_channels == 0) {
        err = "WAV has no fmt or data chunk";
        return false;
    }
    if (fmt_format != 1 /*PCM*/ && fmt_format != 3 /*FLOAT*/) {
        err = "WAV format " + std::to_string(fmt_format)
            + " unsupported (need PCM=1 or FLOAT=3)";
        return false;
    }

    out.original_sample_rate = static_cast<int>(fmt_rate);
    out.original_channels = fmt_channels;
    out.original_bits = fmt_bits;

    // Decode + downmix to mono float
    const size_t bytes_per_sample = fmt_bits / 8;
    const size_t frame_bytes = bytes_per_sample * fmt_channels;
    const size_t frame_count = data_len / frame_bytes;
    std::vector<float> mono(frame_count);
    for (size_t i = 0; i < frame_count; i++) {
        float sum = 0.0f;
        for (int c = 0; c < fmt_channels; c++) {
            const uint8_t* sp = data_ptr + i * frame_bytes + c * bytes_per_sample;
            float s = 0.0f;
            if (fmt_format == 3 && fmt_bits == 32) {
                std::memcpy(&s, sp, 4);
            } else if (fmt_format == 1 && fmt_bits == 16) {
                int16_t v = int16_t(uint16_t(sp[0]) | (uint16_t(sp[1]) << 8));
                s = float(v) / 32768.0f;
            } else if (fmt_format == 1 && fmt_bits == 24) {
                int32_t v = int32_t(uint32_t(sp[0])
                          | (uint32_t(sp[1]) << 8) | (uint32_t(sp[2]) << 16));
                if (v & 0x800000) v |= 0xFF000000;  // sign extend
                s = float(v) / 8388608.0f;
            } else if (fmt_format == 1 && fmt_bits == 32) {
                int32_t v = int32_t(read_u32(sp));
                s = float(v) / 2147483648.0f;
            } else {
                err = "unsupported sample width " + std::to_string(fmt_bits);
                return false;
            }
            sum += s;
        }
        mono[i] = sum / float(fmt_channels);
    }

    // Linear-interpolation resample to kTargetSampleRate. Cheap, but
    // adequate for diagnosing model output — TTS bandwidth is well below
    // 8 kHz so aliasing isn't a meaningful concern here.
    if (static_cast<int>(fmt_rate) == kTargetSampleRate) {
        out.samples = std::move(mono);
    } else {
        const double ratio = double(fmt_rate) / double(kTargetSampleRate);
        const size_t out_len = size_t(double(mono.size()) / ratio);
        out.samples.resize(out_len);
        for (size_t i = 0; i < out_len; i++) {
            double src = double(i) * ratio;
            size_t i0 = size_t(src);
            double frac = src - double(i0);
            float a = mono[i0];
            float b = (i0 + 1 < mono.size()) ? mono[i0 + 1] : a;
            out.samples[i] = float(double(a) + (double(b) - double(a)) * frac);
        }
    }
    out.sample_rate = kTargetSampleRate;
    return true;
}

// ---------------------------------------------------------------------------
// Pipeline event handler
// ---------------------------------------------------------------------------

struct Result {
    std::mutex mu;
    std::condition_variable cv;
    std::string text;
    float confidence = 0.0f;
    bool done = false;
    bool error = false;
};

static void on_event(const speech_event_t* event, void* ctx) {
    auto* r = static_cast<Result*>(ctx);
    std::unique_lock<std::mutex> lock(r->mu);
    switch (event->type) {
        case SPEECH_EVENT_TRANSCRIPTION:
            if (event->text) r->text = event->text;
            r->confidence = event->confidence;
            r->done = true;
            r->cv.notify_all();
            break;
        case SPEECH_EVENT_PARTIAL_TRANSCRIPTION:
            if (event->text) {
                std::cerr << "  [partial] " << event->text << "\r" << std::flush;
            }
            break;
        case SPEECH_EVENT_ERROR:
            std::cerr << "  [error] " << (event->text ? event->text : "") << "\n";
            r->error = true;
            r->done = true;
            r->cv.notify_all();
            break;
        default:
            break;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "usage: %s <model_dir> <input.wav>\n"
            "  model_dir : directory holding parakeet-* + silero-vad.onnx\n"
            "  input.wav : audio to transcribe (mono or stereo, 16-bit/24-bit/float)\n",
            argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    const std::string wav_path  = argv[2];

    WavData wav;
    std::string err;
    if (!load_wav(wav_path, wav, err)) {
        std::fprintf(stderr, "wav: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "loaded %s: %d Hz × %dch × %d-bit → %.2fs of 16 kHz mono\n",
        wav_path.c_str(),
        wav.original_sample_rate, wav.original_channels, wav.original_bits,
        double(wav.samples.size()) / double(wav.sample_rate));

    speech_config_t cfg = speech_config_default();
    cfg.model_dir = model_dir.c_str();
    cfg.transcribe_only = true;

    Result result;
    speech_pipeline_t pipeline = speech_create(cfg, on_event, &result);
    if (!pipeline) {
        std::fprintf(stderr, "speech_create failed (model dir? missing files?)\n");
        return 1;
    }
    speech_start(pipeline);

    // Push real audio
    for (size_t off = 0; off < wav.samples.size(); off += kChunkSamples) {
        size_t n = std::min(kChunkSamples, wav.samples.size() - off);
        speech_push_audio(pipeline, wav.samples.data() + off, n);
    }
    // Trailing 1.5 s of silence so VAD sees end-of-utterance and Parakeet flushes
    std::vector<float> silence(kChunkSamples, 0.0f);
    for (int i = 0; i < 47; i++) {
        speech_push_audio(pipeline, silence.data(), silence.size());
    }

    // Wait up to 30 s for the transcription event
    {
        std::unique_lock<std::mutex> lock(result.mu);
        result.cv.wait_for(lock, std::chrono::seconds(30),
                           [&]{ return result.done; });
    }

    speech_destroy(pipeline);

    if (!result.done || result.error) {
        std::fprintf(stderr, "transcription did not complete\n");
        return 1;
    }
    // Result on stdout — single line, useful for piping
    std::printf("%s\n", result.text.c_str());
    std::fprintf(stderr, "confidence: %.3f\n", result.confidence);
    return 0;
}
