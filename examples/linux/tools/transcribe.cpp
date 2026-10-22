// Tiny CLI that runs Parakeet STT on a WAV file and prints what it heard.
//
// Usage: speech_transcribe [model_dir] <input.wav>
//        (model_dir defaults to $SPEECH_MODEL_DIR, else ~/.cache/speech-core/models)
//
// Reads PCM Int16 / Int24 / Int32 or Float32 mono or stereo at any sample rate, then
// resamples + downmixes to 16 kHz mono Float32 and feeds it through the
// pipeline. Useful for diagnosing TTS round-trip quality (synthesise speech,
// transcribe it back, compare to the original prompt).
//
// No external deps beyond libspeech.

#include "speech.h"
#include "speech_core/audio/resampler.h"
#include "speech_core/audio/wav_io.h"

#include "../../common/default_model_dir.h"
#include "../../common/utf8_args.h"

#include <algorithm>
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
    const auto args = speech_examples::utf8_args(argc, argv);
    argc = static_cast<int>(args.size());
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr,
            "usage: %s [model_dir] <input.wav>\n"
            "  model_dir : directory holding parakeet-* + silero-vad.onnx\n"
            "              (default: $SPEECH_MODEL_DIR, else %s)\n"
            "  input.wav : audio to transcribe (mono or stereo, 16-bit/24-bit/32-bit PCM or float32)\n",
            args.empty() ? "speech_transcribe" : args[0].c_str(),
            speech_example_model_dir().c_str());
        return 2;
    }
    const std::string model_dir = (argc == 3) ? args[1] : speech_example_model_dir();
    const std::string wav_path  = (argc == 3) ? args[2] : args[1];

    speech_core::WavData wav;
    if (!wav.load_mono(wav_path)) {
        std::fprintf(stderr, "could not read WAV: %s\n", wav_path.c_str());
        return 1;
    }
    std::fprintf(stderr,
        "loaded %s: %.2fs of mono audio at %d Hz → 16 kHz\n",
        wav_path.c_str(), wav.duration(), wav.sample_rate);
    if (wav.sample_rate != kTargetSampleRate) {
        wav.samples = speech_core::Resampler::resample(
            wav.samples.data(), wav.samples.size(), wav.sample_rate, kTargetSampleRate);
        wav.sample_rate = kTargetSampleRate;
    }

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
