// FDB M1 smoke test — exercises the corpus iterator, WAV I/O, and the
// driver loop end-to-end against the tiny tests/data/fdb_mini/ fixture
// without requiring real models or a live Ollama server. The MockOllama
// HTTP server pattern is the same one M0 introduced in
// test_pipeline_ollama_e2e.cpp.

// Force-enable assert() under release builds and sanitizer-enabled
// configurations (RelWithDebInfo defines NDEBUG, which would silently
// no-op every check). All scenarios use assert() as their gate.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "fdb_corpus.h"
#include "wav_io.h"

#include "speech_core/audio/pcm_codec.h"
#include "speech_core/audio/resampler.h"
#include "speech_core/llm/ollama_llm.h"
#include "speech_core/pipeline/voice_pipeline.h"

#ifdef SPEECH_CORE_FDB_WITH_ONNX
#  include "speech_core/models/parakeet_stt.h"
#  include "speech_core/models/kokoro_tts.h"
#  include "speech_core/models/silero_vad.h"
#endif

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace speech_core;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

#ifndef SPEECH_CORE_FDB_MINI_DIR
#  error "SPEECH_CORE_FDB_MINI_DIR must be defined by CMake"
#endif

namespace {

// ---------------------------------------------------------------------------
// Mocks — minimal MockSTT / MockTTS / ScriptedVAD.
// ---------------------------------------------------------------------------

class MockSTT : public STTInterface {
public:
    std::string next_text = "hello world";
    int call_count = 0;
    TranscriptionResult transcribe(const float*, size_t, int) override {
        call_count++;
        return {next_text, "", 0.9f, 0.f, 1.f};
    }
    int input_sample_rate() const override { return 16000; }
};

class MockTTS : public TTSInterface {
public:
    std::string last_text;
    int call_count = 0;
    std::atomic<bool> cancelled{false};
    void synthesize(const std::string& text, const std::string&,
                    TTSChunkCallback on_chunk) override {
        call_count++;
        last_text = text;
        cancelled.store(false);
        float samples[8] = {0.f, 0.05f, 0.f, -0.05f, 0.f, 0.05f, 0.f, -0.05f};
        for (int i = 0; i < 5; ++i) {
            if (cancelled.load()) break;
            on_chunk(samples, 8, i == 4);
        }
    }
    int output_sample_rate() const override { return 24000; }
    void cancel() override { cancelled.store(true); }
};

class ScriptedVAD : public VADInterface {
public:
    std::vector<float> probs;
    std::atomic<size_t> idx{0};
    float process_chunk(const float*, size_t) override {
        size_t i = idx.fetch_add(1);
        return (i < probs.size()) ? probs[i] : 0.0f;
    }
    void reset() override { idx.store(0); }
    int input_sample_rate() const override { return 16000; }
    size_t chunk_size() const override { return 512; }
};

// ---------------------------------------------------------------------------
// MockOllama — copied from tests/test_pipeline_ollama_e2e.cpp.
// ---------------------------------------------------------------------------

using ChatHandler =
    std::function<void(const nlohmann::json& req, httplib::DataSink& sink)>;

class MockOllama {
public:
    explicit MockOllama(ChatHandler h) : chat_handler_(std::move(h)) {
        server_.Post("/api/chat",
            [this](const httplib::Request& req, httplib::Response& res) {
                request_count_.fetch_add(1);
                nlohmann::json parsed;
                try { parsed = nlohmann::json::parse(req.body); }
                catch (...) {
                    res.status = 400;
                    res.set_content("{\"error\":\"bad request\"}",
                                    "application/json");
                    return;
                }
                res.set_chunked_content_provider(
                    "application/x-ndjson",
                    [parsed, h = chat_handler_](size_t,
                                                httplib::DataSink& sink) {
                        h(parsed, sink);
                        sink.done();
                        return true;
                    });
            });
        port_ = server_.bind_to_any_port("127.0.0.1");
        if (port_ < 0) throw std::runtime_error("MockOllama: bind failed");
        thread_ = std::thread([this] { server_.listen_after_bind(); });
        while (!server_.is_running()) std::this_thread::sleep_for(2ms);
    }
    ~MockOllama() {
        server_.stop();
        if (thread_.joinable()) thread_.join();
    }
    std::string base_url() const {
        return "http://127.0.0.1:" + std::to_string(port_);
    }
    int request_count() const { return request_count_.load(); }
private:
    httplib::Server  server_;
    ChatHandler      chat_handler_;
    int              port_ = -1;
    std::thread      thread_;
    std::atomic<int> request_count_{0};
};

void write_json_line(httplib::DataSink& sink, const nlohmann::json& j) {
    std::string s = j.dump();
    s.push_back('\n');
    sink.write(s.data(), s.size());
}

// ---------------------------------------------------------------------------
// VAD prob script — speech for the duration of the audio, then silence.
// ---------------------------------------------------------------------------

std::vector<float> make_vad_script(size_t samples_16k, size_t chunk_samples) {
    const size_t speech_chunks = samples_16k / chunk_samples + 1;
    std::vector<float> out;
    out.push_back(0.0f);
    for (size_t i = 0; i < speech_chunks; ++i) out.push_back(0.85f);
    for (size_t i = 0; i < 10; ++i) out.push_back(0.05f);
    return out;
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

void test_wav_io_roundtrip() {
    constexpr int sr = 16000;
    constexpr int n = sr;
    std::vector<float> in(n);
    for (int i = 0; i < n; ++i) {
        in[i] = 0.25f * std::sin(2.0f * 3.14159265f * 220.0f * i / sr);
    }
    auto tmp = fs::temp_directory_path() / "fdb_smoke_roundtrip.wav";
    bool wrote = fdb_bench::write_wav_mono_pcm16(tmp.string(),
                                                  in.data(), in.size(), sr);
    if (!wrote) { std::fprintf(stderr, "write failed\n"); std::abort(); }
    fdb_bench::WavData out;
    bool loaded = fdb_bench::load_wav_mono_pcm16(tmp.string(), &out);
    if (!loaded || out.sample_rate != sr ||
        static_cast<int>(out.samples.size()) < n - 1 ||
        static_cast<int>(out.samples.size()) > n + 1) {
        std::fprintf(stderr, "load failed or bad shape\n");
        std::abort();
    }
    size_t cmp = std::min(static_cast<size_t>(n), out.samples.size());
    for (size_t i = 0; i < cmp; ++i) {
        float diff = std::fabs(in[i] - out.samples[i]);
        if (diff >= 1.5f / 32767.0f) {
            std::fprintf(stderr, "diff too large at %zu\n", i);
            std::abort();
        }
    }
    fs::remove(tmp);
    std::printf("  PASS: wav_io_roundtrip\n");
}

void test_corpus_iterator_against_fdb_mini() {
    fdb_bench::FdbCorpusOptions opts;
    opts.corpus_root = SPEECH_CORE_FDB_MINI_DIR;
    auto samples = fdb_bench::FdbCorpus::load(opts);
    assert(samples.size() == 4);

    std::set<fdb_bench::FdbCategory> seen;
    bool backchannel_no_annotation = false;
    bool other_have_annotation = true;
    for (const auto& s : samples) {
        seen.insert(s.category);
        // Every sample must have a loadable input.wav.
        fdb_bench::WavData wav;
        assert(fdb_bench::load_wav_mono_pcm16(s.input_wav_path, &wav));
        assert(wav.sample_rate == 16000);
        if (s.category == fdb_bench::FdbCategory::Backchannel) {
            backchannel_no_annotation = s.annotation_path.empty();
        } else {
            if (s.annotation_path.empty()) other_have_annotation = false;
        }
    }
    assert(seen.count(fdb_bench::FdbCategory::CandorPauseHandling));
    assert(seen.count(fdb_bench::FdbCategory::SmoothTurnTaking));
    assert(seen.count(fdb_bench::FdbCategory::UserInterruption));
    assert(seen.count(fdb_bench::FdbCategory::Backchannel));
    assert(backchannel_no_annotation);
    assert(other_have_annotation);
    std::printf("  PASS: corpus_iterator_against_fdb_mini (4 samples)\n");
}

void test_corpus_category_filter() {
    fdb_bench::FdbCorpusOptions opts;
    opts.corpus_root = SPEECH_CORE_FDB_MINI_DIR;
    opts.category = fdb_bench::FdbCategory::Backchannel;
    auto samples = fdb_bench::FdbCorpus::load(opts);
    assert(samples.size() == 1);
    assert(samples[0].category == fdb_bench::FdbCategory::Backchannel);
    std::printf("  PASS: corpus_category_filter\n");
}

void test_ground_truth_transcript_extraction() {
    fdb_bench::FdbCorpusOptions opts;
    opts.corpus_root = SPEECH_CORE_FDB_MINI_DIR;
    auto samples = fdb_bench::FdbCorpus::load(opts);
    bool saw_non_empty = false;
    for (const auto& s : samples) {
        if (s.transcription_path.empty()) continue;
        auto gt = fdb_bench::FdbCorpus::extract_ground_truth_transcript(
            s.transcription_path);
        if (!gt.empty()) {
            // None of the sentinel tokens should leak through.
            assert(gt.find("[PAUSE]")       == std::string::npos);
            assert(gt.find("[TURN-TAKING]") == std::string::npos);
            assert(gt.find("[INTERRUPT]")   == std::string::npos);
            saw_non_empty = true;
        }
    }
    assert(saw_non_empty);
    std::printf("  PASS: ground_truth_transcript_extraction\n");
}

// ---------------------------------------------------------------------------
// End-to-end smoke through the same per-sample path the driver uses.
// ---------------------------------------------------------------------------

void test_smoke_driver_with_mock_ollama() {
    // Server emits a small streamed reply for every request.
    MockOllama server([](const nlohmann::json& /*req*/,
                         httplib::DataSink& sink) {
        for (const char* d : {"got", " it"}) {
            write_json_line(sink, {
                {"model", "test"},
                {"message", {{"role", "assistant"}, {"content", d}}},
                {"done", false}
            });
        }
        write_json_line(sink, {
            {"model", "test"},
            {"message", {{"role", "assistant"}, {"content", ""}}},
            {"done", true}, {"done_reason", "stop"}
        });
    });

    OllamaLLM::Options lopts;
    lopts.base_url = server.base_url();
    lopts.model = "test";
    OllamaLLM llm(lopts);

    fdb_bench::FdbCorpusOptions copts;
    copts.corpus_root = SPEECH_CORE_FDB_MINI_DIR;
    auto samples = fdb_bench::FdbCorpus::load(copts);
    assert(!samples.empty());

    auto out_dir = fs::temp_directory_path() /
                   ("fdb_smoke_out_" + std::to_string(::getpid()));
    fs::create_directories(out_dir);

    MockSTT stt;
    MockTTS tts;
    ScriptedVAD vad;

    for (const auto& s : samples) {
        // Load + resample
        fdb_bench::WavData wav;
        assert(fdb_bench::load_wav_mono_pcm16(s.input_wav_path, &wav));
        std::vector<float> audio16k = (wav.sample_rate == 16000)
            ? wav.samples
            : Resampler::resample(wav.samples.data(), wav.samples.size(),
                                  wav.sample_rate, 16000);

        stt.next_text = fdb_bench::FdbCorpus::extract_ground_truth_transcript(
            s.transcription_path);
        if (stt.next_text.empty()) stt.next_text = "...";

        vad.reset();
        vad.probs = make_vad_script(audio16k.size(), vad.chunk_size());

        AgentConfig cfg;
        cfg.mode = AgentConfig::Mode::Pipeline;
        cfg.warmup_stt = false;
        cfg.eager_stt = false;
        cfg.post_playback_guard = 0.0f;
        cfg.min_interruption_duration = 0.0f;
        cfg.max_response_duration = 60.0f;

        std::vector<uint8_t> tts_pcm16;
        std::atomic<int> response_done{0};
        std::atomic<int> errors{0};

        auto cb = [&](const PipelineEvent& e) {
            switch (e.type) {
            case EventType::ResponseAudioDelta:
                tts_pcm16.insert(tts_pcm16.end(),
                                 e.audio_data.begin(), e.audio_data.end());
                break;
            case EventType::ResponseDone:
                response_done.fetch_add(1);
                break;
            case EventType::Error:
                errors.fetch_add(1);
                break;
            default: break;
            }
        };

        VoicePipeline pipe(stt, tts, &llm, vad, cfg, cb);
        pipe.start();
        const size_t chunk = vad.chunk_size();
        for (size_t off = 0; off < audio16k.size(); off += chunk) {
            size_t n = std::min(chunk, audio16k.size() - off);
            pipe.push_audio(audio16k.data() + off, n);
        }
        // Push silence tail so the VAD script's silence probs are
        // consumed and end-of-speech is triggered.
        std::vector<float> tail(chunk, 0.0f);
        for (int i = 0; i < 12; ++i) {
            pipe.push_audio(tail.data(), tail.size());
        }
        pipe.wait_idle();
        pipe.stop();

        assert(errors.load() == 0);
        assert(response_done.load() == 1);
        assert(!tts_pcm16.empty());

        // Write output.wav for this sample so we can verify the writer.
        auto float_audio = PCMCodec::pcm16_to_float(
            tts_pcm16.data(), tts_pcm16.size());
        std::string wav_path = (out_dir / (s.sample_id + ".wav")).string();
        assert(fdb_bench::write_wav_mono_pcm16(
            wav_path, float_audio.data(), float_audio.size(),
            tts.output_sample_rate()));
        assert(fs::file_size(wav_path) > 44);  // > header size
    }

    // Every sample should have triggered exactly one Ollama request.
    assert(server.request_count() == static_cast<int>(samples.size()));

    // Cleanup
    std::error_code ec;
    fs::remove_all(out_dir, ec);

    std::printf("  PASS: smoke_driver_with_mock_ollama (%zu samples, "
                "%d Ollama requests)\n",
                samples.size(), server.request_count());
}

#ifdef SPEECH_CORE_FDB_WITH_ONNX
// Opt-in integration test — loads real Parakeet STT + Kokoro TTS + Silero
// VAD from SPEECH_MODEL_DIR and runs one real-speech sample through the
// same per-sample driver path the unit smoke uses. Skipped silently
// unless SPEECH_FDB_BENCH_INTEGRATION=1 is set and the model files
// exist on disk.
void test_real_models_integration() {
    const char* enabled = std::getenv("SPEECH_FDB_BENCH_INTEGRATION");
    if (!enabled || std::string(enabled) != "1") {
        std::printf("  SKIP: real_models_integration "
                    "(set SPEECH_FDB_BENCH_INTEGRATION=1 to enable)\n");
        return;
    }
    const char* dir_env = std::getenv("SPEECH_MODEL_DIR");
    std::string dir = dir_env ? dir_env : "";
    if (dir.empty()) {
        std::printf("  SKIP: real_models_integration — "
                    "SPEECH_MODEL_DIR unset\n");
        return;
    }
    const std::string enc   = dir + "/parakeet-encoder-int8.onnx";
    const std::string dec   = dir + "/parakeet-decoder-joint-int8.onnx";
    const std::string vocab = dir + "/vocab.json";
    const std::string vad_m = dir + "/silero-vad.onnx";
    const std::string kok_m = dir + "/kokoro-e2e.onnx";

    auto path_exists = [](const std::string& p) {
        std::error_code ec;
        return fs::exists(p, ec);
    };
    if (!path_exists(enc) || !path_exists(dec) || !path_exists(vocab) ||
        !path_exists(vad_m) || !path_exists(kok_m)) {
        std::printf("  SKIP: real_models_integration — model files "
                    "missing in %s\n", dir.c_str());
        return;
    }
    const std::string sample_wav =
        std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
    if (!path_exists(sample_wav)) {
        std::printf("  SKIP: real_models_integration — %s missing\n",
                    sample_wav.c_str());
        return;
    }

    // Mock Ollama (real model integration is about STT/TTS, not LLM).
    MockOllama server([](const nlohmann::json&, httplib::DataSink& sink) {
        for (const char* d : {"ack", " response"}) {
            write_json_line(sink, {
                {"model", "test"},
                {"message", {{"role", "assistant"}, {"content", d}}},
                {"done", false}});
        }
        write_json_line(sink, {
            {"model", "test"},
            {"message", {{"role", "assistant"}, {"content", ""}}},
            {"done", true}, {"done_reason", "stop"}});
    });
    OllamaLLM::Options lopts;
    lopts.base_url = server.base_url();
    lopts.model    = "test";
    OllamaLLM llm(lopts);

    // Build the real backends. Silero is constructed to verify it loads
    // even though the orchestrator below consumes the deterministic
    // ScriptedVAD trajectory — this keeps the unit smoke deterministic
    // and the turn boundary predictable.
    speech_core::ParakeetStt stt(enc, dec, vocab, /*hw_accel=*/false);
    speech_core::KokoroTts   tts(kok_m, dir + "/voices", dir,
                                 /*hw_accel=*/false);
    speech_core::SileroVad   real_vad(vad_m, /*hw_accel=*/false);
    (void)real_vad;  // exercised by construction
    ScriptedVAD vad;

    fdb_bench::WavData wav;
    assert(fdb_bench::load_wav_mono_pcm16(sample_wav, &wav));
    std::vector<float> audio16k = (wav.sample_rate == 16000)
        ? wav.samples
        : Resampler::resample(wav.samples.data(), wav.samples.size(),
                              wav.sample_rate, 16000);
    vad.reset();
    vad.probs = make_vad_script(audio16k.size(), vad.chunk_size());

    AgentConfig cfg;
    cfg.mode = AgentConfig::Mode::Pipeline;
    cfg.warmup_stt = false;
    cfg.eager_stt = false;
    cfg.post_playback_guard = 0.0f;
    cfg.min_interruption_duration = 0.0f;
    cfg.max_response_duration = 60.0f;

    std::vector<uint8_t> tts_pcm16;
    std::string captured_transcript;
    std::atomic<int> response_done{0};
    std::atomic<int> errors{0};
    auto cb = [&](const PipelineEvent& e) {
        switch (e.type) {
        case EventType::TranscriptionCompleted:
            captured_transcript = e.text; break;
        case EventType::ResponseAudioDelta:
            tts_pcm16.insert(tts_pcm16.end(),
                             e.audio_data.begin(), e.audio_data.end());
            break;
        case EventType::ResponseDone: response_done.fetch_add(1); break;
        case EventType::Error:        errors.fetch_add(1); break;
        default: break;
        }
    };

    VoicePipeline pipe(stt, tts, &llm, vad, cfg, cb);
    pipe.start();
    const size_t chunk = vad.chunk_size();
    for (size_t off = 0; off < audio16k.size(); off += chunk) {
        size_t n = std::min(chunk, audio16k.size() - off);
        pipe.push_audio(audio16k.data() + off, n);
    }
    std::vector<float> tail(chunk, 0.0f);
    for (int i = 0; i < 12; ++i) {
        pipe.push_audio(tail.data(), tail.size());
    }
    pipe.wait_idle();
    pipe.stop();

    assert(errors.load() == 0);
    assert(response_done.load() == 1);
    assert(!tts_pcm16.empty());
    assert(!captured_transcript.empty() &&
           "real Parakeet must produce a non-empty transcript");

    // Round-trip the TTS output to disk to exercise the writer at the
    // real backend's native rate.
    auto out_dir = fs::temp_directory_path() /
                   ("fdb_real_out_" + std::to_string(::getpid()));
    fs::create_directories(out_dir);
    auto float_audio = PCMCodec::pcm16_to_float(
        tts_pcm16.data(), tts_pcm16.size());
    std::string wav_out = (out_dir / "real_models_integration.wav").string();
    assert(fdb_bench::write_wav_mono_pcm16(
        wav_out, float_audio.data(), float_audio.size(),
        tts.output_sample_rate()));
    assert(fs::file_size(wav_out) > 44);

    std::error_code ec;
    fs::remove_all(out_dir, ec);

    std::printf("  PASS: real_models_integration "
                "(transcript=\"%s\", tts_bytes=%zu)\n",
                captured_transcript.c_str(), tts_pcm16.size());
}
#endif  // SPEECH_CORE_FDB_WITH_ONNX

}  // namespace

int main() {
    std::printf("test_fdb_bench_smoke:\n");
    test_wav_io_roundtrip();
    test_corpus_iterator_against_fdb_mini();
    test_corpus_category_filter();
    test_ground_truth_transcript_extraction();
    test_smoke_driver_with_mock_ollama();
#ifdef SPEECH_CORE_FDB_WITH_ONNX
    test_real_models_integration();
#endif
    std::printf("All fdb_bench smoke tests passed.\n");
    return 0;
}
