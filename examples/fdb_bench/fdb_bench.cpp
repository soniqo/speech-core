// FDB driver — runs each FDB v1.0 sample through VoicePipeline and dumps
// <out-dir>/<sample_id>.wav + <out-dir>/<sample_id>.json with timing.
//
// M1 supports mock STT + mock TTS by default so the driver runs anywhere
// (no model files, no ORT). When SPEECH_CORE_FDB_WITH_ONNX is set at
// build time, --stt parakeet and --tts kokoro switch to real backends.
// The LLM is always OllamaLLM; --llm-base-url defaults to a local
// Ollama server.

#include "fdb_corpus.h"
#include "wav_io.h"

#include "speech_core/audio/pcm_codec.h"
#include "speech_core/audio/resampler.h"
#include "speech_core/llm/ollama_llm.h"
#include "speech_core/pipeline/voice_pipeline.h"

#ifdef SPEECH_CORE_FDB_WITH_ONNX
#  include "speech_core/models/parakeet_stt.h"
#  include "speech_core/models/kokoro_tts.h"
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace speech_core;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Mock backends — minimal implementations matching the test_pipeline_e2e
// pattern. Inlined for self-containment; ScriptedVAD plays a fixed
// speech-then-silence trajectory.
// ---------------------------------------------------------------------------

class MockSTT : public STTInterface {
public:
    std::string next_text = "hello world";
    float next_confidence = 0.95f;
    int call_count = 0;
    TranscriptionResult transcribe(const float*, size_t, int) override {
        call_count++;
        return {next_text, "", next_confidence, 0.0f, 1.0f};
    }
    int input_sample_rate() const override { return 16000; }
};

class MockTTS : public TTSInterface {
public:
    std::string last_text;
    int call_count = 0;
    std::atomic<bool> cancelled{false};
    int output_rate = 24000;

    void synthesize(const std::string& text, const std::string&,
                    TTSChunkCallback on_chunk) override {
        call_count++;
        last_text = text;
        cancelled.store(false);
        // Emit ~0.5 s of low-amplitude noise across 10 chunks so the
        // output WAV has measurable content for downstream eval.
        const int chunks = 10;
        const int samples_per_chunk = output_rate / (chunks * 2);
        std::vector<float> buf(static_cast<size_t>(samples_per_chunk));
        for (int i = 0; i < chunks; ++i) {
            if (cancelled.load()) break;
            for (int j = 0; j < samples_per_chunk; ++j) {
                buf[j] = 0.02f * std::sin(0.05f * (i * samples_per_chunk + j));
            }
            on_chunk(buf.data(), buf.size(), i == chunks - 1);
        }
    }
    int output_sample_rate() const override { return output_rate; }
    void cancel() override { cancelled.store(true); }
};

// ScriptedVAD plays back a per-driver script of probabilities. The
// driver sets the script for each sample before pushing audio.
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
// CLI parsing — bespoke; no external dep. Throws on unknown flag.
// ---------------------------------------------------------------------------

struct Args {
    std::string corpus_dir;
    std::string out_dir;
    std::optional<fdb_bench::FdbCategory> category;
    std::string stt = "mock";
    std::string tts = "mock";
    std::string llm_base_url = "http://localhost:11434";
    std::string llm_model;
    size_t limit = 0;
    std::string models_dir;     // flat scripts/models/ layout shortcut
    std::string parakeet_dir;   // overrides models_dir for Parakeet files
    std::string kokoro_dir;     // overrides models_dir for Kokoro files
    std::string kokoro_voice = "af_heart";
    bool hw_accel = false;      // pass through to ORT models
    bool show_help = false;
};

std::string resolve_parakeet_dir(const Args& a) {
    return a.parakeet_dir.empty() ? a.models_dir : a.parakeet_dir;
}
std::string resolve_kokoro_dir(const Args& a) {
    return a.kokoro_dir.empty() ? a.models_dir : a.kokoro_dir;
}

void print_usage() {
    std::fprintf(stdout,
        "fdb_bench — FDB v1.0 per-sample driver\n"
        "\n"
        "Usage: fdb_bench --corpus-dir <path> --out-dir <path> --llm-model <tag> [options]\n"
        "\n"
        "Required:\n"
        "  --corpus-dir <path>    Path to the FDB v1.0 v1_0/ directory\n"
        "  --out-dir <path>       Where to write <sample_id>.wav + .json\n"
        "  --llm-model <tag>      Ollama model tag (e.g. llama3.2:3b)\n"
        "\n"
        "Optional:\n"
        "  --category <name>      candor_pause_handling | synthetic_pause_handling |\n"
        "                         smooth_turn_taking | user_interruption | backchannel\n"
        "                         (default: all five)\n"
        "  --stt <backend>        mock | parakeet  (default: mock)\n"
        "  --tts <backend>        mock | kokoro    (default: mock)\n"
        "  --llm-base-url <url>   default http://localhost:11434\n"
        "  --limit <N>            cap at first N samples after sort (default: no cap)\n"
        "  --models-dir <path>    flat directory holding parakeet-encoder-int8.onnx,\n"
        "                         parakeet-decoder-joint-int8.onnx, vocab.json,\n"
        "                         kokoro-e2e.onnx (+ voices/), etc. — matches the\n"
        "                         layout produced by scripts/download_models.sh\n"
        "  --parakeet-dir <path>  override Parakeet location (--stt parakeet)\n"
        "  --kokoro-dir <path>    override Kokoro location (--tts kokoro)\n"
        "  --kokoro-voice <name>  default af_heart (only --tts kokoro)\n"
        "  --hw-accel             enable ORT hardware EP (default off, for repro)\n"
        "  -h, --help             this help\n");
}

Args parse_args(int argc, char** argv) {
    Args a;
    auto eat = [&](int& i, const char* flag) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("missing value for ") + flag);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        if (f == "-h" || f == "--help")            { a.show_help = true; }
        else if (f == "--corpus-dir")              { a.corpus_dir   = eat(i, "--corpus-dir"); }
        else if (f == "--out-dir")                 { a.out_dir      = eat(i, "--out-dir"); }
        else if (f == "--category") {
            std::string s = eat(i, "--category");
            auto c = fdb_bench::FdbCorpus::parse_category(s);
            if (!c) throw std::runtime_error("unknown --category: " + s);
            a.category = *c;
        }
        else if (f == "--stt")           { a.stt = eat(i, "--stt"); }
        else if (f == "--tts")           { a.tts = eat(i, "--tts"); }
        else if (f == "--llm-base-url")  { a.llm_base_url = eat(i, "--llm-base-url"); }
        else if (f == "--llm-model")     { a.llm_model = eat(i, "--llm-model"); }
        else if (f == "--limit")         { a.limit = std::stoul(eat(i, "--limit")); }
        else if (f == "--models-dir")    { a.models_dir = eat(i, "--models-dir"); }
        else if (f == "--parakeet-dir")  { a.parakeet_dir = eat(i, "--parakeet-dir"); }
        else if (f == "--kokoro-dir")    { a.kokoro_dir = eat(i, "--kokoro-dir"); }
        else if (f == "--kokoro-voice")  { a.kokoro_voice = eat(i, "--kokoro-voice"); }
        else if (f == "--hw-accel")      { a.hw_accel = true; }
        else throw std::runtime_error("unknown flag: " + f);
    }
    return a;
}

void validate(const Args& a) {
    if (a.show_help) return;
    if (a.corpus_dir.empty()) throw std::runtime_error("--corpus-dir is required");
    if (a.out_dir.empty())    throw std::runtime_error("--out-dir is required");
    if (a.llm_model.empty())  throw std::runtime_error("--llm-model is required");
    if (a.stt != "mock" && a.stt != "parakeet")
        throw std::runtime_error("--stt must be mock or parakeet");
    if (a.tts != "mock" && a.tts != "kokoro")
        throw std::runtime_error("--tts must be mock or kokoro");
#ifdef SPEECH_CORE_FDB_WITH_ONNX
    if (a.stt == "parakeet" && a.parakeet_dir.empty() && a.models_dir.empty())
        throw std::runtime_error(
            "--stt parakeet requires --parakeet-dir or --models-dir");
    if (a.tts == "kokoro" && a.kokoro_dir.empty() && a.models_dir.empty())
        throw std::runtime_error(
            "--tts kokoro requires --kokoro-dir or --models-dir");
#else
    if (a.stt == "parakeet")
        throw std::runtime_error("--stt parakeet requires SPEECH_CORE_WITH_ONNX=ON");
    if (a.tts == "kokoro")
        throw std::runtime_error("--tts kokoro requires SPEECH_CORE_WITH_ONNX=ON");
#endif
}

// ---------------------------------------------------------------------------
// JSON writer — minimal escape, hand-rolled to avoid pulling nlohmann into
// the driver binary's surface.
// ---------------------------------------------------------------------------

std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 2);
    for (char c : in) {
        switch (c) {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x",
                              static_cast<unsigned char>(c));
                out += buf;
            } else {
                out += c;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Per-sample processing — kept in a free function so the smoke test can
// reuse it. The pipeline is constructed per sample so events accumulated
// from prior samples don't leak.
// ---------------------------------------------------------------------------

struct SampleResult {
    bool ok = false;
    long long stt_ms = 0;
    long long llm_ms = 0;
    long long tts_ms = 0;
    long long ttft_first_audio_from_speech_end_ms = 0;
    long long total_wall_ms = 0;
    std::string agent_transcript_input;
    double output_duration_sec = 0.0;
    int output_sample_rate = 0;
    std::string error;
};

long long ms_between(std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
    if (a.time_since_epoch().count() == 0 ||
        b.time_since_epoch().count() == 0) return 0;
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

// Build a VAD prob script that mirrors the sample's speech-then-silence
// envelope: enough above-threshold probs to confirm speech onset for the
// full duration, then enough silence probs to trigger end-of-speech.
std::vector<float> make_vad_script_for_audio(size_t samples_16k,
                                             size_t chunk_samples) {
    const size_t speech_chunks = samples_16k / chunk_samples + 1;
    const size_t silence_tail  = 10;
    std::vector<float> out;
    out.reserve(speech_chunks + silence_tail + 2);
    out.push_back(0.0f);  // one prime silence
    for (size_t i = 0; i < speech_chunks; ++i) out.push_back(0.85f);
    for (size_t i = 0; i < silence_tail; ++i) out.push_back(0.05f);
    return out;
}

SampleResult process_one_sample(
    const fdb_bench::FdbSample& s,
    const std::string& out_dir,
    STTInterface& stt,
    TTSInterface& tts,
    ScriptedVAD& vad,
    OllamaLLM& llm,
    bool is_mock_stt)
{
    SampleResult r;

    fdb_bench::WavData wav;
    if (!fdb_bench::load_wav_mono_pcm16(s.input_wav_path, &wav)) {
        r.error = "load_wav_mono_pcm16 failed: " + s.input_wav_path;
        return r;
    }
    std::vector<float> audio16k;
    if (wav.sample_rate == 16000) {
        audio16k = std::move(wav.samples);
    } else {
        audio16k = Resampler::resample(
            wav.samples.data(), wav.samples.size(),
            wav.sample_rate, 16000);
    }

    // Prime mock STT with the ground-truth transcript so the LLM gets a
    // meaningful prompt. Real STT ignores this.
    std::string gt = fdb_bench::FdbCorpus::extract_ground_truth_transcript(
        s.transcription_path);
    if (is_mock_stt) {
        if (auto* m = dynamic_cast<MockSTT*>(&stt)) {
            m->next_text = gt.empty() ? "..." : gt;
        }
    }

    AgentConfig cfg;
    cfg.mode = AgentConfig::Mode::Pipeline;
    cfg.warmup_stt = false;
    cfg.eager_stt = false;
    cfg.allow_interruptions = true;
    cfg.min_interruption_duration = 0.0f;
    cfg.post_playback_guard = 0.0f;
    cfg.max_response_duration = 60.0f;
    // Disable max-utterance force-split. FDB samples can run 20-30 s of
    // real speech (Candor monologues, long interruption contexts); the
    // default 15 s force-split would emit a spurious second
    // UserSpeechStarted via streaming_vad_ reset, and the next worker-
    // thread set_agent_speaking(true) inside process_utterance would
    // arm the retroactive interruption path and cancel the LLM call.
    // Benchmark replay has a known-bounded input length, so we don't
    // need force-split protection here.
    cfg.max_utterance_duration = 0.0f;

    vad.reset();
    vad.probs = make_vad_script_for_audio(audio16k.size(), vad.chunk_size());

    struct Tm {
        std::chrono::steady_clock::time_point t0_push;
        std::chrono::steady_clock::time_point speech_started;
        std::chrono::steady_clock::time_point speech_ended;
        std::chrono::steady_clock::time_point txn_done;
        std::chrono::steady_clock::time_point resp_created;
        std::chrono::steady_clock::time_point first_audio;
        std::chrono::steady_clock::time_point resp_done;
    } tm;

    std::vector<uint8_t> tts_pcm16;
    std::string captured_transcript;
    float captured_stt_ms = 0;
    float captured_llm_ms = 0;
    float captured_tts_ms = 0;

    auto cb = [&](const PipelineEvent& e) {
        auto now = std::chrono::steady_clock::now();
        switch (e.type) {
        case EventType::SpeechStarted:        tm.speech_started = now; break;
        case EventType::SpeechEnded:          tm.speech_ended = now; break;
        case EventType::TranscriptionCompleted:
            tm.txn_done = now;
            captured_transcript = e.text;
            captured_stt_ms = e.stt_duration_ms;
            break;
        case EventType::ResponseCreated:
            tm.resp_created = now;
            captured_llm_ms = e.llm_duration_ms;
            break;
        case EventType::ResponseAudioDelta:
            if (tts_pcm16.empty()) tm.first_audio = now;
            tts_pcm16.insert(tts_pcm16.end(),
                             e.audio_data.begin(), e.audio_data.end());
            break;
        case EventType::ResponseDone:
            tm.resp_done = now;
            captured_tts_ms = e.tts_duration_ms;
            break;
        case EventType::Error:
            r.error = e.text;
            break;
        default: break;
        }
    };

    VoicePipeline pipe(stt, tts, &llm, vad, cfg, cb);
    pipe.start();
    tm.t0_push = std::chrono::steady_clock::now();

    const size_t chunk = vad.chunk_size();
    for (size_t off = 0; off < audio16k.size(); off += chunk) {
        size_t n = std::min(chunk, audio16k.size() - off);
        pipe.push_audio(audio16k.data() + off, n);
    }
    // Push silence tail so the VAD script's silence probs are consumed
    // and end-of-speech is detected. Twelve chunks ≈ 384 ms of silence,
    // comfortably above the default 100 ms min_silence_duration.
    std::vector<float> silence_tail(chunk, 0.0f);
    for (int i = 0; i < 12; ++i) {
        pipe.push_audio(silence_tail.data(), silence_tail.size());
    }
    pipe.wait_idle();
    pipe.stop();

    if (!r.error.empty()) return r;

    // Write output.wav at the TTS native rate.
    auto float_audio = PCMCodec::pcm16_to_float(
        tts_pcm16.data(), tts_pcm16.size());
    int out_rate = tts.output_sample_rate();
    fs::create_directories(out_dir);
    std::string wav_out = out_dir + "/" + s.category_dir_name +
                          "__" + s.sample_id + ".wav";
    fdb_bench::write_wav_mono_pcm16(wav_out, float_audio.data(),
                                    float_audio.size(), out_rate);

    r.output_sample_rate = out_rate;
    r.output_duration_sec = out_rate > 0
        ? static_cast<double>(float_audio.size()) / out_rate
        : 0.0;
    r.stt_ms = static_cast<long long>(captured_stt_ms);
    r.llm_ms = static_cast<long long>(captured_llm_ms);
    r.tts_ms = static_cast<long long>(captured_tts_ms);
    r.ttft_first_audio_from_speech_end_ms =
        ms_between(tm.speech_ended, tm.first_audio);
    r.total_wall_ms = ms_between(tm.t0_push, tm.resp_done);
    r.agent_transcript_input = captured_transcript;
    r.ok = true;
    return r;
}

void write_sample_json(const std::string& path,
                       const fdb_bench::FdbSample& s,
                       const SampleResult& r,
                       const std::string& stt_backend,
                       const std::string& tts_backend,
                       const std::string& llm_model,
                       double input_duration_sec)
{
    std::ofstream os(path);
    os << "{\n"
       << "  \"sample_id\": \""        << json_escape(s.sample_id) << "\",\n"
       << "  \"category\": \""         << json_escape(fdb_bench::FdbCorpus::category_name(s.category)) << "\",\n"
       << "  \"category_dir\": \""     << json_escape(s.category_dir_name) << "\",\n"
       << "  \"input_wav\": \""        << json_escape(s.input_wav_path) << "\",\n"
       << "  \"input_duration_sec\": " << input_duration_sec << ",\n"
       << "  \"ground_truth_transcript\": \""
            << json_escape(fdb_bench::FdbCorpus::extract_ground_truth_transcript(s.transcription_path))
            << "\",\n"
       << "  \"agent_transcript_input\": \"" << json_escape(r.agent_transcript_input) << "\",\n"
       << "  \"output_wav\": \""
            << json_escape(s.category_dir_name + "__" + s.sample_id + ".wav")
            << "\",\n"
       << "  \"output_duration_sec\": " << r.output_duration_sec << ",\n"
       << "  \"output_sample_rate\": "  << r.output_sample_rate << ",\n"
       << "  \"timings_ms\": {\n"
       << "    \"stt\": "                                          << r.stt_ms << ",\n"
       << "    \"llm\": "                                          << r.llm_ms << ",\n"
       << "    \"tts\": "                                          << r.tts_ms << ",\n"
       << "    \"ttft_first_audio_from_speech_end\": "             << r.ttft_first_audio_from_speech_end_ms << ",\n"
       << "    \"total_wall\": "                                   << r.total_wall_ms << "\n"
       << "  },\n"
       << "  \"stt_backend\": \"" << json_escape(stt_backend) << "\",\n"
       << "  \"tts_backend\": \"" << json_escape(tts_backend) << "\",\n"
       << "  \"llm_model\": \""   << json_escape(llm_model)   << "\",\n"
       << "  \"error\": \""       << json_escape(r.error)     << "\"\n"
       << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
        validate(args);
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "fdb_bench: %s\n\n", ex.what());
        print_usage();
        return 2;
    }
    if (args.show_help) { print_usage(); return 0; }

    // STT
    std::unique_ptr<STTInterface> stt;
    bool is_mock_stt = (args.stt == "mock");
    if (is_mock_stt) {
        stt = std::make_unique<MockSTT>();
    } else {
#ifdef SPEECH_CORE_FDB_WITH_ONNX
        const std::string pdir = resolve_parakeet_dir(args);
        stt = std::make_unique<ParakeetStt>(
            pdir + "/parakeet-encoder-int8.onnx",
            pdir + "/parakeet-decoder-joint-int8.onnx",
            pdir + "/vocab.json",
            args.hw_accel);
#else
        std::fprintf(stderr, "fdb_bench: --stt parakeet built out\n");
        return 2;
#endif
    }

    // TTS
    std::unique_ptr<TTSInterface> tts;
    if (args.tts == "mock") {
        tts = std::make_unique<MockTTS>();
    } else {
#ifdef SPEECH_CORE_FDB_WITH_ONNX
        const std::string kdir = resolve_kokoro_dir(args);
        auto k = std::make_unique<KokoroTts>(
            kdir + "/kokoro-e2e.onnx",
            kdir + "/voices",
            kdir,
            args.hw_accel);
        k->set_voice(args.kokoro_voice);
        tts = std::move(k);
#else
        std::fprintf(stderr, "fdb_bench: --tts kokoro built out\n");
        return 2;
#endif
    }

    ScriptedVAD vad;

    OllamaLLM::Options lopts;
    lopts.base_url = args.llm_base_url;
    lopts.model    = args.llm_model;
    OllamaLLM llm(lopts);

    fdb_bench::FdbCorpusOptions copts;
    copts.corpus_root = args.corpus_dir;
    copts.category    = args.category;
    copts.limit       = args.limit;
    auto samples = fdb_bench::FdbCorpus::load(copts);
    if (samples.empty()) {
        std::fprintf(stderr, "fdb_bench: no samples under %s\n",
                     args.corpus_dir.c_str());
        return 1;
    }

    fs::create_directories(args.out_dir);

    size_t ok = 0, err = 0;
    long long sum_ttft = 0;

    for (const auto& s : samples) {
        // Reload input WAV briefly just for duration recording. The
        // process_one_sample call below loads it again — small cost, keeps
        // the helper's signature minimal.
        fdb_bench::WavData wav;
        double in_dur = 0.0;
        if (fdb_bench::load_wav_mono_pcm16(s.input_wav_path, &wav)) {
            in_dur = wav.sample_rate > 0
                ? static_cast<double>(wav.samples.size()) / wav.sample_rate
                : 0.0;
        }

        SampleResult r;
        try {
            r = process_one_sample(s, args.out_dir, *stt, *tts, vad, llm,
                                   is_mock_stt);
        } catch (const std::exception& ex) {
            r.error = ex.what();
        }

        if (r.ok) {
            ok++;
            sum_ttft += r.ttft_first_audio_from_speech_end_ms;
        } else {
            err++;
            std::fprintf(stderr, "fdb_bench: [%s/%s] %s\n",
                         s.category_dir_name.c_str(),
                         s.sample_id.c_str(),
                         r.error.empty() ? "(no error message)" : r.error.c_str());
        }

        std::string json_out = args.out_dir + "/" + s.category_dir_name +
                               "__" + s.sample_id + ".json";
        write_sample_json(json_out, s, r, args.stt, args.tts, args.llm_model,
                          in_dur);
    }

    double avg_ttft = ok > 0 ? static_cast<double>(sum_ttft) / ok : 0.0;
    std::fprintf(stdout,
        "fdb_bench: samples_ok=%zu errors=%zu avg_ttft_ms=%.1f out_dir=%s\n",
        ok, err, avg_ttft, args.out_dir.c_str());
    return err == 0 ? 0 : 1;
}
