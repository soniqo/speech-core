// CLI-boundary tests for speech_voxcpm2_clone.
//
// These spawn the REAL executable, so the OS command-line crossing is part of
// the test surface. The field failure that motivated them: on Windows,
// `char** argv` is converted through the active code page (e.g. 1252), which
// turned a Devanagari text argument into a row of '?' — the model spoke
// fifteen question marks while every in-process test (compiled-in strings,
// stdin protocols) passed. On Windows the child is launched via
// CreateProcessW with a wide command line, exactly what cmd/PowerShell do.
//
// Layers, cheap to expensive:
//   1. usage / exit code                — no models needed
//   2. UTF-8 argv echo                  — no models needed: a Devanagari
//      reference path must round-trip into the CLI's error message intact
//   3. clone → ASR roundtrip            — needs SPEECH_LITERT_MODEL_DIR with
//      the VoxCPM2 bundle; ASR coverage additionally needs Omnilingual
//      (omnilingual-ctc-300m.tflite + tokenizer.model). Skips cleanly.
//
// The roundtrip asserts duration bounds and word coverage. Both historical
// failure modes violate them: argv mojibake → non-speech audio, empty
// transcript; the old flat 32-step stop floor → ~6 s render for a 2 s line.

#include "speech_core/audio/resampler.h"
#include "speech_core/models/litert_omnilingual_stt.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

int failures = 0;

#define REQUIRE(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d)\n", #cond, __LINE__); \
        ++failures; \
        return; \
    } \
} while (0)

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return s;
}

// ---------------------------------------------------------------------------
// Locating the CLI binary: same directory as this test executable.
// ---------------------------------------------------------------------------

std::string self_dir() {
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return ".";
    std::wstring w(buf, n);
    size_t cut = w.find_last_of(L"\\/");
    w = (cut == std::wstring::npos) ? L"." : w.substr(0, cut);
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string s(len > 0 ? len - 1 : 0, '\0');
    if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), len, nullptr, nullptr);
    return s;
#else
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    std::string s(buf);
    size_t cut = s.find_last_of('/');
    return (cut == std::string::npos) ? "." : s.substr(0, cut);
#endif
}

std::string cli_path() {
#if defined(_WIN32)
    return self_dir() + "\\speech_voxcpm2_clone.exe";
#else
    return self_dir() + "/speech_voxcpm2_clone";
#endif
}

// ---------------------------------------------------------------------------
// Spawning the CLI with UTF-8 args. On Windows the command line is built and
// passed WIDE — argv conversion inside the child is what we're testing.
// ---------------------------------------------------------------------------

#if defined(_WIN32)
std::wstring widen(const std::string& utf8) {
    int len = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    std::wstring w(len > 0 ? len - 1 : 0, L'\0');
    if (len > 1) MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, w.data(), len);
    return w;
}

// Standard argv quoting: quote every argument; double backslash runs that
// precede a quote; escape embedded quotes. Empty args become "".
std::wstring quote_arg(const std::wstring& a) {
    std::wstring out = L"\"";
    size_t bs = 0;
    for (wchar_t c : a) {
        if (c == L'\\') {
            ++bs;
        } else if (c == L'"') {
            out.append(bs * 2 + 1, L'\\');
            out += L'"';
            bs = 0;
        } else {
            out.append(bs, L'\\');
            out += c;
            bs = 0;
        }
    }
    out.append(bs * 2, L'\\');
    out += L'"';
    return out;
}

int run_cli(const std::vector<std::string>& args, const std::string& stderr_path) {
    std::wstring cmd = quote_arg(widen(cli_path()));
    for (const auto& a : args) {
        cmd += L' ';
        cmd += quote_arg(widen(a));
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE err = CreateFileW(widen(stderr_path).c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                             &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (err == INVALID_HANDLE_VALUE) return -1;

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = err;
    si.hStdError = err;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi{};
    std::wstring mutable_cmd = cmd;  // CreateProcessW may modify the buffer
    BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr,
                             /*inherit=*/TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(err);
    if (!ok) return -1;
    WaitForSingleObject(pi.hProcess, 10 * 60 * 1000);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(code);
}
#else
std::string shell_quote(const std::string& a) {
    std::string out = "'";
    for (char c : a) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

int run_cli(const std::vector<std::string>& args, const std::string& stderr_path) {
    std::string cmd = shell_quote(cli_path());
    for (const auto& a : args) {
        cmd += ' ';
        cmd += shell_quote(a);
    }
    cmd += " >" + shell_quote(stderr_path) + " 2>&1";
    int rc = std::system(cmd.c_str());
    return (rc == -1) ? -1 : WEXITSTATUS(rc);
}
#endif

// ---------------------------------------------------------------------------
// Minimal mono PCM-16 WAV reader (mirrors the loaders elsewhere in tests/).
// ---------------------------------------------------------------------------

bool load_wav_mono(const std::string& path, std::vector<float>& out, int& rate) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char id[4];
    uint32_t sz = 0;
    f.read(id, 4);
    f.read(reinterpret_cast<char*>(&sz), 4);
    f.read(id, 4);
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t r = 0;
    while (f.read(id, 4)) {
        f.read(reinterpret_cast<char*>(&sz), 4);
        if (!std::memcmp(id, "fmt ", 4)) {
            f.read(reinterpret_cast<char*>(&fmt), 2);
            f.read(reinterpret_cast<char*>(&ch), 2);
            f.read(reinterpret_cast<char*>(&r), 4);
            f.seekg(6, std::ios::cur);
            f.read(reinterpret_cast<char*>(&bits), 2);
            if (sz > 16) f.seekg(sz - 16, std::ios::cur);
        } else if (!std::memcmp(id, "data", 4)) {
            if (fmt != 1 || bits != 16 || ch == 0) return false;
            size_t n = sz / 2;
            std::vector<int16_t> pcm(n);
            f.read(reinterpret_cast<char*>(pcm.data()), sz);
            size_t frames = n / ch;
            out.resize(frames);
            for (size_t i = 0; i < frames; ++i) {
                int acc = 0;
                for (uint16_t c = 0; c < ch; ++c) acc += pcm[i * ch + c];
                out[i] = static_cast<float>(acc) / (ch * 32768.0f);
            }
            rate = static_cast<int>(r);
            return true;
        } else {
            f.seekg(sz, std::ios::cur);
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Cross-script token matching for the ASR assertion. Omnilingual decodes
// Hindustani in mixed Devanagari/Urdu script (no language pin), so "राहुल"
// may come back as "راہول". Both scripts are phonetic: fold each token to a
// consonant-class skeleton, then compare with a 1-edit tolerance. Mirrors
// the grader in speech-studio (src-tauri lib.rs).
// ---------------------------------------------------------------------------

std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp = 0;
        int len = 1;
        if ((c & 0x80) == 0x00) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        if (i + static_cast<size_t>(len) > s.size()) break;
        for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        cps.push_back(cp);
        i += static_cast<size_t>(len);
    }
    return cps;
}

std::string fold_skeleton(const std::string& token) {
    const std::vector<uint32_t> cps = codepoints(token);
    std::string out;
    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t c = cps[i];
        // Decomposed nukta forms (base + U+093C) change the consonant class.
        if (i + 1 < cps.size() && cps[i + 1] == 0x093C) {
            char k = 0;
            switch (c) {
                case 0x0921: case 0x0922: k = 'r'; break;  // ड़ ढ़
                case 0x091C: k = 'j'; break;               // ज़
                case 0x092B: k = 'f'; break;               // फ़
                case 0x0915: case 0x0916: k = 'k'; break;  // क़ ख़
                case 0x0917: k = 'g'; break;               // ग़
                default: break;
            }
            if (k) { out += k; ++i; continue; }
        }
        char k = 0;
        switch (c) {
            // Devanagari consonants (and precomposed nukta forms)
            case 0x0915: case 0x0916: case 0x0958: case 0x0959: k = 'k'; break;
            case 0x0917: case 0x0918: case 0x095A: k = 'g'; break;
            case 0x091A: case 0x091B: k = 'c'; break;
            case 0x091C: case 0x091D: case 0x095B: k = 'j'; break;
            case 0x091F: case 0x0920: case 0x0924: case 0x0925: k = 't'; break;
            case 0x0921: case 0x0922: case 0x0926: case 0x0927: k = 'd'; break;
            case 0x095C: case 0x095D: k = 'r'; break;
            case 0x0923: case 0x0928: case 0x0919: case 0x091E: k = 'n'; break;
            case 0x092A: case 0x092B: k = 'p'; break;
            case 0x095E: k = 'f'; break;
            case 0x092C: case 0x092D: k = 'b'; break;
            case 0x092E: k = 'm'; break;
            case 0x092F: k = 'y'; break;
            case 0x0930: k = 'r'; break;
            case 0x0932: k = 'l'; break;
            case 0x0935: k = 'v'; break;
            case 0x0936: case 0x0937: case 0x0938: k = 's'; break;
            case 0x0939: k = 'h'; break;
            // Urdu/Arabic consonants
            case 0x06A9: case 0x0642: case 0x062E: k = 'k'; break;
            case 0x06AF: case 0x063A: k = 'g'; break;
            case 0x0686: k = 'c'; break;
            case 0x062C: case 0x0632: case 0x0630: case 0x0636: case 0x0638:
            case 0x0698: k = 'j'; break;
            case 0x062A: case 0x0679: case 0x0637: k = 't'; break;
            case 0x062F: case 0x0688: k = 'd'; break;
            case 0x0691: k = 'r'; break;
            case 0x0646: k = 'n'; break;
            case 0x067E: k = 'p'; break;
            case 0x0641: k = 'f'; break;
            case 0x0628: k = 'b'; break;
            case 0x0645: k = 'm'; break;
            case 0x06CC: case 0x0626: k = 'y'; break;
            case 0x0631: k = 'r'; break;
            case 0x0644: k = 'l'; break;
            case 0x0648: k = 'v'; break;
            case 0x0633: case 0x0634: case 0x0635: case 0x062B: k = 's'; break;
            case 0x06C1: case 0x0647: case 0x062D: case 0x06C2: k = 'h'; break;
            default: break;
        }
        if (k) { out += k; continue; }
        if (c < 0x80 && std::isalnum(static_cast<int>(c))) {
            out += static_cast<char>(std::tolower(static_cast<int>(c)));
        }
        // Everything else (vowels, matras, nasalisation ں, aspiration ھ,
        // tashkeel, punctuation) drops.
    }
    return out;
}

size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> prev(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 0; i < a.size(); ++i) {
        std::vector<size_t> cur(b.size() + 1);
        cur[0] = i + 1;
        for (size_t j = 0; j < b.size(); ++j) {
            size_t cost = (a[i] == b[j]) ? 0 : 1;
            cur[j + 1] = std::min(std::min(prev[j] + cost, prev[j + 1] + 1), cur[j] + 1);
        }
        prev = cur;
    }
    return prev[b.size()];
}

bool tokens_match(const std::string& a, const std::string& b) {
    if (a == b) return true;
    const std::string fa = fold_skeleton(a);
    const std::string fb = fold_skeleton(b);
    if (fa.empty() || fb.empty()) return false;
    if (fa == fb) return true;
    const size_t min_len = std::min(fa.size(), fb.size());
    const size_t tol = (min_len >= 6) ? 2 : (min_len >= 2 ? 1 : 0);
    return edit_distance(fa, fb) <= tol;
}

// Whitespace/danda tokenizer (ASCII punctuation also separates).
std::vector<std::string> tokenize_words(const std::string& s) {
    std::vector<std::string> toks;
    std::string cur;
    const std::vector<uint32_t> cps = codepoints(s);
    for (uint32_t cp : cps) {
        bool sep = (cp <= 0x20) || cp == 0x0964 || cp == 0x0965;
        if (cp < 0x80 && std::ispunct(static_cast<int>(cp)) && cp != '\'') sep = true;
        if (sep) {
            if (!cur.empty()) { toks.push_back(cur); cur.clear(); }
            continue;
        }
        // re-encode the codepoint as UTF-8
        if (cp < 0x80) cur += static_cast<char>(cp);
        else if (cp < 0x800) {
            cur += static_cast<char>(0xC0 | (cp >> 6));
            cur += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            cur += static_cast<char>(0xE0 | (cp >> 12));
            cur += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            cur += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            cur += static_cast<char>(0xF0 | (cp >> 18));
            cur += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            cur += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            cur += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    if (!cur.empty()) toks.push_back(cur);
    return toks;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

std::string test_data_path(const char* name) {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/" + name;
#else
    return std::string("tests/data/") + name;
#endif
}

std::string tmp_path(const char* name) {
    return self_dir() + "/" + name;
}

// ---------------------------------------------------------------------------
// 1. No args → usage on stderr, exit code 2.
// ---------------------------------------------------------------------------

void test_cli_usage() {
    std::printf("  test_cli_usage ...\n");
    const std::string err_file = tmp_path("cli_test_usage.txt");
    int code = run_cli({}, err_file);
    REQUIRE(code == 2);
    const std::string err = read_file(err_file);
    REQUIRE(err.find("usage:") != std::string::npos);
    std::remove(err_file.c_str());
}

// ---------------------------------------------------------------------------
// 2. UTF-8 argv integrity, no models needed. A Devanagari reference path
// must be echoed back intact in the error message. Under the historical
// active-code-page conversion this arrived (and echoed) as "????-...".
// ---------------------------------------------------------------------------

void test_cli_utf8_argv_echo() {
    std::printf("  test_cli_utf8_argv_echo ...\n");
    const std::string err_file = tmp_path("cli_test_utf8.txt");
    const std::string devanagari_ref = "मेरा-missing-fixture.wav";
    // bundle_dir is optional (defaults to the model cache); omit it so the
    // Devanagari path is argv[1] = ref.wav. The fixture is missing, so the ref
    // load fails and echoes the path before any model is touched — no models
    // needed. (A non-existent first arg can't double as the bundle: the CLI
    // disambiguates by std::filesystem::is_directory, so it'd be read as ref.)
    int code = run_cli({devanagari_ref, "hello", tmp_path("cli_test_out.wav")},
                       err_file);
    REQUIRE(code == 1);
    const std::string err = read_file(err_file);
    REQUIRE(err.find("मेरा-missing-fixture.wav") != std::string::npos);
    REQUIRE(err.find("????") == std::string::npos);
    std::remove(err_file.c_str());
}

// ---------------------------------------------------------------------------
// 3. Clone → ASR roundtrip through the real process boundary.
//
// The bare CLI has no take grader (the studio retries graded seeds), so a
// single seed can legitimately land on an AR-overshoot take — measured: the
// same 4-word line rendered 1.92 s on one seed and 6.56 s (babble tail) on
// another, depending on reference conditioning. The contract this test pins
// down is "a clean take exists within a small seed ladder": every historical
// CLI-layer regression fails ALL seeds (argv mojibake → empty transcript on
// every take; the old flat 32-step stop floor → ≥5.1 s on every take).
// ---------------------------------------------------------------------------

void test_cli_clone_roundtrip(const std::string& dir) {
    const char* names[] = {"voxcpm2-text-prefill.tflite", "voxcpm2-token-step.tflite",
                           "voxcpm2-audio-encoder.tflite", "voxcpm2-audio-decoder.tflite",
                           "tokenizer.json"};
    for (const char* n : names) {
        if (!file_exists(dir + "/" + n)) {
            std::printf("  [skip] clone roundtrip: %s missing in %s\n", n, dir.c_str());
            return;
        }
    }
    std::string ref = test_data_path("test_hindi_ref.wav");
    if (!file_exists(ref)) ref = test_data_path("test_audio.wav");
    if (!file_exists(ref)) {
        std::printf("  [skip] clone roundtrip: no reference fixture\n");
        return;
    }
    std::printf("  test_cli_clone_roundtrip ... (ref=%s)\n", ref.c_str());

    const std::string stt_model = dir + "/omnilingual-ctc-300m.tflite";
    const std::string stt_tok = dir + "/tokenizer.model";
    const bool have_asr = file_exists(stt_model) && file_exists(stt_tok);
    std::unique_ptr<speech_core::LiteRTOmnilingualStt> stt;
    if (have_asr) {
        stt = std::make_unique<speech_core::LiteRTOmnilingualStt>(stt_model, stt_tok,
                                                                  /*hw_accel=*/false);
    } else {
        std::printf("    [skip-asr] omnilingual files missing — duration/level checks only\n");
    }

    const std::string out_wav = tmp_path("cli_test_clone.wav");
    const std::string err_file = tmp_path("cli_test_clone_err.txt");
    const std::string text = "मेरा नाम राहुल है।";
    const std::vector<std::string> target = tokenize_words(text);
    const char* seeds[] = {"1000", "1001", "1002"};

    bool clean_take = false;
    for (const char* seed : seeds) {
        // Empty instruction is the default; passed explicitly so the
        // max_steps and seed arguments stay positional.
        int code = run_cli({dir, ref, text, out_wav, "", "256", seed}, err_file);
        if (code != 0) {
            std::fprintf(stderr, "  FAIL: CLI exited %d (seed=%s); stderr:\n%s\n", code, seed,
                         read_file(err_file).c_str());
            ++failures;
            return;
        }

        std::vector<float> audio;
        int rate = 0;
        REQUIRE(load_wav_mono(out_wav, audio, rate));
        REQUIRE(rate == 48000);
        const double dur = static_cast<double>(audio.size()) / rate;
        double rms_sq = 0.0;
        for (float s : audio) rms_sq += static_cast<double>(s) * s;
        const double rms = std::sqrt(rms_sq / std::max<size_t>(audio.size(), 1));

        int matched = -1;
        std::string heard_text = "(asr off)";
        if (have_asr) {
            std::vector<float> a16 = (rate == 16000)
                ? audio
                : speech_core::Resampler::resample(audio.data(), audio.size(), rate, 16000);
            const auto res = stt->transcribe(a16.data(), a16.size(), 16000);
            heard_text = res.text;
            const std::vector<std::string> heard = tokenize_words(res.text);
            matched = 0;
            for (const auto& t : target) {
                for (const auto& h : heard) {
                    if (tokens_match(t, h)) { ++matched; break; }
                }
            }
        }
        std::printf("    [seed=%s dur=%.2fs rms=%.4f cov=%d/%zu asr=\"%s\"]\n",
                    seed, dur, rms, matched, target.size(), heard_text.c_str());

        // A 4-word line reads in ~1.5-2.5 s; ≤4.5 s rejects babble-tail takes
        // and both historical regressions. Coverage: cross-script ASR is
        // fuzzy, so require half the words rather than all four — the
        // mojibake regression transcribes to an EMPTY string (0 matches).
        const bool dur_ok = dur >= 1.0 && dur <= 4.5;
        const bool rms_ok = rms >= 0.01;
        const bool cov_ok = !have_asr || matched * 2 >= static_cast<int>(target.size());
        if (dur_ok && rms_ok && cov_ok) {
            clean_take = true;
            break;
        }
    }
    REQUIRE(clean_take);

    std::remove(out_wav.c_str());
    std::remove(err_file.c_str());
}

}  // namespace

int main() {
    std::printf("test_litert_voxcpm2_clone_cli (cli=%s)\n", cli_path().c_str());
    if (!file_exists(cli_path())) {
        std::printf("  [skip] speech_voxcpm2_clone binary not found next to the test\n");
        return 0;
    }
    test_cli_usage();
    test_cli_utf8_argv_echo();

    const char* env = std::getenv("SPEECH_LITERT_MODEL_DIR");
    if (env && *env) {
        test_cli_clone_roundtrip(env);
    } else {
        std::printf("  [skip] clone roundtrip: SPEECH_LITERT_MODEL_DIR not set\n");
    }

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all CLI tests passed\n");
    return 0;
}
