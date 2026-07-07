// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

// Indic-Mio LiteRT end-to-end smoke: bundle load → Hindi synthesis with a
// suffix emotion tag → audio sanity, then a cloned take conditioned on
// tests/data/test_audio.wav. Skips unless SPEECH_CORE_INDIC_MIO_BUNDLE points
// at a bundle directory (soniqo/Indic-Mio-LiteRT layout).

#include "speech_core/indic_mio_c.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<float> g_wav;

void on_chunk(const float* samples, size_t length, bool /*is_final*/, void* /*ctx*/) {
    if (samples && length) g_wav.insert(g_wav.end(), samples, samples + length);
}

double rms(const std::vector<float>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (float x : v) s += double(x) * x;
    return std::sqrt(s / v.size());
}

// Minimal 16-bit PCM WAV reader (mono or first channel), enough for the
// bundled test clip.
std::vector<float> read_wav_16(const std::string& path, int& sample_rate) {
    std::ifstream f(path, std::ios::binary);
    assert(f && "test wav missing");
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
    assert(data.size() > 44);
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0;
    size_t pos = 12, data_off = 0, data_len = 0;
    while (pos + 8 <= data.size()) {
        const std::string id(data.data() + pos, 4);
        uint32_t sz = 0;
        std::memcpy(&sz, data.data() + pos + 4, 4);
        if (id == "fmt ") {
            std::memcpy(&channels, data.data() + pos + 10, 2);
            std::memcpy(&rate, data.data() + pos + 12, 4);
            std::memcpy(&bits, data.data() + pos + 22, 2);
        } else if (id == "data") {
            data_off = pos + 8;
            data_len = sz;
            break;
        }
        pos += 8 + sz + (sz & 1);
    }
    assert(bits == 16 && channels >= 1 && data_off > 0);
    sample_rate = static_cast<int>(rate);
    const size_t frames = data_len / (2 * channels);
    std::vector<float> out(frames);
    for (size_t i = 0; i < frames; ++i) {
        int16_t s = 0;
        std::memcpy(&s, data.data() + data_off + (i * channels) * 2, 2);
        out[i] = static_cast<float>(s) / 32768.0f;
    }
    return out;
}

}  // namespace

int main() {
    const char* bundle = std::getenv("SPEECH_CORE_INDIC_MIO_BUNDLE");
    if (!bundle || !*bundle) {
        std::printf("[skip] SPEECH_CORE_INDIC_MIO_BUNDLE not set\n");
        return 0;
    }

    sc_indic_mio_t m = sc_indic_mio_create(bundle);
    if (!m) { std::fprintf(stderr, "create failed\n"); return 1; }
    assert(sc_indic_mio_output_sample_rate(m) == 24000);
    assert(sc_indic_mio_max_text_tokens(m) == 64);
    assert(sc_indic_mio_has_download_support() == true ||
           sc_indic_mio_has_download_support() == false);  // linkage smoke

    sc_indic_mio_set_seed(m, 1234);

    // --- Default voice, suffix emotion tag.
    g_wav.clear();
    int rc = sc_indic_mio_synthesize(
        m, "नमस्ते, आज मौसम बहुत अच्छा है। <happy>", on_chunk, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "synthesize: %s\n", sc_indic_mio_last_error(m));
        sc_indic_mio_destroy(m);
        return 1;
    }
    const double dur = g_wav.size() / 24000.0;
    const double level = rms(g_wav);
    std::printf("default voice: tokens=%d wav=%zu (%.2fs) rms=%.4f eos=%d seed=%u\n",
                sc_indic_mio_tokens_generated(m), g_wav.size(), dur, level,
                sc_indic_mio_stopped_on_eos(m) ? 1 : 0, sc_indic_mio_seed_used(m));
    // Optional artifact for listening / host-side ASR grading.
    if (const char* dump = std::getenv("SPEECH_CORE_INDIC_MIO_DUMP_WAV");
        dump && *dump) {
        std::ofstream w(dump, std::ios::binary);
        auto le32 = [&](uint32_t v) { w.write(reinterpret_cast<char*>(&v), 4); };
        auto le16 = [&](uint16_t v) { w.write(reinterpret_cast<char*>(&v), 2); };
        const uint32_t bytes = static_cast<uint32_t>(g_wav.size() * 2);
        w.write("RIFF", 4); le32(36 + bytes); w.write("WAVEfmt ", 8);
        le32(16); le16(1); le16(1); le32(24000); le32(48000); le16(2); le16(16);
        w.write("data", 4); le32(bytes);
        for (float s : g_wav) {
            const int16_t q = static_cast<int16_t>(
                std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
            w.write(reinterpret_cast<const char*>(&q), 2);
        }
        std::printf("dumped %s\n", dump);
    }
    assert(!g_wav.empty());
    assert(dur > 0.3 && dur < 20.0);
    assert(level > 0.005);
    assert(sc_indic_mio_tokens_generated(m) > 0);
    assert(g_wav.size() ==
           static_cast<size_t>(sc_indic_mio_tokens_generated(m)) * 960);

    // --- Cloned voice from the bundled reference clip.
    {
        const std::string wav_path =
            std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
        int sr = 0;
        const auto ref = read_wav_16(wav_path, sr);
        assert(!ref.empty());
        rc = sc_indic_mio_set_reference(m, ref.data(), ref.size(), sr);
        if (rc != 0) {
            std::fprintf(stderr, "set_reference: %s\n", sc_indic_mio_last_error(m));
            sc_indic_mio_destroy(m);
            return 1;
        }
        g_wav.clear();
        rc = sc_indic_mio_synthesize(m, "यह एक छोटा परीक्षण है। <sad>", on_chunk, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "synthesize(cloned): %s\n", sc_indic_mio_last_error(m));
            sc_indic_mio_destroy(m);
            return 1;
        }
        std::printf("cloned voice: tokens=%d wav=%zu (%.2fs) rms=%.4f\n",
                    sc_indic_mio_tokens_generated(m), g_wav.size(),
                    g_wav.size() / 24000.0, rms(g_wav));
        assert(!g_wav.empty());
        assert(rms(g_wav) > 0.005);
        sc_indic_mio_clear_reference(m);
    }

    // --- Over-long prompt is rejected with a useful error, not truncated.
    {
        std::string longtext;
        for (int i = 0; i < 60; ++i) longtext += "नमस्ते ";
        g_wav.clear();
        rc = sc_indic_mio_synthesize(m, longtext.c_str(), on_chunk, nullptr);
        assert(rc != 0);
        assert(std::strlen(sc_indic_mio_last_error(m)) > 0);
        std::printf("overlong prompt: rejected as expected (%s)\n",
                    sc_indic_mio_last_error(m));
    }

    sc_indic_mio_destroy(m);
    std::printf("indic-mio e2e: PASS\n");
    return 0;
}
