// Smoke test for LiteRTChatterboxTts + the C ABI. Skips (passes) when no bundle
// is available, so CI without the model stays green. Point it at a bundle via the
// SPEECH_CORE_CHATTERBOX_BUNDLE env var.

#include "speech_core/chatterbox_c.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> g_wav;
static void on_chunk(const float* s, size_t n, bool, void*) { if (s && n) g_wav.assign(s, s + n); }

int main() {
    const char* dir = std::getenv("SPEECH_CORE_CHATTERBOX_BUNDLE");
    if (!dir || !*dir) {
        std::printf("[skip] SPEECH_CORE_CHATTERBOX_BUNDLE not set\n");
        return 0;
    }
    sc_chatterbox_t c = sc_chatterbox_create(dir);
    if (!c) { std::fprintf(stderr, "create failed\n"); return 1; }
    if (sc_chatterbox_output_sample_rate(c) != 24000) { std::fprintf(stderr, "bad sample rate\n"); return 1; }
    sc_chatterbox_set_seed(c, 1234);
    int rc = sc_chatterbox_synthesize(c, "This is a test.", "en", on_chunk, nullptr);
    if (rc != 0) { std::fprintf(stderr, "synthesize: %s\n", sc_chatterbox_last_error(c)); sc_chatterbox_destroy(c); return 1; }
    int ok = !g_wav.empty() && sc_chatterbox_tokens_generated(c) > 0;
    std::printf("tokens=%d wav=%zu (%.2fs) -> %s\n", sc_chatterbox_tokens_generated(c), g_wav.size(),
                g_wav.size() / 24000.0, ok ? "PASS" : "FAIL");
    sc_chatterbox_destroy(c);
    return ok ? 0 : 1;
}
