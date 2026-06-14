// Unit tests for the SeamlessM4T / Wav2Vec2-BERT log-mel front-end used by the
// Sidon restorer. The pure-DSP front-end lives in speech_core (no ONNX), so
// these run in the default build. Golden values were captured from the
// canonical transformers.SeamlessM4TFeatureExtractor (facebook/w2v-bert-2.0)
// on the exact synthetic signal reproduced in make_signal() below.

#include "speech_core/audio/seamless_fbank.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace speech_core;

static constexpr double kPi = 3.14159265358979323846;

// Deterministic 440 Hz + 1000 Hz sine mix, 16 kHz, 0.5 s (8000 samples).
static std::vector<float> make_signal() {
    const int sr = 16000, N = 8000;
    std::vector<float> y(N);
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / sr;
        y[i] = static_cast<float>(0.5 * std::sin(2.0 * kPi * 440.0 * t)
                                + 0.3 * std::sin(2.0 * kPi * 1000.0 * t));
    }
    return y;
}

static bool approx(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Shape: 8000 samples -> num_frames = 1 + floor((8000-400)/160) = 48 -> T = 24.
void test_shape() {
    auto y = make_signal();
    int T = 0;
    auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
    assert(T == 24);
    assert(f.size() == static_cast<size_t>(T) * 160);
    printf("  PASS: test_shape (T=%d, %zu floats)\n", T, f.size());
}

// Numerical parity vs the HuggingFace reference extractor (golden values).
// Tolerance accommodates the float32 radix-2 FFT vs HF's float64 spectrogram.
void test_golden_values() {
    auto y = make_signal();
    int T = 0;
    auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
    assert(T == 24);

    // Row 0, first 8 mel-stacked features.
    const float row0[8] = {-0.64536f, -0.83175f, -0.93238f, -0.71677f,
                            1.51440f, 0.75650f, 0.08107f, -0.18327f};
    for (int j = 0; j < 8; ++j) {
        assert(approx(f[j], row0[j], 2e-3f));
    }
    // Last stacked row (index 23), first 4 features.
    const float row23[4] = {0.16382f, 0.33804f, 0.42365f, 0.21589f};
    for (int j = 0; j < 4; ++j) {
        assert(approx(f[static_cast<size_t>(23) * 160 + j], row23[j], 2e-3f));
    }
    printf("  PASS: test_golden_values (max golden tol 2e-3)\n");
}

// CMVN invariant: each of the 80 mel bins, across the (pre-stack) time frames,
// is normalised to ~zero mean. Because the layout concatenates frame 2t and
// frame 2t+1, mel bin m appears at columns m and 80+m; averaging both halves
// over all stacked rows recovers the per-bin mean over all original frames.
void test_cmvn_zero_mean() {
    auto y = make_signal();
    int T = 0;
    auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
    assert(T > 0);
    float max_abs_mean = 0.0f;
    for (int m = 0; m < 80; ++m) {
        double sum = 0.0;
        for (int t = 0; t < T; ++t) {
            sum += f[static_cast<size_t>(t) * 160 + m];        // even frame
            sum += f[static_cast<size_t>(t) * 160 + 80 + m];   // odd frame
        }
        const double mean = sum / (2.0 * T);
        max_abs_mean = std::max(max_abs_mean, static_cast<float>(std::fabs(mean)));
    }
    // Exactly zero in exact arithmetic; allow float slack.
    assert(max_abs_mean < 1e-3f);
    printf("  PASS: test_cmvn_zero_mean (max |bin mean|=%.2e)\n", max_abs_mean);
}

// Guard rails: wrong sample rate throws; sub-frame clips return empty.
void test_edge_cases() {
    auto y = make_signal();
    int T = -1;

    bool threw = false;
    try {
        audio::seamless_log_mel(y.data(), y.size(), 44100, T);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Clip shorter than one frame -> empty, T = 0.
    T = -1;
    std::vector<float> tiny(100, 0.1f);
    auto f = audio::seamless_log_mel(tiny.data(), tiny.size(), 16000, T);
    assert(f.empty() && T == 0);

    // Null input -> empty.
    T = -1;
    auto f2 = audio::seamless_log_mel(nullptr, 0, 16000, T);
    assert(f2.empty() && T == 0);

    printf("  PASS: test_edge_cases\n");
}

int main() {
    printf("test_seamless_fbank:\n");
    test_shape();
    test_golden_values();
    test_cmvn_zero_mean();
    test_edge_cases();
    printf("All seamless_fbank tests passed.\n");
    return 0;
}
