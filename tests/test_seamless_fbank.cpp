// Unit tests for the SeamlessM4T / Wav2Vec2-BERT log-mel front-end used by the
// Sidon restorer. The pure-DSP front-end lives in speech_core (no ONNX), so
// these run in the default build. Golden values were captured from the
// canonical transformers.SeamlessM4TFeatureExtractor (facebook/w2v-bert-2.0)
// on the exact synthetic signal reproduced in make_signal() below.

// Force-enable asserts even under Release / RelWithDebInfo builds (the
// documented build sets CMAKE_BUILD_TYPE=Release, which defines NDEBUG and
// would otherwise compile every assert() below to a no-op).
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/audio/seamless_fbank.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
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

// The front-end needs ≥2 raw frames to emit one ×2-stacked output frame.
// Exactly one raw frame (400 samples) yields num_frames == 1, which must
// return empty / T == 0; the first length that gives two raw frames
// (400 + 160 = 560 samples) must produce exactly one stacked frame.
void test_stack_boundary() {
    int T = -1;
    std::vector<float> one_frame(400, 0.2f);
    auto f0 = audio::seamless_log_mel(one_frame.data(), one_frame.size(), 16000, T);
    assert(f0.empty() && T == 0);

    T = -1;
    std::vector<float> two_frames(560, 0.2f);   // 1 + (560-400)/160 = 2 raw frames
    auto f1 = audio::seamless_log_mel(two_frames.data(), two_frames.size(), 16000, T);
    assert(T == 1);
    assert(f1.size() == 160u);

    // 720 samples -> 3 raw frames -> trailing odd frame dropped -> still T == 1.
    T = -1;
    std::vector<float> three_frames(720, 0.2f);
    auto f2 = audio::seamless_log_mel(three_frames.data(), three_frames.size(), 16000, T);
    assert(T == 1);
    assert(f2.size() == 160u);

    printf("  PASS: test_stack_boundary\n");
}

// Output shape T == floor(num_raw_frames / 2) for a sweep of input lengths,
// where num_raw_frames = 1 + floor((N - 400) / 160). Each output frame is
// exactly 160 floats.
void test_shape_sweep() {
    struct Case { int samples; int expect_T; };
    const Case cases[] = {
        {560,   1},    // 2 raw frames
        {1000,  2},    // 1 + (600/160)=4 raw -> 2
        {4000,  11},   // 1 + (3600/160)=23 raw -> 11
        {8000,  24},   // matches make_signal()
        {16000, 49},   // 1 + (15600/160)=98 raw -> 49
    };
    for (const auto& c : cases) {
        int T = -1;
        std::vector<float> y(c.samples, 0.0f);
        for (int i = 0; i < c.samples; ++i) {
            y[i] = static_cast<float>(0.3 * std::sin(2.0 * kPi * 300.0 * i / 16000.0));
        }
        auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
        assert(T == c.expect_T);
        assert(f.size() == static_cast<size_t>(c.expect_T) * 160);
    }
    printf("  PASS: test_shape_sweep\n");
}

// Determinism: identical input -> bit-identical output (the front-end caches
// immutable tables in a magic static; a second call must not perturb them).
void test_determinism() {
    auto y = make_signal();
    int T1 = 0, T2 = 0;
    auto a = audio::seamless_log_mel(y.data(), y.size(), 16000, T1);
    auto b = audio::seamless_log_mel(y.data(), y.size(), 16000, T2);
    assert(T1 == T2);
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        assert(a[i] == b[i]);  // exact, not approximate
    }
    printf("  PASS: test_determinism\n");
}

// CMVN unit-variance invariant: for each mel bin that carries real energy
// spread across frames, the post-CMVN sample variance (ddof = 1) over all
// original (pre-stack) frames is ~1. The stacked layout puts bin m of even
// frame 2t at column m and bin m of odd frame 2t+1 at column 80+m, so iterating
// both halves over all T stacked rows reconstructs the full per-bin frame
// sequence (length 2T == num_frames when even).
//
// We drive the front-end with a deterministic broadband signal (summed
// harmonics + a reproducible PRNG hiss) so every mel bin has genuine
// frame-to-frame variation. With a pure tone many bins sit on the mel floor
// (constant -> ~0 variance, and the kNormEps guard keeps them ~0), which is
// correct behaviour but makes the unit-variance check inapplicable to them —
// hence the broadband excitation here.
void test_cmvn_unit_variance() {
    const int N = 8000;       // -> 48 raw frames (even) -> T = 24
    std::vector<float> y(N);
    uint32_t s = 0x12345678u; // deterministic LCG
    for (int i = 0; i < N; ++i) {
        const double t = static_cast<double>(i) / 16000.0;
        double v = 0.0;
        for (int h = 1; h <= 12; ++h)
            v += (0.25 / h) * std::sin(2.0 * kPi * (180.0 * h) * t);
        s = s * 1664525u + 1013904223u;
        const double hiss = (static_cast<double>(s >> 8) / 16777216.0 - 0.5) * 0.2;
        y[i] = static_cast<float>(v + hiss);
    }
    int T = 0;
    auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
    assert(T == 24);
    const int n = 2 * T;      // reconstructed raw-frame count
    float max_var_err = 0.0f;
    int checked = 0;
    for (int m = 0; m < 80; ++m) {
        double sum = 0.0;
        for (int t = 0; t < T; ++t) {
            sum += f[static_cast<size_t>(t) * 160 + m];
            sum += f[static_cast<size_t>(t) * 160 + 80 + m];
        }
        const double mu = sum / n;
        double var = 0.0;
        for (int t = 0; t < T; ++t) {
            double d0 = f[static_cast<size_t>(t) * 160 + m] - mu;
            double d1 = f[static_cast<size_t>(t) * 160 + 80 + m] - mu;
            var += d0 * d0 + d1 * d1;
        }
        var /= (n - 1);   // sample variance, ddof = 1 — matches the front-end
        // Only bins that actually got normalised toward unit variance apply:
        // a bin floored to a constant pre-CMVN normalises to ~0, not 1.
        if (var < 0.5) continue;
        ++checked;
        max_var_err = std::max(max_var_err, static_cast<float>(std::fabs(var - 1.0)));
    }
    // Broadband input must exercise the great majority of bins.
    assert(checked >= 70);
    // kNormEps (1e-5) under the sqrt keeps variance a hair below 1; the inverse-
    // std applied to the same data recovers ~1. Allow generous float slack.
    assert(max_var_err < 5e-3f);
    printf("  PASS: test_cmvn_unit_variance (checked=%d/80, max |var-1|=%.2e)\n",
           checked, max_var_err);
}

// Frame-stacking layout: re-running the front-end on a sub-clip that starts at
// the same sample boundary reproduces the corresponding rows. Because framing
// is snip_edges (no centering) and CMVN is global, an exact row match across
// clips is not expected — but the *shape* of the stacking (even row == first
// 80 cols, odd row == last 80 cols, both finite) must hold for every cell.
void test_stacking_finite_and_split() {
    auto y = make_signal();
    int T = 0;
    auto f = audio::seamless_log_mel(y.data(), y.size(), 16000, T);
    assert(T == 24);
    for (size_t i = 0; i < f.size(); ++i) {
        assert(std::isfinite(f[i]));   // no NaN/Inf anywhere
    }
    // Even/odd halves of a stacked row come from distinct source frames, so for
    // a non-stationary-ish signal they should not be byte-identical for all 80
    // bins on every row. Confirm at least one row has a differing half.
    bool any_diff = false;
    for (int t = 0; t < T && !any_diff; ++t) {
        for (int m = 0; m < 80; ++m) {
            if (f[static_cast<size_t>(t) * 160 + m]
                != f[static_cast<size_t>(t) * 160 + 80 + m]) {
                any_diff = true;
                break;
            }
        }
    }
    assert(any_diff);
    printf("  PASS: test_stacking_finite_and_split\n");
}

// Silence (all zeros) must not crash or produce NaN. With a zero signal every
// frame hits the mel floor identically, so per-bin variance is ~0 and CMVN
// divides by sqrt(kNormEps); the output stays finite (and ~0).
void test_silence() {
    int T = 0;
    std::vector<float> zeros(8000, 0.0f);
    auto f = audio::seamless_log_mel(zeros.data(), zeros.size(), 16000, T);
    assert(T == 24);
    assert(f.size() == static_cast<size_t>(T) * 160);
    for (float v : f) assert(std::isfinite(v));
    printf("  PASS: test_silence\n");
}

int main() {
    printf("test_seamless_fbank:\n");
    test_shape();
    test_golden_values();
    test_cmvn_zero_mean();
    test_edge_cases();
    test_stack_boundary();
    test_shape_sweep();
    test_determinism();
    test_cmvn_unit_variance();
    test_stacking_finite_and_split();
    test_silence();
    printf("All seamless_fbank tests passed.\n");
    return 0;
}
