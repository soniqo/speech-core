// Force-enable asserts even under Release builds.
#ifdef NDEBUG
#  undef NDEBUG
#endif

#include "speech_core/audio/fft.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool approx(float a, double b, double tol) {
    return std::fabs(static_cast<double>(a) - b) <= tol;
}

size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

void naive_fft(const std::vector<float>& input, size_t bins,
               std::vector<double>& re, std::vector<double>& im) {
    const size_t transform_size = next_pow2(input.size());
    re.assign(bins, 0.0);
    im.assign(bins, 0.0);

    for (size_t k = 0; k < bins; ++k) {
        for (size_t t = 0; t < input.size(); ++t) {
            const double angle = 2.0 * kPi * static_cast<double>(k * t)
                               / static_cast<double>(transform_size);
            re[k] += input[t] * std::cos(angle);
            im[k] -= input[t] * std::sin(angle);
        }
    }
}

}  // namespace

void test_impulse_forward() {
    const size_t n = 8;
    const size_t bins = n / 2 + 1;
    std::vector<float> input(n, 0.0f);
    input[0] = 1.0f;

    std::vector<float> re(bins), im(bins);
    audio::fft_real(input.data(), n, re.data(), im.data());

    for (size_t i = 0; i < bins; ++i) {
        assert(approx(re[i], 1.0, 1e-6));
        assert(approx(im[i], 0.0, 1e-6));
    }
    std::printf("  PASS: impulse_forward\n");
}

void test_power_of_two_roundtrip() {
    const size_t n = 16;
    const size_t bins = n / 2 + 1;
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n);
        input[i] = static_cast<float>(0.7 * std::sin(2.0 * kPi * 3.0 * t)
                                    + 0.2 * std::cos(2.0 * kPi * 2.0 * t));
    }

    std::vector<float> re(bins), im(bins), output(n);
    audio::fft_real(input.data(), n, re.data(), im.data());
    audio::ifft_real(re.data(), im.data(), n, output.data());

    for (size_t i = 0; i < n; ++i) {
        assert(approx(output[i], input[i], 2e-5));
    }
    std::printf("  PASS: power_of_two_roundtrip\n");
}

void test_inverse_cosine_bin() {
    const size_t n = 8;
    const size_t bins = n / 2 + 1;
    std::vector<float> re(bins, 0.0f), im(bins, 0.0f), output(n);
    re[1] = static_cast<float>(n) / 2.0f;

    audio::ifft_real(re.data(), im.data(), n, output.data());

    for (size_t i = 0; i < n; ++i) {
        const double expected = std::cos(2.0 * kPi * static_cast<double>(i)
                                       / static_cast<double>(n));
        assert(approx(output[i], expected, 2e-5));
    }
    std::printf("  PASS: inverse_cosine_bin\n");
}

void test_non_power_of_two_forward_zero_pads() {
    const size_t n = 6;
    const size_t bins = n / 2 + 1;
    std::vector<float> input = {1.0f, -0.5f, 0.25f, 0.75f, -0.125f, 0.5f};

    std::vector<float> re(bins), im(bins);
    audio::fft_real(input.data(), n, re.data(), im.data());

    std::vector<double> expect_re, expect_im;
    naive_fft(input, bins, expect_re, expect_im);

    for (size_t i = 0; i < bins; ++i) {
        assert(approx(re[i], expect_re[i], 2e-5));
        assert(approx(im[i], expect_im[i], 2e-5));
    }
    std::printf("  PASS: non_power_of_two_forward_zero_pads\n");
}

int main() {
    std::printf("test_fft:\n");
    test_impulse_forward();
    test_power_of_two_roundtrip();
    test_inverse_cosine_bin();
    test_non_power_of_two_forward_zero_pads();
    std::printf("All FFT tests passed.\n");
    return 0;
}
