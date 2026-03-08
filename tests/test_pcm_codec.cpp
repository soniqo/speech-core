#include "speech_core/audio/pcm_codec.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace speech_core;

void test_float_to_pcm16_roundtrip() {
    float samples[] = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 0.123f};
    size_t count = sizeof(samples) / sizeof(samples[0]);

    auto pcm = PCMCodec::float_to_pcm16(samples, count);
    assert(pcm.size() == count * 2);

    auto back = PCMCodec::pcm16_to_float(pcm.data(), pcm.size());
    assert(back.size() == count);

    for (size_t i = 0; i < count; i++) {
        float diff = std::abs(back[i] - samples[i]);
        assert(diff < 0.001f);  // PCM16 quantization error
    }
    printf("  PASS: float_to_pcm16_roundtrip\n");
}

void test_base64_roundtrip() {
    uint8_t data[] = {0, 1, 2, 3, 255, 128, 64, 32, 16, 8};
    size_t len = sizeof(data);

    auto encoded = PCMCodec::to_base64(data, len);
    auto decoded = PCMCodec::from_base64(encoded);

    assert(decoded.size() == len);
    for (size_t i = 0; i < len; i++) {
        assert(decoded[i] == data[i]);
    }
    printf("  PASS: base64_roundtrip\n");
}

void test_base64_padding() {
    // Test 1 byte (needs ==)
    uint8_t one[] = {42};
    auto e1 = PCMCodec::to_base64(one, 1);
    auto d1 = PCMCodec::from_base64(e1);
    assert(d1.size() == 1 && d1[0] == 42);

    // Test 2 bytes (needs =)
    uint8_t two[] = {42, 99};
    auto e2 = PCMCodec::to_base64(two, 2);
    auto d2 = PCMCodec::from_base64(e2);
    assert(d2.size() == 2 && d2[0] == 42 && d2[1] == 99);

    // Test 3 bytes (no padding)
    uint8_t three[] = {42, 99, 7};
    auto e3 = PCMCodec::to_base64(three, 3);
    auto d3 = PCMCodec::from_base64(e3);
    assert(d3.size() == 3);

    printf("  PASS: base64_padding\n");
}

void test_clamping() {
    float extremes[] = {2.0f, -2.0f};
    auto pcm = PCMCodec::float_to_pcm16(extremes, 2);
    auto back = PCMCodec::pcm16_to_float(pcm.data(), pcm.size());
    // Should clamp to [-1, 1]
    assert(std::abs(back[0] - 1.0f) < 0.001f);
    assert(std::abs(back[1] - (-1.0f)) < 0.001f);
    printf("  PASS: clamping\n");
}

int main() {
    printf("test_pcm_codec:\n");
    test_float_to_pcm16_roundtrip();
    test_base64_roundtrip();
    test_base64_padding();
    test_clamping();
    printf("All PCM codec tests passed.\n");
    return 0;
}
