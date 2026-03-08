#include "speech_core/audio/audio_buffer.h"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace speech_core;

void test_write_read() {
    AudioBuffer buf(1024);
    float input[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    size_t written = buf.write(input, 5);
    assert(written == 5);
    assert(buf.available() == 5);

    float output[5];
    size_t read = buf.read(output, 5);
    assert(read == 5);
    assert(buf.available() == 0);

    for (int i = 0; i < 5; i++) {
        assert(std::abs(output[i] - input[i]) < 1e-6f);
    }
    printf("  PASS: write_read\n");
}

void test_partial_read() {
    AudioBuffer buf(1024);
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    buf.write(input, 4);

    float out1[2], out2[2];
    assert(buf.read(out1, 2) == 2);
    assert(buf.read(out2, 2) == 2);

    assert(std::abs(out1[0] - 1.0f) < 1e-6f);
    assert(std::abs(out2[0] - 3.0f) < 1e-6f);
    printf("  PASS: partial_read\n");
}

void test_empty_read() {
    AudioBuffer buf(1024);
    float output[10];
    assert(buf.read(output, 10) == 0);
    printf("  PASS: empty_read\n");
}

void test_clear() {
    AudioBuffer buf(1024);
    float input[] = {1.0f, 2.0f};
    buf.write(input, 2);
    assert(buf.available() == 2);
    buf.clear();
    assert(buf.available() == 0);
    printf("  PASS: clear\n");
}

void test_wrap_around() {
    AudioBuffer buf(4);  // small buffer
    float a[] = {1.0f, 2.0f, 3.0f};
    buf.write(a, 3);

    float out[3];
    buf.read(out, 3);

    // Write again — wraps around
    float b[] = {4.0f, 5.0f, 6.0f};
    size_t written = buf.write(b, 3);
    assert(written == 3);

    float out2[3];
    size_t read = buf.read(out2, 3);
    assert(read == 3);
    assert(std::abs(out2[0] - 4.0f) < 1e-6f);
    assert(std::abs(out2[1] - 5.0f) < 1e-6f);
    assert(std::abs(out2[2] - 6.0f) < 1e-6f);
    printf("  PASS: wrap_around\n");
}

int main() {
    printf("test_audio_buffer:\n");
    test_write_read();
    test_partial_read();
    test_empty_read();
    test_clear();
    test_wrap_around();
    printf("All audio buffer tests passed.\n");
    return 0;
}
