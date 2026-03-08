#include "speech_core/audio/pcm_codec.h"

#include <algorithm>
#include <cmath>

namespace speech_core {

static constexpr char kBase64Chars[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::vector<uint8_t> PCMCodec::float_to_pcm16(const float* samples, size_t count) {
    std::vector<uint8_t> data(count * 2);
    for (size_t i = 0; i < count; i++) {
        float clamped = std::max(-1.0f, std::min(1.0f, samples[i]));
        int16_t val = static_cast<int16_t>(clamped * 32767.0f);
        data[i * 2]     = static_cast<uint8_t>(val & 0xFF);
        data[i * 2 + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
    }
    return data;
}

std::vector<float> PCMCodec::pcm16_to_float(const uint8_t* data, size_t byte_count) {
    size_t sample_count = byte_count / 2;
    std::vector<float> samples(sample_count);
    for (size_t i = 0; i < sample_count; i++) {
        int16_t val = static_cast<int16_t>(
            static_cast<uint16_t>(data[i * 2]) |
            (static_cast<uint16_t>(data[i * 2 + 1]) << 8));
        samples[i] = static_cast<float>(val) / 32768.0f;
    }
    return samples;
}

std::string PCMCodec::to_base64(const uint8_t* data, size_t length) {
    std::string result;
    result.reserve(((length + 2) / 3) * 4);

    for (size_t i = 0; i < length; i += 3) {
        uint32_t triple = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < length) triple |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < length) triple |= static_cast<uint32_t>(data[i + 2]);

        result += kBase64Chars[(triple >> 18) & 0x3F];
        result += kBase64Chars[(triple >> 12) & 0x3F];
        result += (i + 1 < length) ? kBase64Chars[(triple >> 6) & 0x3F] : '=';
        result += (i + 2 < length) ? kBase64Chars[triple & 0x3F] : '=';
    }

    return result;
}

std::vector<uint8_t> PCMCodec::from_base64(const std::string& encoded) {
    static const uint8_t kDecodeTable[256] = {
        ['A'] = 0,  ['B'] = 1,  ['C'] = 2,  ['D'] = 3,  ['E'] = 4,
        ['F'] = 5,  ['G'] = 6,  ['H'] = 7,  ['I'] = 8,  ['J'] = 9,
        ['K'] = 10, ['L'] = 11, ['M'] = 12, ['N'] = 13, ['O'] = 14,
        ['P'] = 15, ['Q'] = 16, ['R'] = 17, ['S'] = 18, ['T'] = 19,
        ['U'] = 20, ['V'] = 21, ['W'] = 22, ['X'] = 23, ['Y'] = 24,
        ['Z'] = 25, ['a'] = 26, ['b'] = 27, ['c'] = 28, ['d'] = 29,
        ['e'] = 30, ['f'] = 31, ['g'] = 32, ['h'] = 33, ['i'] = 34,
        ['j'] = 35, ['k'] = 36, ['l'] = 37, ['m'] = 38, ['n'] = 39,
        ['o'] = 40, ['p'] = 41, ['q'] = 42, ['r'] = 43, ['s'] = 44,
        ['t'] = 45, ['u'] = 46, ['v'] = 47, ['w'] = 48, ['x'] = 49,
        ['y'] = 50, ['z'] = 51, ['0'] = 52, ['1'] = 53, ['2'] = 54,
        ['3'] = 55, ['4'] = 56, ['5'] = 57, ['6'] = 58, ['7'] = 59,
        ['8'] = 60, ['9'] = 61, ['+'] = 62, ['/'] = 63
    };

    std::vector<uint8_t> result;
    result.reserve(encoded.size() * 3 / 4);

    for (size_t i = 0; i + 3 < encoded.size(); i += 4) {
        uint32_t triple =
            (kDecodeTable[static_cast<uint8_t>(encoded[i])]     << 18) |
            (kDecodeTable[static_cast<uint8_t>(encoded[i + 1])] << 12) |
            (kDecodeTable[static_cast<uint8_t>(encoded[i + 2])] << 6) |
            (kDecodeTable[static_cast<uint8_t>(encoded[i + 3])]);

        result.push_back(static_cast<uint8_t>((triple >> 16) & 0xFF));
        if (encoded[i + 2] != '=')
            result.push_back(static_cast<uint8_t>((triple >> 8) & 0xFF));
        if (encoded[i + 3] != '=')
            result.push_back(static_cast<uint8_t>(triple & 0xFF));
    }

    return result;
}

}  // namespace speech_core
