#include "speech_core/audio/pcm_codec.h"

#include <algorithm>
#include <array>
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
    // GCC rejects C99 designated initializers for non-trivially-default
    // types in C++ mode, so we build the table once at first call via an
    // immediately-invoked lambda. Apple-clang previously accepted the
    // designator syntax as a -Wc99-designator extension.
    static const auto kDecodeTable = []() {
        std::array<uint8_t, 256> t{};
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
        for (uint8_t i = 0; i < 64; ++i) {
            t[static_cast<uint8_t>(alphabet[i])] = i;
        }
        return t;
    }();

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
