#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace speech_core {

/// PCM format conversions for audio transport.
class PCMCodec {
public:
    /// Convert Float32 samples to PCM16 little-endian bytes.
    static std::vector<uint8_t> float_to_pcm16(const float* samples, size_t count);

    /// Convert PCM16 little-endian bytes to Float32 samples.
    static std::vector<float> pcm16_to_float(const uint8_t* data, size_t byte_count);

    /// Encode raw bytes to base64 string.
    static std::string to_base64(const uint8_t* data, size_t length);

    /// Decode base64 string to raw bytes.
    static std::vector<uint8_t> from_base64(const std::string& encoded);
};

}  // namespace speech_core
