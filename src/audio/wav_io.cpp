#include "speech_core/audio/wav_io.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace speech_core {
namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8));
}
uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

float decode_sample(const uint8_t* p, int audio_format, int bits_per_sample) {
    if (audio_format == 3) {
        const uint32_t bits = read_u32_le(p);
        float sample;
        static_assert(sizeof(sample) == sizeof(bits), "WAV requires 32-bit float");
        std::memcpy(&sample, &bits, sizeof(sample));
        return sample;
    }

    uint32_t raw;
    if (bits_per_sample == 16) {
        raw = read_u16_le(p);
    } else if (bits_per_sample == 24) {
        raw = static_cast<uint32_t>(p[0]) |
              (static_cast<uint32_t>(p[1]) << 8) |
              (static_cast<uint32_t>(p[2]) << 16);
    } else {
        raw = read_u32_le(p);
    }
    int64_t value = raw;
    if (raw & (uint32_t{1} << (bits_per_sample - 1))) {
        value -= int64_t{1} << bits_per_sample;
    }
    return static_cast<float>(static_cast<double>(value) /
                              static_cast<double>(int64_t{1} << (bits_per_sample - 1)));
}

void write_u16_le(std::ostream& os, uint16_t v) {
    char buf[2] = {static_cast<char>(v & 0xff),
                   static_cast<char>((v >> 8) & 0xff)};
    os.write(buf, 2);
}

void write_u32_le(std::ostream& os, uint32_t v) {
    char buf[4] = {static_cast<char>(v & 0xff),
                   static_cast<char>((v >> 8) & 0xff),
                   static_cast<char>((v >> 16) & 0xff),
                   static_cast<char>((v >> 24) & 0xff)};
    os.write(buf, 4);
}

}  // namespace

bool load_wav_mono_pcm16(const std::string& path, WavData* out) {
    return out && out->load_mono(path);
}

bool write_wav_mono_pcm16(const std::string& path,
                          const std::vector<float>& data, int sample_rate) {
    return WavData::write_mono(std::filesystem::u8path(path), data, sample_rate);
}

bool write_wav_mono_pcm16(const std::string& path,
                          const float* samples, size_t count, int sample_rate) {
    if (!samples || count == 0 || sample_rate <= 0) return false;
    std::ofstream os(std::filesystem::u8path(path), std::ios::binary);
    return WavData::write_mono(os, samples, count, sample_rate);
}

bool WavData::load_mono(const char* path) {
    return path && load_mono(std::string(path));
}

bool WavData::load_mono(const std::string& path) {
    return load_mono(std::filesystem::u8path(path));
}

bool WavData::load_mono(const std::filesystem::path& path) {
    std::ifstream is(path, std::ios::binary);
    return load_mono(is);
}

bool WavData::load_mono(std::istream& is) {
    if (!is) return false;

    // Keep the file buffer alive while walking and decoding its chunks.
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(is)),
                              std::istreambuf_iterator<char>());
    if (buf.size() < 44) return false;

    if (std::memcmp(buf.data(), "RIFF", 4) != 0) return false;
    if (std::memcmp(buf.data() + 8, "WAVE", 4) != 0) return false;

    size_t pos = 12;
    int    channels = 0;
    int    bits_per_sample = 0;
    int    audio_format = 0;
    int    sample_rate = 0;
    const uint8_t* pcm_data = nullptr;
    size_t pcm_bytes = 0;

    while (buf.size() - pos >= 8) {
        const uint8_t* chunk_id = buf.data() + pos;
        uint32_t chunk_size = read_u32_le(buf.data() + pos + 4);
        pos += 8;
        if (chunk_size > buf.size() - pos) return false;

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) return false;
            audio_format    = read_u16_le(buf.data() + pos);
            channels        = read_u16_le(buf.data() + pos + 2);
            sample_rate     = static_cast<int>(read_u32_le(buf.data() + pos + 4));
            bits_per_sample = read_u16_le(buf.data() + pos + 14);
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            pcm_data  = buf.data() + pos;
            pcm_bytes = chunk_size;
        }
        // Skip to next chunk (chunks are word-aligned).
        pos += chunk_size;
        if ((chunk_size & 1u) && pos < buf.size()) pos += 1;
    }

    if (channels < 1 || sample_rate <= 0 || pcm_data == nullptr) return false;

    const bool is_pcm = audio_format == 1 &&
        (bits_per_sample == 16 || bits_per_sample == 24 || bits_per_sample == 32);
    const bool is_float32 = (audio_format == 3 && bits_per_sample == 32);
    if (!is_pcm && !is_float32) return false;

    const size_t bytes_per_sample = static_cast<size_t>(bits_per_sample / 8);
    const size_t bytes_per_frame = static_cast<size_t>(channels) * bytes_per_sample;
    if (pcm_bytes < bytes_per_frame || pcm_bytes % bytes_per_frame != 0) return false;
    const size_t num_frames = pcm_bytes / bytes_per_frame;

    this->samples.resize(num_frames);
    this->sample_rate = sample_rate;

    for (size_t i = 0; i < num_frames; ++i) {
        double sum = 0.0;
        for (int c = 0; c < channels; ++c) {
            const auto* sample = pcm_data +
                (i * static_cast<size_t>(channels) + static_cast<size_t>(c)) * bytes_per_sample;
            sum += decode_sample(sample, audio_format, bits_per_sample);
        }
        this->samples[i] = static_cast<float>(sum / channels);
    }
    return true;
}

bool WavData::load(const std::string& path) {
    return load_mono(path);
}

bool WavData::save(const char* path) const {
    return path && save(std::string(path));
}

bool WavData::save(const std::string& path) const {
    return save(std::filesystem::u8path(path));
}

bool WavData::save(const std::filesystem::path& path) const {
    return write_mono(path, samples, sample_rate);
}

bool WavData::save(std::ostream& os) const {
    return write_mono(os, samples.data(), samples.size(), sample_rate);
}

bool WavData::write_mono(const std::filesystem::path& path,
                         const std::vector<float>& data, int sample_rate) {
    if (data.empty() || sample_rate <= 0) return false;
    std::ofstream os(path, std::ios::binary);
    return write_mono(os, data.data(), data.size(), sample_rate);
}

bool WavData::write_mono(std::ostream& os,
                         const float* samples, size_t count, int sample_rate) {
    if (!samples || count == 0 || sample_rate <= 0) return false;

    if (!os) return false;

    const uint32_t data_bytes = static_cast<uint32_t>(count * 2);
    const uint32_t chunk_size = 36u + data_bytes;

    // RIFF header
    os.write("RIFF", 4);
    write_u32_le(os, chunk_size);
    os.write("WAVE", 4);

    // fmt chunk (PCM = 1)
    os.write("fmt ", 4);
    write_u32_le(os, 16);                   // subchunk1 size
    write_u16_le(os, 1);                    // audio format = PCM
    write_u16_le(os, 1);                    // channels = mono
    write_u32_le(os, static_cast<uint32_t>(sample_rate));
    write_u32_le(os, static_cast<uint32_t>(sample_rate) * 2);  // byte rate
    write_u16_le(os, 2);                    // block align
    write_u16_le(os, 16);                   // bits per sample

    // data chunk
    os.write("data", 4);
    write_u32_le(os, data_bytes);

    for (size_t i = 0; i < count; ++i) {
        float v = samples[i];
        if (v < -1.0f) v = -1.0f;
        if (v >  1.0f) v =  1.0f;
        int16_t s = static_cast<int16_t>(v * 32767.0f);
        char buf[2] = {static_cast<char>(s & 0xff),
                       static_cast<char>((s >> 8) & 0xff)};
        os.write(buf, 2);
    }

    return static_cast<bool>(os);
}

}  // namespace speech_core
