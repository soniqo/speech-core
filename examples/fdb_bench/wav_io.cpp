#include "wav_io.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace fdb_bench {

namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8));
}
uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
void write_u16_le(std::ofstream& os, uint16_t v) {
    char buf[2] = {static_cast<char>(v & 0xff),
                   static_cast<char>((v >> 8) & 0xff)};
    os.write(buf, 2);
}
void write_u32_le(std::ofstream& os, uint32_t v) {
    char buf[4] = {static_cast<char>(v & 0xff),
                   static_cast<char>((v >> 8) & 0xff),
                   static_cast<char>((v >> 16) & 0xff),
                   static_cast<char>((v >> 24) & 0xff)};
    os.write(buf, 4);
}

}  // namespace

bool load_wav_mono_pcm16(const std::string& path, WavData* out) {
    if (!out) return false;
    out->samples.clear();
    out->sample_rate = 0;

    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    // Read full file into a buffer — FDB clips are < 30 s so size is bounded.
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

    while (pos + 8 <= buf.size()) {
        const uint8_t* chunk_id = buf.data() + pos;
        uint32_t chunk_size = read_u32_le(buf.data() + pos + 4);
        pos += 8;
        if (pos + chunk_size > buf.size()) return false;

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
        if (chunk_size & 1u) pos += 1;
    }

    // FDB v1.0 ships two WAV flavours:
    //   - PCM int16          (audio_format=1, bits=16)  — candor_pause_handling, icc_backchannel, synthetic_user_interruption
    //   - IEEE Float 32-bit  (audio_format=3, bits=32)  — candor_turn_taking, synthetic_pause_handling
    // The function name says pcm16 for historical reasons (PR #43), but the
    // contract here is "mono float32 audio in [-1, 1], whatever the on-disk
    // encoding was". Keeping the name avoids churn in all the call sites.
    if (channels < 1 || sample_rate <= 0 || pcm_data == nullptr) return false;

    const bool is_pcm16   = (audio_format == 1 && bits_per_sample == 16);
    const bool is_float32 = (audio_format == 3 && bits_per_sample == 32);
    if (!is_pcm16 && !is_float32) return false;

    const size_t bytes_per_sample = is_pcm16 ? 2u : 4u;
    const size_t bytes_per_frame = static_cast<size_t>(channels) * bytes_per_sample;
    if (pcm_bytes < bytes_per_frame) return false;
    const size_t num_frames = pcm_bytes / bytes_per_frame;

    out->samples.resize(num_frames);
    out->sample_rate = sample_rate;

    if (is_pcm16) {
        const auto* sp = reinterpret_cast<const int16_t*>(pcm_data);
        for (size_t i = 0; i < num_frames; ++i) {
            int acc = 0;
            for (int c = 0; c < channels; ++c) acc += sp[i * channels + c];
            const float avg = static_cast<float>(acc) /
                              static_cast<float>(channels);
            out->samples[i] = avg / 32768.0f;
        }
    } else {
        // Float32 PCM is already in [-1, 1] — read with memcpy so we don't
        // assume the pcm_data pointer is 4-byte aligned (the chunk offset
        // depends on prior chunk sizes and isn't guaranteed to be aligned).
        for (size_t i = 0; i < num_frames; ++i) {
            float acc = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float v;
                std::memcpy(&v, pcm_data + (i * channels + c) * 4, 4);
                acc += v;
            }
            out->samples[i] = acc / static_cast<float>(channels);
        }
    }
    return true;
}

bool write_wav_mono_pcm16(const std::string& path,
                          const float* samples, size_t count,
                          int sample_rate)
{
    if (!samples || sample_rate <= 0) return false;

    std::ofstream os(path, std::ios::binary);
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

}  // namespace fdb_bench
