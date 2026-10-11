#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

namespace speech_core {

/// RIFF/WAVE audio decoded to mono float32 at the file's native sample rate.
struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
    double duration() const {
        return sample_rate > 0
            ? static_cast<double>(samples.size()) / sample_rate : 0.0;
    }
    void clear() {
        samples.clear();
        sample_rate = 0;
    }

    /// Load PCM16/24/32 or IEEE float32 WAV audio, averaging channels to mono.
    /// The native sample rate is preserved; callers resample when needed.
    /// Paths are UTF-8. Returns false on I/O or format errors and keep object unchanged.
    bool load_mono(const char* path);
    bool load_mono(const std::string& path);
    bool load_mono(const std::filesystem::path& path);
    bool load_mono(std::istream& file);

    /// Original UTF-8 string-path entry point, equivalent to load_mono().
    bool load(const std::string& path);

    /// Save mono PCM16 RIFF/WAVE audio, clamping float samples to [-1, 1].
    /// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
    /// Invalid arguments leave the destination untouched.
    bool save(const char* path) const;
    bool save(const std::string& path) const;
    bool save(const std::filesystem::path& path) const;
    bool save(std::ostream& file) const;

    /// Write mono PCM16 RIFF/WAVE audio, clamping float samples to [-1, 1].
    /// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
    /// Invalid arguments leave the destination untouched.
    static bool write_mono(const std::filesystem::path& path,
                           const std::vector<float>& data, int sample_rate);
    static bool write_mono(std::ostream& os,
                           const float* samples, size_t count,
                           int sample_rate);
};

/// Load PCM16/24/32 or IEEE float32 WAV audio, averaging channels to mono.
/// The native sample rate is preserved; callers resample when needed.
/// The historical pcm16 name also covers the other supported encodings.
/// Paths are UTF-8. Returns false on I/O or format errors.
bool load_wav_mono_pcm16(const std::string& path, WavData* out);

/// Write mono PCM16 RIFF/WAVE audio, clamping float samples to [-1, 1].
/// Paths are UTF-8. Returns false on I/O errors or invalid arguments.
bool write_wav_mono_pcm16(const std::string& path,
                          const std::vector<float>& data, int sample_rate);
bool write_wav_mono_pcm16(const std::string& path,
                          const float* samples, size_t count,
                          int sample_rate);

}  // namespace speech_core
