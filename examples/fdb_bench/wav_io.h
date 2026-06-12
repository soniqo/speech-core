// Minimal RIFF/WAVE I/O for the FDB driver. Vendored under examples/ —
// not promoted to a public speech_core API because no consumer outside
// this driver needs it yet (see audit notes in docs/fdb-bench.md).
//
// Reader: accepts mono OR multi-channel PCM16 (downmixed to mono by
// averaging). Returns 16-bit-quantised float32 in [-1, 1] at the file's
// native sample rate. The driver resamples to 16 kHz via
// speech_core::Resampler when needed.
//
// Writer: mono PCM16 only. Clamps float32 to [-1, 1].
//
// Adapted from the proven WAV reader in
// examples/litert/voxcpm2_clone.cpp:42-129 — same algorithm, scoped
// down to what M1 needs.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fdb_bench {

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};

// Returns true on success and populates *out. Returns false on any I/O
// or format error (file missing, not RIFF/WAVE, not PCM16, etc.).
bool load_wav_mono_pcm16(const std::string& path, WavData* out);

// Returns true on success. Writes a mono PCM16 RIFF/WAVE file.
bool write_wav_mono_pcm16(const std::string& path,
                          const float* samples, size_t count,
                          int sample_rate);

}  // namespace fdb_bench
