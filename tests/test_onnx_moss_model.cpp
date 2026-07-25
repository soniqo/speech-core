#include "speech_core/models/onnx_moss_transcribe_diarize.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct WavData {
    std::vector<float> samples;
    int sample_rate = 0;
};

WavData load_pcm16_mono(const std::string& path) {
    WavData output;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return output;

    char riff[4] = {};
    char wave[4] = {};
    std::uint32_t file_size = 0;
    stream.read(riff, sizeof(riff));
    stream.read(reinterpret_cast<char*>(&file_size), sizeof(file_size));
    stream.read(wave, sizeof(wave));
    if (std::memcmp(riff, "RIFF", sizeof(riff)) != 0
        || std::memcmp(wave, "WAVE", sizeof(wave)) != 0) {
        return output;
    }

    std::uint16_t audio_format = 0;
    std::uint16_t channels = 0;
    std::uint16_t bits_per_sample = 0;
    std::uint32_t sample_rate = 0;
    bool have_format = false;
    char chunk_id[4] = {};
    std::uint32_t chunk_size = 0;
    while (stream.read(chunk_id, sizeof(chunk_id))) {
        stream.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
        if (!stream) return {};
        if (std::memcmp(chunk_id, "fmt ", sizeof(chunk_id)) == 0) {
            if (chunk_size < 16) return {};
            stream.read(
                reinterpret_cast<char*>(&audio_format),
                sizeof(audio_format));
            stream.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            stream.read(
                reinterpret_cast<char*>(&sample_rate),
                sizeof(sample_rate));
            stream.seekg(6, std::ios::cur);
            stream.read(
                reinterpret_cast<char*>(&bits_per_sample),
                sizeof(bits_per_sample));
            if (chunk_size > 16) {
                stream.seekg(
                    static_cast<std::streamoff>(chunk_size - 16),
                    std::ios::cur);
            }
            have_format = true;
        } else if (std::memcmp(chunk_id, "data", sizeof(chunk_id)) == 0) {
            if (!have_format || audio_format != 1 || channels != 1
                || bits_per_sample != 16 || chunk_size % 2 != 0) {
                return {};
            }
            std::vector<std::int16_t> pcm(chunk_size / 2);
            stream.read(
                reinterpret_cast<char*>(pcm.data()),
                static_cast<std::streamsize>(chunk_size));
            if (!stream) return {};
            output.samples.resize(pcm.size());
            std::transform(
                pcm.begin(), pcm.end(), output.samples.begin(),
                [](std::int16_t sample) {
                    return static_cast<float>(sample) / 32768.0f;
                });
            output.sample_rate = static_cast<int>(sample_rate);
            return output;
        } else {
            stream.seekg(
                static_cast<std::streamoff>(chunk_size), std::ios::cur);
        }
        if ((chunk_size & 1u) != 0u) stream.seekg(1, std::ios::cur);
    }
    return {};
}

std::string test_audio_path() {
#ifdef SPEECH_CORE_TEST_DATA_DIR
    return std::string(SPEECH_CORE_TEST_DATA_DIR) + "/test_audio.wav";
#else
    return "tests/data/test_audio.wav";
#endif
}

bool contains_wire_control(const std::string& text) {
    return text.find("[S") != std::string::npos
        || text.find("<|") != std::string::npos;
}

}  // namespace

int main() {
    const char* bundle = std::getenv("SPEECH_MOSS_ONNX_DIR");
    if (!bundle || std::string(bundle).empty()) {
        std::cout << "Skipping MOSS ONNX model test: "
                     "SPEECH_MOSS_ONNX_DIR is not set\n";
        return 0;
    }

    const WavData wav = load_pcm16_mono(test_audio_path());
    if (wav.samples.empty() || wav.sample_rate <= 0) {
        std::cerr << "Could not load the MOSS test WAV\n";
        return 1;
    }

    // The fixture contains two timestamped speakers. Use the complete clip so
    // an arbitrary cut cannot turn a valid final segment into malformed wire.
    const std::size_t sample_count = wav.samples.size();
    speech_core::OnnxMossTranscribeDiarize::Config config;
    config.max_new_tokens = 128;
    config.audio_hardware_acceleration = false;
    config.decoder_hardware_acceleration = false;
    speech_core::OnnxMossTranscribeDiarize model(bundle, config);
    const auto result = model.transcribe_diarized(
        wav.samples.data(), sample_count, wav.sample_rate);
    const auto profile = model.last_profile();
    std::cout << "MOSS raw: " << result.raw_text << '\n'
              << "MOSS text: " << result.text << '\n';

    if (result.raw_text.empty() || result.text.empty()
        || result.segments.size() != 2) {
        std::cerr << "MOSS produced no usable transcript\n";
        return 1;
    }
    if (contains_wire_control(result.text)) {
        std::cerr << "MOSS published a wire-control token\n";
        return 1;
    }
    if (profile.audio_chunks != 1 || profile.generated_tokens <= 0
        || profile.total_ms <= 0.0) {
        std::cerr << "MOSS profile did not cover a complete inference\n";
        return 1;
    }

    std::cout << "MOSS ONNX model test passed: " << result.text << '\n';
    return 0;
}
