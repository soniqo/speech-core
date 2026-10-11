#include "speech_core/models/onnx_moss_transcribe_diarize.h"
#include "speech_core/audio/wav_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

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

    speech_core::WavData wav;
    wav.load_mono(test_audio_path());
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
