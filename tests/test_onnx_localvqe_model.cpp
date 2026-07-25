#include "speech_core/models/onnx_localvqe_echo_canceller.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

bool all_finite(const std::vector<float>& values) {
    return std::all_of(
        values.begin(), values.end(),
        [](float value) { return std::isfinite(value); });
}

float maximum_difference(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(
            maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

}  // namespace

int main() {
    const char* bundle = std::getenv("SPEECH_LOCALVQE_ONNX_DIR");
    if (!bundle || std::string(bundle).empty()) {
        std::cout << "Skipping LocalVQE ONNX model test: "
                     "SPEECH_LOCALVQE_ONNX_DIR is not set\n";
        return 0;
    }

    using Canceller = speech_core::OnnxLocalVQEEchoCanceller;
    Canceller::Config config;
    config.hardware_acceleration = false;
    config.enable_prealignment = false;
    Canceller model(bundle, config);

    constexpr std::size_t frames = 6;
    const std::size_t sample_count = frames * Canceller::kFrameSize;
    std::vector<float> microphone(sample_count);
    std::vector<float> reference(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index) {
        const double time = static_cast<double>(index)
            / Canceller::kSampleRate;
        reference[index] = static_cast<float>(
            0.08 * std::sin(2.0 * kPi * 440.0 * time));
        microphone[index] = static_cast<float>(
            0.04 * std::sin(2.0 * kPi * 173.0 * time)
            + 0.35 * reference[index]);
    }

    auto process_direct = [&]() {
        std::vector<float> output(sample_count, 0.0f);
        for (std::size_t offset = 0; offset < sample_count;
             offset += Canceller::kFrameSize) {
            model.process_frame(
                microphone.data() + offset,
                reference.data() + offset,
                output.data() + offset);
        }
        return output;
    };

    const std::vector<float> first = process_direct();
    if (!all_finite(first)
        || std::all_of(
            first.begin(), first.end(),
            [](float value) { return value == 0.0f; })) {
        std::cerr << "LocalVQE produced invalid or empty audio\n";
        return 1;
    }
    const auto profile = model.last_profile();
    if (profile.total_ms <= 0.0 || profile.neural_mask_ms <= 0.0) {
        std::cerr << "LocalVQE profile did not cover inference\n";
        return 1;
    }

    model.reset();
    const std::vector<float> second = process_direct();
    if (maximum_difference(first, second) > 1e-6f) {
        std::cerr << "LocalVQE reset did not reproduce the stream\n";
        return 1;
    }

    model.reset();
    model.feed_reference(reference.data(), reference.size());
    std::vector<float> queued(sample_count, 0.0f);
    model.cancel_echo(
        microphone.data(), microphone.size(), queued.data());
    if (maximum_difference(first, queued) > 1e-6f) {
        std::cerr << "LocalVQE queued-reference path changed output\n";
        return 1;
    }

    model.reset();
    std::vector<float> untouched(
        Canceller::kFrameSize, 123.0f);
    bool rejected = false;
    try {
        model.cancel_echo(
            microphone.data(),
            Canceller::kFrameSize,
            untouched.data());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected || !std::all_of(
            untouched.begin(), untouched.end(),
            [](float value) { return value == 123.0f; })) {
        std::cerr << "LocalVQE did not fail closed on reference underrun\n";
        return 1;
    }

    model.reset();
    std::vector<float> silence(Canceller::kFrameSize, 0.0f);
    std::vector<float> silent_output(Canceller::kFrameSize, 1.0f);
    model.process_frame(
        silence.data(), silence.data(), silent_output.data());
    if (!std::all_of(
            silent_output.begin(), silent_output.end(),
            [](float value) { return std::abs(value) < 1e-8f; })) {
        std::cerr << "LocalVQE silence did not stay silent\n";
        return 1;
    }

    std::cout << "LocalVQE ONNX model test passed\n";
    return 0;
}
