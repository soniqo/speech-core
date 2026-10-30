#pragma once

#include <onnxruntime_c_api.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace speech_core {

struct VapForecast {
    // Channel order is user, agent. These are forecasts, not trigger decisions.
    std::array<float, 2> p_now;
    std::array<float, 2> p_future;
    std::array<float, 2> voice_activity;
};

/// otoSpeech Voice Activity Projection, using synchronized 16 kHz channels.
/// Each request recomputes the last 20 s; shorter audio receives leading silence.
/// The application chooses its response/interruption policy from the forecasts.
/// Serialize calls to one model instance.
class OnnxVap final {
public:
    static constexpr int kSampleRate = 16000;
    static constexpr std::size_t kWindowSamples = 320000;
    static constexpr std::size_t kFrames = 1000;

    explicit OnnxVap(const std::string& model_path, bool hardware_acceleration = false);
    ~OnnxVap();
    OnnxVap(const OnnxVap&) = delete;
    OnnxVap& operator=(const OnnxVap&) = delete;

    VapForecast forecast(const float* user, const float* agent,
                         std::size_t length, int sample_rate = kSampleRate);

    /// Channel-major [1, 2, 320000] waveform; channel 0 is the user.
    static std::vector<float> prepare_window(
        const float* user, const float* agent, std::size_t length);

private:
    void validate_contract();
    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;
};

}  // namespace speech_core
