#include "speech_core/models/onnx_vap.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
int fail(const char* message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}
bool close(float value, float expected, float tolerance = 0.0005f) {
    return std::isfinite(value) && std::fabs(value - expected) <= tolerance;
}
template <class F> bool rejects(F function) {
    try { function(); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}
}  // namespace

int main() {
    using speech_core::OnnxVap;
    const float user[] = {1.0f, 2.0f}, agent[] = {3.0f, 4.0f};
    const auto window = OnnxVap::prepare_window(user, agent, 2);
    if (window.size() != 640000 || window[319997] != 0.0f
        || window[319998] != 1.0f || window[319999] != 2.0f
        || window[639998] != 3.0f || window[639999] != 4.0f) {
        return fail("channel order or synchronized leading padding");
    }
    if (!rejects([&] { OnnxVap::prepare_window(nullptr, agent, 2); })
        || !rejects([&] { OnnxVap::prepare_window(user, nullptr, 2); })
        || !rejects([&] { OnnxVap::prepare_window(user, agent, 0); })) {
        return fail("missing channels must be rejected");
    }
    const float invalid[] = {std::numeric_limits<float>::quiet_NaN(), 0};
    if (!rejects([&] { OnnxVap::prepare_window(invalid, agent, 2); })) {
        return fail("nonfinite audio must be rejected");
    }
    std::vector<float> long_user(320001, 0.1f), long_agent(320001, 0.2f);
    long_user[1] = 0.3f;
    const auto tail = OnnxVap::prepare_window(long_user.data(), long_agent.data(), long_user.size());
    if (tail[0] != 0.3f || tail[320000] != 0.2f) return fail("recent window selection");

    const char* path = std::getenv("SPEECH_VAP_ONNX");
    if (!path || !*path) {
        std::cout << "VAP audio preparation passed; set SPEECH_VAP_ONNX for model checks\n";
        return 77;
    }
    OnnxVap model(path);
    std::vector<float> zero(320000, 0.0f);
    if (!rejects([&] { model.forecast(zero.data(), zero.data(), zero.size(), 48000); })) {
        return fail("wrong sample rate must be rejected");
    }
    const auto silence = model.forecast(zero.data(), zero.data(), zero.size());
    if (!close(silence.p_now[0], 0.44378302f) || !close(silence.p_now[1], 0.55563056f)
        || !close(silence.p_future[0], 0.34941691f) || !close(silence.p_future[1], 0.65039903f)
        || !close(silence.voice_activity[0], 0.00018012f, 0.00001f)) {
        return fail("silence differs from the original checkpoint");
    }
    std::vector<float> signal(320000);
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t index = 0; index < signal.size(); ++index) {
        const double time = static_cast<double>(index) / 16000.0;
        signal[index] = (index / 2400) % 3 == 0 ? 0.0f
            : static_cast<float>(0.05 * std::sin(2 * pi * 173 * time)
                                 + 0.02 * std::sin(2 * pi * 587 * time));
    }
    const auto first = model.forecast(signal.data(), zero.data(), signal.size());
    const auto second = model.forecast(zero.data(), signal.data(), signal.size());
    if (!close(first.p_now[0], 0.98009735f) || !close(first.p_now[1], 0.01986181f)
        || !close(second.p_now[0], 0.01861823f) || !close(second.p_now[1], 0.98133320f)) {
        return fail("speaker channels differ from the original checkpoint");
    }
    const auto repeat = model.forecast(signal.data(), zero.data(), signal.size());
    if (!close(first.p_now[0], repeat.p_now[0], 1e-6f)) return fail("repeat inference changed");
    std::cout << "VAP ONNX input contract and original-checkpoint references passed\n";
    return 0;
}
