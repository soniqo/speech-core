#include "speech_core/pipeline/timestamped_echo_cancellation_stream.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace speech_core;

namespace {

constexpr int kSampleRate = 16000;
constexpr std::size_t kFrameSize = 256;

std::int64_t time_for_sample(std::int64_t sample) {
    return static_cast<std::int64_t>(std::llround(
        static_cast<long double>(sample) * 1000000000.0L
        / kSampleRate));
}

class FakeCanceller final : public FrameEchoCancellerInterface {
public:
    int input_sample_rate() const override { return kSampleRate; }
    std::size_t frame_size() const override { return kFrameSize; }

    void process_frame(
        const float* microphone,
        const float* reference,
        float* output) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (should_fail) throw std::runtime_error("fake failure");
        microphone_frames.emplace_back(
            microphone, microphone + kFrameSize);
        reference_frames.emplace_back(
            reference, reference + kFrameSize);
        std::copy_n(microphone, kFrameSize, output);
    }

    bool prime_delay(
        const float* microphone,
        const float* reference,
        std::size_t sample_count) override {
        std::lock_guard<std::mutex> lock(mutex);
        if (should_fail) throw std::runtime_error("fake failure");
        primed_microphone.emplace_back(
            microphone, microphone + sample_count);
        primed_reference.emplace_back(
            reference, reference + sample_count);
        return true;
    }

    int current_delay_samples() const override { return 37; }
    float delay_confidence() const override { return 0.9f; }

    void reset() override {
        std::lock_guard<std::mutex> lock(mutex);
        ++reset_count;
    }

    mutable std::mutex mutex;
    bool should_fail = false;
    int reset_count = 0;
    std::vector<std::vector<float>> microphone_frames;
    std::vector<std::vector<float>> reference_frames;
    std::vector<std::vector<float>> primed_microphone;
    std::vector<std::vector<float>> primed_reference;
};

struct Harness {
    FakeCanceller canceller;
    std::atomic<std::int64_t> now_ns{0};
    std::mutex mutex;
    std::vector<EchoCancelledFrame> frames;
    std::vector<EchoCancellationPrimingEvent> priming;
    std::vector<std::string> failures;

    TimestampedEchoCancellationStream::Config config() {
        TimestampedEchoCancellationStream::Config value;
        value.playback_priming_samples = 0;
        value.current_time_ns = [this] { return now_ns.load(); };
        return value;
    }

    std::unique_ptr<TimestampedEchoCancellationStream> make(
        TimestampedEchoCancellationStream::Config value) {
        return std::make_unique<TimestampedEchoCancellationStream>(
            canceller,
            std::move(value),
            [this](const EchoCancelledFrame& frame) {
                std::lock_guard<std::mutex> lock(mutex);
                frames.push_back(frame);
            },
            [this](const std::string& failure) {
                std::lock_guard<std::mutex> lock(mutex);
                failures.push_back(failure);
            },
            [this](const EchoCancellationPrimingEvent& event) {
                std::lock_guard<std::mutex> lock(mutex);
                priming.push_back(event);
            });
    }
};

void test_timestamp_alignment_keeps_sources_separate() {
    Harness harness;
    auto stream = harness.make(harness.config());
    std::vector<float> reference(kFrameSize * 2, 0.1f);
    std::fill(
        reference.begin() + static_cast<std::ptrdiff_t>(kFrameSize),
        reference.end(),
        0.2f);
    std::vector<float> microphone(kFrameSize, 0.7f);
    stream->push_reference(
        reference.data(), reference.size(), time_for_sample(0));
    stream->push_microphone(
        microphone.data(), microphone.size(),
        time_for_sample(kFrameSize));
    stream->finish();

    assert(harness.frames.size() == 1);
    assert(harness.frames[0].raw_microphone == microphone);
    assert(harness.frames[0].cleaned_microphone == microphone);
    assert(harness.frames[0].playback_reference
           == std::vector<float>(kFrameSize, 0.2f));
    assert(harness.frames[0].start_time_ns
           == time_for_sample(kFrameSize));
}

void test_worker_waits_for_late_reference() {
    Harness harness;
    harness.now_ns.store(time_for_sample(500));
    auto stream = harness.make(harness.config());
    std::vector<float> microphone(kFrameSize, 0.8f);
    std::vector<float> reference(kFrameSize, 0.2f);
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    {
        std::lock_guard<std::mutex> lock(harness.mutex);
        assert(harness.frames.empty());
    }
    stream->push_reference(
        reference.data(), reference.size(), time_for_sample(0));
    stream->finish();
    assert(harness.frames.size() == 1);
    assert(harness.frames[0].playback_reference == reference);
}

void test_active_playback_is_primed_before_publish() {
    Harness harness;
    auto config = harness.config();
    config.playback_priming_samples = kFrameSize * 2;
    auto stream = harness.make(config);
    std::vector<float> reference(kFrameSize * 2, 0.2f);
    std::vector<float> microphone(kFrameSize * 2, 0.7f);
    stream->push_reference(
        reference.data(), reference.size(), time_for_sample(0));
    stream->push_microphone(
        microphone.data(), kFrameSize, time_for_sample(0));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    {
        std::lock_guard<std::mutex> lock(harness.mutex);
        assert(harness.frames.empty());
    }
    stream->push_microphone(
        microphone.data() + kFrameSize,
        kFrameSize,
        time_for_sample(kFrameSize));
    stream->finish();

    assert(harness.canceller.primed_microphone.size() == 1);
    assert(harness.canceller.primed_microphone[0] == microphone);
    assert(harness.canceller.primed_reference[0] == reference);
    assert(harness.frames.size() == 2);
    assert(harness.priming.size() == 1);
    assert(harness.priming[0].reason == "initial_playback");
    assert(harness.priming[0].delay_samples == 37);
}

void test_quiet_reference_uses_zeros() {
    Harness harness;
    harness.now_ns.store(time_for_sample(5000));
    auto stream = harness.make(harness.config());
    std::vector<float> microphone(kFrameSize, 0.6f);
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    stream->finish();
    assert(harness.frames.size() == 1);
    assert(std::all_of(
        harness.frames[0].playback_reference.begin(),
        harness.frames[0].playback_reference.end(),
        [](float value) { return value == 0.0f; }));
    assert(harness.frames[0].cleaned_microphone == microphone);
}

void test_model_failure_never_publishes_raw_microphone() {
    Harness harness;
    harness.canceller.should_fail = true;
    harness.now_ns.store(time_for_sample(5000));
    auto stream = harness.make(harness.config());
    std::vector<float> microphone(kFrameSize, 0.9f);
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    bool threw = false;
    try {
        stream->finish();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(harness.frames.empty());
    assert(!harness.failures.empty());
}

void test_large_backward_timestamp_fails_closed() {
    Harness harness;
    auto stream = harness.make(harness.config());
    std::vector<float> microphone(kFrameSize, 0.4f);
    stream->push_microphone(
        microphone.data(), microphone.size(),
        time_for_sample(2000));
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    bool threw = false;
    try {
        stream->finish();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(harness.frames.empty());
}

void test_timestamp_gap_resets_before_next_frame() {
    Harness harness;
    auto config = harness.config();
    auto stream = harness.make(config);
    std::vector<float> reference(kFrameSize, 0.2f);
    std::vector<float> microphone(kFrameSize, 0.5f);
    stream->push_reference(
        reference.data(), reference.size(), time_for_sample(0));
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    stream->push_reference(
        reference.data(), reference.size(), time_for_sample(2256));
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(2256));
    stream->finish();
    assert(harness.frames.size() == 2);
    assert(!harness.frames[0].discontinuity);
    assert(harness.frames[1].discontinuity);
    assert(harness.canceller.reset_count >= 2);
}

void test_finish_pads_model_only() {
    Harness harness;
    auto stream = harness.make(harness.config());
    std::vector<float> microphone(100, 0.6f);
    stream->push_microphone(
        microphone.data(), microphone.size(), time_for_sample(0));
    stream->finish();
    assert(harness.canceller.microphone_frames.size() == 1);
    assert(harness.canceller.microphone_frames[0].size()
           == kFrameSize);
    assert(harness.frames.size() == 1);
    assert(harness.frames[0].sample_count == 100);
    assert(harness.frames[0].cleaned_microphone == microphone);
}

}  // namespace

int main() {
    test_timestamp_alignment_keeps_sources_separate();
    test_worker_waits_for_late_reference();
    test_active_playback_is_primed_before_publish();
    test_quiet_reference_uses_zeros();
    test_model_failure_never_publishes_raw_microphone();
    test_large_backward_timestamp_fails_closed();
    test_timestamp_gap_resets_before_next_frame();
    test_finish_pads_model_only();
    std::puts("Timestamped echo-cancellation stream tests passed");
    return 0;
}
