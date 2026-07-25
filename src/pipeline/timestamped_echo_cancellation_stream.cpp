#include "speech_core/pipeline/timestamped_echo_cancellation_stream.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace speech_core {
namespace {

std::int64_t steady_time_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::int64_t duration_to_samples(
    std::int64_t duration_ns, int sample_rate) {
    const long double samples =
        static_cast<long double>(duration_ns)
        * static_cast<long double>(sample_rate)
        / 1000000000.0L;
    if (!std::isfinite(samples)
        || samples < static_cast<long double>(
            std::numeric_limits<std::int64_t>::min())
        || samples > static_cast<long double>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "Echo-cancellation timestamp cannot map to a sample clock");
    }
    return static_cast<std::int64_t>(std::llround(samples));
}

std::int64_t samples_to_time_ns(
    std::int64_t sample, int sample_rate) {
    const long double nanoseconds =
        static_cast<long double>(sample)
        * 1000000000.0L
        / static_cast<long double>(sample_rate);
    if (!std::isfinite(nanoseconds)
        || nanoseconds < static_cast<long double>(
            std::numeric_limits<std::int64_t>::min())
        || nanoseconds > static_cast<long double>(
            std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "Echo-cancellation sample clock cannot map to a timestamp");
    }
    return static_cast<std::int64_t>(std::llround(nanoseconds));
}

void validate_samples(
    const float* samples, std::size_t count, const char* source) {
    if (!samples && count != 0) {
        throw std::invalid_argument(
            std::string(source) + " samples must not be null");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::isfinite(samples[index])) {
            throw std::invalid_argument(
                std::string(source)
                + " samples contain NaN or infinity");
        }
    }
}

}  // namespace

class TimestampedEchoCancellationStream::Impl {
public:
    struct MicrophoneSegment {
        std::int64_t start_sample = 0;
        std::vector<float> samples;
        std::size_t offset = 0;
        bool reset_before = false;

        std::int64_t current_start() const {
            return start_sample + static_cast<std::int64_t>(offset);
        }
        std::int64_t end_sample() const {
            return start_sample
                + static_cast<std::int64_t>(samples.size());
        }
        std::size_t available() const {
            return samples.size() - offset;
        }
    };

    struct ReferenceSegment {
        std::int64_t start_sample = 0;
        std::vector<float> samples;
        std::size_t offset = 0;

        std::int64_t current_start() const {
            return start_sample + static_cast<std::int64_t>(offset);
        }
        std::int64_t end_sample() const {
            return start_sample
                + static_cast<std::int64_t>(samples.size());
        }
    };

    struct WorkFrame {
        std::vector<float> microphone;
        std::vector<float> reference;
        std::size_t actual_count = 0;
        std::int64_t start_sample = 0;
        bool reset_before = false;
    };

    Impl(
        FrameEchoCancellerInterface& canceller_value,
        Config config_value,
        OutputCallback output_value,
        FailureCallback failure_value,
        PrimingCallback priming_value)
        : canceller(canceller_value),
          config(std::move(config_value)),
          output(std::move(output_value)),
          failure_callback(std::move(failure_value)),
          priming_callback(std::move(priming_value)),
          needs_playback_priming(
              config.playback_priming_samples > 0),
          playback_priming_reason(
              config.playback_priming_samples > 0
              ? "initial_playback" : "") {
        validate_config();
        if (!config.current_time_ns) {
            config.current_time_ns = steady_time_ns;
        }
        canceller.reset();
        worker = std::thread([this] { worker_loop(); });
    }

    ~Impl() {
        cancel();
    }

    void push_microphone(
        const float* samples,
        std::size_t count,
        std::int64_t start_time_ns,
        bool discontinuity) {
        validate_samples(samples, count, "Microphone");
        if (start_time_ns < 0) {
            fail_from_capture(
                "Microphone did not provide a valid monotonic timestamp");
            return;
        }
        if (count == 0) return;
        const std::int64_t reported =
            duration_to_samples(start_time_ns, config.sample_rate);

        std::string failure_to_emit;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (capture_ended || cancelled || failure_message) return;
            std::int64_t normalized = reported;
            bool reset_before = discontinuity;
            if (last_microphone_end) {
                const std::int64_t difference =
                    reported - *last_microphone_end;
                if (difference <= -discontinuity_samples) {
                    failure_to_emit =
                        "Microphone capture timestamps moved backwards";
                } else if (std::llabs(difference)
                           < discontinuity_samples
                           && !discontinuity) {
                    // Both routine converter carry and bounded callback jitter
                    // describe continuous PCM. Preserve the established local
                    // sample clock without repeatedly resetting adapted AEC.
                    normalized = *last_microphone_end;
                } else if (difference != 0 || discontinuity) {
                    normalized = reported;
                    reset_before = true;
                    schedule_model_reset_locked(normalized);
                }
            } else if (discontinuity) {
                schedule_model_reset_locked(normalized);
            }
            if (failure_to_emit.empty()
                && queued_microphone_samples + count
                    > config.capacity_samples) {
                failure_to_emit =
                    "Microphone echo cancellation exceeded its bounded queue";
            }
            if (!failure_to_emit.empty()) {
                set_failure_locked(failure_to_emit);
            } else {
                if (!reset_before && !microphone_segments.empty()
                    && microphone_segments.back().end_sample()
                        == normalized) {
                    auto& tail = microphone_segments.back().samples;
                    tail.insert(tail.end(), samples, samples + count);
                } else {
                    microphone_segments.push_back({
                        normalized,
                        std::vector<float>(samples, samples + count),
                        0,
                        reset_before,
                    });
                }
                queued_microphone_samples += count;
                last_microphone_end =
                    normalized + static_cast<std::int64_t>(count);
                condition.notify_all();
            }
        }
        if (!failure_to_emit.empty()) {
            emit_failure_once(failure_to_emit);
        }
    }

    void push_reference(
        const float* samples,
        std::size_t count,
        std::int64_t start_time_ns,
        bool discontinuity) {
        validate_samples(samples, count, "Playback reference");
        if (start_time_ns < 0) {
            fail_from_capture(
                "Playback reference did not provide a valid monotonic timestamp");
            return;
        }
        if (count == 0) return;
        const std::int64_t reported =
            duration_to_samples(start_time_ns, config.sample_rate);

        std::string failure_to_emit;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (capture_ended || cancelled || failure_message) return;
            std::int64_t normalized = reported;
            bool reset_required = discontinuity;
            if (last_reference_end) {
                const std::int64_t difference =
                    reported - *last_reference_end;
                if (difference <= -discontinuity_samples) {
                    failure_to_emit =
                        "Playback-reference timestamps moved backwards";
                } else if (std::llabs(difference)
                           < discontinuity_samples
                           && !discontinuity) {
                    normalized = *last_reference_end;
                } else if (difference != 0 || discontinuity) {
                    normalized = reported;
                    reset_required = true;
                }
            }
            if (failure_to_emit.empty()
                && queued_reference_samples + count
                    > config.capacity_samples) {
                failure_to_emit =
                    "Playback-reference synchronization exceeded its bounded queue";
            }
            if (!failure_to_emit.empty()) {
                set_failure_locked(failure_to_emit);
            } else {
                if (reset_required) {
                    schedule_model_reset_locked(normalized);
                }
                reference_segments.push_back({
                    normalized,
                    std::vector<float>(samples, samples + count),
                    0,
                });
                queued_reference_samples += count;
                const std::int64_t end =
                    normalized + static_cast<std::int64_t>(count);
                last_reference_end = end;
                latest_reference_end = latest_reference_end
                    ? std::max(*latest_reference_end, end) : end;
                condition.notify_all();
            }
        }
        if (!failure_to_emit.empty()) {
            emit_failure_once(failure_to_emit);
        }
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!cancelled) capture_ended = true;
            condition.notify_all();
        }
        if (worker.joinable()
            && worker.get_id() != std::this_thread::get_id()) {
            worker.join();
        }
        const auto failed = failure();
        if (failed) {
            throw std::runtime_error(*failed);
        }
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (cancelled && !worker.joinable()) return;
            cancelled = true;
            microphone_segments.clear();
            reference_segments.clear();
            queued_microphone_samples = 0;
            queued_reference_samples = 0;
            condition.notify_all();
        }
        if (worker.joinable()
            && worker.get_id() != std::this_thread::get_id()) {
            worker.join();
        }
        canceller.reset();
    }

    std::optional<std::string> failure() const {
        std::lock_guard<std::mutex> lock(mutex);
        return failure_message;
    }

private:
    void validate_config() {
        if (config.sample_rate <= 0
            || config.frame_size == 0
            || config.capacity_samples < config.frame_size
            || config.reference_wait_ns < 0
            || config.clock_skew_tolerance_ns < 0
            || config.timestamp_discontinuity_ns
                <= config.clock_skew_tolerance_ns
            || config.playback_activation_rms < 0.0f
            || !std::isfinite(config.playback_activation_rms)) {
            throw std::invalid_argument(
                "Timestamped echo-cancellation configuration is invalid");
        }
        if (canceller.input_sample_rate() != config.sample_rate
            || canceller.frame_size() != config.frame_size) {
            throw std::invalid_argument(
                "Echo canceller frame contract does not match the stream");
        }
        if (!output) {
            throw std::invalid_argument(
                "Echo-cancellation output callback is required");
        }
        clock_skew_tolerance_samples = std::max<std::int64_t>(
            0,
            duration_to_samples(
                config.clock_skew_tolerance_ns,
                config.sample_rate));
        discontinuity_samples = std::max<std::int64_t>(
            clock_skew_tolerance_samples + 1,
            duration_to_samples(
                config.timestamp_discontinuity_ns,
                config.sample_rate));
        reference_wait_samples = std::max<std::int64_t>(
            0,
            duration_to_samples(
                config.reference_wait_ns,
                config.sample_rate));
    }

    void worker_loop() {
        try {
            for (;;) {
                std::optional<WorkFrame> work;
                bool should_finish = false;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    condition.wait_for(
                        lock,
                        std::chrono::milliseconds(10),
                        [this] {
                            return cancelled || failure_message
                                || capture_ended
                                || !microphone_segments.empty();
                        });
                    if (cancelled || failure_message) return;
                    work = take_work_frame_locked();
                    should_finish =
                        capture_ended && microphone_segments.empty();
                    if (!work && !should_finish) {
                        // A microphone frame can be ready before the
                        // independently scheduled reference callback. Release
                        // the lock briefly instead of spinning until either
                        // that callback arrives or the reference wait expires.
                        condition.wait_for(
                            lock, std::chrono::milliseconds(10));
                    }
                }

                if (work) {
                    process_work_frame(std::move(*work));
                    continue;
                }
                if (should_finish) {
                    flush_playback_priming_frames();
                    return;
                }
            }
        } catch (const std::exception& error) {
            const std::string message =
                "Microphone echo cancellation stopped: "
                + std::string(error.what());
            bool first = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                first = set_failure_locked(message);
            }
            if (first) emit_failure_once(message);
        } catch (...) {
            const std::string message =
                "Microphone echo cancellation stopped with an unknown error";
            bool first = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                first = set_failure_locked(message);
            }
            if (first) emit_failure_once(message);
        }
    }

    std::optional<WorkFrame> take_work_frame_locked() {
        while (!microphone_segments.empty()
               && microphone_segments.front().available() == 0) {
            microphone_segments.pop_front();
        }
        if (microphone_segments.empty()) return std::nullopt;

        auto& segment = microphone_segments.front();
        const std::size_t available = segment.available();
        const bool followed_by_discontinuity =
            microphone_segments.size() > 1
            && microphone_segments[1].reset_before;
        if (available < config.frame_size
            && !followed_by_discontinuity && !capture_ended) {
            return std::nullopt;
        }
        const std::size_t actual_count =
            std::min(config.frame_size, available);
        const std::int64_t frame_start = segment.current_start();
        const std::int64_t frame_end =
            frame_start + static_cast<std::int64_t>(actual_count);
        const std::int64_t wall_sample = duration_to_samples(
            config.current_time_ns(), config.sample_rate);
        const bool reference_ready =
            capture_ended
            || (latest_reference_end
                && *latest_reference_end >= frame_end)
            || wall_sample >= frame_end + reference_wait_samples;
        if (!reference_ready) return std::nullopt;

        WorkFrame work;
        work.actual_count = actual_count;
        work.start_sample = frame_start;
        work.microphone.assign(
            segment.samples.begin()
                + static_cast<std::ptrdiff_t>(segment.offset),
            segment.samples.begin()
                + static_cast<std::ptrdiff_t>(
                    segment.offset + actual_count));
        work.microphone.resize(config.frame_size, 0.0f);
        work.reference =
            reference_frame_locked(frame_start, actual_count);
        const bool scheduled_reset =
            pending_model_reset_at
            && frame_end > *pending_model_reset_at;
        if (scheduled_reset) pending_model_reset_at.reset();
        work.reset_before =
            (segment.reset_before && segment.offset == 0)
            || scheduled_reset;

        segment.offset += actual_count;
        queued_microphone_samples -= actual_count;
        if (segment.offset == segment.samples.size()) {
            microphone_segments.pop_front();
        }
        discard_reference_locked(frame_end);
        return work;
    }

    std::vector<float> reference_frame_locked(
        std::int64_t start_sample,
        std::size_t actual_count) const {
        std::vector<float> frame(config.frame_size, 0.0f);
        const std::int64_t requested_end =
            start_sample + static_cast<std::int64_t>(actual_count);
        for (const auto& segment : reference_segments) {
            if (segment.current_start() >= requested_end) break;
            if (segment.end_sample() <= start_sample) continue;
            const std::int64_t overlap_start =
                std::max(start_sample, segment.current_start());
            const std::int64_t overlap_end =
                std::min(requested_end, segment.end_sample());
            if (overlap_end <= overlap_start) continue;
            const std::size_t source_start = static_cast<std::size_t>(
                overlap_start - segment.start_sample);
            const std::size_t destination_start =
                static_cast<std::size_t>(overlap_start - start_sample);
            const std::size_t count =
                static_cast<std::size_t>(overlap_end - overlap_start);
            std::copy_n(
                segment.samples.data() + source_start,
                count,
                frame.data() + destination_start);
        }
        return frame;
    }

    void discard_reference_locked(std::int64_t before_sample) {
        while (!reference_segments.empty()
               && reference_segments.front().end_sample()
                    <= before_sample) {
            const auto& first = reference_segments.front();
            queued_reference_samples -=
                first.samples.size() - first.offset;
            reference_segments.pop_front();
        }
        if (reference_segments.empty()) return;
        auto& first = reference_segments.front();
        if (first.current_start() >= before_sample
            || before_sample >= first.end_sample()) {
            return;
        }
        const std::size_t discarded = static_cast<std::size_t>(
            before_sample - first.current_start());
        first.offset += discarded;
        queued_reference_samples -= discarded;
    }

    void process_work_frame(WorkFrame work) {
        if (work.reset_before) {
            flush_playback_priming_frames();
            canceller.reset();
            needs_playback_priming =
                config.playback_priming_samples > 0;
            playback_priming_reason =
                needs_playback_priming
                ? "timestamp_realign" : "";
            inactive_reference_samples = 0;
        }

        const bool reference_active = reference_is_active(work);
        if (!needs_playback_priming
            && config.playback_priming_samples > 0
            && config.playback_repriming_silence_samples > 0
            && reference_active
            && inactive_reference_samples
                >= config.playback_repriming_silence_samples) {
            needs_playback_priming = true;
            playback_priming_reason = "reference_resume";
        }

        if (needs_playback_priming
            && (!playback_priming_frames.empty()
                || reference_active)) {
            playback_priming_collected_samples += work.actual_count;
            playback_priming_frames.push_back(std::move(work));
            if (playback_priming_collected_samples
                >= config.playback_priming_samples) {
                flush_playback_priming_frames();
            }
            return;
        }

        record_reference_activity(
            reference_active, work.actual_count);
        std::vector<float> cleaned(config.frame_size, 0.0f);
        canceller.process_frame(
            work.microphone.data(),
            work.reference.data(),
            cleaned.data());
        publish(work, cleaned);
    }

    bool reference_is_active(const WorkFrame& work) const {
        if (work.actual_count == 0) return false;
        const double threshold =
            static_cast<double>(config.playback_activation_rms)
            * config.playback_activation_rms;
        double sum = 0.0;
        for (std::size_t index = 0;
             index < work.actual_count; ++index) {
            sum += static_cast<double>(work.reference[index])
                * work.reference[index];
        }
        return sum / static_cast<double>(work.actual_count)
            >= threshold;
    }

    void record_reference_activity(
        bool active, std::size_t sample_count) {
        if (active) {
            inactive_reference_samples = 0;
            return;
        }
        if (config.playback_repriming_silence_samples == 0) return;
        inactive_reference_samples = std::min(
            config.playback_repriming_silence_samples,
            inactive_reference_samples + sample_count);
    }

    std::size_t trailing_inactive_reference_samples(
        const std::vector<WorkFrame>& frames) const {
        if (config.playback_repriming_silence_samples == 0) {
            return 0;
        }
        std::size_t samples = 0;
        for (auto iterator = frames.rbegin();
             iterator != frames.rend(); ++iterator) {
            if (reference_is_active(*iterator)) break;
            samples = std::min(
                config.playback_repriming_silence_samples,
                samples + iterator->actual_count);
        }
        return samples;
    }

    void flush_playback_priming_frames() {
        if (playback_priming_frames.empty()) return;
        std::vector<WorkFrame> frames;
        frames.swap(playback_priming_frames);
        playback_priming_collected_samples = 0;

        std::size_t total = 0;
        for (const auto& frame : frames) {
            total += frame.actual_count;
        }
        std::vector<float> microphone;
        std::vector<float> reference;
        microphone.reserve(total);
        reference.reserve(total);
        for (const auto& frame : frames) {
            microphone.insert(
                microphone.end(),
                frame.microphone.begin(),
                frame.microphone.begin()
                    + static_cast<std::ptrdiff_t>(
                        frame.actual_count));
            reference.insert(
                reference.end(),
                frame.reference.begin(),
                frame.reference.begin()
                    + static_cast<std::ptrdiff_t>(
                        frame.actual_count));
        }
        canceller.prime_delay(
            microphone.data(), reference.data(), total);

        if (priming_callback && !frames.empty()) {
            EchoCancellationPrimingEvent event;
            event.start_time_ns = samples_to_time_ns(
                frames.front().start_sample, config.sample_rate);
            const auto& last = frames.back();
            event.end_time_ns = samples_to_time_ns(
                last.start_sample
                    + static_cast<std::int64_t>(last.actual_count),
                config.sample_rate);
            event.reason = playback_priming_reason.empty()
                ? "unspecified" : playback_priming_reason;
            event.delay_samples =
                canceller.current_delay_samples();
            event.delay_confidence =
                canceller.delay_confidence();
            priming_callback(event);
        }

        for (const auto& frame : frames) {
            std::vector<float> cleaned(config.frame_size, 0.0f);
            canceller.process_frame(
                frame.microphone.data(),
                frame.reference.data(),
                cleaned.data());
            publish(frame, cleaned);
        }
        inactive_reference_samples =
            trailing_inactive_reference_samples(frames);
        needs_playback_priming = false;
        playback_priming_reason.clear();
    }

    void publish(
        const WorkFrame& work,
        const std::vector<float>& cleaned) {
        if (cleaned.size() != config.frame_size) {
            throw std::runtime_error(
                "Echo canceller returned an invalid frame length");
        }
        EchoCancelledFrame frame;
        frame.start_time_ns = samples_to_time_ns(
            work.start_sample, config.sample_rate);
        frame.sample_count = work.actual_count;
        frame.discontinuity = work.reset_before;
        frame.raw_microphone.assign(
            work.microphone.begin(),
            work.microphone.begin()
                + static_cast<std::ptrdiff_t>(work.actual_count));
        frame.playback_reference.assign(
            work.reference.begin(),
            work.reference.begin()
                + static_cast<std::ptrdiff_t>(work.actual_count));
        frame.cleaned_microphone.assign(
            cleaned.begin(),
            cleaned.begin()
                + static_cast<std::ptrdiff_t>(work.actual_count));
        output(frame);
    }

    void schedule_model_reset_locked(std::int64_t at_sample) {
        pending_model_reset_at = pending_model_reset_at
            ? std::min(*pending_model_reset_at, at_sample)
            : at_sample;
    }

    bool set_failure_locked(const std::string& message) {
        if (failure_message) return false;
        failure_message = message;
        condition.notify_all();
        return true;
    }

    void fail_from_capture(const std::string& message) {
        bool first = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            first = set_failure_locked(message);
        }
        if (first) emit_failure_once(message);
    }

    void emit_failure_once(const std::string& message) {
        if (failure_callback) failure_callback(message);
    }

    FrameEchoCancellerInterface& canceller;
    Config config;
    OutputCallback output;
    FailureCallback failure_callback;
    PrimingCallback priming_callback;

    mutable std::mutex mutex;
    std::condition_variable condition;
    std::thread worker;
    std::deque<MicrophoneSegment> microphone_segments;
    std::deque<ReferenceSegment> reference_segments;
    std::size_t queued_microphone_samples = 0;
    std::size_t queued_reference_samples = 0;
    std::optional<std::int64_t> last_microphone_end;
    std::optional<std::int64_t> last_reference_end;
    std::optional<std::int64_t> latest_reference_end;
    std::optional<std::int64_t> pending_model_reset_at;
    std::optional<std::string> failure_message;
    std::int64_t clock_skew_tolerance_samples = 0;
    std::int64_t discontinuity_samples = 0;
    std::int64_t reference_wait_samples = 0;
    bool capture_ended = false;
    bool cancelled = false;

    // Worker-only model acquisition state.
    std::vector<WorkFrame> playback_priming_frames;
    std::size_t playback_priming_collected_samples = 0;
    bool needs_playback_priming = false;
    std::string playback_priming_reason;
    std::size_t inactive_reference_samples = 0;
};

TimestampedEchoCancellationStream::
TimestampedEchoCancellationStream(
    FrameEchoCancellerInterface& canceller,
    Config config,
    OutputCallback output,
    FailureCallback failure,
    PrimingCallback priming)
    : impl_(std::make_unique<Impl>(
          canceller,
          std::move(config),
          std::move(output),
          std::move(failure),
          std::move(priming))) {}

TimestampedEchoCancellationStream::
~TimestampedEchoCancellationStream() = default;

void TimestampedEchoCancellationStream::push_microphone(
    const float* samples,
    std::size_t count,
    std::int64_t start_time_ns,
    bool discontinuity) {
    impl_->push_microphone(
        samples, count, start_time_ns, discontinuity);
}

void TimestampedEchoCancellationStream::push_reference(
    const float* samples,
    std::size_t count,
    std::int64_t start_time_ns,
    bool discontinuity) {
    impl_->push_reference(
        samples, count, start_time_ns, discontinuity);
}

void TimestampedEchoCancellationStream::finish() {
    impl_->finish();
}

void TimestampedEchoCancellationStream::cancel() {
    impl_->cancel();
}

std::optional<std::string>
TimestampedEchoCancellationStream::failure() const {
    return impl_->failure();
}

}  // namespace speech_core
