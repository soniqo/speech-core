#include "speech_core/pipeline/meeting_transcription_track.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

class FakeVad final : public speech_core::VADInterface {
public:
    explicit FakeVad(int sample_rate, std::size_t chunk)
        : sample_rate_(sample_rate), chunk_(chunk) {}

    float process_chunk(
        const float* samples, std::size_t length) override {
        assert(length == chunk_);
        float peak = 0.0f;
        for (std::size_t index = 0; index < length; ++index) {
            peak = std::max(peak, std::abs(samples[index]));
        }
        return peak > 0.25f ? 0.9f : 0.0f;
    }
    void reset() override {}
    int input_sample_rate() const override { return sample_rate_; }
    std::size_t chunk_size() const override { return chunk_; }

private:
    int sample_rate_;
    std::size_t chunk_;
};

class FakePreview final : public speech_core::STTInterface {
public:
    explicit FakePreview(int sample_rate, std::string text)
        : sample_rate_(sample_rate), text_(std::move(text)) {}

    speech_core::TranscriptionResult transcribe(
        const float*, std::size_t, int) override {
        return {text_, {}, 1.0f, 0.0f, 0.0f};
    }
    int input_sample_rate() const override { return sample_rate_; }
    bool supports_streaming() const override { return true; }
    void begin_stream(int) override { active_ = true; }
    speech_core::PartialResult push_chunk(
        const float*, std::size_t) override {
        return {};
    }
    void flush_stream() override {}
    speech_core::TranscriptionResult end_stream() override {
        active_ = false;
        return {text_, {}, 1.0f, 0.0f, 0.0f};
    }
    void cancel_stream() override { active_ = false; }

private:
    int sample_rate_;
    std::string text_;
    bool active_ = false;
};

class FakeMoss final
    : public speech_core::TranscribeDiarizeInterface {
public:
    FakeMoss(
        int sample_rate,
        std::string text,
        bool structured)
        : sample_rate_(sample_rate),
          text_(std::move(text)),
          structured_(structured) {}

    speech_core::DiarizedTranscriptionResult
    transcribe_diarized(
        const float*, std::size_t length, int sample_rate) override {
        assert(sample_rate == sample_rate_);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lengths_.push_back(length);
        }
        speech_core::DiarizedTranscriptionResult result;
        result.text = text_;
        result.raw_text = structured_
            ? "[0.00][S01]" + text_ + "[0.50]"
            : text_;
        if (structured_) {
            result.segments.push_back(
                {0.0f, 0.5f, "S01", text_});
        }
        return result;
    }
    int input_sample_rate() const override { return sample_rate_; }

    std::vector<std::size_t> lengths() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return lengths_;
    }

private:
    int sample_rate_;
    std::string text_;
    bool structured_;
    mutable std::mutex mutex_;
    std::vector<std::size_t> lengths_;
};

class SegmentMoss final
    : public speech_core::TranscribeDiarizeInterface {
public:
    SegmentMoss(
        int sample_rate,
        std::vector<speech_core::DiarizedTranscriptionSegment>
            segments)
        : sample_rate_(sample_rate),
          segments_(std::move(segments)) {}

    speech_core::DiarizedTranscriptionResult
    transcribe_diarized(
        const float*, std::size_t, int sample_rate) override {
        assert(sample_rate == sample_rate_);
        speech_core::DiarizedTranscriptionResult result;
        result.segments = segments_;
        for (const auto& segment : segments_) {
            if (!result.text.empty()) result.text.push_back(' ');
            result.text += segment.text;
        }
        result.raw_text = "structured";
        return result;
    }

    int input_sample_rate() const override { return sample_rate_; }

private:
    int sample_rate_;
    std::vector<speech_core::DiarizedTranscriptionSegment>
        segments_;
};

class ScriptedMoss final
    : public speech_core::TranscribeDiarizeInterface {
public:
    ScriptedMoss(
        int sample_rate,
        std::vector<
            speech_core::DiarizedTranscriptionResult> results)
        : sample_rate_(sample_rate),
          results_(std::move(results)) {}

    speech_core::DiarizedTranscriptionResult
    transcribe_diarized(
        const float*, std::size_t length,
        int sample_rate) override {
        assert(sample_rate == sample_rate_);
        std::lock_guard<std::mutex> lock(mutex_);
        assert(next_ < results_.size());
        lengths_.push_back(length);
        return results_[next_++];
    }

    int input_sample_rate() const override {
        return sample_rate_;
    }

    std::size_t call_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_;
    }

private:
    int sample_rate_;
    std::vector<speech_core::DiarizedTranscriptionResult>
        results_;
    mutable std::mutex mutex_;
    std::size_t next_ = 0;
    std::vector<std::size_t> lengths_;
};

class FullSpanMoss final
    : public speech_core::TranscribeDiarizeInterface {
public:
    explicit FullSpanMoss(int sample_rate)
        : sample_rate_(sample_rate) {}

    speech_core::DiarizedTranscriptionResult
    transcribe_diarized(
        const float*, std::size_t length,
        int sample_rate) override {
        assert(sample_rate == sample_rate_);
        const float seconds = static_cast<float>(length)
            / static_cast<float>(sample_rate_);
        speech_core::DiarizedTranscriptionResult result;
        result.text = "continuous";
        result.raw_text = "structured";
        result.segments.push_back(
            {0.0f, seconds, "S01", "continuous"});
        return result;
    }

    int input_sample_rate() const override {
        return sample_rate_;
    }

private:
    int sample_rate_;
};

speech_core::DiarizedTranscriptionResult moss_result(
    std::string text,
    std::vector<
        speech_core::DiarizedTranscriptionSegment> segments = {}) {
    speech_core::DiarizedTranscriptionResult result;
    result.text = std::move(text);
    result.raw_text = result.text;
    result.segments = std::move(segments);
    return result;
}

std::vector<speech_core::MeetingTrackEvent> events_with_lock(
    std::mutex& mutex,
    const std::vector<speech_core::MeetingTrackEvent>& events) {
    std::lock_guard<std::mutex> lock(mutex);
    return events;
}

void push_chunks(
    speech_core::MeetingTranscriptionTrack& track,
    int sample_rate,
    int speech_chunks,
    int silence_chunks,
    std::size_t chunk,
    std::int64_t start_ns = 1'000'000'000) {
    std::vector<float> speech(chunk, 0.5f);
    std::vector<float> silence(chunk, 0.0f);
    std::size_t offset = 0;
    auto push = [&](const std::vector<float>& audio) {
        const std::int64_t time = start_ns
            + static_cast<std::int64_t>(
                std::llround(
                    static_cast<long double>(offset)
                    * 1'000'000'000.0L / sample_rate));
        track.push_audio(audio.data(), audio.size(), time);
        offset += audio.size();
    };
    for (int index = 0; index < speech_chunks; ++index) push(speech);
    for (int index = 0; index < silence_chunks; ++index) push(silence);
}

void test_final_structured_revision() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "hello world");
    FakeMoss moss(sample_rate, "hello world", true);
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    config.continuous_update_seconds = 10.0f;
    config.maximum_window_seconds = 20.0f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });
    push_chunks(track, sample_rate, 10, 5, chunk);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    const auto revision = std::find_if(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                == speech_core::MeetingTrackEventType::Revision;
        });
    assert(revision != captured.end());
    assert(revision->paragraph_final);
    assert(revision->blocks.size() == 1);
    assert(revision->blocks[0].text == "hello world");
    assert(revision->blocks[0].activity_label == "S01");
    assert(revision->blocks[0].identity_audio.size() == 500);
    assert(revision->blocks[0].start_time_ns
        >= revision->replace_start_time_ns);
    assert(revision->blocks[0].end_time_ns
        <= revision->replace_end_time_ns);
}

void test_short_microphone_requires_preview_agreement() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "different words");
    FakeMoss moss(sample_rate, "hallucinated words", false);
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.microphone = true;
    config.minimum_speech_seconds = 0.05f;
    config.silence_close_seconds = 0.5f;
    config.continuous_update_seconds = 10.0f;
    config.maximum_window_seconds = 20.0f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });
    push_chunks(track, sample_rate, 1, 5, chunk);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    assert(std::none_of(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                    == speech_core::MeetingTrackEventType::Revision
                && !event.blocks.empty();
        }));
}

void test_short_microphone_agreement_is_per_segment() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "different long words");
    SegmentMoss moss(
        sample_rate,
        {
            {0.0f, 0.2f, "S01", "hallucinated"},
            {0.2f, 1.0f, "S01", "long words"},
        });
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.microphone = true;
    config.minimum_speech_seconds = 0.05f;
    config.silence_close_seconds = 0.5f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });
    push_chunks(track, sample_rate, 10, 5, chunk);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    const auto revision = std::find_if(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                    == speech_core::MeetingTrackEventType::Revision
                && !event.blocks.empty();
        });
    assert(revision != captured.end());
    assert(revision->blocks.size() == 1);
    assert(revision->blocks[0].text == "long words");
}

void test_short_microphone_agreement_preserves_languages() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    auto run = [&](const std::string& preview_text,
                   const std::string& moss_text) {
        FakeVad vad(sample_rate, chunk);
        FakePreview preview(sample_rate, preview_text);
        FakeMoss moss(sample_rate, moss_text, false);
        std::mutex event_mutex;
        std::vector<speech_core::MeetingTrackEvent> events;
        speech_core::MeetingTranscriptionTrack::Config config;
        config.sample_rate = sample_rate;
        config.microphone = true;
        config.minimum_speech_seconds = 0.05f;
        config.silence_close_seconds = 0.5f;
        speech_core::MeetingTranscriptionTrack track(
            preview, moss, vad, config,
            [&](const speech_core::MeetingTrackEvent& event) {
                std::lock_guard<std::mutex> lock(event_mutex);
                events.push_back(event);
            });
        push_chunks(track, sample_rate, 1, 5, chunk);
        track.wait_idle();
        const auto captured = events_with_lock(event_mutex, events);
        return std::any_of(
            captured.begin(), captured.end(),
            [](const auto& event) {
                return event.type
                        == speech_core::MeetingTrackEventType::Revision
                    && !event.blocks.empty();
            });
    };
    assert(run("ПРИВЕТ", "Привет!"));
    assert(run("你好", "你好。"));
}

void test_numeric_moss_wire_marker_is_never_published() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "preview");
    FakeMoss moss(sample_rate, "[0.63]", false);
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.minimum_speech_seconds = 0.05f;
    config.silence_close_seconds = 0.5f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });
    push_chunks(track, sample_rate, 5, 5, chunk);
    track.wait_idle();
    const auto captured = events_with_lock(event_mutex, events);
    assert(std::none_of(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                    == speech_core::MeetingTrackEventType::Revision
                && !event.blocks.empty();
        }));
}

void test_continuous_windows_are_bounded() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "continuous");
    FakeMoss moss(sample_rate, "continuous", true);

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    config.continuous_update_seconds = 1.0f;
    config.maximum_window_seconds = 2.0f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [](const speech_core::MeetingTrackEvent&) {});
    push_chunks(track, sample_rate, 31, 5, chunk);
    track.wait_idle();
    const auto lengths = moss.lengths();
    assert(!lengths.empty());
    assert(std::all_of(
        lengths.begin(), lengths.end(),
        [](std::size_t length) { return length <= 2000; }));
}

void test_continuous_windows_consume_identity_audio_once() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "continuous");
    FullSpanMoss moss(sample_rate);
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;

    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    config.continuous_update_seconds = 1.0f;
    config.maximum_window_seconds = 2.0f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });
    push_chunks(track, sample_rate, 31, 5, chunk);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    std::size_t identity_samples = 0;
    for (const auto& event : captured) {
        for (const auto& block : event.blocks) {
            identity_samples += block.identity_audio.size();
        }
    }
    if (identity_samples != 3100) {
        std::cerr << "identity samples: "
                  << identity_samples << '\n';
    }
    assert(identity_samples == 3100);
}

void test_preceding_speech_recovers_activity_without_inheritance() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "yes");
    ScriptedMoss moss(
        sample_rate,
        {
            moss_result(
                "first paragraph",
                {{0.0f, 1.0f, "S01", "first paragraph"}}),
            moss_result("yes"),
            moss_result(
                "yes",
                {{1.5f, 2.5f, "S02", "yes"}}),
        });
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;
    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });

    push_chunks(track, sample_rate, 10, 5, chunk);
    track.wait_idle();
    push_chunks(
        track, sample_rate, 10, 5, chunk,
        2'500'000'000);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    assert(moss.call_count() == 3);
    assert(std::any_of(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                    == speech_core::MeetingTrackEventType::Revision
                && event.blocks.size() == 1
                && event.blocks[0].text == "yes"
                && event.blocks[0].activity_label == "S02";
        }));
}

void test_following_speech_backfills_only_compatible_fragment() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "yes");
    ScriptedMoss moss(
        sample_rate,
        {
            moss_result("yes"),
            moss_result(
                "current words",
                {{0.2f, 1.2f, "S02", "current words"}}),
            moss_result(
                "yes current words",
                {
                    {0.0f, 1.0f, "S01", "yes"},
                    {1.5f, 2.5f, "S02", "current words"},
                }),
        });
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;
    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });

    push_chunks(track, sample_rate, 10, 5, chunk);
    track.wait_idle();
    push_chunks(
        track, sample_rate, 10, 5, chunk,
        2'500'000'000);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    assert(moss.call_count() == 3);
    std::size_t matching_revisions = 0;
    bool recovered_activity = false;
    for (const auto& event : captured) {
        if (event.type
                != speech_core::MeetingTrackEventType::Revision
            || event.blocks.size() != 1
            || event.blocks[0].text != "yes") {
            continue;
        }
        ++matching_revisions;
        recovered_activity =
            recovered_activity
            || event.blocks[0].activity_label == "S01";
    }
    assert(matching_revisions == 2);
    assert(recovered_activity);
}

void test_following_speech_does_not_guess_from_mismatched_text() {
    constexpr int sample_rate = 1000;
    constexpr std::size_t chunk = 100;
    FakeVad vad(sample_rate, chunk);
    FakePreview preview(sample_rate, "yes");
    ScriptedMoss moss(
        sample_rate,
        {
            moss_result("yes"),
            moss_result(
                "current words",
                {{0.2f, 1.2f, "S02", "current words"}}),
            moss_result(
                "unrelated words",
                {{0.0f, 1.0f, "S01", "unrelated words"}}),
        });
    std::mutex event_mutex;
    std::vector<speech_core::MeetingTrackEvent> events;
    speech_core::MeetingTranscriptionTrack::Config config;
    config.sample_rate = sample_rate;
    config.silence_close_seconds = 0.5f;
    speech_core::MeetingTranscriptionTrack track(
        preview, moss, vad, config,
        [&](const speech_core::MeetingTrackEvent& event) {
            std::lock_guard<std::mutex> lock(event_mutex);
            events.push_back(event);
        });

    push_chunks(track, sample_rate, 10, 5, chunk);
    track.wait_idle();
    push_chunks(
        track, sample_rate, 10, 5, chunk,
        2'500'000'000);
    track.wait_idle();

    const auto captured = events_with_lock(event_mutex, events);
    assert(moss.call_count() == 3);
    assert(std::none_of(
        captured.begin(), captured.end(),
        [](const auto& event) {
            return event.type
                    == speech_core::MeetingTrackEventType::Revision
                && event.blocks.size() == 1
                && event.blocks[0].text == "unrelated words";
        }));
}

}  // namespace

int main() {
    test_final_structured_revision();
    test_short_microphone_requires_preview_agreement();
    test_short_microphone_agreement_is_per_segment();
    test_short_microphone_agreement_preserves_languages();
    test_numeric_moss_wire_marker_is_never_published();
    test_continuous_windows_are_bounded();
    test_continuous_windows_consume_identity_audio_once();
    test_preceding_speech_recovers_activity_without_inheritance();
    test_following_speech_backfills_only_compatible_fragment();
    test_following_speech_does_not_guess_from_mismatched_text();
    std::cout << "Meeting transcription track tests passed\n";
    return 0;
}
