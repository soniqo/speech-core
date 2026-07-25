#include "speech_core/pipeline/meeting_transcription_track.h"
#include "speech_core/transcription/moss_transcript_parser.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace speech_core {
namespace {

using Clock = std::chrono::steady_clock;

std::size_t seconds_to_samples(float seconds, int sample_rate) {
    if (!std::isfinite(seconds) || seconds < 0.0f || sample_rate <= 0) {
        throw std::invalid_argument(
            "Meeting track duration/sample rate is invalid");
    }
    return static_cast<std::size_t>(std::llround(
        static_cast<double>(seconds) * sample_rate));
}

std::string trim(std::string value) {
    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](unsigned char item) { return std::isspace(item); });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](unsigned char item) { return std::isspace(item); }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

bool locale_control_at(const std::string& text, std::size_t offset) {
    if (offset + 7 > text.size()
        || text[offset] != '<' || text[offset + 6] != '>') {
        return false;
    }
    const auto lower = [](char value) {
        return value >= 'a' && value <= 'z';
    };
    const auto upper = [](char value) {
        return value >= 'A' && value <= 'Z';
    };
    return lower(text[offset + 1])
        && lower(text[offset + 2])
        && text[offset + 3] == '-'
        && upper(text[offset + 4])
        && upper(text[offset + 5]);
}

std::string sanitize_preview(const std::string& text) {
    std::string output;
    output.reserve(text.size());
    for (std::size_t index = 0; index < text.size();) {
        if (locale_control_at(text, index)) {
            index += 7;
        } else {
            output.push_back(text[index++]);
        }
    }
    return trim(std::move(output));
}

bool has_wire_control(const std::string& text) {
    return text.find("<|") != std::string::npos
        || contains_moss_wire_marker(text);
}

bool decode_utf8(
    const std::string& text,
    std::size_t& offset,
    std::uint32_t& codepoint) {
    if (offset >= text.size()) return false;
    const unsigned char first =
        static_cast<unsigned char>(text[offset++]);
    if (first < 0x80) {
        codepoint = first;
        return true;
    }
    int continuation_count = 0;
    if ((first & 0xe0u) == 0xc0u) {
        codepoint = first & 0x1fu;
        continuation_count = 1;
    } else if ((first & 0xf0u) == 0xe0u) {
        codepoint = first & 0x0fu;
        continuation_count = 2;
    } else if ((first & 0xf8u) == 0xf0u) {
        codepoint = first & 0x07u;
        continuation_count = 3;
    } else {
        return false;
    }
    if (offset + static_cast<std::size_t>(continuation_count)
        > text.size()) {
        return false;
    }
    for (int index = 0; index < continuation_count; ++index) {
        const unsigned char next =
            static_cast<unsigned char>(text[offset++]);
        if ((next & 0xc0u) != 0x80u) return false;
        codepoint = (codepoint << 6u) | (next & 0x3fu);
    }
    return true;
}

void append_utf8(std::string& output, std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
        output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
        output.push_back(static_cast<char>(
            0x80u | ((codepoint >> 6u) & 0x3fu)));
        output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    } else {
        output.push_back(static_cast<char>(0xf0u | (codepoint >> 18u)));
        output.push_back(static_cast<char>(
            0x80u | ((codepoint >> 12u) & 0x3fu)));
        output.push_back(static_cast<char>(
            0x80u | ((codepoint >> 6u) & 0x3fu)));
        output.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    }
}

std::uint32_t simple_lower(std::uint32_t codepoint) {
    if (codepoint >= 'A' && codepoint <= 'Z') return codepoint + 32;
    if ((codepoint >= 0x00c0 && codepoint <= 0x00d6)
        || (codepoint >= 0x00d8 && codepoint <= 0x00de)
        || (codepoint >= 0x0391 && codepoint <= 0x03a1)
        || (codepoint >= 0x03a3 && codepoint <= 0x03ab)
        || (codepoint >= 0x0410 && codepoint <= 0x042f)) {
        return codepoint + 32;
    }
    if (codepoint == 0x0401) return 0x0451;
    return codepoint;
}

bool unicode_space(std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        return std::isspace(static_cast<unsigned char>(codepoint));
    }
    return codepoint == 0x00a0
        || codepoint == 0x1680
        || (codepoint >= 0x2000 && codepoint <= 0x200a)
        || codepoint == 0x2028
        || codepoint == 0x2029
        || codepoint == 0x202f
        || codepoint == 0x205f
        || codepoint == 0x3000;
}

bool unicode_punctuation(std::uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        return !std::isalnum(static_cast<unsigned char>(codepoint))
            && !std::isspace(static_cast<unsigned char>(codepoint));
    }
    return (codepoint >= 0x2000 && codepoint <= 0x206f)
        || (codepoint >= 0x2e00 && codepoint <= 0x2e7f)
        || (codepoint >= 0x3000 && codepoint <= 0x303f)
        || (codepoint >= 0xfe10 && codepoint <= 0xfe1f)
        || (codepoint >= 0xfe30 && codepoint <= 0xfe4f)
        || (codepoint >= 0xff01 && codepoint <= 0xff0f)
        || (codepoint >= 0xff1a && codepoint <= 0xff20)
        || (codepoint >= 0xff3b && codepoint <= 0xff40)
        || (codepoint >= 0xff5b && codepoint <= 0xff65)
        || codepoint == 0x060c
        || codepoint == 0x061b
        || codepoint == 0x061f
        || codepoint == 0x0964
        || codepoint == 0x0965;
}

std::vector<std::string> normalized_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;
    std::size_t offset = 0;
    while (offset < text.size()) {
        std::uint32_t codepoint = 0;
        const std::size_t previous = offset;
        if (!decode_utf8(text, offset, codepoint)) {
            offset = previous + 1;
            continue;
        }
        if (unicode_space(codepoint)) {
            if (!token.empty()) {
                tokens.push_back(std::move(token));
                token.clear();
            }
            continue;
        }
        if (unicode_punctuation(codepoint)) continue;
        append_utf8(token, simple_lower(codepoint));
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    return tokens;
}

bool contains_contiguous(
    const std::vector<std::string>& text,
    const std::vector<std::string>& sequence) {
    if (sequence.empty() || text.size() < sequence.size()) return false;
    for (std::size_t start = 0;
         start + sequence.size() <= text.size(); ++start) {
        if (std::equal(
                sequence.begin(), sequence.end(),
                text.begin() + static_cast<std::ptrdiff_t>(start))) {
            return true;
        }
    }
    return false;
}

bool short_microphone_agrees(
    const std::string& result,
    const std::string& preview,
    std::size_t index,
    std::size_t count) {
    const auto result_tokens = normalized_tokens(result);
    const auto preview_tokens = normalized_tokens(preview);
    if (result_tokens.empty() || preview_tokens.empty()) return false;
    if (count <= 1) return result_tokens == preview_tokens;
    if (index == 0) {
        return preview_tokens.size() >= result_tokens.size()
            && std::equal(
                result_tokens.begin(), result_tokens.end(),
                preview_tokens.begin());
    }
    if (index + 1 == count) {
        return preview_tokens.size() >= result_tokens.size()
            && std::equal(
                result_tokens.rbegin(), result_tokens.rend(),
                preview_tokens.rbegin());
    }
    return contains_contiguous(preview_tokens, result_tokens);
}

double text_similarity(
    const std::vector<std::string>& left,
    const std::vector<std::string>& right) {
    if (left.empty() || right.empty()) return 0.0;
    std::vector<std::size_t> previous(right.size() + 1, 0);
    for (const auto& left_word : left) {
        std::vector<std::size_t> current(
            right.size() + 1, 0);
        for (std::size_t index = 0;
             index < right.size(); ++index) {
            current[index + 1] =
                left_word == right[index]
                ? previous[index] + 1
                : std::max(
                    previous[index + 1], current[index]);
        }
        previous = std::move(current);
    }
    return static_cast<double>(previous.back())
        / static_cast<double>(
            std::max(left.size(), right.size()));
}

bool activity_recovery_compatible(
    const std::string& original,
    const std::string& recovered) {
    return text_similarity(
        normalized_tokens(original),
        normalized_tokens(recovered)) >= 0.60;
}

double elapsed_ms(const Clock::time_point& started) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - started).count();
}

}  // namespace

class MeetingTranscriptionTrack::Impl {
public:
    struct TimeSegment {
        std::uint64_t first_sample = 0;
        std::uint64_t sample_count = 0;
        std::int64_t start_time_ns = 0;
    };

    struct Request {
        std::uint64_t generation = 0;
        std::uint64_t sequence = 0;
        std::uint64_t audio_start_sample = 0;
        std::int64_t audio_start_time_ns = 0;
        std::uint64_t speech_start_sample = 0;
        std::uint64_t speech_end_sample = 0;
        std::uint64_t identity_evidence_start_sample = 0;
        std::vector<float> audio;
        std::vector<float> recovery_audio;
        std::uint64_t recovery_start_sample = 0;
        std::int64_t recovery_start_time_ns = 0;
        std::string preview_text;
        bool paragraph_final = false;
    };

    struct PendingActivityRecovery {
        std::uint64_t generation = 0;
        std::uint64_t speech_start_sample = 0;
        std::uint64_t speech_end_sample = 0;
        std::vector<MeetingTranscriptBlock> blocks;
    };

    Impl(
        STTInterface& preview_value,
        TranscribeDiarizeInterface& final_model_value,
        VADInterface& vad_value,
        Config config_value,
        EventCallback callback_value)
        : preview(preview_value),
          final_model(final_model_value),
          vad(vad_value),
          config(config_value),
          callback(std::move(callback_value)) {
        validate_config();
        if (!preview.supports_streaming()) {
            throw std::invalid_argument(
                "Meeting track preview must support streaming");
        }
        if (preview.input_sample_rate() != config.sample_rate
            || final_model.input_sample_rate() != config.sample_rate
            || vad.input_sample_rate() != config.sample_rate) {
            throw std::invalid_argument(
                "Meeting track models must share the configured sample rate");
        }
        if (vad.chunk_size() == 0) {
            throw std::invalid_argument(
                "Meeting track VAD chunk size must be positive");
        }
        worker = std::thread([this] { worker_loop(); });
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            requests.clear();
            request_cv.notify_all();
        }
        if (worker.joinable()) worker.join();
    }

    void push_audio(
        const float* samples,
        std::size_t length,
        std::int64_t start_time_ns,
        bool discontinuity) {
        if ((!samples && length != 0) || start_time_ns < 0) {
            throw std::invalid_argument(
                "Meeting track received invalid audio metadata");
        }
        for (std::size_t index = 0; index < length; ++index) {
            if (!std::isfinite(samples[index])) {
                throw std::invalid_argument(
                    "Meeting track audio contains NaN or infinity");
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (discontinuity) reset_locked(false);
        if (length == 0) return;

        const std::uint64_t first_sample = next_sample;
        time_segments.push_back({
            first_sample,
            static_cast<std::uint64_t>(length),
            start_time_ns,
        });
        ring.insert(ring.end(), samples, samples + length);
        pending.insert(pending.end(), samples, samples + length);
        next_sample += static_cast<std::uint64_t>(length);
        process_vad_locked();
        prune_ring_locked();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (speech_active) close_paragraph_locked(true);
        }
        wait_idle();
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        reset_locked(false);
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lock(mutex);
        idle_cv.wait(lock, [this] {
            return requests.empty() && !worker_busy;
        });
    }

private:
    void validate_config() const {
        if (config.sample_rate <= 0
            || config.vad_offset < 0.0f
            || config.vad_onset > 1.0f
            || config.vad_offset > config.vad_onset
            || config.minimum_speech_seconds <= 0.0f
            || config.silence_close_seconds <= 0.0f
            || config.continuous_update_seconds <= 0.0f
            || config.maximum_window_seconds
                < config.continuous_update_seconds) {
            throw std::invalid_argument(
                "Meeting track configuration is invalid");
        }
    }

    std::uint64_t pending_first_sample_locked() const {
        return next_sample - static_cast<std::uint64_t>(pending.size());
    }

    void process_vad_locked() {
        const std::size_t chunk = vad.chunk_size();
        while (pending.size() >= chunk) {
            const std::uint64_t chunk_start =
                pending_first_sample_locked();
            const std::uint64_t chunk_end =
                chunk_start + static_cast<std::uint64_t>(chunk);
            vad_processed_sample = chunk_end;
            const float probability =
                vad.process_chunk(pending.data(), chunk);
            if (!std::isfinite(probability)) {
                emit_error_locked(
                    "Meeting track VAD returned a non-finite probability");
                reset_locked(false);
                return;
            }

            if (!speech_active) {
                if (probability >= config.vad_onset) {
                    start_paragraph_locked(chunk_start);
                    handle_active_chunk_locked(
                        pending.data(), chunk, chunk_end, probability);
                }
            } else {
                handle_active_chunk_locked(
                    pending.data(), chunk, chunk_end, probability);
            }
            pending.erase(
                pending.begin(),
                pending.begin() + static_cast<std::ptrdiff_t>(chunk));
        }
    }

    void start_paragraph_locked(std::uint64_t speech_start) {
        speech_active = true;
        paragraph_speech_start = speech_start;
        paragraph_last_voiced_end = speech_start;
        paragraph_preview.clear();
        silence_samples = 0;
        const std::uint64_t pre_roll = seconds_to_samples(
            config.pre_roll_seconds, config.sample_rate);
        paragraph_audio_start = speech_start > pre_roll
            ? speech_start - pre_roll : 0;
        paragraph_audio_start =
            std::max(paragraph_audio_start, ring_start_sample);
        next_continuous_update = speech_start
            + seconds_to_samples(
                config.continuous_update_seconds, config.sample_rate);
        preview.begin_stream(config.sample_rate);
        const std::vector<float> context = audio_slice_locked(
            paragraph_audio_start, speech_start);
        if (!context.empty()) {
            append_preview_locked(context.data(), context.size());
        }
    }

    void handle_active_chunk_locked(
        const float* samples,
        std::size_t length,
        std::uint64_t chunk_end,
        float probability) {
        append_preview_locked(samples, length);
        if (probability >= config.vad_offset) {
            paragraph_last_voiced_end = chunk_end;
            silence_samples = 0;
        } else {
            silence_samples += length;
        }

        while (speech_active
               && chunk_end >= next_continuous_update) {
            queue_request_locked(
                next_continuous_update, false, paragraph_preview);
            next_continuous_update += seconds_to_samples(
                config.continuous_update_seconds, config.sample_rate);
        }
        const std::size_t close_samples = seconds_to_samples(
            config.silence_close_seconds, config.sample_rate);
        if (silence_samples >= close_samples) {
            close_paragraph_locked(false);
        }
    }

    void append_preview_locked(
        const float* samples, std::size_t length) {
        try {
            const PartialResult partial =
                preview.push_chunk(samples, length);
            const std::string delta = sanitize_preview(partial.text);
            if (!delta.empty()) paragraph_preview += delta;
            MeetingTrackEvent event;
            event.type = MeetingTrackEventType::Preview;
            event.preview_text = trim(paragraph_preview);
            emit_unlocked(event);
        } catch (const std::exception& error) {
            emit_error_locked(
                "Meeting track preview failed: "
                + std::string(error.what()));
            reset_locked(false);
        }
    }

    void close_paragraph_locked(bool forced) {
        if (!speech_active) return;
        try {
            preview.flush_stream();
            const TranscriptionResult final_preview =
                preview.end_stream();
            const std::string full =
                sanitize_preview(final_preview.text);
            if (!full.empty()) paragraph_preview = full;
        } catch (const std::exception& error) {
            emit_error_locked(
                "Meeting track preview finalization failed: "
                + std::string(error.what()));
            preview.cancel_stream();
        }

        const std::size_t minimum_samples = seconds_to_samples(
            config.minimum_speech_seconds, config.sample_rate);
        const std::uint64_t voiced =
            paragraph_last_voiced_end > paragraph_speech_start
            ? paragraph_last_voiced_end - paragraph_speech_start : 0;
        if (voiced >= minimum_samples) {
            const std::uint64_t post_roll = seconds_to_samples(
                config.post_roll_seconds, config.sample_rate);
            const std::uint64_t available_end =
                forced ? next_sample : vad_processed_sample;
            const std::uint64_t audio_end = std::min(
                available_end,
                paragraph_last_voiced_end + post_roll);
            queue_request_locked(
                audio_end, true, paragraph_preview);
        }
        if (paragraph_last_voiced_end > paragraph_speech_start) {
            previous_completed_speech_start =
                paragraph_speech_start;
            previous_completed_speech_end =
                paragraph_last_voiced_end;
            has_previous_completed_speech = true;
        }
        MeetingTrackEvent clear;
        clear.type = MeetingTrackEventType::Preview;
        emit_unlocked(clear);
        speech_active = false;
        silence_samples = 0;
        paragraph_preview.clear();
        if (forced) pending.clear();
    }

    void queue_request_locked(
        std::uint64_t requested_end,
        bool paragraph_final,
        const std::string& preview_text) {
        const std::uint64_t maximum_samples = seconds_to_samples(
            config.maximum_window_seconds, config.sample_rate);
        const std::uint64_t window_start = requested_end > maximum_samples
            ? requested_end - maximum_samples : 0;
        const std::uint64_t audio_start = std::max(
            {paragraph_audio_start, window_start, ring_start_sample});
        const std::uint64_t speech_start = std::max(
            paragraph_speech_start, audio_start);
        const std::uint64_t speech_end = std::min(
            paragraph_last_voiced_end, requested_end);
        if (speech_end <= speech_start || requested_end <= audio_start) {
            return;
        }
        Request request;
        request.generation = generation;
        request.sequence = ++next_sequence;
        request.audio_start_sample = audio_start;
        request.audio_start_time_ns =
            time_for_sample_locked(audio_start);
        request.speech_start_sample = speech_start;
        request.speech_end_sample = speech_end;
        request.audio =
            audio_slice_locked(audio_start, requested_end);
        request.preview_text = preview_text;
        request.paragraph_final = paragraph_final;

        if (!config.microphone
            && has_previous_completed_speech
            && previous_completed_speech_end <= speech_start) {
            const std::uint64_t recovery_samples =
                seconds_to_samples(
                    config.activity_recovery_context_seconds,
                    config.sample_rate);
            const std::uint64_t maximum_start =
                requested_end > maximum_samples
                ? requested_end - maximum_samples : 0;
            const std::uint64_t context_start =
                speech_start > recovery_samples
                ? speech_start - recovery_samples : 0;
            request.recovery_start_sample = std::max({
                ring_start_sample,
                previous_completed_speech_start,
                maximum_start,
                context_start,
            });
            const std::uint64_t retained_prior =
                std::min(
                    previous_completed_speech_end,
                    speech_start)
                    > request.recovery_start_sample
                ? std::min(
                    previous_completed_speech_end,
                    speech_start)
                    - request.recovery_start_sample
                : 0;
            const std::uint64_t previous_length =
                previous_completed_speech_end
                    - previous_completed_speech_start;
            const std::uint64_t required_prior = std::min<
                std::uint64_t>(
                previous_length,
                std::max<std::uint64_t>(
                    1,
                    seconds_to_samples(
                        config.pre_roll_seconds,
                        config.sample_rate)));
            if (retained_prior >= required_prior
                && request.recovery_start_sample < audio_start) {
                request.recovery_start_time_ns =
                    time_for_sample_locked(
                        request.recovery_start_sample);
                request.recovery_audio = audio_slice_locked(
                    request.recovery_start_sample,
                    requested_end);
            }
        }

        // Source-local requests are serialized in capture order. Earlier
        // rolling windows carry stable transcript and identity evidence that
        // a final latest-20-second window cannot reconstruct, so finalization
        // must not drop them.
        constexpr std::size_t kMaximumPendingRequests = 8;
        if (requests.size() >= kMaximumPendingRequests) {
            emit_error_locked(
                "Meeting track final inference queue overran");
            reset_locked(false);
            return;
        }
        requests.push_back(std::move(request));
        request_cv.notify_one();
    }

    std::vector<float> audio_slice_locked(
        std::uint64_t first,
        std::uint64_t last) const {
        first = std::max(first, ring_start_sample);
        last = std::min(last, next_sample);
        if (last <= first) return {};
        const std::size_t offset =
            static_cast<std::size_t>(first - ring_start_sample);
        const std::size_t count =
            static_cast<std::size_t>(last - first);
        if (offset > ring.size() || count > ring.size() - offset) {
            return {};
        }
        return std::vector<float>(
            ring.begin() + static_cast<std::ptrdiff_t>(offset),
            ring.begin() + static_cast<std::ptrdiff_t>(offset + count));
    }

    std::int64_t time_for_sample_locked(
        std::uint64_t sample) const {
        for (const auto& segment : time_segments) {
            if (sample >= segment.first_sample
                && sample <= segment.first_sample
                    + segment.sample_count) {
                const std::uint64_t offset =
                    sample - segment.first_sample;
                const long double nanoseconds =
                    static_cast<long double>(offset)
                    * 1000000000.0L
                    / static_cast<long double>(config.sample_rate);
                return segment.start_time_ns
                    + static_cast<std::int64_t>(
                        std::llround(nanoseconds));
            }
        }
        if (!time_segments.empty()) {
            const auto& segment = time_segments.back();
            const std::uint64_t end =
                segment.first_sample + segment.sample_count;
            if (sample >= end) {
                const long double nanoseconds =
                    static_cast<long double>(sample - segment.first_sample)
                    * 1000000000.0L
                    / static_cast<long double>(config.sample_rate);
                return segment.start_time_ns
                    + static_cast<std::int64_t>(
                        std::llround(nanoseconds));
            }
        }
        throw std::runtime_error(
            "Meeting track could not map a sample to capture time");
    }

    void prune_ring_locked() {
        const std::uint64_t keep_samples = seconds_to_samples(
            config.maximum_window_seconds
                + config.activity_recovery_context_seconds
                + config.pre_roll_seconds + 2.0f,
            config.sample_rate);
        const std::uint64_t protected_start = speech_active
            ? paragraph_audio_start
            : (next_sample > keep_samples
                ? next_sample - keep_samples : 0);
        const std::uint64_t desired_start = next_sample > keep_samples
            ? next_sample - keep_samples : 0;
        const std::uint64_t next_start =
            std::min(protected_start, desired_start);
        if (next_start <= ring_start_sample) return;
        const std::size_t remove = static_cast<std::size_t>(
            std::min<std::uint64_t>(
                next_start - ring_start_sample, ring.size()));
        ring.erase(
            ring.begin(),
            ring.begin() + static_cast<std::ptrdiff_t>(remove));
        ring_start_sample += remove;
        while (!time_segments.empty()
               && time_segments.front().first_sample
                    + time_segments.front().sample_count
                    < ring_start_sample) {
            time_segments.pop_front();
        }
    }

    void reset_locked(bool preserve_timeline) {
        ++generation;
        requests.clear();
        speech_active = false;
        silence_samples = 0;
        paragraph_preview.clear();
        pending.clear();
        ring.clear();
        ring_start_sample = next_sample;
        time_segments.clear();
        has_previous_completed_speech = false;
        previous_completed_speech_start = 0;
        previous_completed_speech_end = 0;
        vad.reset();
        preview.cancel_stream();
        MeetingTrackEvent clear;
        clear.type = MeetingTrackEventType::Preview;
        emit_unlocked(clear);
        if (!preserve_timeline) {
            // Absolute sample indices remain monotonic so stale worker results
            // cannot alias a new recording; generation is the primary guard.
        }
    }

    void worker_loop() {
        for (;;) {
            Request request;
            {
                std::unique_lock<std::mutex> lock(mutex);
                request_cv.wait(lock, [this] {
                    return stopping || !requests.empty();
                });
                if (stopping) return;
                request = std::move(requests.front());
                requests.pop_front();
                worker_busy = true;
            }

            MeetingTrackEvent event;
            std::optional<MeetingTrackEvent>
                following_recovery;
            const bool may_retry_pending =
                pending_activity_recovery
                && pending_activity_recovery->generation
                    == request.generation
                && request.speech_start_sample
                    >= pending_activity_recovery
                        ->speech_end_sample;
            try {
                Request inference_request = request;
                if (!config.microphone) {
                    if (worker_evidence_generation
                        != request.generation) {
                        worker_evidence_generation =
                            request.generation;
                        worker_evidence_through = 0;
                    }
                    inference_request
                        .identity_evidence_start_sample =
                        std::max(
                            request.speech_start_sample,
                            worker_evidence_through);
                }
                event = run_request(inference_request);
                if (may_retry_pending
                    && event.type
                        != MeetingTrackEventType::Error) {
                    following_recovery =
                        run_following_activity_recovery(
                            *pending_activity_recovery,
                            request);
                }
            } catch (const std::exception& error) {
                event.type = MeetingTrackEventType::Error;
                event.error =
                    "Meeting track MOSS inference failed: "
                    + std::string(error.what());
            }

            bool publish = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                publish = request.generation == generation;
                worker_busy = false;
                if (requests.empty()) idle_cv.notify_all();
            }
            if (publish
                && (event.type != MeetingTrackEventType::Revision
                    || !event.blocks.empty())) {
                emit_unlocked(event);
            } else if (publish
                       && event.type == MeetingTrackEventType::Revision
                       && request.paragraph_final) {
                // A confident abstention still clears the revisable caption;
                // it deliberately does not erase already-published blocks.
                MeetingTrackEvent clear;
                clear.type = MeetingTrackEventType::Preview;
                emit_unlocked(clear);
            }
            if (!publish) continue;
            if (!config.microphone
                && event.type
                    == MeetingTrackEventType::Revision
                && !event.blocks.empty()) {
                worker_evidence_through = std::max(
                    worker_evidence_through,
                    request.speech_end_sample);
            }
            if (may_retry_pending) {
                pending_activity_recovery.reset();
                if (following_recovery
                    && !following_recovery->blocks.empty()) {
                    emit_unlocked(*following_recovery);
                }
            } else if (pending_activity_recovery
                       && pending_activity_recovery->generation
                            != request.generation) {
                pending_activity_recovery.reset();
            }
            if (request.paragraph_final
                && event.type
                    == MeetingTrackEventType::Revision
                && !event.blocks.empty()
                && !blocks_contain_activity(event.blocks)
                && !config.microphone) {
                pending_activity_recovery =
                    PendingActivityRecovery{
                        request.generation,
                        request.speech_start_sample,
                        request.speech_end_sample,
                        event.blocks,
                    };
            }
        }
    }

    MeetingTrackEvent run_request(const Request& request) {
        const auto started = Clock::now();
        DiarizedTranscriptionResult result =
            final_model.transcribe_diarized(
                request.audio.data(),
                request.audio.size(),
                config.sample_rate);
        if (result.text.empty() || has_wire_control(result.text)) {
            return revision_base(request, elapsed_ms(started));
        }

        std::vector<DiarizedTranscriptionSegment> segments =
            result.segments;
        if (segments.empty() && !config.microphone
            && !request.recovery_audio.empty()) {
            const auto recovered = final_model.transcribe_diarized(
                request.recovery_audio.data(),
                request.recovery_audio.size(),
                config.sample_rate);
            std::string current_text;
            std::vector<DiarizedTranscriptionSegment>
                current_segments;
            const double current_start_seconds =
                static_cast<double>(
                    request.speech_start_sample
                    - request.recovery_start_sample)
                / config.sample_rate;
            const double current_end_seconds =
                static_cast<double>(
                    request.speech_end_sample
                    - request.recovery_start_sample)
                / config.sample_rate;
            for (const auto& segment : recovered.segments) {
                if (segment.end_time <= current_start_seconds
                    || segment.start_time >= current_end_seconds) {
                    continue;
                }
                current_segments.push_back(segment);
                if (!current_text.empty()) current_text.push_back(' ');
                current_text += segment.text;
            }
            if (!current_segments.empty()
                && activity_recovery_compatible(
                    current_text, result.text)) {
                const double shift = static_cast<double>(
                    request.audio_start_sample
                    - request.recovery_start_sample)
                    / config.sample_rate;
                for (auto& segment : current_segments) {
                    segment.start_time -= static_cast<float>(shift);
                    segment.end_time -= static_cast<float>(shift);
                }
                segments = std::move(current_segments);
            }
        }

        MeetingTrackEvent event =
            revision_base(request, elapsed_ms(started));
        if (segments.empty()) {
            const std::uint64_t voiced_samples =
                request.speech_end_sample
                    - request.speech_start_sample;
            if (config.microphone
                && voiced_samples < seconds_to_samples(
                    config.microphone_short_agreement_seconds,
                    config.sample_rate)
                && !short_microphone_agrees(
                    result.text, request.preview_text, 0, 1)) {
                return event;
            }
            MeetingTranscriptBlock block;
            block.start_time_ns = map_request_time(
                request, request.speech_start_sample);
            block.end_time_ns = map_request_time(
                request, request.speech_end_sample);
            block.text = trim(result.text);
            event.blocks.push_back(std::move(block));
            return event;
        }

        struct Candidate {
            std::uint64_t start = 0;
            std::uint64_t end = 0;
            std::string speaker;
            std::string text;
        };
        std::vector<Candidate> candidates;
        for (const auto& segment : segments) {
            const auto relative_start =
                static_cast<std::int64_t>(std::llround(
                    static_cast<double>(segment.start_time)
                    * config.sample_rate));
            const auto relative_end =
                static_cast<std::int64_t>(std::llround(
                    static_cast<double>(segment.end_time)
                    * config.sample_rate));
            const std::uint64_t start = std::clamp<std::uint64_t>(
                relative_start > 0
                    ? request.audio_start_sample
                        + static_cast<std::uint64_t>(relative_start)
                    : request.audio_start_sample,
                request.speech_start_sample,
                request.speech_end_sample);
            const std::uint64_t end = std::clamp<std::uint64_t>(
                relative_end > 0
                    ? request.audio_start_sample
                        + static_cast<std::uint64_t>(relative_end)
                    : request.audio_start_sample,
                request.speech_start_sample,
                request.speech_end_sample);
            const std::string text = trim(segment.text);
            if (end > start && !text.empty()
                && !has_wire_control(text)) {
                candidates.push_back({
                    start, end, segment.speaker, text,
                });
            }
        }

        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (config.microphone
                && candidate.end - candidate.start
                    < seconds_to_samples(
                        config.microphone_short_agreement_seconds,
                        config.sample_rate)
                && !short_microphone_agrees(
                    candidate.text,
                    request.preview_text,
                    index,
                    candidates.size())) {
                continue;
            }
            bool overlaps = false;
            for (std::size_t other = 0;
                 other < candidates.size(); ++other) {
                if (other == index) continue;
                overlaps = candidate.start < candidates[other].end
                    && candidates[other].start < candidate.end;
                if (overlaps) break;
            }
            MeetingTranscriptBlock block;
            block.start_time_ns =
                map_request_time(request, candidate.start);
            block.end_time_ns =
                map_request_time(request, candidate.end);
            block.text = candidate.text;
            if (!config.microphone) {
                block.activity_label = candidate.speaker;
            }
            if (!config.microphone && !overlaps) {
                const std::size_t first = static_cast<std::size_t>(
                    std::max(
                        candidate.start,
                        request.identity_evidence_start_sample)
                    - request.audio_start_sample);
                const std::size_t last = static_cast<std::size_t>(
                    candidate.end - request.audio_start_sample);
                if (last <= request.audio.size() && last > first) {
                    block.identity_audio.assign(
                        request.audio.begin()
                            + static_cast<std::ptrdiff_t>(first),
                        request.audio.begin()
                            + static_cast<std::ptrdiff_t>(last));
                }
            }
            event.blocks.push_back(std::move(block));
        }
        return event;
    }

    static bool blocks_contain_activity(
        const std::vector<MeetingTranscriptBlock>& blocks) {
        return std::any_of(
            blocks.begin(), blocks.end(),
            [](const auto& block) {
                return !trim(block.activity_label).empty();
            });
    }

    static bool following_recovery_compatible(
        const std::vector<MeetingTranscriptBlock>& original,
        const std::vector<MeetingTranscriptBlock>& recovered) {
        if (original.empty()
            || blocks_contain_activity(original)
            || !blocks_contain_activity(recovered)) {
            return false;
        }
        std::string original_text;
        for (const auto& block : original) {
            if (!original_text.empty()) {
                original_text.push_back(' ');
            }
            original_text += block.text;
        }
        std::string activity_text;
        for (const auto& block : recovered) {
            if (trim(block.activity_label).empty()) continue;
            if (!activity_text.empty()) {
                activity_text.push_back(' ');
            }
            activity_text += block.text;
        }
        const auto original_words =
            normalized_tokens(original_text);
        const auto activity_words =
            normalized_tokens(activity_text);
        if (original_words.empty()
            || activity_words.size()
                < original_words.size()) {
            return false;
        }
        const std::vector<std::string> prefix(
            activity_words.begin(),
            activity_words.begin()
                + static_cast<std::ptrdiff_t>(
                    original_words.size()));
        return text_similarity(
            original_words, prefix) >= 0.80;
    }

    std::optional<MeetingTrackEvent>
    run_following_activity_recovery(
        const PendingActivityRecovery& pending_recovery,
        const Request& current) {
        if (current.recovery_audio.empty()
            || current.recovery_start_sample
                > pending_recovery.speech_start_sample
            || current.speech_start_sample
                < pending_recovery.speech_end_sample) {
            return std::nullopt;
        }
        const std::uint64_t pre_roll = seconds_to_samples(
            config.pre_roll_seconds, config.sample_rate);
        const std::uint64_t context = seconds_to_samples(
            config.activity_recovery_context_seconds,
            config.sample_rate);
        const std::uint64_t maximum = seconds_to_samples(
            config.maximum_window_seconds,
            config.sample_rate);
        const std::uint64_t desired_start =
            pending_recovery.speech_start_sample > pre_roll
            ? pending_recovery.speech_start_sample - pre_roll
            : 0;
        const std::uint64_t start = std::max(
            desired_start, current.recovery_start_sample);
        const std::uint64_t available_end =
            current.recovery_start_sample
            + static_cast<std::uint64_t>(
                current.recovery_audio.size());
        const std::uint64_t context_end =
            pending_recovery.speech_end_sample
                > std::numeric_limits<std::uint64_t>::max()
                    - context
            ? std::numeric_limits<std::uint64_t>::max()
            : pending_recovery.speech_end_sample + context;
        const std::uint64_t window_end =
            start > std::numeric_limits<std::uint64_t>::max()
                    - maximum
            ? std::numeric_limits<std::uint64_t>::max()
            : start + maximum;
        const std::uint64_t end = std::min({
            available_end,
            context_end,
            window_end,
        });
        const std::uint64_t following_end =
            std::min(end, current.speech_end_sample);
        const std::uint64_t current_length =
            current.speech_end_sample
                - current.speech_start_sample;
        const std::uint64_t required_following = std::min<
            std::uint64_t>(
            current_length,
            std::max<std::uint64_t>(1, pre_roll));
        if (start > pending_recovery.speech_start_sample
            || end < pending_recovery.speech_end_sample
            || following_end <= current.speech_start_sample
            || following_end - current.speech_start_sample
                < required_following) {
            return std::nullopt;
        }

        const std::size_t first =
            static_cast<std::size_t>(
                start - current.recovery_start_sample);
        const std::size_t last =
            static_cast<std::size_t>(
                end - current.recovery_start_sample);
        if (last > current.recovery_audio.size()
            || first >= last) {
            return std::nullopt;
        }
        Request retry;
        retry.generation = current.generation;
        retry.sequence = current.sequence;
        retry.audio_start_sample = start;
        const long double offset_ns =
            static_cast<long double>(
                start - current.recovery_start_sample)
            * 1000000000.0L
            / static_cast<long double>(config.sample_rate);
        retry.audio_start_time_ns =
            current.recovery_start_time_ns
            + static_cast<std::int64_t>(
                std::llround(offset_ns));
        retry.speech_start_sample =
            pending_recovery.speech_start_sample;
        retry.speech_end_sample =
            pending_recovery.speech_end_sample;
        retry.identity_evidence_start_sample =
            pending_recovery.speech_start_sample;
        retry.audio.assign(
            current.recovery_audio.begin()
                + static_cast<std::ptrdiff_t>(first),
            current.recovery_audio.begin()
                + static_cast<std::ptrdiff_t>(last));
        retry.paragraph_final = true;

        MeetingTrackEvent recovered = run_request(retry);
        if (!following_recovery_compatible(
                pending_recovery.blocks,
                recovered.blocks)) {
            return std::nullopt;
        }
        return recovered;
    }

    MeetingTrackEvent revision_base(
        const Request& request, double final_ms) const {
        MeetingTrackEvent event;
        event.type = MeetingTrackEventType::Revision;
        event.replace_start_time_ns =
            map_request_time(request, request.speech_start_sample);
        event.replace_end_time_ns =
            map_request_time(request, request.speech_end_sample);
        event.paragraph_final = request.paragraph_final;
        event.final_asr_ms = final_ms;
        return event;
    }

    std::int64_t map_request_time(
        const Request& request,
        std::uint64_t sample) const {
        const std::uint64_t offset =
            sample > request.audio_start_sample
            ? sample - request.audio_start_sample : 0;
        const long double nanoseconds =
            static_cast<long double>(offset)
            * 1000000000.0L
            / static_cast<long double>(config.sample_rate);
        return request.audio_start_time_ns
            + static_cast<std::int64_t>(std::llround(nanoseconds));
    }

    void emit_error_locked(const std::string& text) {
        MeetingTrackEvent event;
        event.type = MeetingTrackEventType::Error;
        event.error = text;
        emit_unlocked(event);
    }

    void emit_unlocked(const MeetingTrackEvent& event) const {
        if (callback) callback(event);
    }

    STTInterface& preview;
    TranscribeDiarizeInterface& final_model;
    VADInterface& vad;
    Config config;
    EventCallback callback;

    mutable std::mutex mutex;
    std::condition_variable request_cv;
    std::condition_variable idle_cv;
    std::thread worker;
    bool stopping = false;
    bool worker_busy = false;
    std::deque<Request> requests;
    std::uint64_t generation = 1;
    std::uint64_t next_sequence = 0;

    std::vector<float> ring;
    std::uint64_t ring_start_sample = 0;
    std::uint64_t next_sample = 0;
    std::uint64_t vad_processed_sample = 0;
    std::deque<TimeSegment> time_segments;
    std::vector<float> pending;

    bool speech_active = false;
    std::uint64_t paragraph_audio_start = 0;
    std::uint64_t paragraph_speech_start = 0;
    std::uint64_t paragraph_last_voiced_end = 0;
    std::uint64_t next_continuous_update = 0;
    std::size_t silence_samples = 0;
    std::string paragraph_preview;
    bool has_previous_completed_speech = false;
    std::uint64_t previous_completed_speech_start = 0;
    std::uint64_t previous_completed_speech_end = 0;
    std::optional<PendingActivityRecovery>
        pending_activity_recovery;
    std::uint64_t worker_evidence_generation = 0;
    std::uint64_t worker_evidence_through = 0;
};

MeetingTranscriptionTrack::MeetingTranscriptionTrack(
    STTInterface& preview,
    TranscribeDiarizeInterface& final_model,
    VADInterface& vad,
    Config config,
    EventCallback callback)
    : impl_(std::make_unique<Impl>(
        preview,
        final_model,
        vad,
        config,
        std::move(callback))) {}

MeetingTranscriptionTrack::~MeetingTranscriptionTrack() = default;

void MeetingTranscriptionTrack::push_audio(
    const float* samples,
    std::size_t length,
    std::int64_t start_time_ns,
    bool discontinuity) {
    impl_->push_audio(
        samples, length, start_time_ns, discontinuity);
}

void MeetingTranscriptionTrack::finish() {
    impl_->finish();
}

void MeetingTranscriptionTrack::cancel() {
    impl_->cancel();
}

void MeetingTranscriptionTrack::wait_idle() {
    impl_->wait_idle();
}

}  // namespace speech_core
