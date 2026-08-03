#pragma once

#include "speech_core/interfaces.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace speech_core {

/// One authoritative paragraph block from a source-local meeting track.
struct MeetingTranscriptBlock {
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 0;
    std::string text;
    /// MOSS activity label scoped to this one inference result. Empty means
    /// attribution abstained or this is the microphone track.
    std::string activity_label;
    /// Clean, non-overlapped source-local evidence for an identity encoder.
    /// Empty when the turn is mixed, overlapping, or unalignable.
    std::vector<float> identity_audio;
};

enum class MeetingTrackEventType {
    Preview,
    Revision,
    Error,
};

struct MeetingTrackEvent {
    MeetingTrackEventType type = MeetingTrackEventType::Preview;
    /// Full revisable Nemotron caption for Preview; empty clears it.
    std::string preview_text;
    /// Revision replaces source-local blocks intersecting this interval.
    std::int64_t replace_start_time_ns = 0;
    std::int64_t replace_end_time_ns = 0;
    std::vector<MeetingTranscriptBlock> blocks;
    bool paragraph_final = false;
    double final_asr_ms = 0.0;
    std::string error;
};

/// Source-local fixed meeting pipeline:
/// Silero VAD -> Nemotron previews -> MOSS paragraph-final text/activity.
///
/// Construct one instance per captured source. PCM, VAD state, preview state,
/// paragraph state, and timestamps never cross instances. The MOSS model may
/// be shared; its implementation is responsible for serializing inference.
class MeetingTranscriptionTrack {
public:
    struct Config {
        int sample_rate = 16000;
        float vad_onset = 0.50f;
        float vad_offset = 0.35f;
        float minimum_speech_seconds = 0.20f;
        float silence_close_seconds = 0.55f;
        float pre_roll_seconds = 0.20f;
        float post_roll_seconds = 0.20f;
        float activity_recovery_context_seconds = 4.0f;
        float microphone_short_agreement_seconds = 0.40f;
        float continuous_update_seconds = 10.0f;
        float maximum_window_seconds = 20.0f;
        /// How much un-transcribed audio may wait for the final model.
        ///
        /// A queued request owns its copy of its window, so the backlog costs
        /// the audio in it rather than the number of entries; 600 s of 16 kHz
        /// mono float is about 38 MB. When the bound is reached the arriving
        /// paragraph is refused with an error and the backlog is kept, because
        /// a track that fell behind should finish late rather than lose the
        /// paragraphs it has already read.
        float maximum_pending_seconds = 600.0f;
        /// Microphone keeps MOSS text/timing but discards activity labels and
        /// gates sub-400 ms results on independent preview agreement.
        bool microphone = false;

        /// Whether a retry over the SAME audio may supply this paragraph's
        /// speaker activity. Both decodes heard the same speech, so the
        /// caller's rule should be symmetric.
        ///
        /// Deciding whether a re-decode still says what the paragraph said is
        /// application policy, not engine mechanics: it is a judgement about
        /// the product's transcripts and it carries a measured threshold the
        /// engine cannot own. When this is unset the retry is REJECTED and the
        /// original paragraph is published unchanged. There is deliberately no
        /// default rule — an engine that has not been told how to judge a
        /// re-decode should not judge it. Do not restore one.
        std::function<bool(const std::string& original,
                           const std::string& recovered)>
            activity_recovery_compatible;

        /// Whether a retry's speaker-marked text may supply a pending
        /// paragraph's activity when the marked segment runs on past it.
        ///
        /// Same ownership and the same default: unset REJECTS the retry and
        /// the original paragraph is published unchanged. Do not restore a
        /// default rule here either.
        std::function<bool(const std::string& original,
                           const std::string& activity)>
            following_recovery_compatible;
    };

    using EventCallback = std::function<void(const MeetingTrackEvent&)>;

    MeetingTranscriptionTrack(
        STTInterface& preview,
        TranscribeDiarizeInterface& final_model,
        VADInterface& vad,
        Config config,
        EventCallback callback);
    ~MeetingTranscriptionTrack();

    MeetingTranscriptionTrack(
        const MeetingTranscriptionTrack&) = delete;
    MeetingTranscriptionTrack& operator=(
        const MeetingTranscriptionTrack&) = delete;

    /// Push 16 kHz mono PCM and the capture time of its first sample.
    /// `discontinuity` resets all source-local state before accepting the
    /// block. Callers should then fail the enclosing required track if that
    /// policy is stricter than a reset.
    void push_audio(
        const float* samples,
        std::size_t length,
        std::int64_t start_time_ns,
        bool discontinuity = false);

    /// Close an open paragraph, publish its final MOSS result, and wait until
    /// all source-local inference is complete.
    void finish();

    /// Discard revisable work and reset VAD/ASR state without publishing it.
    void cancel();

    /// Wait until every already queued final inference has completed AND its
    /// events have been delivered, so what was published can be read after
    /// this returns.
    ///
    /// Never call this, or `finish()`, from the event callback: the track is
    /// not idle while a callback is running, so it would wait on itself.
    void wait_idle();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace speech_core
