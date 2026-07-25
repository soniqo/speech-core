#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/pipeline/meeting_transcription_track.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace speech_core {

struct StoredSpeakerIdentityProfile {
    std::string name;
    std::vector<float> embedding;
    std::size_t sample_count = 1;
};

struct SpeakerIdentityBackfill {
    std::int64_t start_time_ns = 0;
    std::int64_t end_time_ns = 0;
    std::string text;
    std::string speaker;
};

struct SpeakerIdentityResolution {
    /// One optional recording/global identity per input block.
    std::vector<std::optional<std::string>> speakers;
    /// Exact earlier short-fragment ranges that supplied newly resolved
    /// recording-local evidence. Callers must also require text compatibility.
    std::vector<SpeakerIdentityBackfill> backfills;
};

/// Precision-first ReDimNet identity layer for MOSS paragraph-local activity.
///
/// MOSS S01... labels are accepted only as within-result different-speaker
/// constraints and never escape this class. Long clean evidence may match or
/// create a recording identity. A 0.6-to-2-second probe can only retrieve an
/// existing profile/session identity. Unsupported or ambiguous evidence
/// remains unlabelled.
class RecordingSpeakerIdentity {
public:
    struct Config {
        int sample_rate = 16000;
        float minimum_long_seconds = 2.0f;
        float minimum_short_seconds = 0.60f;
        float match_threshold = 0.55f;
        float novelty_threshold = 0.45f;
        float ambiguity_margin = 0.08f;
        float global_short_threshold = 0.85f;
        float global_short_margin = 0.20f;
        float recording_short_threshold = 0.70f;
        float recording_short_margin = 0.18f;
        float gallery_bootstrap_threshold = 0.60f;
        float gallery_support_threshold = 0.35f;
        float gallery_support_margin = 0.10f;
        std::size_t gallery_minimum_fragments = 3;
        std::size_t maximum_session_speakers = 12;
        std::size_t maximum_gallery_candidates = 12;
        std::size_t maximum_gallery_fragments = 8;
    };

    explicit RecordingSpeakerIdentity(
        EmbeddingInterface& embedder);
    RecordingSpeakerIdentity(
        EmbeddingInterface& embedder,
        Config config);
    ~RecordingSpeakerIdentity();

    RecordingSpeakerIdentity(
        const RecordingSpeakerIdentity&) = delete;
    RecordingSpeakerIdentity& operator=(
        const RecordingSpeakerIdentity&) = delete;

    void set_profiles(
        std::vector<StoredSpeakerIdentityProfile> profiles);
    const std::vector<StoredSpeakerIdentityProfile>& profiles() const {
        return profiles_;
    }

    void begin_recording();

    SpeakerIdentityResolution resolve(
        const std::vector<MeetingTranscriptBlock>& blocks);

    /// Promote/rename an established recording identity as a durable profile.
    /// Returns false when the recording label is unknown.
    bool promote_recording_identity(
        const std::string& label,
        const std::string& name);

    void upsert_profile(
        const std::string& name,
        const std::vector<float>& embedding,
        std::size_t sample_count = 1);
    bool delete_profile(const std::string& name);
    bool has_profile(const std::string& name) const;

private:
    struct SessionVoice;
    struct GalleryCandidate;
    struct GroupEvidence;

    std::optional<std::string> resolve_long(
        const std::vector<float>& embedding,
        const std::vector<std::string>& claimed);
    std::optional<std::string> resolve_short(
        const std::vector<float>& embedding,
        const std::vector<std::string>& claimed) const;
    void add_gallery_evidence(
        const GroupEvidence& group,
        const std::vector<float>& embedding,
        const std::vector<std::string>& claimed,
        SpeakerIdentityResolution& resolution,
        std::size_t group_index);

    EmbeddingInterface& embedder_;
    Config config_;
    std::vector<StoredSpeakerIdentityProfile> profiles_;
    std::vector<SessionVoice> session_voices_;
    std::vector<GalleryCandidate> gallery_;
    std::uint64_t paragraph_sequence_ = 0;
    std::uint64_t gallery_tick_ = 0;
    std::size_t next_session_label_ = 1;
};

}  // namespace speech_core
