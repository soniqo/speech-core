#include "speech_core/pipeline/recording_speaker_identity.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

class FakeEmbedder final : public EmbeddingInterface {
public:
    std::vector<float> embed(
        const float* audio,
        std::size_t length,
        int sample_rate) override {
        assert(sample_rate == 16000);
        assert(length >= 2);
        return {audio[0], audio[1]};
    }

    std::vector<float> embed_short_utterance(
        const float* audio,
        std::size_t length,
        int sample_rate) override {
        return embed(audio, length, sample_rate);
    }

    int embedding_dim() const override { return 2; }
    int input_sample_rate() const override { return 16000; }
};

std::vector<float> evidence(
    std::size_t samples, float x, float y) {
    std::vector<float> audio(samples, 0.0f);
    audio[0] = x;
    audio[1] = y;
    return audio;
}

MeetingTranscriptBlock block(
    const char* activity,
    std::int64_t start,
    std::int64_t end,
    std::size_t samples,
    float x = 1.0f,
    float y = 0.0f) {
    MeetingTranscriptBlock value;
    value.start_time_ns = start;
    value.end_time_ns = end;
    value.text = "words";
    value.activity_label = activity;
    value.identity_audio = evidence(samples, x, y);
    return value;
}

void test_joint_activity_labels_never_collapse() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.begin_recording();
    const auto result = identity.resolve({
        block("S01", 0, 2000000000, 32000),
        block("S02", 0, 2000000000, 32000),
    });
    assert(result.speakers.size() == 2);
    assert(result.speakers[0] == "S1");
    assert(result.speakers[1] == "S2");
}

void test_short_probe_retrieves_but_never_creates() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.set_profiles({{"Alice", {1.0f, 0.0f}, 1}});
    identity.begin_recording();
    auto result = identity.resolve({
        block("S01", 0, 1000000000, 16000),
    });
    assert(result.speakers[0] == "Alice");

    RecordingSpeakerIdentity empty(embedder);
    empty.begin_recording();
    result = empty.resolve({
        block("S01", 0, 1000000000, 16000),
    });
    assert(!result.speakers[0]);
    assert(result.backfills.empty());
}

void test_ambiguous_short_probe_abstains() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.set_profiles({
        {"Alice", {1.0f, 0.0f}, 1},
        {"Bob", {0.99f, 0.1f}, 1},
    });
    identity.begin_recording();
    const auto result = identity.resolve({
        block("S01", 0, 1000000000, 16000),
    });
    assert(!result.speakers[0]);
}

void test_three_supported_fragments_create_and_exactly_backfill() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.begin_recording();
    for (int paragraph = 0; paragraph < 2; ++paragraph) {
        const auto result = identity.resolve({
            block(
                "S01",
                paragraph * 1000000000LL,
                paragraph * 1000000000LL + 700000000,
                11200),
        });
        assert(!result.speakers[0]);
        assert(result.backfills.empty());
    }
    const auto result = identity.resolve({
        block("S01", 2000000000, 2700000000, 11200),
    });
    assert(result.speakers[0] == "S1");
    assert(result.backfills.size() == 3);
    assert(result.backfills[0].start_time_ns == 0);
    assert(result.backfills[1].start_time_ns == 1000000000);
    assert(result.backfills[2].start_time_ns == 2000000000);
    for (const auto& backfill : result.backfills) {
        assert(backfill.text == "words");
        assert(backfill.speaker == "S1");
    }
}

void test_different_labels_in_one_paragraph_cannot_share_gallery() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.begin_recording();
    const auto first = identity.resolve({
        block("S01", 0, 700000000, 11200),
        block(
            "S02", 700000000, 1400000000,
            11200, 0.0f, 1.0f),
    });
    assert(!first.speakers[0]);
    assert(!first.speakers[1]);

    // Two further S01 fragments can resolve its candidate. The same-paragraph
    // S02 fragment was forbidden from contributing to that candidate.
    auto second = identity.resolve({
        block("S01", 2000000000, 2700000000, 11200),
    });
    assert(!second.speakers[0]);
    auto third = identity.resolve({
        block("S01", 3000000000, 3700000000, 11200),
    });
    assert(third.speakers[0] == "S1");
    assert(third.backfills.size() == 3);
}

void test_promote_recording_identity() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.begin_recording();
    const auto result = identity.resolve({
        block("S01", 0, 2000000000, 32000),
    });
    assert(result.speakers[0] == "S1");
    assert(identity.promote_recording_identity("S1", "Viktor"));
    assert(identity.has_profile("Viktor"));
    assert(!identity.promote_recording_identity("missing", "Nobody"));
    assert(identity.delete_profile("Viktor"));
}

void test_profile_names_starting_with_s_are_not_evictable() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity::Config config;
    config.maximum_session_speakers = 1;
    RecordingSpeakerIdentity identity(embedder, config);
    identity.set_profiles({{"Sam", {1.0f, 0.0f}, 1}});
    identity.begin_recording();
    auto result = identity.resolve({
        block("S01", 0, 2000000000, 32000),
    });
    assert(result.speakers[0] == "Sam");
    result = identity.resolve({
        block(
            "S01", 3000000000, 5000000000,
            32000, 0.0f, 1.0f),
    });
    assert(!result.speakers[0]);
}

void test_recording_label_does_not_collide_with_profile_name() {
    FakeEmbedder embedder;
    RecordingSpeakerIdentity identity(embedder);
    identity.set_profiles({{"S1", {1.0f, 0.0f}, 1}});
    identity.begin_recording();
    const auto result = identity.resolve({
        block(
            "S01", 0, 2000000000,
            32000, 0.0f, 1.0f),
    });
    assert(result.speakers[0] == "S2");
}

}  // namespace

int main() {
    test_joint_activity_labels_never_collapse();
    test_short_probe_retrieves_but_never_creates();
    test_ambiguous_short_probe_abstains();
    test_three_supported_fragments_create_and_exactly_backfill();
    test_different_labels_in_one_paragraph_cannot_share_gallery();
    test_promote_recording_identity();
    test_profile_names_starting_with_s_are_not_evictable();
    test_recording_label_does_not_collide_with_profile_name();
    std::puts("Recording speaker-identity tests passed");
    return 0;
}
