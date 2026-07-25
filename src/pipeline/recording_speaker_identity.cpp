#include "speech_core/pipeline/recording_speaker_identity.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace speech_core {
namespace {

std::vector<float> normalized(std::vector<float> value) {
    double squared = 0.0;
    for (float item : value) {
        if (!std::isfinite(item)) return {};
        squared += static_cast<double>(item) * item;
    }
    const double norm = std::sqrt(squared);
    if (!std::isfinite(norm) || norm <= 1e-8) return {};
    for (float& item : value) {
        item = static_cast<float>(item / norm);
    }
    return value;
}

float cosine(
    const std::vector<float>& left,
    const std::vector<float>& right) {
    if (left.empty() || left.size() != right.size()) return -1.0f;
    double score = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        score += static_cast<double>(left[index]) * right[index];
    }
    return static_cast<float>(score);
}

bool same_label(const std::string& left, const std::string& right) {
    return left == right;
}

bool contains_label(
    const std::vector<std::string>& labels,
    const std::string& candidate) {
    return std::any_of(
        labels.begin(), labels.end(),
        [&](const std::string& value) {
            return same_label(value, candidate);
        });
}

std::size_t seconds_to_samples(float seconds, int sample_rate) {
    return static_cast<std::size_t>(std::llround(
        static_cast<double>(seconds) * sample_rate));
}

}  // namespace

struct RecordingSpeakerIdentity::SessionVoice {
    std::string label;
    std::vector<float> centroid;
    std::size_t sample_count = 1;
    std::uint64_t last_match_tick = 0;
    bool recording_local = false;
};

struct RecordingSpeakerIdentity::GroupEvidence {
    std::string activity_label;
    std::vector<std::size_t> block_indices;
    std::vector<float> audio;
    std::vector<SpeakerIdentityBackfill> ranges;
    std::uint64_t paragraph = 0;
};

struct RecordingSpeakerIdentity::GalleryCandidate {
    struct Fragment {
        std::vector<float> audio;
        std::vector<float> embedding;
        std::vector<SpeakerIdentityBackfill> ranges;
        std::uint64_t paragraph = 0;
        std::string activity_label;
    };

    std::vector<float> centroid;
    std::vector<Fragment> fragments;
    std::map<std::uint64_t, std::string> paragraph_labels;
    std::uint64_t last_tick = 0;
};

RecordingSpeakerIdentity::~RecordingSpeakerIdentity() = default;

RecordingSpeakerIdentity::RecordingSpeakerIdentity(
    EmbeddingInterface& embedder)
    : RecordingSpeakerIdentity(embedder, Config{}) {}

RecordingSpeakerIdentity::RecordingSpeakerIdentity(
    EmbeddingInterface& embedder,
    Config config)
    : embedder_(embedder), config_(config) {
    if (config.sample_rate <= 0
        || config.minimum_short_seconds <= 0.0f
        || config.minimum_long_seconds
            < config.minimum_short_seconds
        || config.novelty_threshold > config.match_threshold
        || config.gallery_minimum_fragments < 3
        || config.maximum_session_speakers == 0
        || config.maximum_gallery_candidates == 0
        || config.maximum_gallery_fragments
            < config.gallery_minimum_fragments
        || embedder.input_sample_rate() != config.sample_rate
        || embedder.embedding_dim() <= 0) {
        throw std::invalid_argument(
            "Recording speaker-identity configuration is invalid");
    }
}

void RecordingSpeakerIdentity::set_profiles(
    std::vector<StoredSpeakerIdentityProfile> profiles) {
    std::vector<StoredSpeakerIdentityProfile> accepted;
    accepted.reserve(profiles.size());
    for (auto& profile : profiles) {
        profile.embedding = normalized(
            std::move(profile.embedding));
        if (profile.name.empty()
            || profile.embedding.size()
                != static_cast<std::size_t>(
                    embedder_.embedding_dim())) {
            throw std::invalid_argument(
                "Stored speaker profile is invalid");
        }
        if (std::any_of(
                accepted.begin(), accepted.end(),
                [&](const auto& existing) {
                    return same_label(
                        existing.name, profile.name);
                })) {
            throw std::invalid_argument(
                "Stored speaker profile names are not unique");
        }
        profile.sample_count =
            std::max<std::size_t>(1, profile.sample_count);
        accepted.push_back(std::move(profile));
    }
    profiles_ = std::move(accepted);
}

void RecordingSpeakerIdentity::begin_recording() {
    session_voices_.clear();
    gallery_.clear();
    paragraph_sequence_ = 0;
    gallery_tick_ = 0;
    next_session_label_ = 1;
}

SpeakerIdentityResolution RecordingSpeakerIdentity::resolve(
    const std::vector<MeetingTranscriptBlock>& blocks) {
    SpeakerIdentityResolution resolution;
    resolution.speakers.resize(blocks.size());
    const std::uint64_t paragraph = ++paragraph_sequence_;

    std::vector<GroupEvidence> groups;
    std::unordered_map<std::string, std::size_t> group_index;
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto& block = blocks[index];
        if (block.activity_label.empty()) continue;
        auto [iterator, inserted] = group_index.emplace(
            block.activity_label, groups.size());
        if (inserted) {
            groups.push_back({
                block.activity_label, {}, {}, {}, paragraph,
            });
        }
        auto& group = groups[iterator->second];
        group.block_indices.push_back(index);
        group.audio.insert(
            group.audio.end(),
            block.identity_audio.begin(),
            block.identity_audio.end());
        group.ranges.push_back({
            block.start_time_ns,
            block.end_time_ns,
            block.text,
            {},
        });
    }

    struct EmbeddedGroup {
        std::size_t index = 0;
        std::vector<float> embedding;
        bool long_form = false;
        float existing_confidence = -1.0f;
    };
    std::vector<EmbeddedGroup> embedded;
    const std::size_t short_minimum = seconds_to_samples(
        config_.minimum_short_seconds, config_.sample_rate);
    const std::size_t long_minimum = seconds_to_samples(
        config_.minimum_long_seconds, config_.sample_rate);
    for (std::size_t index = 0; index < groups.size(); ++index) {
        const auto& audio = groups[index].audio;
        if (audio.size() < short_minimum) continue;
        const bool long_form = audio.size() >= long_minimum;
        std::vector<float> vector = long_form
            ? embedder_.embed(
                audio.data(), audio.size(), config_.sample_rate)
            : embedder_.embed_short_utterance(
                audio.data(), audio.size(), config_.sample_rate);
        vector = normalized(std::move(vector));
        if (vector.size()
            != static_cast<std::size_t>(
                embedder_.embedding_dim())) {
            continue;
        }
        float confidence = -1.0f;
        for (const auto& profile : profiles_) {
            confidence = std::max(
                confidence,
                cosine(profile.embedding, vector));
        }
        for (const auto& session : session_voices_) {
            confidence = std::max(
                confidence,
                cosine(session.centroid, vector));
        }
        embedded.push_back({
            index,
            std::move(vector),
            long_form,
            confidence,
        });
    }
    std::stable_sort(
        embedded.begin(), embedded.end(),
        [](const auto& left, const auto& right) {
            return left.existing_confidence
                > right.existing_confidence;
        });

    std::vector<std::string> claimed;
    for (const auto& item : embedded) {
        auto& group = groups[item.index];
        std::optional<std::string> speaker =
            item.long_form
            ? resolve_long(item.embedding, claimed)
            : resolve_short(item.embedding, claimed);
        if (speaker) {
            claimed.push_back(*speaker);
            for (std::size_t block : group.block_indices) {
                resolution.speakers[block] = *speaker;
            }
        } else if (!item.long_form) {
            add_gallery_evidence(
                group,
                item.embedding,
                claimed,
                resolution,
                item.index);
            const auto assigned =
                resolution.speakers[group.block_indices.front()];
            if (assigned) claimed.push_back(*assigned);
        }
    }
    return resolution;
}

std::optional<std::string>
RecordingSpeakerIdentity::resolve_long(
    const std::vector<float>& embedding,
    const std::vector<std::string>& claimed) {
    enum class Kind { Profile, Session };
    struct Candidate {
        Kind kind;
        std::size_t index;
        std::string label;
        float score;
    };
    std::vector<Candidate> candidates;
    for (std::size_t index = 0; index < profiles_.size(); ++index) {
        candidates.push_back({
            Kind::Profile,
            index,
            profiles_[index].name,
            cosine(profiles_[index].embedding, embedding),
        });
    }
    for (std::size_t index = 0;
         index < session_voices_.size(); ++index) {
        candidates.push_back({
            Kind::Session,
            index,
            session_voices_[index].label,
            cosine(session_voices_[index].centroid, embedding),
        });
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            if (left.score == right.score) {
                return left.label < right.label;
            }
            return left.score > right.score;
        });
    const auto best_overall =
        candidates.empty()
        ? candidates.end() : candidates.begin();
    const auto best = std::find_if(
        candidates.begin(), candidates.end(),
        [&](const auto& candidate) {
            return !contains_label(claimed, candidate.label);
        });
    const bool excluded_conflict =
        best_overall != candidates.end()
        && contains_label(claimed, best_overall->label)
        && best_overall->score >= config_.match_threshold;

    if (best != candidates.end()) {
        const auto competitor = std::find_if(
            std::next(best), candidates.end(),
            [&](const auto& candidate) {
                return !contains_label(
                    claimed, candidate.label)
                    && !same_label(
                        candidate.label, best->label);
            });
        const bool margin_ok =
            competitor == candidates.end()
            || best->score - competitor->score
                >= config_.ambiguity_margin;
        if (best->score >= config_.match_threshold
            && margin_ok) {
            std::string label = best->label;
            if (best->kind == Kind::Profile) {
                auto existing = std::find_if(
                    session_voices_.begin(),
                    session_voices_.end(),
                    [&](const auto& voice) {
                        return same_label(
                            voice.label, label);
                    });
                if (existing == session_voices_.end()) {
                    session_voices_.push_back({
                        label, embedding, 1, ++gallery_tick_, false,
                    });
                }
            } else {
                auto& voice = session_voices_[best->index];
                const std::size_t previous =
                    std::min<std::size_t>(
                        voice.sample_count, 20);
                std::vector<float> merged(
                    embedding.size(), 0.0f);
                for (std::size_t index = 0;
                     index < embedding.size(); ++index) {
                    merged[index] = static_cast<float>(
                        (static_cast<double>(
                             voice.centroid[index])
                             * previous
                         + embedding[index])
                        / static_cast<double>(
                            previous + 1));
                }
                voice.centroid = normalized(
                    std::move(merged));
                voice.sample_count = previous + 1;
                voice.last_match_tick = ++gallery_tick_;
            }
            return label;
        }
    }

    const bool novel =
        candidates.empty()
        || (best != candidates.end()
            && best->score < config_.novelty_threshold);
    if (!novel && !excluded_conflict) return std::nullopt;
    if (session_voices_.size()
        >= config_.maximum_session_speakers) {
        const auto evict = std::min_element(
            session_voices_.begin(),
            session_voices_.end(),
            [](const auto& left, const auto& right) {
                const bool left_temporary =
                    left.recording_local;
                const bool right_temporary =
                    right.recording_local;
                if (left_temporary != right_temporary) {
                    return left_temporary;
                }
                return left.last_match_tick
                    < right.last_match_tick;
            });
        if (evict == session_voices_.end()
            || !evict->recording_local) {
            return std::nullopt;
        }
        session_voices_.erase(evict);
    }
    std::string label;
    do {
        label = "S" + std::to_string(next_session_label_++);
    } while (has_profile(label)
             || std::any_of(
                 session_voices_.begin(),
                 session_voices_.end(),
                 [&](const auto& voice) {
                     return same_label(voice.label, label);
                 }));
    session_voices_.push_back({
        label, embedding, 1, ++gallery_tick_, true,
    });
    return label;
}

std::optional<std::string>
RecordingSpeakerIdentity::resolve_short(
    const std::vector<float>& embedding,
    const std::vector<std::string>& claimed) const {
    struct Candidate {
        std::string label;
        float score = -1.0f;
        float threshold = 1.0f;
        float margin = 1.0f;
    };
    std::vector<Candidate> candidates;
    for (const auto& profile : profiles_) {
        candidates.push_back({
            profile.name,
            cosine(profile.embedding, embedding),
            config_.global_short_threshold,
            config_.global_short_margin,
        });
    }
    for (const auto& session : session_voices_) {
        candidates.push_back({
            session.label,
            cosine(session.centroid, embedding),
            config_.recording_short_threshold,
            config_.recording_short_margin,
        });
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            if (left.score == right.score) {
                return left.label < right.label;
            }
            return left.score > right.score;
        });
    const auto best = std::find_if(
        candidates.begin(), candidates.end(),
        [&](const auto& candidate) {
            return !contains_label(claimed, candidate.label);
        });
    if (best == candidates.end()
        || best->score < best->threshold) {
        return std::nullopt;
    }
    const auto competitor = std::find_if(
        std::next(best), candidates.end(),
        [&](const auto& candidate) {
            return !contains_label(claimed, candidate.label)
                && !same_label(
                    candidate.label, best->label);
        });
    if (competitor != candidates.end()
        && best->score - competitor->score < best->margin) {
        return std::nullopt;
    }
    return best->label;
}

void RecordingSpeakerIdentity::add_gallery_evidence(
    const GroupEvidence& group,
    const std::vector<float>& embedding,
    const std::vector<std::string>& claimed,
    SpeakerIdentityResolution& resolution,
    std::size_t /*group_index*/) {
    struct Match {
        std::size_t index = 0;
        float score = -1.0f;
    };
    std::vector<Match> matches;
    for (std::size_t index = 0; index < gallery_.size(); ++index) {
        const auto paragraph =
            gallery_[index].paragraph_labels.find(group.paragraph);
        if (paragraph
            != gallery_[index].paragraph_labels.end()
            && paragraph->second != group.activity_label) {
            continue;
        }
        matches.push_back({
            index,
            cosine(gallery_[index].centroid, embedding),
        });
    }
    std::sort(
        matches.begin(), matches.end(),
        [](const auto& left, const auto& right) {
            return left.score > right.score;
        });

    std::optional<std::size_t> selected;
    if (!matches.empty()) {
        const auto& best = matches.front();
        const auto& candidate = gallery_[best.index];
        const float threshold =
            candidate.fragments.size() == 1
            ? config_.gallery_bootstrap_threshold
            : config_.gallery_support_threshold;
        const bool margin_ok =
            matches.size() == 1
            || best.score - matches[1].score
                >= config_.gallery_support_margin;
        if (best.score >= threshold && margin_ok) {
            selected = best.index;
        }
    }
    if (!selected) {
        if (gallery_.size()
            >= config_.maximum_gallery_candidates) {
            const auto oldest = std::min_element(
                gallery_.begin(), gallery_.end(),
                [](const auto& left, const auto& right) {
                    return left.last_tick < right.last_tick;
                });
            gallery_.erase(oldest);
        }
        GalleryCandidate candidate;
        candidate.centroid = embedding;
        candidate.last_tick = ++gallery_tick_;
        gallery_.push_back(std::move(candidate));
        selected = gallery_.size() - 1;
    }

    auto& candidate = gallery_[*selected];
    if (candidate.fragments.size()
        >= config_.maximum_gallery_fragments) {
        candidate.fragments.erase(
            candidate.fragments.begin());
    }
    candidate.fragments.push_back({
        group.audio,
        embedding,
        group.ranges,
        group.paragraph,
        group.activity_label,
    });
    candidate.paragraph_labels[group.paragraph] =
        group.activity_label;
    std::vector<float> centroid(
        embedding.size(), 0.0f);
    for (const auto& fragment : candidate.fragments) {
        for (std::size_t index = 0;
             index < centroid.size(); ++index) {
            centroid[index] += fragment.embedding[index];
        }
    }
    candidate.centroid = normalized(std::move(centroid));
    candidate.last_tick = ++gallery_tick_;

    std::size_t combined_samples = 0;
    for (const auto& fragment : candidate.fragments) {
        combined_samples += fragment.audio.size();
    }
    if (candidate.fragments.size()
            < config_.gallery_minimum_fragments
        || combined_samples < seconds_to_samples(
            config_.minimum_long_seconds,
            config_.sample_rate)) {
        return;
    }
    std::vector<float> combined;
    combined.reserve(combined_samples);
    for (const auto& fragment : candidate.fragments) {
        combined.insert(
            combined.end(),
            fragment.audio.begin(),
            fragment.audio.end());
    }
    std::vector<float> long_embedding = normalized(
        embedder_.embed(
            combined.data(),
            combined.size(),
            config_.sample_rate));
    if (long_embedding.size()
        != static_cast<std::size_t>(
            embedder_.embedding_dim())) {
        return;
    }
    const auto speaker = resolve_long(
        long_embedding, claimed);
    if (!speaker) return;

    for (const auto& fragment : candidate.fragments) {
        for (auto range : fragment.ranges) {
            range.speaker = *speaker;
            resolution.backfills.push_back(std::move(range));
        }
    }
    for (std::size_t block : group.block_indices) {
        resolution.speakers[block] = *speaker;
    }
    gallery_.erase(
        gallery_.begin()
        + static_cast<std::ptrdiff_t>(*selected));
}

bool RecordingSpeakerIdentity::promote_recording_identity(
    const std::string& label,
    const std::string& name) {
    if (name.empty()) {
        throw std::invalid_argument(
            "Speaker profile name must not be empty");
    }
    const auto voice = std::find_if(
        session_voices_.begin(), session_voices_.end(),
        [&](const auto& value) {
            return same_label(value.label, label);
        });
    if (voice == session_voices_.end()) return false;
    upsert_profile(
        name, voice->centroid, voice->sample_count);
    voice->label = name;
    voice->recording_local = false;
    return true;
}

void RecordingSpeakerIdentity::upsert_profile(
    const std::string& name,
    const std::vector<float>& embedding,
    std::size_t sample_count) {
    std::vector<float> value = normalized(embedding);
    if (name.empty()
        || value.size()
            != static_cast<std::size_t>(
                embedder_.embedding_dim())) {
        throw std::invalid_argument(
            "Speaker profile is invalid");
    }
    const auto existing = std::find_if(
        profiles_.begin(), profiles_.end(),
        [&](const auto& profile) {
            return same_label(profile.name, name);
        });
    if (existing == profiles_.end()) {
        profiles_.push_back({
            name, std::move(value),
            std::max<std::size_t>(1, sample_count),
        });
    } else {
        existing->embedding = std::move(value);
        existing->sample_count =
            std::max<std::size_t>(1, sample_count);
    }
}

bool RecordingSpeakerIdentity::delete_profile(
    const std::string& name) {
    const auto iterator = std::find_if(
        profiles_.begin(), profiles_.end(),
        [&](const auto& profile) {
            return same_label(profile.name, name);
        });
    if (iterator == profiles_.end()) return false;
    profiles_.erase(iterator);
    return true;
}

bool RecordingSpeakerIdentity::has_profile(
    const std::string& name) const {
    return std::any_of(
        profiles_.begin(), profiles_.end(),
        [&](const auto& profile) {
            return same_label(profile.name, name);
        });
}

}  // namespace speech_core
