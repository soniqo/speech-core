#include "speech_core/models/sortformer_speaker_cache.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace speech_core {
namespace {

constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

/// Per-frame, per-speaker importance.
///
/// High for a confident, non-overlapped speaker: the sum term rewards every
/// *other* speaker being confidently silent, so a frame with two people talking
/// scores below one with a single clear voice.
void log_prediction_scores(
    const float* predictions,
    std::size_t frames,
    int speakers,
    float floor_value,
    std::vector<float>& scores) {
    scores.assign(frames * static_cast<std::size_t>(speakers), 0.0f);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        const float* row = predictions + frame * speakers;
        float inverse_sum = 0.0f;
        for (int speaker = 0; speaker < speakers; ++speaker) {
            inverse_sum += std::log(
                std::max(1.0f - row[speaker], floor_value));
        }
        for (int speaker = 0; speaker < speakers; ++speaker) {
            const float positive = std::log(std::max(row[speaker], floor_value));
            const float negative =
                std::log(std::max(1.0f - row[speaker], floor_value));
            scores[frame * speakers + speaker] =
                positive - negative + inverse_sum - std::log(0.5f);
        }
    }
}

/// Take non-speech out of contention, and overlap too when there is enough
/// clean speech to fill a speaker's share without it.
void disable_low_scores(
    const float* predictions,
    std::size_t frames,
    int speakers,
    int minimum_positive,
    std::vector<float>& scores) {
    for (int speaker = 0; speaker < speakers; ++speaker) {
        int positives = 0;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const std::size_t index = frame * speakers + speaker;
            if (predictions[index] <= 0.5f) {
                scores[index] = kNegativeInfinity;
            } else if (scores[index] > 0.0f) {
                ++positives;
            }
        }
        if (positives < minimum_positive) continue;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const std::size_t index = frame * speakers + speaker;
            if (predictions[index] > 0.5f && scores[index] <= 0.0f) {
                scores[index] = kNegativeInfinity;
            }
        }
    }
}

/// Lift each speaker's own best frames so the cache cannot be filled by
/// whoever happened to talk most.
///
/// Applied twice by the caller: strongly, to guarantee every speaker a minimum,
/// then weakly, to cap dominance. Disabled entries stay disabled, because
/// adding a finite amount to negative infinity changes nothing.
void boost_top_scores(
    std::size_t frames,
    int speakers,
    int count,
    float scale,
    std::vector<float>& scores) {
    if (count <= 0) return;
    const std::size_t take =
        std::min(static_cast<std::size_t>(count), frames);
    std::vector<std::size_t> order(frames);
    for (int speaker = 0; speaker < speakers; ++speaker) {
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::partial_sort(
            order.begin(), order.begin() + take, order.end(),
            [&](std::size_t a, std::size_t b) {
                return scores[a * speakers + speaker]
                    > scores[b * speakers + speaker];
            });
        for (std::size_t rank = 0; rank < take; ++rank) {
            scores[order[rank] * speakers + speaker] -= scale * std::log(0.5f);
        }
    }
}

}  // namespace

SortformerSpeakerCache::SortformerSpeakerCache(Config config)
    : config_(config) {
    reset();
}

void SortformerSpeakerCache::reset() {
    const std::size_t width = static_cast<std::size_t>(config_.embedding_dim);
    cache_.assign(static_cast<std::size_t>(config_.cache_frames) * width, 0.0f);
    cache_frames_ = static_cast<std::size_t>(config_.cache_frames);
    cache_predictions_.clear();
    cache_scored_ = false;
    fifo_.clear();
    fifo_predictions_.clear();
    fifo_frames_ = 0;
    mean_silence_.assign(width, 0.0f);
    silence_frames_ = 0;
}

void SortformerSpeakerCache::update_silence_profile(
    const float* embeddings, const float* predictions, std::size_t frames) {
    const std::size_t width = static_cast<std::size_t>(config_.embedding_dim);
    const int speakers = config_.speakers;
    std::vector<float> sum(width, 0.0f);
    std::size_t counted = 0;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        float activity = 0.0f;
        for (int speaker = 0; speaker < speakers; ++speaker) {
            activity += predictions[frame * speakers + speaker];
        }
        if (activity >= config_.silence_threshold) continue;
        const float* row = embeddings + frame * width;
        for (std::size_t index = 0; index < width; ++index) {
            sum[index] += row[index];
        }
        ++counted;
    }
    if (counted == 0) return;
    // A running mean, so what silence sounds like is learned from the whole
    // recording rather than from whichever frames happened to be evicted last.
    for (std::size_t index = 0; index < width; ++index) {
        const float total = mean_silence_[index]
            * static_cast<float>(silence_frames_) + sum[index];
        mean_silence_[index] =
            total / static_cast<float>(silence_frames_ + counted);
    }
    silence_frames_ += counted;
}

void SortformerSpeakerCache::compress() {
    const std::size_t width = static_cast<std::size_t>(config_.embedding_dim);
    const int speakers = config_.speakers;
    const std::size_t frames = cache_frames_;
    const int keep = config_.cache_frames;
    const int per_speaker =
        keep / speakers - config_.silence_frames_per_speaker;
    const int strong =
        static_cast<int>(std::floor(per_speaker * config_.strong_boost_rate));
    const int weak =
        static_cast<int>(std::floor(per_speaker * config_.weak_boost_rate));
    const int minimum_positive = static_cast<int>(
        std::floor(per_speaker * config_.minimum_positive_rate));

    std::vector<float> scores;
    log_prediction_scores(
        cache_predictions_.data(), frames, speakers,
        config_.prediction_floor, scores);
    disable_low_scores(
        cache_predictions_.data(), frames, speakers, minimum_positive, scores);
    if (config_.latest_boost > 0.0f) {
        // Newly arrived frames get a nudge, so a voice heard recently is not
        // evicted in favour of an older frame that merely scored higher.
        for (std::size_t frame = static_cast<std::size_t>(keep);
             frame < frames; ++frame) {
            for (int speaker = 0; speaker < speakers; ++speaker) {
                scores[frame * speakers + speaker] += config_.latest_boost;
            }
        }
    }
    boost_top_scores(frames, speakers, strong, 2.0f, scores);
    boost_top_scores(frames, speakers, weak, 1.0f, scores);

    // Selection is speaker-major, so one frame may be chosen on behalf of more
    // than one speaker; the modulo below maps back. Sorting afterwards is what
    // preserves chronological order inside the cache.
    const std::size_t silence_slots =
        static_cast<std::size_t>(config_.silence_frames_per_speaker);
    const std::size_t padded = frames + silence_slots;
    std::vector<std::size_t> candidates(padded * speakers);
    std::iota(candidates.begin(), candidates.end(), std::size_t{0});
    const auto score_of = [&](std::size_t flat) -> float {
        const std::size_t speaker = flat / padded;
        const std::size_t frame = flat % padded;
        if (frame >= frames) return std::numeric_limits<float>::infinity();
        return scores[frame * speakers + static_cast<int>(speaker)];
    };
    const std::size_t take =
        std::min(static_cast<std::size_t>(keep), candidates.size());
    std::partial_sort(
        candidates.begin(), candidates.begin() + take, candidates.end(),
        [&](std::size_t a, std::size_t b) { return score_of(a) > score_of(b); });
    candidates.resize(take);

    std::vector<std::size_t> chosen;
    chosen.reserve(take);
    for (const std::size_t flat : candidates) {
        // A disabled score means the slot has no frame worth keeping, and is
        // filled with mean silence instead.
        chosen.push_back(
            score_of(flat) == kNegativeInfinity ? padded : flat % padded);
    }
    std::sort(chosen.begin(), chosen.end());

    std::vector<float> next_cache(static_cast<std::size_t>(keep) * width, 0.0f);
    std::vector<float> next_predictions(
        static_cast<std::size_t>(keep) * speakers, 0.0f);
    for (std::size_t slot = 0; slot < chosen.size(); ++slot) {
        const std::size_t frame = chosen[slot];
        if (frame >= frames) {
            std::copy(
                mean_silence_.begin(), mean_silence_.end(),
                next_cache.begin() + slot * width);
            continue;
        }
        std::copy_n(
            cache_.begin() + frame * width, width,
            next_cache.begin() + slot * width);
        std::copy_n(
            cache_predictions_.begin() + frame * speakers, speakers,
            next_predictions.begin() + slot * speakers);
    }
    cache_ = std::move(next_cache);
    cache_predictions_ = std::move(next_predictions);
    cache_frames_ = static_cast<std::size_t>(keep);
}

std::vector<float> SortformerSpeakerCache::advance(
    const float* chunk_embeddings,
    std::size_t chunk_frames,
    const float* predictions,
    std::size_t prediction_frames,
    int left_context,
    int right_context) {
    const std::size_t width = static_cast<std::size_t>(config_.embedding_dim);
    const int speakers = config_.speakers;
    const std::size_t owned = chunk_frames
        - static_cast<std::size_t>(left_context)
        - static_cast<std::size_t>(right_context);

    // Predictions arrive as [cache + fifo + chunk]. The FIFO's slice is the
    // model's latest opinion of frames already queued, which is what scores
    // them when they graduate into the cache.
    const std::size_t fifo_offset = cache_frames_;
    const std::size_t chunk_offset = cache_frames_ + fifo_frames_;
    if (prediction_frames < chunk_offset + chunk_frames) return {};

    fifo_predictions_.assign(
        predictions + fifo_offset * speakers,
        predictions + (fifo_offset + fifo_frames_) * speakers);

    const float* owned_embeddings =
        chunk_embeddings + static_cast<std::size_t>(left_context) * width;
    const float* owned_predictions = predictions
        + (chunk_offset + static_cast<std::size_t>(left_context)) * speakers;
    std::vector<float> chunk_predictions(
        owned_predictions, owned_predictions + owned * speakers);

    fifo_.insert(
        fifo_.end(), owned_embeddings, owned_embeddings + owned * width);
    fifo_predictions_.insert(
        fifo_predictions_.end(), owned_predictions,
        owned_predictions + owned * speakers);
    const std::size_t previous_fifo = fifo_frames_;
    fifo_frames_ += owned;

    if (static_cast<int>(fifo_frames_) > config_.fifo_frames) {
        std::size_t pop = static_cast<std::size_t>(config_.update_period);
        pop = std::max<std::size_t>(
            pop,
            owned + previous_fifo
                - static_cast<std::size_t>(config_.fifo_frames));
        pop = std::min(pop, fifo_frames_);

        update_silence_profile(
            fifo_.data(), fifo_predictions_.data(), pop);

        if (!cache_scored_) {
            // First graduation: nothing has scored the cache's own frames
            // before, so this pass supplies them.
            cache_predictions_.assign(
                predictions, predictions + cache_frames_ * speakers);
            cache_scored_ = true;
        }
        cache_.insert(
            cache_.end(), fifo_.begin(), fifo_.begin() + pop * width);
        cache_predictions_.insert(
            cache_predictions_.end(), fifo_predictions_.begin(),
            fifo_predictions_.begin() + pop * speakers);
        cache_frames_ += pop;

        fifo_.erase(fifo_.begin(), fifo_.begin() + pop * width);
        fifo_predictions_.erase(
            fifo_predictions_.begin(),
            fifo_predictions_.begin() + pop * speakers);
        fifo_frames_ -= pop;

        if (static_cast<int>(cache_frames_) > config_.cache_frames) compress();
    }
    return chunk_predictions;
}

}  // namespace speech_core
