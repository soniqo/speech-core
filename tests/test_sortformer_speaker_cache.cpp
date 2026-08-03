// Unit tests for SortformerSpeakerCache — runs in the default build, with no
// model files and no ONNX Runtime.
//
// The cache is what makes a Sortformer speaker index mean the same person for a
// whole recording: the graph takes it as an input and returns embeddings, and
// every decision about what to keep and what to evict happens here. Getting it
// wrong is silent — the cache still holds its full complement of plausible
// embeddings and simply keeps the wrong ones, so labels drift partway through a
// recording with nothing logged.
//
// The expected values below were produced by the numpy port of NeMo's
// `streaming_update_async` (scripts/sortformer_speaker_cache.py in the
// consuming app) driven with the same deterministic sequence. Regenerate them
// from that port rather than from this implementation if the algorithm changes,
// or the test only asserts that the code still does what it currently does.

#include "speech_core/models/sortformer_speaker_cache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace speech_core;

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void check_near(double got, double want, double tolerance, const char* what) {
    if (std::fabs(got - want) <= tolerance) return;
    std::fprintf(stderr, "FAIL: %s (got %.6f, want %.6f)\n", what, got, want);
    ++failures;
}

/// A fixed linear congruential sequence rather than <random>, so the reference
/// values can be reproduced in another language without both agreeing on a
/// generator implementation.
struct Sequence {
    std::uint32_t state = 12345;
    float next() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>((state >> 8) & 0xFFFF) / 65535.0f;
    }
};

SortformerSpeakerCache::Config small_config() {
    SortformerSpeakerCache::Config config;
    config.embedding_dim = 32;
    config.cache_frames = 40;
    config.fifo_frames = 12;
    config.update_period = 8;
    return config;
}

void test_matches_reference_through_eviction() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    Sequence sequence;

    const int speakers = config.speakers;
    const int width = config.embedding_dim;
    const int chunk = 14;
    const int left = 1;
    const int right = 7;
    const std::size_t owned =
        static_cast<std::size_t>(chunk - left - right);

    double last_sum = 0.0;
    double last_ordered = 0.0;
    for (int step = 0; step < 24; ++step) {
        std::vector<float> embeddings(
            static_cast<std::size_t>(chunk) * width);
        for (auto& value : embeddings) value = sequence.next() * 2.0f - 1.0f;

        const std::size_t total =
            cache.cache_frames() + cache.fifo_frames() + chunk;
        std::vector<float> predictions(total * speakers);
        // TWO speakers active, and which two alternates with the frame.
        //
        // A single active speaker cannot exercise the ordering at all: the
        // selection is speaker-major, so when every kept frame belongs to one
        // speaker, sorting the chosen indices before mapping them back to
        // frames and sorting after give the identical cache. This suite ran
        // that way and passed against a build whose cache was in the wrong
        // order — the defect reached a real model and cost its predictions
        // from the call after the first compression.
        const int active = step % speakers;
        for (std::size_t frame = 0; frame < total; ++frame) {
            const int alternate =
                (active + 1 + static_cast<int>(frame % 2)) % speakers;
            for (int speaker = 0; speaker < speakers; ++speaker) {
                const bool speaking =
                    speaker == active || speaker == alternate;
                predictions[frame * speakers + speaker] =
                    speaking ? 0.80f + sequence.next() * 0.19f
                             : sequence.next() * 0.10f;
            }
        }

        const auto chunk_predictions = cache.advance(
            embeddings.data(), chunk, predictions.data(), total, left, right,
            SortformerSpeakerCache::PredictionLayout::Packed);
        check(chunk_predictions.size() / speakers == owned,
              "a step owns its chunk minus the context either side");
        check(cache.cache_frames()
                  == static_cast<std::size_t>(config.cache_frames),
              "the cache stays at its configured length");
        check(cache.fifo_frames()
                  <= static_cast<std::size_t>(config.fifo_frames),
              "the fifo never exceeds its configured length");

        last_sum = 0.0;
        last_ordered = 0.0;
        std::size_t position = 0;
        for (const float value : cache.cache()) {
            last_sum += value;
            last_ordered +=
                static_cast<double>(position++ % 9973) * value;
        }
    }
    // Reference values from the numpy port over the same sequence. It is a
    // checksum of the retained embeddings, so it moves if eviction keeps
    // different frames — which is the failure this guards against.
    check_near(last_sum, 17.995605, 1e-3,
               "retained embeddings match the reference implementation");
    // And a position-weighted one, because a plain sum cannot see ORDER, and
    // order is a real failure mode here rather than a hypothetical: the
    // selection is speaker-major, so sorting the chosen indices before mapping
    // them back to frames gives strict chronological order while sorting after
    // gives the reference's speaker-major order. Both keep exactly the same
    // frames, so the sum above agreed to six figures while the model saw a
    // different cache and its predictions diverged from the call after the
    // first compression. This suite passed throughout.
    check_near(last_ordered, 4088.0877, 1e-1,
               "retained embeddings are in the reference's order");
}

void test_reset_restarts_arrival_order() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    Sequence sequence;

    std::vector<float> embeddings(14 * config.embedding_dim, 0.5f);
    const std::size_t total = cache.cache_frames() + cache.fifo_frames() + 14;
    std::vector<float> predictions(total * config.speakers, 0.9f);
    cache.advance(
        embeddings.data(), 14, predictions.data(), total, 1, 7,
        SortformerSpeakerCache::PredictionLayout::Packed);
    check(cache.fifo_frames() > 0, "a step leaves work in the fifo");

    cache.reset();
    check(cache.fifo_frames() == 0, "reset empties the fifo");
    check(cache.cache_frames()
              == static_cast<std::size_t>(config.cache_frames),
          "reset restores the cache to its configured length");
    double sum = 0.0;
    for (const float value : cache.cache()) sum += value;
    check_near(sum, 0.0, 1e-6, "reset clears the retained embeddings");
    (void)sequence;
}

void test_short_predictions_are_refused() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    std::vector<float> embeddings(14 * config.embedding_dim, 0.1f);
    // Fewer prediction frames than [cache + fifo + chunk] means the caller fed
    // a graph output that does not describe this state. Returning nothing is
    // better than indexing past the end of it.
    std::vector<float> predictions(4 * config.speakers, 0.5f);
    const auto out = cache.advance(
        embeddings.data(), 14, predictions.data(), 4, 1, 7,
        SortformerSpeakerCache::PredictionLayout::Packed);
    check(out.empty(), "predictions shorter than the state are refused");
}

void test_chunk_shorter_than_its_context_is_refused() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    // Six frames with one left and seven right of context describes nothing:
    // the chunk owns a negative number of frames. Unsigned arithmetic would
    // wrap that into an enormous read past both buffers, so it is refused.
    std::vector<float> embeddings(6 * config.embedding_dim, 0.1f);
    const std::size_t total = cache.cache_frames() + cache.fifo_frames() + 6;
    std::vector<float> predictions(total * config.speakers, 0.5f);
    const auto out = cache.advance(
        embeddings.data(), 6, predictions.data(), total, 1, 7,
        SortformerSpeakerCache::PredictionLayout::Packed);
    check(out.empty(), "a chunk shorter than its context is refused");
    check(cache.fifo_frames() == 0, "a refused chunk leaves no state behind");
}

void test_negative_context_is_refused() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    std::vector<float> embeddings(14 * config.embedding_dim, 0.1f);
    const std::size_t total = cache.cache_frames() + cache.fifo_frames() + 14;
    std::vector<float> predictions(total * config.speakers, 0.5f);
    const auto out = cache.advance(
        embeddings.data(), 14, predictions.data(), total, -1, 7,
        SortformerSpeakerCache::PredictionLayout::Packed);
    check(out.empty(), "negative context is refused");
}

/// The layout trap, which is the reason `advance` makes the caller say.
///
/// The exported graph's FIFO block is a fixed width with the unused frames
/// zeroed; the synchronous reference's FIFO is only as long as it holds. On the
/// first call of a recording the FIFO is empty, so the two put the chunk
/// `fifo_frames` apart — and reading the graph's buffer as packed silently
/// reports on the wrong slice.
///
/// Both readings are exercised on one buffer built so the difference is
/// visible: the FIFO block carries a marker value the chunk does not.
void test_the_fifo_block_width_decides_where_the_chunk_starts() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    const std::size_t chunk = 14;
    const int left = 1;
    const int right = 7;
    const std::size_t owned = chunk - left - right;

    check(cache.fifo_frames() == 0, "a fresh cache holds nothing in its fifo");

    // [cache | fifo block, all marker | chunk], as the graph emits it.
    const std::size_t block =
        cache.cache_frames() + static_cast<std::size_t>(config.fifo_frames)
        + chunk;
    std::vector<float> predictions(block * config.speakers, 0.25f);
    const std::size_t fifo_start = cache.cache_frames() * config.speakers;
    const std::size_t chunk_start =
        (cache.cache_frames() + config.fifo_frames) * config.speakers;
    for (std::size_t index = fifo_start; index < chunk_start; ++index) {
        predictions[index] = 0.99f;
    }
    for (std::size_t index = chunk_start;
         index < block * config.speakers; ++index) {
        predictions[index] = 0.10f;
    }
    std::vector<float> embeddings(chunk * config.embedding_dim, 0.1f);

    const auto graph_reading = cache.advance(
        embeddings.data(), chunk, predictions.data(), block, left, right,
        SortformerSpeakerCache::PredictionLayout::FixedFifoBlock);
    check(graph_reading.size() / config.speakers == owned,
          "the graph reading owns its chunk minus the context");
    for (float value : graph_reading) {
        check_near(value, 0.10f, 1e-6, "the graph reading lands in the chunk");
    }

    // The same buffer read as packed starts the chunk where the FIFO block
    // does, so it reports the marker instead — which is the defect, reproduced.
    SortformerSpeakerCache other(config);
    const auto packed_reading = other.advance(
        embeddings.data(), chunk, predictions.data(), block, left, right,
        SortformerSpeakerCache::PredictionLayout::Packed);
    check(!packed_reading.empty(), "the packed reading still returns a chunk");
    check_near(packed_reading.front(), 0.99f, 1e-6,
               "reading the graph's buffer as packed takes the fifo's frames");
}

}  // namespace

int main() {
    test_matches_reference_through_eviction();
    test_reset_restarts_arrival_order();
    test_short_predictions_are_refused();
    test_chunk_shorter_than_its_context_is_refused();
    test_negative_context_is_refused();
    test_the_fifo_block_width_decides_where_the_chunk_starts();
    if (failures == 0) std::printf("sortformer speaker cache: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
