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
    for (int step = 0; step < 24; ++step) {
        std::vector<float> embeddings(
            static_cast<std::size_t>(chunk) * width);
        for (auto& value : embeddings) value = sequence.next() * 2.0f - 1.0f;

        const std::size_t total =
            cache.cache_frames() + cache.fifo_frames() + chunk;
        std::vector<float> predictions(total * speakers);
        const int active = step % speakers;
        for (std::size_t frame = 0; frame < total; ++frame) {
            for (int speaker = 0; speaker < speakers; ++speaker) {
                predictions[frame * speakers + speaker] =
                    speaker == active ? 0.80f + sequence.next() * 0.19f
                                      : sequence.next() * 0.10f;
            }
        }

        const auto chunk_predictions = cache.advance(
            embeddings.data(), chunk, predictions.data(), total, left, right);
        check(chunk_predictions.size() / speakers == owned,
              "a step owns its chunk minus the context either side");
        check(cache.cache_frames()
                  == static_cast<std::size_t>(config.cache_frames),
              "the cache stays at its configured length");
        check(cache.fifo_frames()
                  <= static_cast<std::size_t>(config.fifo_frames),
              "the fifo never exceeds its configured length");

        last_sum = 0.0;
        for (const float value : cache.cache()) last_sum += value;
    }
    // Reference value from the numpy port over the same sequence. It is a
    // checksum of the retained embeddings, so it moves if eviction keeps
    // different frames — which is the failure this guards against.
    check_near(last_sum, 0.810773, 1e-3,
               "retained embeddings match the reference implementation");
}

void test_reset_restarts_arrival_order() {
    const auto config = small_config();
    SortformerSpeakerCache cache(config);
    Sequence sequence;

    std::vector<float> embeddings(14 * config.embedding_dim, 0.5f);
    const std::size_t total = cache.cache_frames() + cache.fifo_frames() + 14;
    std::vector<float> predictions(total * config.speakers, 0.9f);
    cache.advance(embeddings.data(), 14, predictions.data(), total, 1, 7);
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
        embeddings.data(), 14, predictions.data(), 4, 1, 7);
    check(out.empty(), "predictions shorter than the state are refused");
}

}  // namespace

int main() {
    test_matches_reference_through_eviction();
    test_reset_restarts_arrival_order();
    test_short_predictions_are_refused();
    if (failures == 0) std::printf("sortformer speaker cache: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
