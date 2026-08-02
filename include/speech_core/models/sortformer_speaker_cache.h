#pragma once

#include <cstddef>
#include <vector>

namespace speech_core {

/// Arrival-Order Speaker Cache for streaming Sortformer.
///
/// The exported graph takes the speaker cache and FIFO as *inputs* and returns
/// fresh embeddings as an *output*; it never updates them. Deciding what enters
/// the cache, what waits, and what is evicted happens here, and that
/// bookkeeping is what makes a speaker index mean the same person for a whole
/// recording. A caller that skips it gets per-call speaker numbering and none
/// of the model's advantage over a window-local segmenter.
///
/// Deliberately free of ONNX Runtime and of the model: it takes plain
/// embeddings and predictions. That makes it testable against the reference on
/// any machine, which matters because getting it wrong is quiet — the model
/// still runs, still emits four plausible probabilities per frame, and simply
/// stops meaning the same person.
///
/// Ported from NeMo's `SortformerModules.streaming_update_async`. Batch is one:
/// this serves a single recording, and a batch axis would only obscure the
/// arithmetic.
class SortformerSpeakerCache {
public:
    struct Config {
        /// These belong to the *exported variant*, not to NeMo's class
        /// defaults. Pairing one variant's graph with another's periods evicts
        /// the wrong frames while looking healthy, so they are read from the
        /// bundle's config rather than assumed.
        int speakers = 4;
        int cache_frames = 188;
        int fifo_frames = 40;
        int update_period = 188;
        int silence_frames_per_speaker = 3;
        int embedding_dim = 512;

        float prediction_floor = 0.25f;
        float latest_boost = 0.05f;
        float silence_threshold = 0.2f;
        float strong_boost_rate = 0.75f;
        float weak_boost_rate = 1.5f;
        float minimum_positive_rate = 0.5f;
    };

    explicit SortformerSpeakerCache(Config config);

    /// Advance one step and return this chunk's predictions.
    ///
    /// `predictions` covers [cache + fifo + chunk] exactly as the graph emitted
    /// it. `chunk_embeddings` is the graph's per-chunk output, which is what
    /// enters the FIFO and eventually the cache.
    ///
    /// `left_context` and `right_context` are in encoder frames: the chunk
    /// carries context either side that the model needed but that this step
    /// does not own, and including it would publish the same audio twice.
    std::vector<float> advance(
        const float* chunk_embeddings,
        std::size_t chunk_frames,
        const float* predictions,
        std::size_t prediction_frames,
        int left_context,
        int right_context);

    /// The cache and FIFO the next call must feed back to the graph.
    const std::vector<float>& cache() const { return cache_; }
    const std::vector<float>& fifo() const { return fifo_; }
    std::size_t cache_frames() const { return cache_frames_; }
    std::size_t fifo_frames() const { return fifo_frames_; }

    /// Forget everything. Arrival order restarts, so a speaker index after this
    /// has no relationship to one before it.
    void reset();

private:
    void compress();
    void update_silence_profile(
        const float* embeddings, const float* predictions, std::size_t frames);

    Config config_;
    /// [cache_frames_ x embedding_dim], the graph's `spkcache` input.
    std::vector<float> cache_;
    /// Predictions for the cached frames, which is what scores them at
    /// eviction. Empty until the cache has been compressed at least once.
    std::vector<float> cache_predictions_;
    std::vector<float> fifo_;
    std::vector<float> fifo_predictions_;
    std::vector<float> mean_silence_;
    std::size_t cache_frames_ = 0;
    std::size_t fifo_frames_ = 0;
    std::size_t silence_frames_ = 0;
    bool cache_scored_ = false;
};

}  // namespace speech_core
