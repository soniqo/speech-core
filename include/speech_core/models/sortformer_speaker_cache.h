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
/// Ported from NeMo's `SortformerModules.streaming_update` -- the synchronous
/// one. The async variant is a different algorithm, not a batched wrapper: it
/// carries `spkcache_lengths` and reads each batch item's predictions at its
/// own valid offset. Batch is one here: this serves a single recording, and a
/// batch axis would only obscure the arithmetic.
///
/// One deliberate departure. NeMo grows the cache from empty; the exported
/// graph cannot, because `spkcache` is declared [1, 188, 512] and every call
/// must fill it. So the cache starts as `cache_frames` zero frames and every
/// slot is treated as live, which is the async variant's fixed-buffer
/// representation driven by the synchronous variant's arithmetic. The padding
/// leaves on its own: a zero embedding scores as non-speech, `disable_low_scores`
/// sends it to negative infinity, and the first compression replaces it with
/// mean silence. That argument holds for predictions a real graph produces and
/// not for arbitrary ones, so a harness that feeds every frame confident speech
/// will see the padding survive and should not read that as a defect.
///
/// Checked against NeMo 2.7.3 driven on identical input, at both the streaming
/// variant's periods and the shipped default's: worst relative cache difference
/// 4.9e-07 and 5.0e-06 respectively, which is float32 accumulation over 96,256
/// values. Give NeMo the same zero-prefilled cache when repeating this, and call
/// `.eval()` first -- `_compress_spkcache` takes `permute_spk=self.training`,
/// and an nn.Module defaults to training mode, so a stock instance permutes
/// speakers at random and matches nothing.
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
