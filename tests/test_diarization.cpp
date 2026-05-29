// Unit tests for DiarizationPipeline — runs in the default build (no LiteRT, no
// model files). Drives the clustering + segment-building logic with mock
// Segmentation/Embedding implementations so the algorithm is exercised in PR CI,
// not only the model-gated nightly e2e.

#include "speech_core/diarization/diarization_pipeline.h"
#include "speech_core/interfaces.h"

#include <cassert>
#include <cstdio>
#include <set>
#include <vector>

using namespace speech_core;

namespace {

// Returns preset windows verbatim; ignores the audio.
class MockSegmentation : public SegmentationInterface {
public:
    std::vector<SegmentationWindow> windows;
    int max_spk = 2;

    std::vector<SegmentationWindow> segment(const float*, size_t, int) override {
        return windows;
    }
    int input_sample_rate() const override { return 16000; }
    int max_local_speakers() const override { return max_spk; }
};

// Direction-distinct embedding keyed on the sign of the region's mean sample —
// so the test can make two regions "the same speaker" (same sign) or "different
// speakers" (opposite sign) by how it fills the audio buffer.
class MockEmbedding : public EmbeddingInterface {
public:
    std::vector<float> embed(const float* audio, size_t len, int) override {
        double s = 0.0;
        for (size_t i = 0; i < len; ++i) s += audio[i];
        float mean = len ? static_cast<float>(s / len) : 0.0f;
        if (mean > 0.01f)  return {1.0f, 0.0f, 0.0f};
        if (mean < -0.01f) return {0.0f, 1.0f, 0.0f};
        return {0.0f, 0.0f, 1.0f};
    }
    int embedding_dim() const override { return 3; }
    int input_sample_rate() const override { return 16000; }
};

// One window [start,end] where local speaker 0 is active across all frames.
SegmentationWindow window_spk0(float start, float end, int K, int nf) {
    SegmentationWindow w;
    w.start_time = start;
    w.end_time   = end;
    w.speaker_activity.assign(static_cast<size_t>(nf) * K, 0.0f);
    for (int f = 0; f < nf; ++f) w.speaker_activity[f * K + 0] = 1.0f;
    return w;
}

void assert_well_formed(const std::vector<DiarizedSegment>& segs) {
    for (size_t i = 0; i < segs.size(); ++i) {
        assert(segs[i].speaker >= 0);
        assert(segs[i].end > segs[i].start);
        if (i > 0) assert(segs[i].start >= segs[i - 1].start);  // time-sorted
    }
}

std::set<int> distinct_speakers(const std::vector<DiarizedSegment>& segs) {
    std::set<int> s;
    for (auto& g : segs) s.insert(g.speaker);
    return s;
}

// Two windows, one active speaker each, with DIFFERENT embeddings → the
// clusterer must keep them apart (cosine distance 1 > threshold 0.715).
void test_distinct_speakers_stay_separate() {
    MockSegmentation seg;
    MockEmbedding    emb;
    seg.windows = {window_spk0(0.0f, 1.0f, 2, 10), window_spk0(1.0f, 2.0f, 2, 10)};

    std::vector<float> audio(32000, 0.0f);
    for (int i = 0;     i < 16000; ++i) audio[i] =  0.5f;  // window 0 → [1,0,0]
    for (int i = 16000; i < 32000; ++i) audio[i] = -0.5f;  // window 1 → [0,1,0]

    DiarizationPipeline diar(seg, emb);
    auto segs = diar.diarize(audio.data(), audio.size(), 16000, DiarizerConfig{});

    assert(!segs.empty());
    assert_well_formed(segs);
    assert(distinct_speakers(segs).size() == 2);
    printf("  PASS: distinct_speakers_stay_separate (%zu segments)\n", segs.size());
}

// Two windows, one active speaker each, with the SAME embedding → the clusterer
// must merge them into one speaker (cosine distance 0 < threshold).
void test_same_speaker_merges_across_windows() {
    MockSegmentation seg;
    MockEmbedding    emb;
    seg.windows = {window_spk0(0.0f, 1.0f, 2, 10), window_spk0(1.0f, 2.0f, 2, 10)};

    std::vector<float> audio(32000, 0.5f);  // both regions → [1,0,0]

    DiarizationPipeline diar(seg, emb);
    auto segs = diar.diarize(audio.data(), audio.size(), 16000, DiarizerConfig{});

    assert(!segs.empty());
    assert_well_formed(segs);
    assert(distinct_speakers(segs).size() == 1);
    printf("  PASS: same_speaker_merges_across_windows (%zu segments)\n", segs.size());
}

// Empty segmentation → empty result, no crash.
void test_empty_segmentation() {
    MockSegmentation seg;  // no windows
    MockEmbedding    emb;
    std::vector<float> audio(16000, 0.5f);

    DiarizationPipeline diar(seg, emb);
    auto segs = diar.diarize(audio.data(), audio.size(), 16000, DiarizerConfig{});
    assert(segs.empty());
    printf("  PASS: empty_segmentation\n");
}

}  // namespace

int main() {
    printf("test_diarization\n");
    test_distinct_speakers_stay_separate();
    test_same_speaker_merges_across_windows();
    test_empty_segmentation();
    printf("All diarization tests passed\n");
    return 0;
}
