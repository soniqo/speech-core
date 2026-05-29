#pragma once

#include "speech_core/interfaces.h"

#include <vector>

namespace speech_core {

/// Full diarization: segmentation → embedding → constrained agglomerative
/// clustering. Composes a SegmentationInterface (per-window local speaker
/// activity) with an EmbeddingInterface (per-speaker vectors). Backend-agnostic
/// — pure orchestration, no ML runtime dependency.
class DiarizationPipeline : public DiarizerInterface {
public:
    DiarizationPipeline(SegmentationInterface& seg,
                        EmbeddingInterface& emb,
                        float clustering_threshold = 0.715f);

    std::vector<DiarizedSegment> diarize(
        const float* audio, size_t length, int sample_rate,
        const DiarizerConfig& config) override;

private:
    struct LocalSpeaker {
        int   window_idx;
        int   local_speaker;
        float start;
        float end;
        std::vector<float> embedding;
    };

    std::vector<LocalSpeaker> extract_embeddings(
        const float* audio, size_t length,
        const std::vector<SegmentationWindow>& windows,
        const DiarizerConfig& config);

    std::vector<int> cluster(
        const std::vector<LocalSpeaker>& speakers,
        float threshold, int min_clusters, int max_clusters);

    std::vector<DiarizedSegment> build_segments(
        const std::vector<SegmentationWindow>& windows,
        const std::vector<LocalSpeaker>& local_speakers,
        const std::vector<int>& cluster_ids,
        const DiarizerConfig& config);

    SegmentationInterface& seg_;
    EmbeddingInterface&    emb_;
    float default_threshold_;
};

}  // namespace speech_core
