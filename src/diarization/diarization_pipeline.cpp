#include "speech_core/diarization/diarization_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace speech_core {

namespace {
/// Cosine similarity over two equal-length vectors. 0 if shapes mismatch or
/// either side has ~zero norm.
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-8f) ? dot / denom : 0.0f;
}
}  // namespace

DiarizationPipeline::DiarizationPipeline(
    SegmentationInterface& seg, EmbeddingInterface& emb, float clustering_threshold)
    : seg_(seg), emb_(emb), default_threshold_(clustering_threshold) {}

std::vector<DiarizedSegment> DiarizationPipeline::diarize(
    const float* audio, size_t length, int sample_rate, const DiarizerConfig& config)
{
    auto windows = seg_.segment(audio, length, sample_rate);
    if (windows.empty()) return {};

    auto local_speakers = extract_embeddings(audio, length, windows, config);
    if (local_speakers.empty()) return {};

    float thresh = (config.clustering_threshold > 0)
        ? config.clustering_threshold : default_threshold_;
    auto cluster_ids = cluster(local_speakers, thresh,
                               config.min_speakers, config.max_speakers);

    return build_segments(windows, local_speakers, cluster_ids, config);
}

std::vector<DiarizationPipeline::LocalSpeaker>
DiarizationPipeline::extract_embeddings(
    const float* audio, size_t length,
    const std::vector<SegmentationWindow>& windows,
    const DiarizerConfig& config)
{
    std::vector<LocalSpeaker> speakers;
    const int K  = seg_.max_local_speakers();
    const int sr = seg_.input_sample_rate();

    for (int w = 0; w < static_cast<int>(windows.size()); ++w) {
        auto& win = windows[w];
        int nf = static_cast<int>(win.speaker_activity.size()) / K;
        if (nf <= 0) continue;
        float frame_dur = (win.end_time - win.start_time) / static_cast<float>(nf);

        for (int spk = 0; spk < K; ++spk) {
            float best_start = 0, best_end = 0, best_dur = 0;
            float cur_start = 0;
            bool  active = false;

            for (int f = 0; f < nf; ++f) {
                float act = win.speaker_activity[f * K + spk];
                float t   = win.start_time + f * frame_dur;
                if (!active && act >= config.onset) {
                    active = true;
                    cur_start = t;
                } else if (active && act < config.offset) {
                    float dur = t - cur_start;
                    if (dur > best_dur) { best_dur = dur; best_start = cur_start; best_end = t; }
                    active = false;
                }
            }
            if (active) {
                float dur = win.end_time - cur_start;
                if (dur > best_dur) { best_dur = dur; best_start = cur_start; best_end = win.end_time; }
            }

            if (best_dur < 0.5f) continue;

            size_t s0 = static_cast<size_t>(best_start * sr);
            size_t s1 = std::min(static_cast<size_t>(best_end * sr), length);
            if (s1 <= s0) continue;

            auto emb = emb_.embed(audio + s0, s1 - s0, sr);
            if (emb.empty()) continue;

            speakers.push_back({w, spk, best_start, best_end, std::move(emb)});
        }
    }

    return speakers;
}

std::vector<int> DiarizationPipeline::cluster(
    const std::vector<LocalSpeaker>& speakers,
    float threshold, int min_clusters, int max_clusters)
{
    int n = static_cast<int>(speakers.size());
    if (n == 0) return {};

    std::vector<int> labels(n);
    std::iota(labels.begin(), labels.end(), 0);

    std::vector<std::vector<float>> dist(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            float sim = cosine_similarity(speakers[i].embedding, speakers[j].embedding);
            float d = 1.0f - sim;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    auto get_cluster = [&](int idx) -> std::set<int> {
        std::set<int> members;
        int c = labels[idx];
        for (int i = 0; i < n; ++i) if (labels[i] == c) members.insert(i);
        return members;
    };

    while (true) {
        int num_clusters = static_cast<int>(std::set<int>(labels.begin(), labels.end()).size());

        if (max_clusters > 0 && num_clusters <= max_clusters &&
            min_clusters > 0 && num_clusters <= min_clusters) break;

        float best_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;

        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (labels[i] == labels[j]) continue;

                auto ci = get_cluster(i);
                auto cj = get_cluster(j);
                bool same_window = false;
                for (int a : ci) {
                    for (int b : cj) {
                        if (speakers[a].window_idx == speakers[b].window_idx) { same_window = true; break; }
                    }
                    if (same_window) break;
                }
                if (same_window) continue;

                float avg_dist = 0;
                int count = 0;
                for (int a : ci) for (int b : cj) { avg_dist += dist[a][b]; ++count; }
                avg_dist /= static_cast<float>(count);

                if (avg_dist < best_dist) { best_dist = avg_dist; best_i = i; best_j = j; }
            }
        }

        if (best_i < 0 || best_dist > threshold) break;

        int old_label = labels[best_j];
        int new_label = labels[best_i];
        for (int i = 0; i < n; ++i) if (labels[i] == old_label) labels[i] = new_label;
    }

    std::set<int> unique(labels.begin(), labels.end());
    std::unordered_map<int, int> remap;
    int idx = 0;
    for (int c : unique) remap[c] = idx++;
    for (auto& l : labels) l = remap[l];

    return labels;
}

std::vector<DiarizedSegment> DiarizationPipeline::build_segments(
    const std::vector<SegmentationWindow>& windows,
    const std::vector<LocalSpeaker>& local_speakers,
    const std::vector<int>& cluster_ids,
    const DiarizerConfig& config)
{
    std::map<std::pair<int, int>, int> speaker_map;
    for (size_t i = 0; i < local_speakers.size(); ++i) {
        speaker_map[{local_speakers[i].window_idx, local_speakers[i].local_speaker}] = cluster_ids[i];
    }

    std::vector<DiarizedSegment> segments;
    const int K = seg_.max_local_speakers();

    for (int w = 0; w < static_cast<int>(windows.size()); ++w) {
        auto& win = windows[w];
        int nf = static_cast<int>(win.speaker_activity.size()) / K;
        if (nf <= 0) continue;
        float frame_dur = (win.end_time - win.start_time) / static_cast<float>(nf);

        for (int spk = 0; spk < K; ++spk) {
            auto it = speaker_map.find({w, spk});
            if (it == speaker_map.end()) continue;
            int global_spk = it->second;

            bool  active = false;
            float seg_start = 0;

            for (int f = 0; f < nf; ++f) {
                float act = win.speaker_activity[f * K + spk];
                float t   = win.start_time + f * frame_dur;
                if (!active && act >= config.onset) {
                    active = true;
                    seg_start = t;
                } else if (active && act < config.offset) {
                    if (t - seg_start >= config.min_speech_duration)
                        segments.push_back({seg_start, t, global_spk});
                    active = false;
                }
            }
            if (active) {
                float t = win.end_time;
                if (t - seg_start >= config.min_speech_duration)
                    segments.push_back({seg_start, t, global_spk});
            }
        }
    }

    std::sort(segments.begin(), segments.end(),
              [](const DiarizedSegment& a, const DiarizedSegment& b) { return a.start < b.start; });

    std::vector<DiarizedSegment> merged;
    for (auto& seg : segments) {
        if (!merged.empty() && merged.back().speaker == seg.speaker &&
            seg.start - merged.back().end < 0.5f) {
            merged.back().end = seg.end;
        } else {
            merged.push_back(seg);
        }
    }

    return merged;
}

}  // namespace speech_core
