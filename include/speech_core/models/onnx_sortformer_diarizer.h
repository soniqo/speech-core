#pragma once

#include "speech_core/models/sortformer_speaker_cache.h"

#include <onnxruntime_c_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// Streaming Sortformer diarization via ONNX Runtime.
///
/// Answers *who spoke when* as a continuous per-frame timeline, and nothing
/// else. It produces no speaker embeddings, so naming a voice remains a
/// separate question for an identity encoder — the same division a caller
/// already has with any joint transcribe-diarize model.
///
/// Matches soniqo/Sortformer-Diarization-4spk-ONNX, the `default` export:
///
///   in:  chunk[1, 3048, 128]  spkcache[1, 188, 512]  fifo[1, 40, 512]
///   out: spkcache_fifo_chunk_preds[1, 609, 4]
///        chunk_pre_encode_embs[1, 381, 512]
///
/// 609 = 188 cache + 40 fifo + 381 window frames, and 3048 = 381 x 8. One call
/// consumes 340 encoder frames — 27.2 s — of new audio and needs 40 more frames
/// of audio after them, so a label is about 30 s behind the microphone. That
/// latency is the design: labels are meant to reach text some other model has
/// already displayed.
///
/// ## Why this is a stream and not a function
///
/// Arrival order is per *stream*. The speaker cache is what makes index 2 mean
/// the same person in minute one and minute forty, and it is built up call by
/// call. Constructing one of these per utterance, or calling `reset()` between
/// paragraphs, restarts the numbering and throws away the entire advantage this
/// model has over a window-local segmenter — after paying for its download and
/// its arrival-order bookkeeping. Feed one instance the whole recording and ask
/// it about time ranges afterwards.
///
/// ## What is deliberately absent
///
/// No threshold, and no notion of a turn, a segment or a speaker label. This
/// returns probabilities. Deciding what counts as active speech, and how a
/// range of frames becomes somebody's turn, is a product judgement with a
/// measured constant behind it; an engine that invented one would be making
/// that decision on the application's behalf, in the repository least able to
/// tell when it had changed.
///
/// Not thread-safe — one per recording, driven from one thread.
class OnnxSortformerDiarizer {
public:
    struct Config {
        int sample_rate = 16000;

        /// Encoder frames the model needs either side of the frames a call
        /// reports on. They belong to neighbouring calls; publishing them would
        /// report the same audio twice.
        int left_context = 1;
        int right_context = 40;

        /// Mel frames per encoder frame, and the mel hop. Together these fix
        /// how much wall-clock one output frame covers: 8 x 10 ms = 80 ms.
        int subsampling = 8;
        int mel_hop = 160;
        int mel_bins = 128;
        int n_fft = 512;
        int win_length = 400;

        /// The arrival-order cache's own parameters. Its defaults belong to
        /// this exported variant; pairing one variant's graph with another's
        /// periods evicts the wrong frames while looking healthy.
        SortformerSpeakerCache::Config cache;
    };

    /// `hw_accel` selects a hardware execution provider where one is available.
    /// The geometry is read from the graph rather than assumed — a bundle
    /// exported at a different chunk length would otherwise be driven with this
    /// one's windowing and produce confident nonsense.
    explicit OnnxSortformerDiarizer(
        const std::string& model_path, bool hw_accel = true);
    ~OnnxSortformerDiarizer();

    OnnxSortformerDiarizer(const OnnxSortformerDiarizer&) = delete;
    OnnxSortformerDiarizer& operator=(const OnnxSortformerDiarizer&) = delete;

    /// Feed 16 kHz mono PCM in arrival order.
    ///
    /// Returns whatever frames became final during this call, as
    /// `[frames x speakers]` probabilities in chronological order — usually
    /// nothing, then 340 frames at once when enough audio has arrived. Append
    /// them; frame `i` of the recording covers
    /// `[i * frame_seconds(), (i + 1) * frame_seconds())`.
    std::vector<float> push_audio(const float* samples, std::size_t length);

    /// Finalise the tail. The remaining audio is padded to a full window, so
    /// the last frames see zeros where the recording simply stopped.
    ///
    /// Returns the frames that padding made final.
    std::vector<float> end_stream();

    /// Forget everything. Arrival order restarts, so a speaker index after this
    /// has no relationship to one before it.
    void reset();

    int speakers() const { return speakers_; }
    /// Wall-clock covered by one output frame.
    float frame_seconds() const;
    /// Encoder frames finalised so far, which is also the timeline's length.
    std::size_t frames_emitted() const { return frames_emitted_; }

    /// Encoder frames one call reports on — read from the graph.
    int chunk_frames() const { return chunk_frames_; }

private:
    /// Run one call for step `step_`, or return false if `flush` is false and
    /// the audio it needs has not arrived.
    bool advance_step(bool flush, std::vector<float>& out);

    /// Mel for the window this step needs, zero-filled outside the recording.
    std::vector<float> window_features(std::int64_t first_frame) const;

    const OrtApi* api_ = nullptr;
    OrtSession* session_ = nullptr;

    Config cfg_;
    /// Built after the graph has been read, because its widths come from the
    /// graph rather than from this class's defaults.
    std::unique_ptr<SortformerSpeakerCache> cache_;

    /// Read from the graph, never assumed.
    int window_frames_ = 0;   // 381
    int window_mels_ = 0;     // 3048
    int chunk_frames_ = 0;    // 340 = window - left - right
    int speakers_ = 0;        // 4
    int embedding_dim_ = 0;   // 512
    int cache_frames_ = 0;    // 188
    int fifo_frames_ = 0;     // 40

    /// Audio still needed by a future step, and where it starts. Trimmed as
    /// steps complete so a long recording does not accumulate.
    std::vector<float> audio_;
    std::size_t audio_start_sample_ = 0;
    std::size_t samples_seen_ = 0;

    std::size_t step_ = 0;
    std::size_t frames_emitted_ = 0;
    bool finished_ = false;
};

}  // namespace speech_core
