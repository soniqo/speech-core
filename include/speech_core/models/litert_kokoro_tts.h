#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/kokoro_phonemizer.h"

#include <atomic>
#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#ifdef SPEECH_CORE_HAS_LITERT_KOKORO

struct TfLiteModel;
struct TfLiteInterpreter;
struct TfLiteOpaqueDelegate;

namespace speech_core {

/// Kokoro 82M text-to-speech using the validated staged 60-frame FP32 LiteRT bundle.
///
/// The exported bundle accepts at most 32 active phonemes and produces a fixed
/// 36,000-sample buffer. The final convolutional network needs unused right
/// context, so this wrapper accepts at most 56 predicted frames per invocation.
/// Longer text is split and an overflowing piece is discarded and retried as
/// smaller pieces; a merely in-bounds 60-frame output is not treated as safe.
class LiteRTKokoroTts : public TTSInterface {
public:
    LiteRTKokoroTts(const std::string& model_path,
                    const std::string& voices_dir,
                    const std::string& data_dir,
                    bool hw_accel = false,
                    int num_threads = 4);
    /// Three-stage export with the compact recurrent graph.  The stage tensor
    /// contracts are fixed and validated at construction; all three graphs stay
    /// resident so repeated synthesis does not pay model reload overhead.
    LiteRTKokoroTts(const std::string& encoder_path,
                    const std::string& recurrent_path,
                    const std::string& vocoder_path,
                    const std::string& voices_dir,
                    const std::string& data_dir,
                    bool hw_accel = false,
                    int num_threads = 4);
    ~LiteRTKokoroTts() override;

    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    void set_voice(const std::string& name);
    void set_speed(float speed);
    void set_seed(uint32_t seed) { seed_ = seed; }

    uint32_t seed_used() const { return seed_used_; }
    int model_runs_last_synthesis() const { return model_runs_last_; }
    int num_threads() const { return num_threads_; }
    std::array<double, 3> last_stage_ms() const { return last_stage_ms_; }

    static constexpr int max_active_phonemes() { return kActivePhonemes; }
    static constexpr int preferred_chunk_phonemes() { return kPreferredPhonemes; }
    static constexpr int max_safe_frames() { return kSafeFrames; }

private:
    struct InferenceResult {
        std::vector<float> audio;
        int64_t valid_samples = 0;
    };

    std::vector<float> load_voice_embedding(const std::string& name) const;
    void auto_switch_voice(const std::string& language);
    size_t phoneme_count(const std::string& text);
    std::vector<std::string> split_for_phoneme_limit(const std::string& text);
    std::vector<std::string> split_in_half(const std::string& text) const;
    std::vector<float> synthesize_piece(const std::string& text,
                                        std::mt19937& rng,
                                        int depth);
    InferenceResult invoke_tokens(const std::vector<int64_t>& tokens,
                                  std::mt19937& rng);
    InferenceResult invoke_staged_tokens(const std::vector<int64_t>& tokens,
                                         std::mt19937& rng);
    void load_support_data(const std::string& data_dir);
    static std::vector<float> finish_audio(std::vector<float> audio,
                                           size_t valid_samples,
                                           const std::string& text);

    static constexpr int kInputPhonemes = 128;
    static constexpr int kActivePhonemes = 32;
    static constexpr int kPreferredPhonemes = 14;
    static constexpr int kMaxFrames = 60;
    static constexpr int kSafeFrames = 56;
    static constexpr int kSamplesPerFrame = 600;
    static constexpr int kOutputSamples = kMaxFrames * kSamplesPerFrame;

    TfLiteModel* model_ = nullptr;
    TfLiteInterpreter* interpreter_ = nullptr;
    TfLiteOpaqueDelegate* xnnpack_delegate_ = nullptr;
    int audio_output_idx_ = -1;
    int length_output_idx_ = -1;
    int duration_output_idx_ = -1;

    static constexpr size_t kStageCount = 3;
    bool staged_ = false;
    std::array<TfLiteModel*, kStageCount> staged_models_{};
    std::array<TfLiteInterpreter*, kStageCount> staged_interpreters_{};
    std::array<TfLiteOpaqueDelegate*, kStageCount> staged_delegates_{};
    std::vector<float> duration_features_;
    std::vector<float> text_features_;
    std::vector<float> aligned_text_;
    std::vector<float> shared_features_;
    std::array<double, kStageCount> last_stage_ms_{};

    KokoroPhonemizer phonemizer_;
    std::vector<float> voice_embedding_;
    std::string voices_dir_;
    std::string current_lang_;
    float speed_ = 1.0f;
    uint32_t seed_ = 0;
    uint32_t seed_used_ = 0;
    int num_threads_ = 4;
    int model_runs_last_ = 0;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core

#endif  // SPEECH_CORE_HAS_LITERT_KOKORO
