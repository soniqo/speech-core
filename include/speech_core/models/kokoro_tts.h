#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/kokoro_phonemizer.h"

#include <onnxruntime_c_api.h>
#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

namespace speech_core {

/// Kokoro 82M — lightweight text-to-speech via ONNX Runtime.
/// Non-autoregressive synthesis, sentence-chunked for bounded model exports.
/// Output: 24 kHz PCM Float32.
class KokoroTts : public TTSInterface {
public:
    struct Config {
        bool hw_accel = true;

        /// Preferred packing budget and absolute phoneme-token ceiling for
        /// each internal model run. Tiny tails may first merge with the prior
        /// text chunk, but a preflight split keeps every actual run at or
        /// below `chunk_token_hard_cap`.
        size_t chunk_token_budget = 72;
        size_t chunk_token_hard_cap = 128;

        /// Optional additional output limit validated for this graph. The
        /// runtime always reserves a structural tail margin below the tensor
        /// capacity; nonzero values can lower that safe limit further.
        size_t max_safe_output_samples = 0;

        /// Published 120-frame (3.0 s physical / 2.8 s guarded) graph profile
        /// for tightly bounded voice-agent replies. The caller must supply the
        /// matching graph; unsafe outputs split and retry safely.
        static Config short_turn_3s(bool hw_accel = true);

        /// Selects the matching profile for an official Kokoro graph path.
        /// `kokoro-e2e-realtime.onnx` uses the 3.0 s short-turn default;
        /// all other paths retain the full-graph configuration.
        static Config default_for_model_path(const std::string& model_path,
                                             bool hw_accel = true);

        /// Validated 140-frame (3.5 s physical / 3.3 s guarded) graph profile
        /// for bounded Android voice-agent replies.
        static Config short_turn_3p5s(bool hw_accel = true);
    };

    KokoroTts(const std::string& model_path,
              const std::string& voices_dir,
              const std::string& data_dir,
              bool hw_accel = true);
    KokoroTts(const std::string& model_path,
              const std::string& voices_dir,
              const std::string& data_dir,
              const Config& config);
    ~KokoroTts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    /// Select an explicit voice for subsequent synthesis calls. Once set,
    /// language changes no longer replace it with a language-default voice.
    void set_voice(const std::string& name);
    /// Set the model duration scalar used by the next synthesis. Values match
    /// the OpenAI speech API range: 0.25 (slowest) through 4.0 (fastest).
    void set_speed(float speed);

private:
    enum class ChunkResult { Emitted, RetrySmaller, Cancelled };

    std::vector<float> load_voice_embedding(const std::string& name);
    void auto_switch_voice(const std::string& language);
    bool synthesize_with_retry(const std::string& text,
                               const TTSChunkCallback& on_chunk,
                               bool is_final,
                               size_t depth,
                               size_t& inference_attempts);
    ChunkResult synthesize_chunk(const std::string& text,
                                 const TTSChunkCallback& on_chunk,
                                 bool is_final);

    const OrtApi* api_;
    OrtSession* session_ = nullptr;

    KokoroPhonemizer phonemizer_;
    std::vector<float> voice_embedding_;
    std::string voices_dir_;
    std::string current_lang_;
    bool voice_overridden_ = false;
    float speed_ = 0.85f;
    Config config_;
    std::atomic<bool> cancelled_{false};
};

}  // namespace speech_core
