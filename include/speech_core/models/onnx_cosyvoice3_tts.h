#pragma once

#include "speech_core/interfaces.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace speech_core {

/// CosyVoice3 0.5B TTS via ONNX Runtime.
///
/// This wrapper owns the tested deployment graphs exported by
/// soniqo/CosyVoice3-0.5B-ONNX:
///   llm_prefill -> llm_step xN -> flow_frontend -> flow estimator -> hift.
///
/// CosyVoice3 zero-shot cloning needs prompt-derived conditioning tensors
/// (prompt text ids, prompt speech tokens, prompt mel, speaker embedding).
/// Those are intentionally explicit so cloud callers can compute them once at
/// voice-create time and reuse them for many synth requests.
class OnnxCosyVoice3Tts : public TTSInterface {
public:
    struct Conditioning {
        std::vector<int64_t> prompt_text_ids;
        std::vector<int64_t> llm_prompt_speech_tokens;
        std::vector<int64_t> flow_prompt_speech_tokens;
        std::vector<float>   prompt_speech_feat;  // [frames, 80], row-major
        int64_t              prompt_speech_feat_frames = 0;
        std::vector<float>   embedding;           // [192]
    };

    explicit OnnxCosyVoice3Tts(const std::string& bundle_dir,
                               bool hw_accel = true);
    ~OnnxCosyVoice3Tts() override;

    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    void set_conditioning(Conditioning conditioning);
    void clear_conditioning();
    bool has_conditioning() const;

    void set_seed(uint32_t seed) { seed_ = seed; }
    uint32_t seed_used() const { return seed_used_; }

    void set_max_steps(int max_steps) { max_steps_ = max_steps; }
    void set_flow_steps(int steps);
    int flow_steps() const { return flow_steps_; }
    void set_cfg_rate(float cfg_rate);
    float cfg_rate() const { return cfg_rate_; }
    int tokens_generated() const { return tokens_generated_; }
    bool stopped_on_stop_token() const { return stopped_on_stop_token_; }

    void set_instruction(std::string instruction) { instruction_ = std::move(instruction); }

    int64_t prefill_ms() const { return prefill_ms_; }
    int64_t ar_ms() const { return ar_ms_; }
    int64_t audio_decode_ms() const { return audio_decode_ms_; }
    int64_t flow_frontend_ms() const { return flow_frontend_ms_; }
    int64_t flow_estimator_ms() const { return flow_estimator_ms_; }
    int64_t hift_ms() const { return hift_ms_; }

    static std::string helper_prompt_prefix();
    static std::string prompt_text_from_transcript(const std::string& transcript);

    static std::vector<uint8_t> encode_conditioning_blob(const Conditioning& c);
    static Conditioning decode_conditioning_blob(const uint8_t* data,
                                                 size_t size);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::string instruction_;
    uint32_t seed_ = 0;
    uint32_t seed_used_ = 0;
    int max_steps_ = 256;
    int flow_steps_ = 10;
    float cfg_rate_ = 0.7f;
    int tokens_generated_ = 0;
    bool stopped_on_stop_token_ = false;
    int64_t prefill_ms_ = -1;
    int64_t ar_ms_ = -1;
    int64_t audio_decode_ms_ = -1;
    int64_t flow_frontend_ms_ = -1;
    int64_t flow_estimator_ms_ = -1;
    int64_t hift_ms_ = -1;
};

}  // namespace speech_core
