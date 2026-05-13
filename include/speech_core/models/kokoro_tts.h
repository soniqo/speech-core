#pragma once

#include "speech_core/interfaces.h"
#include "speech_core/models/kokoro_phonemizer.h"

#include <onnxruntime_c_api.h>
#include <string>
#include <vector>

namespace speech_core {

/// Kokoro 82M — lightweight text-to-speech via ONNX Runtime.
/// Non-autoregressive, single-pass synthesis.
/// Output: 24 kHz PCM Float32.
class KokoroTts : public TTSInterface {
public:
    KokoroTts(const std::string& model_path,
              const std::string& voices_dir,
              const std::string& data_dir,
              bool hw_accel = true);
    ~KokoroTts() override;

    // --- TTSInterface ---
    void synthesize(const std::string& text,
                    const std::string& language,
                    TTSChunkCallback on_chunk) override;
    int output_sample_rate() const override { return 24000; }
    void cancel() override;

    void set_voice(const std::string& name);

private:
    std::vector<float> load_voice_embedding(const std::string& name);
    void auto_switch_voice(const std::string& language);

    const OrtApi* api_;
    OrtSession* session_ = nullptr;

    KokoroPhonemizer phonemizer_;
    std::vector<float> voice_embedding_;
    std::string voices_dir_;
    std::string current_lang_;
    bool cancelled_ = false;
};

}  // namespace speech_core
