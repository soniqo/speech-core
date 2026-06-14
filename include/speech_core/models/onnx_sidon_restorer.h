#pragma once

#include "speech_core/interfaces.h"

#include <onnxruntime_c_api.h>

#include <memory>
#include <string>
#include <vector>

namespace speech_core {

/// Sidon — combined denoise + dereverb speech restoration via ONNX Runtime.
///
/// Two ONNX graphs, orchestrated here, with a C++ DSP front-end between the
/// caller's audio and the first graph:
///
///   audio (16 kHz) ──[SeamlessM4T log-mel front-end, C++ DSP]──►
///       input_features [1, T, 160]
///     ─► predictor  (w2v-BERT 2.0 truncated to 8 layers + merged LoRA)
///        ─► features [1, T, 1024]
///          ─► vocoder (DAC decoder, rates [8,5,4,3,2], ×960 upsample)
///             ─► audio (48 kHz) [1, M]
///
/// The predictor's 8-layer `last_hidden_state` IS the cleansed feature (there
/// is no separate head); the vocoder reconstructs 48 kHz audio from it. See
/// speech-models `models/sidon/export/NOTES.md` for the export rationale and
/// parity numbers (ONNX FP32 is bit-exact vs PyTorch; FP16 near-lossless).
///
/// Backend note: this ships on ONNX Runtime only. The DAC decoder's
/// `ConvTranspose1d` does not legalise to TFLite, so there is no LiteRT vocoder
/// (documented in NOTES). Both graphs run through the shared `OnnxEngine`
/// (CPU, or CUDA/TensorRT on a SPEECH_CORE_WITH_CUDA build, with CPU fallback;
/// NNAPI/QNN on Android).
///
/// Use case: clean a reverberant voice-cloning reference clip before feeding it
/// to a TTS voice-cloner. This is an offline / whole-clip operation, not a
/// streaming one.
///
/// Note on the interface: restoration changes both sample rate (16 kHz → 48 kHz)
/// and length, so it does not fit `EnhancerInterface` (which is fixed-rate,
/// equal-length, in-place). The natural API is `restore()`, which returns a
/// freshly sized 48 kHz buffer. `as_enhancer()` adapts to `EnhancerInterface`
/// for callers that only have the abstract handle, by resampling 48 kHz back
/// down to the input rate and length.
class OnnxSidonRestorer {
public:
    /// @param predictor_path  sidon-predictor.onnx (w2v-BERT predictor).
    /// @param vocoder_path    sidon-vocoder.onnx   (DAC decoder).
    /// @param hw_accel        Route sessions to the hardware EP when available
    ///                        (CUDA/TensorRT on desktop GPU builds, NNAPI/QNN on
    ///                        Android); falls back to CPU automatically.
    OnnxSidonRestorer(const std::string& predictor_path,
                      const std::string& vocoder_path,
                      bool hw_accel = true);
    ~OnnxSidonRestorer();

    OnnxSidonRestorer(const OnnxSidonRestorer&) = delete;
    OnnxSidonRestorer& operator=(const OnnxSidonRestorer&) = delete;

    /// Restore a clip: denoise + dereverb, returning 48 kHz mono audio.
    /// @param audio        Input PCM Float32, mono, in [-1, 1].
    /// @param length       Number of input samples.
    /// @param sample_rate  Input sample rate; resampled to 16 kHz internally if
    ///                     it differs.
    /// @return             Restored audio at output_sample_rate() (48 kHz).
    ///                     Empty if the clip is too short to yield a frame.
    std::vector<float> restore(const float* audio, size_t length,
                               int sample_rate);

    /// Restore and resample back to `sample_rate`, written into `output`
    /// (caller-allocated, `length` samples) — the equal-length, equal-rate
    /// adapter used by the EnhancerInterface bridge. Output is truncated /
    /// zero-padded to exactly `length` samples.
    void restore_in_place(const float* audio, size_t length, int sample_rate,
                          float* output);

    /// Restoration output sample rate (DAC vocoder): 48 kHz.
    int output_sample_rate() const { return 48000; }

    /// The model's required input sample rate (w2v-BERT front-end): 16 kHz.
    int input_sample_rate() const { return 16000; }

    /// Adapt to the abstract `EnhancerInterface` (equal-length, equal-rate,
    /// in-place). Returns a thin owning adapter — see the class note above on
    /// why restoration is not natively an Enhancer. The adapter holds a
    /// reference to this restorer, which must outlive it.
    std::unique_ptr<EnhancerInterface> as_enhancer();

private:
    // Run the predictor graph: input_features[1,T,160] -> features[1,T,1024].
    std::vector<float> run_predictor(const float* features, int frames);
    // Run the vocoder graph: features[1,T,1024] -> audio[1,M].
    std::vector<float> run_vocoder(const float* features, int frames);

    const OrtApi* api_ = nullptr;
    OrtSession*   predictor_session_ = nullptr;
    OrtSession*   vocoder_session_   = nullptr;

    static constexpr int kInputFeatDim  = 160;   // SeamlessM4T stacked log-mel
    static constexpr int kHiddenDim     = 1024;  // w2v-BERT last_hidden_state
};

}  // namespace speech_core
