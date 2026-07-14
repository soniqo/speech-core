#pragma once

#include <cstddef>
#include <vector>

namespace speech_core {
namespace audio {

/// Kaldi-compatible log-mel fbank for WeSpeaker, [frames × 80] row-major.
///
/// Shared by the LiteRT and ONNX WeSpeaker wrappers so both runtimes are fed
/// BIT-IDENTICAL features. If each computed its own, a runtime-parity test would
/// be comparing two different frontends and could pass while hiding a real
/// feature bug — or fail for a reason that has nothing to do with the runtime.
///
/// Contract (matches pyannote's compute_fbank, which both graphs were exported
/// against): 80 mel bins, 25 ms frame / 10 ms hop, Hamming window, dither off,
/// per-frame DC removal, n_fft = 512, log(max(power, 1e-10)).
///
/// Note we do NOT scale the waveform by 1<<15 the way pyannote does. Both graphs
/// subtract the per-utterance mean internally, and that scaling only adds a
/// constant offset (2·log(32768)) to every log-mel value — the centering removes
/// it exactly. Verified: cosine 1.000000 against the PyTorch reference.
///
/// `frames` is the number of frames to emit. Input shorter than needed is TILED
/// (not zero-padded) and longer input is truncated — matching the behaviour the
/// LiteRT path has always had, so the swap changes nothing downstream.
std::vector<float> wespeaker_fbank(const float* audio, size_t length, int frames);

}  // namespace audio
}  // namespace speech_core
