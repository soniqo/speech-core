# Third-party notices

Binary distributions of the `speech` package (the `.deb` / `.tar.gz` packages published
on GitHub Releases) bundle the following third-party runtime libraries. The
speech-core source tree also vendors the source-only components noted below.

## ONNX Runtime

- **Files:** `lib/speech/libonnxruntime.so*`
- **Source:** https://github.com/microsoft/onnxruntime (prebuilt release binaries)
- **License:** MIT — https://github.com/microsoft/onnxruntime/blob/main/LICENSE
- Copyright (c) Microsoft Corporation

## LiteRT (`libLiteRt.so`)

- **Files:** `lib/speech/libLiteRt.so`
- **Source:** extracted from Google's [`ai-edge-litert`](https://pypi.org/project/ai-edge-litert/)
  PyPI package (see `scripts/fetch_litert.sh`); upstream repository
  https://github.com/google-ai-edge/LiteRT
- **License:** Apache License 2.0 —
  https://github.com/google-ai-edge/LiteRT/blob/main/LICENSE
- Copyright Google LLC. This product includes software developed at Google
  as part of the LiteRT / TensorFlow Lite projects. Use of this library does
  not imply endorsement by Google.

The LiteRT shared library statically incorporates the following components,
each under its own permissive license:

| Component | License |
|---|---|
| TensorFlow / TensorFlow Lite | Apache-2.0 |
| XNNPACK | BSD-3-Clause |
| Eigen | MPL-2.0 |
| FlatBuffers | Apache-2.0 |
| ruy | Apache-2.0 |
| farmhash | MIT |
| FP16 / pthreadpool / cpuinfo | BSD-2/3-Clause |
| Abseil | Apache-2.0 |

Refer to the upstream LiteRT repository's `LICENSE` and third-party notices
for the complete texts.

## Vendored headers (source tree)

- **Files:** `third_party/litert/` (~44 LiteRT C API headers)
- **License:** Apache License 2.0 (same as LiteRT above)

## Vendored source (source tree)

### sherpa-onnx SentencePiece tokenizer algorithm

- **Files:** `src/models/pocket/pocket_tts_tokenizer.cpp`
- **Source:** adapted from `sherpa-onnx/csrc/sentence-piece-tokenizer.cc`
  in https://github.com/k2-fsa/sherpa-onnx
- **License:** Apache License 2.0
- Copyright 2026 Xiaomi Corporation.
- **Used by:** the Pocket TTS ONNX backend. No sherpa-onnx runtime code or
  library is linked into speech-core.

### Ooura FFT

- **Files:** `third_party/fftooura/`
- **Source:** local C++ port of Takuya Ooura's `fft4g`, plus a small C++ wrapper.
- **License:** Ooura FFT license — permits use, copy, modification, and distribution
  for any purpose, including commercial use, without fee; modified code should
  refer to the package.
- Copyright Takuya OOURA, 1996-2001.
- **Used by:** `speech_core::audio::fft_real` / `speech_core::audio::ifft_real`

## Models

No machine-learning models are included in any speech-core package. Models
are downloaded separately by the user at runtime (see
`scripts/download_models.sh`, `scripts/download_models_litert.sh`, and
`docs/models.md`) and are governed by their own licenses as published on
their respective HuggingFace repositories.

The optional Pocket TTS bundle is published separately at
https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8 under CC BY 4.0. Its
fixed voice was performed by Alba MacKenna; retain that attribution when the
bundle is redistributed or used in product notices.
