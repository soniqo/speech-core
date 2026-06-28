# Speech Core

**English** · [简体中文](README.zh-CN.md)

Voice-agent pipeline engine in **C++17** — on-device speech for **Linux, Windows, and Android** (plus Apple via a [Swift sibling](https://github.com/soniqo/speech-swift)).

On-device voice activity detection, speech-to-text (batch **and** real-time streaming), speaker diarization, and text-to-speech. Runs locally on CPU — no cloud, no Python at inference, no data leaves the machine.

**[📖 Docs](docs/)** · **[🤗 Models](https://huggingface.co/soniqo)** · **[🍎 Apple (Swift)](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="Voice cloning with VoxCPM2 — watch the speech-studio demo on YouTube">
  </a>
</p>
<p align="center"><em>Voice cloning with VoxCPM2 — watch the speech-studio demo on YouTube</em></p>

speech-core is a small orchestration core (state machine, turn detection, interruption handling, audio utilities — zero ML deps) plus a set of abstract interfaces. Model inference is **opt-in** through two interchangeable backends you can enable independently:

- **ONNX Runtime** (`SPEECH_CORE_WITH_ONNX`) — Silero VAD, Parakeet STT, Nemotron-3.5 multilingual streaming STT, Kokoro TTS, DeepFilterNet3, Sidon speech restoration, **PersonaPlex 7B full-duplex speech-to-speech** (CUDA target).
- **LiteRT** (`SPEECH_CORE_WITH_LITERT`) — Silero VAD, Parakeet STT, **Nemotron streaming STT**, **Nemotron-3.5 multilingual streaming STT**, Omnilingual STT, Pyannote diarization, WeSpeaker embeddings, VoxCPM2 TTS. Backed by Google's `ai-edge-litert` (`libLiteRt`).

Consumers can enable either, both, or neither — or bring their own implementations of the interfaces (CPU, GPU, CoreML/MLX, a remote API).

## Supported models

| Model | Task | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) | Voice activity detection | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) | Speech-to-text | ✓ | ✓ |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) | Streaming speech-to-text | ✓ | ✓ |
| [Nemotron-3.5 ASR Streaming Multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) | Streaming speech-to-text (multilingual, prompt-conditioned) | ✓ | ✓ |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) | Speech-to-text (multilingual) | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) | Diarization (segmentation) | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) | Speaker embedding | — | ✓ |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-LiteRT) | Text-to-speech (48 kHz, voice cloning) | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-ONNX) | Text-to-speech | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) | Speech enhancement | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) | Speech restoration — denoise + dereverb (16 kHz → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) | Full-duplex speech-to-speech (CUDA) — 4 variants from 7.6 GB → 17 GB | ✓ | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) | On-device LLM — structured function / tool calls | — | ✓ (`LiteRTFunctionGemmaLLM`, opt-in via `SPEECH_CORE_WITH_LITERT_LM`) |

`LiteRTFunctionGemmaLLM` loads the [FunctionGemma 270M .litertlm bundle](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM)
via Google's `liblitert-lm` runtime — a higher-level shared library than the
`libLiteRt` used for the `.tflite` STT/VAD/TTS models. It builds into its own
static library (`speech_core_models_litert_lm`) so Android consumers can link
the LLM without pulling in `libLiteRt`. On macOS, extract the runtime once
with `scripts/fetch_litert_lm.sh` (PyPI wheel). On Android, neither
`libLiteRt.so` nor `liblitert-lm.so` is published upstream — use
`scripts/build_litert_lm_android.sh` to cross-compile `liblitert-lm.so` from
the pinned `google-ai-edge/LiteRT-LM` v0.13.1 source tree (set
`EMULATOR_SAFE=1` for the Apple-Silicon-hosted Android emulator, where HVF
mistraps KleidiAI's SME `rdsvl` instruction). Then build with
`-DSPEECH_CORE_WITH_LITERT_LM=ON -DLITERT_LM_DIR=...`. The Apple CoreML
build of the same model lives in [speech-swift](https://github.com/soniqo/speech-swift);
see [docs/models.md](docs/models.md#llm-backends-llminterface) for the full
LLM backend picture.

Diarization (`DiarizationPipeline`) is pure C++ and composes a segmenter + embedder into speaker-labelled segments — no ML-runtime dependency of its own.

## Platforms & backends

| Backend | Static lib | Runtime dep | Platforms | Setup |
|---|---|---|---|---|
| ONNX | `speech_core_models` | `onnxruntime` | Linux, macOS, Windows, Android | `ORT_DIR` from an ONNX Runtime release |
| LiteRT | `speech_core_models_litert` | `libLiteRt` | Linux x86_64, Windows x86_64, Android, macOS arm64 | `scripts/fetch_litert.sh` (extracts from the `ai-edge-litert` PyPI wheel) |

**Hardware acceleration.** ONNX: CPU, NNAPI on Android, QNN on Qualcomm Linux (drop `libQnnHtp.so` on the lib path). For other accelerators, install a `SessionOptionsHook` on `OnnxEngine` to append your provider before `CreateSession`. LiteRT runs CPU only today; Hexagon / GPU delegates exist in `libLiteRt` but aren't wired through the C API yet.

## Quick start

Build the core + the LiteRT backend (the runtime library is extracted from the `ai-edge-litert` wheel — no TensorFlow build):

```bash
git clone https://github.com/soniqo/speech-core && cd speech-core
scripts/fetch_litert.sh build/litert          # PYTHON=python3.11 if 'python3' is older
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert
cmake --build build
```

Link the targets you need:

```cmake
target_link_libraries(my_app PRIVATE speech_core)                          # orchestration only
target_link_libraries(my_app PRIVATE speech_core speech_core_models)        # + ONNX models
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert) # + LiteRT models
```

**Transcribe an audio buffer:**

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");

auto r = stt.transcribe(audio, n_samples, 16000);   // r.text / r.language / r.confidence
```

**Real-time streaming with partials (CPU, ~RTF 1.0):**

```cpp
#include <speech_core/models/litert_nemotron_streaming_stt.h>

speech_core::LiteRTNemotronStreamingStt stt(
    "nemotron-streaming-encoder.tflite",
    "nemotron-streaming-decoder.tflite",
    "nemotron-streaming-joint.tflite", "vocab.json");

stt.begin_stream(16000);
for (const auto& chunk : mic_chunks) {              // feed ~80 ms windows as they arrive
    auto partial = stt.push_chunk(chunk.data(), chunk.size());
    if (!partial.text.empty()) std::cout << partial.text << std::flush;
}
auto final = stt.end_stream();
```

**Full voice-agent pipeline (VAD → STT → LLM → TTS):**

```cpp
#include <speech_core/pipeline/voice_pipeline.h>

speech_core::AgentConfig cfg;
cfg.mode = speech_core::AgentConfig::Mode::Pipeline;   // or ::TranscribeOnly / ::Echo

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, cfg,
    [](const speech_core::PipelineEvent& ev) { /* transcripts, audio out, errors */ });

pipeline.start();
pipeline.push_audio(mic_samples, count);               // call from your audio thread
```

`VoicePipeline` is the real-time voice-agent state machine — VAD-driven turn detection, interruption handling, eager STT, conversation tracking, tool calling. It owns no audio I/O or network: the platform feeds audio in and receives events via the callback. Pass `Mode::TranscribeOnly` (and `llm = nullptr`) for a pure transcription pipeline.

## Code examples

### Voice activity detection

```cpp
#include <speech_core/models/litert_silero_vad.h>
speech_core::LiteRTSileroVad vad("silero-vad.tflite");
float p = vad.process_chunk(samples_512, 512);   // speech probability in [0, 1]
```

Feed the probability stream to `StreamingVAD` (`speech_core/vad/streaming_vad.h`) for hysteresis-gated `SpeechStarted` / `SpeechEnded` events.

### Speaker diarization

```cpp
#include <speech_core/models/litert_pyannote_segmentation.h>
#include <speech_core/models/litert_wespeaker_embedding.h>
#include <speech_core/diarization/diarization_pipeline.h>

speech_core::LiteRTPyannoteSegmentation seg("pyannote-segmentation.tflite");
speech_core::LiteRTWeSpeakerEmbedding   emb("wespeaker-resnet34.tflite");
speech_core::DiarizationPipeline        diar(seg, emb);

auto segments = diar.diarize(audio, n_samples, 16000, speech_core::DiarizerConfig{});
for (const auto& s : segments)
    printf("speaker %d: %.2fs - %.2fs\n", s.speaker, s.start, s.end);
```

### Text-to-speech

```cpp
#include <speech_core/models/litert_voxcpm2_tts.h>
speech_core::LiteRTVoxCPM2Tts tts(
    "voxcpm2-text-prefill.tflite", "voxcpm2-token-step.tflite",
    "voxcpm2-audio-encoder.tflite", "voxcpm2-audio-decoder.tflite", "tokenizer.json");

tts.synthesize("Hello world", "en", [](const float* samples, size_t len, bool is_final) {
    // 48 kHz Float32 PCM, streamed in chunks
});
```

For offline post-processing (for example spectral de-essing), use buffered
delivery. `synthesize()` remains the legacy streaming call; buffered mode
accumulates all PCM for that one submitted text input, applies the requested
offline processing chain, then calls the callback once with `is_final=true`.

```cpp
speech_core::TtsSynthesisOptions opts;
opts.mode = speech_core::TtsSynthesisMode::Buffered;
opts.postprocess_flags = speech_core::kTtsPostProcessDeEsser;

tts.synthesize_with_options("Hello world", "en", opts,
    [](const float* samples, size_t len, bool is_final) {
        // Full post-processed utterance for this text input.
    });
```

### Speech restoration (Sidon)

Combined denoise + dereverb. A w2v-BERT 2.0 predictor (with a C++ SeamlessM4T
log-mel front-end) feeds a DAC vocoder; input is resampled to 16 kHz and the
output is 48 kHz mono. Typical use: clean a reverberant voice-cloning reference
before handing it to a TTS voice-cloner. Offline / whole-clip, ONNX only.

```cpp
#include <speech_core/models/onnx_sidon_restorer.h>
speech_core::OnnxSidonRestorer rest("sidon-predictor.onnx", "sidon-vocoder.onnx");

// Any input rate (resampled to 16 kHz internally) -> 48 kHz mono.
std::vector<float> clean = rest.restore(ref.data(), ref.size(), ref_rate);
```

CLI: `speech_sidon_restore <bundle_dir> <in.wav> <out.wav>` (built with
`SPEECH_CORE_WITH_ONNX=ON`; reads any-rate 16-bit PCM WAV, writes 48 kHz).

Each interface and model is documented in **[docs/interfaces.md](docs/interfaces.md)** and **[docs/models.md](docs/models.md)** (download URLs, sizes, preprocessing).

## Architecture

```
┌──────────────────────────────────────────────┐
│            speech_core (always built)         │
│                                              │
│  VoicePipeline / TurnDetector / SpeechQueue  │  orchestration
│  StreamingVAD / AudioBuffer / Resampler      │
│  DiarizationPipeline                         │
│                                              │
│  STT / TTS / VAD / Enhancer / AEC / LLM      │  abstract interfaces
│  Segmentation / Embedding / Diarizer         │
└──────────────────────────────────────────────┘
              ▲                       ▲
              │ implements (optional) │
┌─────────────┴──────────┐  ┌─────────┴──────────────┐
│ speech_core_models     │  │ speech_core_models_litert │
│ (SPEECH_CORE_WITH_ONNX)│  │ (SPEECH_CORE_WITH_LITERT) │
│  ONNX Runtime          │  │  libLiteRt                │
└────────────────────────┘  └───────────────────────────┘
```

The orchestration core depends only on the interfaces — never on a concrete model — so a backend swap is a link-time choice, not a rewrite. Design principles: pure C++17 core, no platform APIs in the core, no network I/O, no audio I/O (operates on float buffers), callback-driven.

Reference: **[interfaces](docs/interfaces.md)** · **[models](docs/models.md)** · **[pipeline / state machine](docs/pipeline.md)** · **[C API (FFI)](docs/c-api.md)** · **[tool calling](docs/tools.md)**

## Build

```bash
# Orchestration only (no ML deps)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# + ONNX backend
cmake -B build -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime && cmake --build build

# + LiteRT backend
scripts/fetch_litert.sh build/litert
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert && cmake --build build
```

LiteRT headers are vendored in `third_party/litert/` (no setup). `LITERT_DIR` points at the directory holding `libLiteRt.{so,dylib,dll}` (Windows also needs `LiteRt.lib`). Add `-DSPEECH_CORE_BUILD_EXAMPLES=ON` for the Linux CLI demos (`speech_transcribe`, `speech_synthesize`, …) — see [`examples/linux`](examples/linux). A voice-cloning CLI (`speech_voxcpm2_clone`) is built automatically whenever `SPEECH_CORE_WITH_LITERT=ON` — see [`examples/litert`](examples/litert).

**On-device model download (optional).** Add `-DSPEECH_CORE_WITH_HF_DOWNLOAD=ON` to fetch model bundles from Hugging Face on first use instead of provisioning them by hand. It links libcurl (`find_package(CURL)` — system libcurl on Linux/macOS, vcpkg on Windows) and adds `sc_voxcpm2_create_from_pretrained("soniqo/VoxCPM2-LiteRT", …)` to the [VoxCPM2 C ABI](include/speech_core/voxcpm2_c.h): a resumable, retrying download (HTTP Range, atomic rename) that tolerates network interruptions and caches under the OS cache dir (`SPEECH_CORE_CACHE_DIR` to override; `HF_ENDPOINT` for a mirror). Off by default so embedded/offline builds carry no HTTP/TLS dependency. The `hf_fetch` debug CLI exercises it directly.

### Install on Linux (prebuilt `speech` package)

Each release ships `.deb` and `.tar.gz` packages with the CLI tools
(`speech_transcribe`, `speech_synthesize`, `speech_phonemize`, `speech_demo`,
and on amd64 the `speech_voxcpm2_clone` voice-cloning CLI), with
`libonnxruntime.so` / `libLiteRt.so` bundled under `/usr/lib/speech/` — no
extra runtime setup:

```bash
# amd64 (arm64: speech_VERSION_arm64.deb — transcribe/synthesize/phonemize only)
curl -LO https://github.com/soniqo/speech-core/releases/latest/download/speech_VERSION_amd64.deb
sudo apt install ./speech_VERSION_amd64.deb

speech download-models           # ONNX set → ~/.cache/speech-core/models (~1.2 GB)
speech transcribe input.wav      # same verb-style CLI as speech-swift's `brew install speech`
speech speak "Hello world"
```

The `speech` command dispatches to standalone `speech_<command>` binaries
(`speech_transcribe`, `speech_synthesize`, …) which expose extra options.

Models are never inside the package. The tools look in `$SPEECH_MODEL_DIR`
(LiteRT: `$SPEECH_LITERT_MODEL_DIR`), falling back to the per-user cache dir
that `speech_download_models` / `speech_download_models_litert` populate; an
explicit model-dir argument always wins. Ubuntu 22.04+ / Debian 12+
(glibc ≥ 2.35).

## Testing & CI

```bash
cd build && ctest --output-on-failure        # core unit tests (no models needed)
```

The orchestration + diarization unit tests need no model files. Integration tests load real `.tflite` / `.onnx` artifacts and **skip cleanly** when `SPEECH_LITERT_MODEL_DIR` / `SPEECH_MODEL_DIR` are unset:

```bash
scripts/fetch_litert.sh build/litert
scripts/download_models_litert.sh            # public soniqo/* models, no token
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert && cmake --build build
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

CI builds + tests across **Linux, Windows, and macOS** (LiteRT on Linux + Windows, ONNX on Linux), plus an **aarch64** cross-compile; a **nightly** lane runs the model integration tests against the public model files.

## Contributing

PRs welcome — model integrations, backends, docs, fixes. Branch off `main`, build + `ctest`, open a PR. No marketing copy in commits or PRs.

## License

Apache 2.0 — see [LICENSE](LICENSE).
