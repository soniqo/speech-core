# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

On-device speech infrastructure in **C++17** for **Linux, Windows, and Android**: voice activity detection, batch and real-time streaming speech-to-text, speaker diarization, text-to-speech, and the voice-agent pipeline that connects them.

Runs locally on CPU. No cloud, no Python at inference, and no audio leaves the machine.

**[📚 Full documentation →](https://soniqo.audio/speech-core)** · **[🐧 Linux](https://soniqo.audio/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/getting-started/windows)** · **[⌨️ Linux CLI](docs/cli.md)**

**[🤗 Models](https://huggingface.co/soniqo)** · **[🍎 Apple sibling](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="Voice cloning with VoxCPM2 — watch the speech-studio demo on YouTube">
  </a>
</p>
<p align="center"><em>Voice cloning with VoxCPM2 — watch the speech-studio demo on YouTube</em></p>

## Why speech-core

speech-core separates a small, model-agnostic orchestration layer from optional inference backends. The core owns turn detection, interruption handling, audio utilities, conversation state, and tool calls; your application chooses the models.

- **Local-first:** pure C++17 core, float audio buffers, no network or platform audio dependency.
- **Built for live agents:** VAD-driven turns, eager STT, partial transcripts, barge-in, streaming TTS, and tool calling.
- **Real streaming ASR:** cache-aware RNN-T decoders, end-of-utterance detection, beam search, and contextual phrase biasing.
- **Backend choice:** enable ONNX Runtime, LiteRT, both, neither, or implement the abstract interfaces yourself.
- **Portable surface:** native C++ API plus C APIs suitable for Kotlin/JNI, Swift/FFI, embedded Linux, and other hosts.
- **Tested across targets:** Linux, Windows, macOS, Android-oriented arm64 builds, sanitizers, and model-backed nightly lanes.

## v0.0.9 highlights

- **Parakeet-EOU 120M:** low-memory multilingual streaming ASR with end-of-utterance tokens, opt-in beam search, contextual phrase biasing, and an over-bias cap.
- **Native Whisper ONNX:** small through large-v3/turbo, language detection or fixed-language prompts, profiling, and CPU tuning controls.
- **Broader TTS:** VoxCPM/VoxCPM2, CosyVoice3, Chatterbox, Supertonic, and Indic-Mio runtimes alongside Kokoro; buffered post-processing and transcript-guided cloning.
- **Faster conversations:** Kokoro short-turn optimizations, sentence chunking for long text, and continuous pre-speech buffering around playback.
- **On-device LLM tools:** FunctionGemma through LiteRT-LM plus the existing Ollama adapter and pipeline tool-call loop.
- **Release-grade Linux CLI:** amd64 and arm64 packages, model download helpers, architecture-aware command availability, and clean-container smoke tests.

## Supported models

| Model | Task | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/vad) | Voice activity detection | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/guides/parakeet) | Speech-to-text | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/guides/whisper) | Multilingual speech-to-text | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/nemotron) | Streaming speech-to-text | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/guides/nemotron) | Prompt-conditioned streaming STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/guides/dictate) | Streaming STT + end-of-utterance | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/omnilingual) | Multilingual speech-to-text | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/diarize) | Diarization segmentation | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/embed-speaker) | Speaker embedding | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + voice cloning | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/guides/voxcpm2) | 48 kHz TTS + voice cloning | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/guides/cosyvoice) | 24 kHz conditioned TTS | staged | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/chatterbox) | 24 kHz text-to-speech | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/supertonic) | Text-to-speech | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/guides/indic-mio) | Hindi/Indic voice cloning + emotion | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-ONNX) · [soniqo.audio](https://soniqo.audio/guides/kokoro) | Text-to-speech | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/guides/denoise) | Speech enhancement | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/guides/sidon) | Denoise + dereverb (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/guides/respond) | Full-duplex speech-to-speech (CUDA) | structural | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/guides/functiongemma) | On-device structured tool calls | — | LiteRT-LM |

See [docs/models.md](docs/models.md) for maturity, bundle layouts, preprocessing, memory notes, and complete examples.

## Platforms and backends

| Backend | Target | Platforms | Runtime setup |
|---|---|---|---|
| Core only | `speech_core` | Linux, Windows, macOS, Android | none |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | extracted ONNX Runtime release via `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android build path | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX can use CPU, Android NNAPI, Qualcomm QNN on Linux, or an application-supplied execution-provider hook. LiteRT currently uses CPU through its C API.

## Quick start

Build the core and LiteRT backend:

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Transcribe an audio buffer; Parakeet v3 detects the language automatically:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite",
    "parakeet-decoder-joint.tflite",
    "vocab.json");

auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Connect any implementations of the abstract VAD, STT, LLM, and TTS interfaces to the live pipeline:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcription, response audio, tool call, or error
    });

pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

Link only what your application uses:

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI packages

Releases ship `.deb` and `.tar.gz` packages for amd64 and arm64. The package bundles runtime libraries but not models.

```bash
VERSION=0.0.9
ARCH="$(dpkg --print-architecture)"   # amd64 or arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
```

The amd64 package also includes the LiteRT VoxCPM2 voice-cloning command. Its x86 bundle is about 13 GB and is downloaded explicitly:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

See the **[Linux CLI reference](docs/cli.md)** for exact syntax, model directories, standalone binaries, and the amd64/arm64 command matrix. [`soniqo.audio/cli`](https://soniqo.audio/cli) documents the larger speech-swift CLI for Apple platforms.

## Architecture

```text
application audio / events
            │
            ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · turn detection       │
│ interruption · tools · audio utils   │
│ abstract VAD / STT / LLM / TTS APIs  │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ reference models│ │ reference models   │
      └─────────────────┘ └─────────────────────┘
```

The orchestration target never depends on a concrete model. A backend swap is a construction and link choice, not a pipeline rewrite.

## Documentation

| Topic | Documentation |
|---|---|
| Product overview and model matrix | [soniqo.audio/speech-core](https://soniqo.audio/speech-core) |
| Linux setup | [soniqo.audio/getting-started/linux](https://soniqo.audio/getting-started/linux) |
| Windows setup | [soniqo.audio/getting-started/windows](https://soniqo.audio/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Interfaces and custom backends | [docs/interfaces.md](docs/interfaces.md) |
| Model implementations | [docs/models.md](docs/models.md) |
| Voice pipeline and state machine | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Tool calling | [docs/tools.md](docs/tools.md) |

## Build variants

```bash
# Orchestration only: no ML runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release

# ONNX models
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime

# LiteRT models
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Testing

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Core tests require no model files. Backend integration tests skip cleanly unless their model-directory environment variables are set. CI covers Linux, Windows, and macOS; model-backed scheduled workflows cover the public ONNX and LiteRT bundles.

## Related projects

- [speech-android](https://github.com/soniqo/speech-android) — Kotlin SDK and JNI integration over speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — native MLX/CoreML speech stack for macOS and iOS.
- [Soniqo documentation](https://soniqo.audio) — guides, architecture, benchmarks, and model pages.

## Contributing

Issues and pull requests are welcome. Branch from `main`, build the affected configurations, run `ctest`, and open a focused PR.

## License

Apache 2.0 — see [LICENSE](LICENSE).
