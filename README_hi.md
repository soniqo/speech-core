# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

**Linux, Windows और Android** के लिए **C++17** में on-device speech infrastructure: voice activity detection, batch और real-time streaming speech-to-text, speaker diarization, text-to-speech और इन सभी को जोड़ने वाली voice-agent pipeline।

यह CPU पर पूरी तरह स्थानीय रूप से चलता है। inference के समय cloud या Python नहीं चाहिए और audio device से बाहर नहीं जाता।

**[📚 पूरा documentation →](https://soniqo.audio/hi/speech-core)** · **[🐧 Linux](https://soniqo.audio/hi/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/hi/getting-started/windows)** · **[⌨️ Linux CLI](docs/cli.md)**

**[🤗 Models](https://huggingface.co/soniqo)** · **[🍎 Apple sibling](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<p align="center"><a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="VoxCPM2 से voice cloning — YouTube पर Speech Studio demo"></a></p>
<p align="center"><em>VoxCPM2 से voice cloning — YouTube पर Speech Studio demo</em></p>

## speech-core क्यों

speech-core एक छोटे, model-agnostic orchestration layer को optional inference backends से अलग रखता है। core turn detection, interruption handling, audio utilities, conversation state और tool calls संभालता है; application models चुनती है।

- **Local-first:** शुद्ध C++17 core, Float audio buffers, network या platform audio dependency नहीं।
- **Live agents के लिए:** VAD-driven turns, eager STT, partial transcripts, barge-in, streaming TTS और tool calling।
- **वास्तविक streaming ASR:** cache-aware RNN-T decoders, end-of-utterance detection, beam search और contextual phrase biasing।
- **Backend का चुनाव:** ONNX Runtime, LiteRT, दोनों, कोई नहीं, या abstract interfaces की अपनी implementation।
- **Portable API:** native C++ और Kotlin/JNI, Swift/FFI, embedded Linux जैसे hosts के लिए C APIs।
- **कई targets पर tested:** Linux, Windows, macOS, Android-oriented arm64 builds, sanitizers और model-backed nightly lanes।

## v0.0.10 की मुख्य बातें

- **Parakeet-EOU 120M:** कम-memory multilingual streaming ASR, end-of-utterance tokens, optional beam search, contextual phrase biasing और over-bias cap।
- **Native Whisper ONNX:** small से large-v3/turbo, language detection या fixed-language prompts, profiling और CPU tuning।
- **अधिक TTS:** Kokoro के साथ VoxCPM/VoxCPM2, CosyVoice3, Chatterbox, Supertonic और Indic-Mio; buffered post-processing और transcript-guided cloning।
- **तेज़ conversations:** छोटे turns के लिए Kokoro optimization, लंबे text का sentence chunking और playback के आसपास continuous pre-speech buffer।
- **On-device LLM tools:** LiteRT-LM से FunctionGemma, Ollama adapter और pipeline tool-call loop।
- **Release-grade Linux CLI:** amd64/arm64 packages, model download helpers, architecture-aware commands और clean-container smoke tests।

## Supported models

| Model | कार्य | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/vad) | Voice activity detection | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/hi/guides/parakeet) | Speech-to-text | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/whisper) | Multilingual speech-to-text | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/nemotron) | Streaming speech-to-text | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/hi/guides/nemotron) | Prompt-conditioned streaming STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/hi/guides/dictate) | Streaming STT + utterance end | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/omnilingual) | Multilingual speech-to-text | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/diarize) | Diarization segmentation | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/embed-speaker) | Speaker embedding | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + voice cloning | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/voxcpm2) | 48 kHz TTS + voice cloning | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/cosyvoice) | 24 kHz conditioned TTS | staged | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/chatterbox) | 24 kHz text-to-speech | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/supertonic) | Text-to-speech | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/indic-mio) | हिन्दी/भारतीय भाषाओं की voice cloning + emotion | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/hi/guides/kokoro) | Text-to-speech | ✓ | ✓ |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/denoise) | Speech enhancement | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/restore) | Denoise + dereverb (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/hi/guides/respond) | Full-duplex speech-to-speech (CUDA) | structural | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/hi/guides/function-calls) | On-device structured tool calls | — | LiteRT-LM |

Maturity, bundle layouts, preprocessing, memory notes और पूरे examples के लिए [docs/models.md](docs/models.md) देखें।

## Platforms और backends

| Backend | Target | Platforms | Runtime setup |
|---|---|---|---|
| केवल core | `speech_core` | Linux, Windows, macOS, Android | कोई नहीं |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | extracted ONNX Runtime को `ORT_DIR` से दें |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android build path | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX CPU, Android NNAPI, Linux पर Qualcomm QNN या application-provided execution-provider hook उपयोग कर सकता है। LiteRT अभी C API से CPU उपयोग करता है।

## Quick start

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Audio buffer transcribe करें; Parakeet v3 भाषा अपने आप पहचानता है:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

किसी भी VAD, STT, LLM और TTS implementation को live pipeline से जोड़ें:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcript, response audio, tool call या error
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI packages

Releases में amd64 और arm64 के लिए `.deb` व `.tar.gz` packages होते हैं। runtime libraries bundled हैं, models नहीं।

```bash
VERSION=0.0.10
ARCH="$(dpkg --print-architecture)"   # amd64 या arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
```

amd64 package में LiteRT VoxCPM2 voice cloning भी है। x86 bundle लगभग 13 GB है:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

सही syntax, model directories, standalone binaries और amd64/arm64 matrix के लिए **[Linux CLI reference](docs/cli.md)** देखें। [`soniqo.audio/cli`](https://soniqo.audio/hi/cli) Apple के बड़े speech-swift CLI का documentation है।

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

Orchestration target किसी concrete model पर निर्भर नहीं है। backend बदलना construction और linking का चुनाव है, pipeline rewrite नहीं।

## Documentation

| विषय | Documentation |
|---|---|
| Overview और model matrix | [soniqo.audio/hi/speech-core](https://soniqo.audio/hi/speech-core) |
| Linux setup | [soniqo.audio/hi/getting-started/linux](https://soniqo.audio/hi/getting-started/linux) |
| Windows setup | [soniqo.audio/hi/getting-started/windows](https://soniqo.audio/hi/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Interfaces और custom backends | [docs/interfaces.md](docs/interfaces.md) |
| Model implementations | [docs/models.md](docs/models.md) |
| Voice pipeline और state machine | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Tool calling | [docs/tools.md](docs/tools.md) |

## Build variants

```bash
# केवल orchestration; ML runtime नहीं
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

Core tests को models नहीं चाहिए। model-directory variables न होने पर backend integrations cleanly skip होती हैं। CI Linux, Windows और macOS cover करता है; scheduled workflows public ONNX और LiteRT bundles test करते हैं।

## संबंधित projects

- [speech-android](https://github.com/soniqo/speech-android) — speech-core पर Kotlin SDK और JNI integration।
- [speech-swift](https://github.com/soniqo/speech-swift) — macOS/iOS के लिए native MLX/CoreML speech stack।
- [Soniqo documentation](https://soniqo.audio/hi) — guides, architecture, benchmarks और model pages।

## Contributing

Issues और pull requests स्वागत योग्य हैं। `main` से branch बनाएं, affected configurations build करें, `ctest` चलाएं और focused PR खोलें।

## License

Apache 2.0 — [LICENSE](LICENSE) देखें।
