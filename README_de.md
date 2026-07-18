# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Lokale Sprachinfrastruktur in **C++17** für **Linux, Windows und Android**: Spracherkennung, Voice Activity Detection, Echtzeit-Streaming, Sprechertrennung, Sprachsynthese und die Voice-Agent-Pipeline, die alles verbindet.

Läuft lokal auf der CPU. Keine Cloud, kein Python bei der Inferenz und keine Audiodaten verlassen das Gerät.

**[📚 Vollständige Dokumentation →](https://soniqo.audio/de/speech-core)** · **[🐧 Linux](https://soniqo.audio/de/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/de/getting-started/windows)** · **[⌨️ Desktop-CLI](docs/cli.md)** · **[🔊 HTTP-Audio](docs/http-server.md)**

**[🤗 Modelle](https://huggingface.co/soniqo)** · **[🍎 Apple-Schwesterprojekt](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<table align="center">
  <tr>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/watch?v=7L7_Uvvxtv0"><img src="https://img.youtube.com/vi/7L7_Uvvxtv0/maxresdefault.jpg" alt="Ein komplett offline laufender Sprachagent in 1,2 GB auf Android — Demo auf YouTube ansehen"></a>
      <br><em>Ein vollständig offline laufender Sprachagent in 1,2 GB auf Android — die speech-android control-demo</em>
    </td>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" alt="Stimmklonen mit VoxCPM2 — Speech-Studio-Demo auf YouTube"></a>
      <br><em>Stimmklonen mit VoxCPM2 — die Speech-Studio-Demo</em>
    </td>
  </tr>
</table>

## Warum speech-core

speech-core trennt eine kleine, modellunabhängige Orchestrierungsschicht von optionalen Inferenz-Backends. Der Kern verwaltet Sprecherwechsel, Unterbrechungen, Audio-Hilfen, Gesprächszustand und Tool-Aufrufe; die Anwendung wählt die Modelle.

- **Local first:** reiner C++17-Kern, Float-Audiopuffer, keine Netzwerk- oder Plattform-Audio-Abhängigkeit.
- **Für Live-Agenten:** VAD-gesteuerte Turns, frühe STT, Teilergebnisse, Barge-in, Streaming-TTS und Tool-Aufrufe.
- **Echtes Streaming-ASR:** cachefähige RNN-T-Decoder, Äußerungsende, Beam Search und kontextuelle Phrasenverstärkung.
- **Freie Backend-Wahl:** ONNX Runtime, LiteRT, beide, keines oder eigene Implementierungen der abstrakten Schnittstellen.
- **Portable API:** natives C++ und C-APIs für Kotlin/JNI, Swift/FFI, Embedded Linux und weitere Hosts.
- **Breit getestet:** Linux, Windows, macOS, Android-orientierte arm64-Builds, Sanitizer und modellgestützte Nightly-Lanes.

## Highlights in v0.0.11

- **OpenAI-kompatibles lokales TTS:** `speech-server` stellt `POST /v1/audio/speech` mit OpenAI-Modellaliasen, nativen und generischen Stimmen, Sprach- und Temporegelung, WAV/PCM sowie optionaler Bearer-Authentifizierung bereit.
- **Windows-Paket:** ein eigenständiges x64-ZIP enthält Server, ONNX-CLI-Tools, `speech.dll`, ONNX Runtime und einen PowerShell-Modell-Downloader; CI entpackt und testet das Paket.
- **DeepFilterNet3-Parität:** libdf-kompatible STFT-Skalierung, ERB-/Komplex-Normalisierung, Deep Filtering, Overlap-Add und 480-Sample-Verzögerungskompensation stellen das Referenz-DSP wieder her.
- **Streaming Pocket TTS:** das ONNX-Backend liefert feste 80-ms-Frames, einen begrenzten Decoder-Cache und eine optionale modellgestützte Roundtrip-Prüfung.
- **Korrekter Silero-v5-Kontext:** jede ONNX-Inferenz erhält nun den erforderlichen linken Kontext von 64 Samples.

## Unterstützte Modelle

| Modell | Aufgabe | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/vad) | Voice Activity Detection | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/de/guides/parakeet) | Speech-to-Text | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/whisper) | Mehrsprachiges Speech-to-Text | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/nemotron) | Streaming-STT | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/de/guides/nemotron) | Prompt-gesteuertes Streaming-STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/de/guides/dictate) | Streaming-STT + Äußerungsende | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/omnilingual) | Mehrsprachiges STT | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/diarize) | Diarisierungssegmentierung | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/embed-speaker) | Sprecher-Embedding | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16-kHz-TTS + Stimmklonen | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/voxcpm2) | 48-kHz-TTS + Stimmklonen | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/cosyvoice) | Konditioniertes 24-kHz-TTS | gestuft | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/chatterbox) | 24-kHz-TTS | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/supertonic) | Text-to-Speech | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/indic-mio) | Hindi/indisches Stimmklonen + Emotion | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/de/guides/kokoro) | Text-to-Speech | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | Streaming-TTS (feste Alba-Stimme) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/denoise) | Sprachverbesserung | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/restore) | Entrauschen + Enthallen (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/de/guides/respond) | Full-Duplex Speech-to-Speech (CUDA) | strukturell | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/de/guides/function-calls) | Strukturierte lokale Tool-Aufrufe | — | LiteRT-LM |

Reifegrad, Bundle-Aufbau, Vorverarbeitung, Speicherhinweise und vollständige Beispiele stehen in [docs/models.md](docs/models.md).

## Plattformen und Backends

| Backend | Target | Plattformen | Laufzeit-Setup |
|---|---|---|---|
| Nur Kern | `speech_core` | Linux, Windows, macOS, Android | keines |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | entpacktes ONNX Runtime über `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android-Buildpfad | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX unterstützt CPU, Android NNAPI, Qualcomm QNN unter Linux oder einen von der App bereitgestellten Execution-Provider-Hook. LiteRT nutzt derzeit die CPU über seine C-API.

## Schnellstart

Kern und LiteRT-Backend bauen:

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Audiopuffer transkribieren; Parakeet v3 erkennt die Sprache automatisch:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Beliebige VAD-, STT-, LLM- und TTS-Implementierungen mit der Live-Pipeline verbinden:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // Transkript, Antwortaudio, Tool-Aufruf oder Fehler
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

Nur benötigte Targets linken:

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux-CLI-Pakete

Releases enthalten `.deb`- und `.tar.gz`-Pakete für amd64 und arm64. Laufzeitbibliotheken sind enthalten, Modelle nicht.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 oder arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

Das amd64-Paket enthält außerdem VoxCPM2-Stimmklonen mit LiteRT. Das x86-Bundle ist etwa 13 GB groß und wird explizit geladen:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

Exakte Syntax, Modellverzeichnisse, Einzelprogramme und die amd64/arm64-Matrix stehen in der **[Linux-CLI-Referenz](docs/cli.md)**. [`soniqo.audio/cli`](https://soniqo.audio/de/cli) dokumentiert die umfangreichere speech-swift-CLI für Apple.

## Architektur

```text
Anwendungs-Audio / Ereignisse
              │
              ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · Turn-Erkennung       │
│ Unterbrechung · Tools · Audio-Hilfen │
│ abstrakte VAD/STT/LLM/TTS-APIs       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ Referenzmodelle │ │ Referenzmodelle     │
      └─────────────────┘ └─────────────────────┘
```

Das Orchestrierungs-Target hängt nie von einem konkreten Modell ab. Ein Backend-Wechsel ist eine Konstruktions- und Linkentscheidung, kein Pipeline-Neubau.

## Dokumentation

| Thema | Dokumentation |
|---|---|
| Überblick und Modellmatrix | [soniqo.audio/de/speech-core](https://soniqo.audio/de/speech-core) |
| Linux-Einrichtung | [soniqo.audio/de/getting-started/linux](https://soniqo.audio/de/getting-started/linux) |
| Windows-Einrichtung | [soniqo.audio/de/getting-started/windows](https://soniqo.audio/de/getting-started/windows) |
| Linux-CLI | [docs/cli.md](docs/cli.md) |
| Schnittstellen und eigene Backends | [docs/interfaces.md](docs/interfaces.md) |
| Modellimplementierungen | [docs/models.md](docs/models.md) |
| Voice-Pipeline und Zustandsautomat | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Tool-Aufrufe | [docs/tools.md](docs/tools.md) |

## Build-Varianten

```bash
# Nur Orchestrierung, keine ML-Laufzeit
cmake -B build -DCMAKE_BUILD_TYPE=Release
# ONNX-Modelle
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# LiteRT-Modelle
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Tests

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Kerntests benötigen keine Modelle. Backend-Integrationstests werden ohne die jeweiligen Verzeichnisvariablen sauber übersprungen. CI deckt Linux, Windows und macOS ab; geplante Modell-Workflows prüfen die öffentlichen ONNX- und LiteRT-Bundles.

## Verwandte Projekte

- [speech-android](https://github.com/soniqo/speech-android) — Kotlin-SDK und JNI-Integration über speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — nativer MLX/CoreML-Sprachstack für macOS und iOS.
- [Soniqo-Dokumentation](https://soniqo.audio/de) — Leitfäden, Architektur, Benchmarks und Modellseiten.

## Mitwirken

Issues und Pull Requests sind willkommen. Von `main` abzweigen, betroffene Konfigurationen bauen, `ctest` ausführen und einen fokussierten PR öffnen.

## Lizenz

Apache 2.0 — siehe [LICENSE](LICENSE).
