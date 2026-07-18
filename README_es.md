# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Infraestructura de voz local en **C++17** para **Linux, Windows y Android**: detección de actividad de voz, transcripción por lotes y en tiempo real, diarización de hablantes, síntesis de voz y el pipeline de agentes de voz que conecta todo.

Se ejecuta localmente en CPU. Sin nube, sin Python durante la inferencia y sin enviar audio fuera del dispositivo.

**[📚 Documentación completa →](https://soniqo.audio/es/speech-core)** · **[🐧 Linux](https://soniqo.audio/es/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/es/getting-started/windows)** · **[⌨️ CLI de escritorio](docs/cli.md)** · **[🔊 Audio HTTP](docs/http-server.md)**

**[🤗 Modelos](https://huggingface.co/soniqo)** · **[🍎 Proyecto hermano para Apple](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demostración

<p align="center"><a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="Clonación de voz con VoxCPM2 — demostración de Speech Studio en YouTube"></a></p>
<p align="center"><em>Clonación de voz con VoxCPM2 — demostración de Speech Studio en YouTube</em></p>

## Por qué speech-core

speech-core separa una pequeña capa de orquestación independiente del modelo de los backends de inferencia opcionales. El núcleo gestiona turnos, interrupciones, utilidades de audio, estado de conversación y llamadas a herramientas; la aplicación elige los modelos.

- **Local por diseño:** núcleo C++17 puro, buffers de audio Float y ninguna dependencia de red o audio de plataforma.
- **Creado para agentes en vivo:** turnos por VAD, STT anticipado, transcripciones parciales, barge-in, TTS en streaming y herramientas.
- **ASR realmente streaming:** RNN-T con caché, detección de fin de enunciado, beam search y sesgo de frases contextuales.
- **Backend a elección:** ONNX Runtime, LiteRT, ambos, ninguno o implementaciones propias de las interfaces abstractas.
- **API portable:** C++ nativo y APIs C para Kotlin/JNI, Swift/FFI, Linux embebido y otros hosts.
- **Pruebas multiplataforma:** Linux, Windows, macOS, builds arm64 orientadas a Android, sanitizers y pruebas nocturnas con modelos.

## Novedades de v0.0.11

- **TTS local compatible con OpenAI:** `speech-server` expone `POST /v1/audio/speech` con alias de modelos OpenAI, voces nativas y genéricas, idioma y velocidad, salida WAV/PCM y autenticación Bearer opcional.
- **Paquete para Windows:** ZIP x64 autocontenido con el servidor, herramientas CLI ONNX, `speech.dll`, ONNX Runtime y descargador de modelos PowerShell; CI extrae y prueba el paquete.
- **Paridad de DeepFilterNet3:** escalado STFT compatible con libdf, normalización ERB/compleja, filtrado profundo, overlap-add y compensación de 480 muestras restauran el DSP de referencia.
- **Pocket TTS en streaming:** el backend ONNX emite frames fijos de 80 ms, usa caché acotada y ofrece validación round-trip opcional con el modelo.
- **Contexto correcto de Silero v5:** cada inferencia ONNX recibe ahora las 64 muestras de contexto izquierdo requeridas.

## Modelos compatibles

| Modelo | Tarea | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/vad) | Detección de actividad de voz | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/es/guides/parakeet) | Voz a texto | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/whisper) | Voz a texto multilingüe | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/nemotron) | Voz a texto en streaming | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/es/guides/nemotron) | STT streaming condicionado por prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/es/guides/dictate) | STT streaming + fin de enunciado | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/omnilingual) | Voz a texto multilingüe | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/diarize) | Segmentación para diarización | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/embed-speaker) | Embedding de hablante | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 kHz + clonación | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/voxcpm2) | TTS 48 kHz + clonación | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/cosyvoice) | TTS condicionado 24 kHz | en preparación | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/chatterbox) | Texto a voz 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/supertonic) | Texto a voz | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/indic-mio) | Clonación hindi/índica + emoción | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/es/guides/kokoro) | Texto a voz | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | TTS en streaming (voz Alba fija) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/denoise) | Mejora de voz | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/restore) | Reducción de ruido y reverberación (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/es/guides/respond) | Voz a voz full-duplex (CUDA) | estructura | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/es/guides/function-calls) | Herramientas estructuradas locales | — | LiteRT-LM |

Consulta [docs/models.md](docs/models.md) para conocer la madurez, estructura de bundles, preprocesado, memoria y ejemplos completos.

## Plataformas y backends

| Backend | Target | Plataformas | Configuración |
|---|---|---|---|
| Solo núcleo | `speech_core` | Linux, Windows, macOS, Android | ninguna |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | release de ONNX Runtime extraída mediante `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, ruta de build Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX puede usar CPU, NNAPI en Android, Qualcomm QNN en Linux o un hook de execution provider de la aplicación. LiteRT usa actualmente CPU mediante su API C.

## Inicio rápido

Compila el núcleo y el backend LiteRT:

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Transcribe un buffer; Parakeet v3 detecta el idioma automáticamente:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Conecta cualquier implementación de VAD, STT, LLM y TTS al pipeline en vivo:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcripción, audio de respuesta, herramienta o error
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

Enlaza solo los targets que uses:

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Paquetes CLI para Linux

Cada release incluye paquetes `.deb` y `.tar.gz` para amd64 y arm64. Incluyen las bibliotecas de runtime, pero no los modelos.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 o arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

El paquete amd64 también incluye clonación VoxCPM2 con LiteRT. Su bundle x86 ocupa unos 13 GB y se descarga explícitamente:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

La **[referencia CLI de Linux](docs/cli.md)** contiene la sintaxis exacta, directorios, binarios y matriz amd64/arm64. [`soniqo.audio/cli`](https://soniqo.audio/es/cli) documenta el CLI más amplio de speech-swift para Apple.

## Arquitectura

```text
audio / eventos de la aplicación
              │
              ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · detección de turnos  │
│ interrupción · herramientas · audio  │
│ APIs abstractas VAD/STT/LLM/TTS      │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ modelos ref.    │ │ modelos ref.        │
      └─────────────────┘ └─────────────────────┘
```

El target de orquestación nunca depende de un modelo concreto. Cambiar de backend es una decisión de construcción y enlace, no una reescritura del pipeline.

## Documentación

| Tema | Documentación |
|---|---|
| Visión general y matriz de modelos | [soniqo.audio/es/speech-core](https://soniqo.audio/es/speech-core) |
| Configuración Linux | [soniqo.audio/es/getting-started/linux](https://soniqo.audio/es/getting-started/linux) |
| Configuración Windows | [soniqo.audio/es/getting-started/windows](https://soniqo.audio/es/getting-started/windows) |
| CLI de Linux | [docs/cli.md](docs/cli.md) |
| Interfaces y backends propios | [docs/interfaces.md](docs/interfaces.md) |
| Implementaciones de modelos | [docs/models.md](docs/models.md) |
| Pipeline y máquina de estados | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Llamadas a herramientas | [docs/tools.md](docs/tools.md) |

## Variantes de compilación

```bash
# Solo orquestación, sin runtime ML
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Modelos ONNX
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# Modelos LiteRT
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Pruebas

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Las pruebas del núcleo no requieren modelos. Las integraciones de backend se omiten correctamente sin sus variables de directorio. CI cubre Linux, Windows y macOS; los workflows programados validan los bundles públicos ONNX y LiteRT.

## Proyectos relacionados

- [speech-android](https://github.com/soniqo/speech-android) — SDK Kotlin e integración JNI sobre speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — stack MLX/CoreML nativo para macOS e iOS.
- [Documentación de Soniqo](https://soniqo.audio/es) — guías, arquitectura, benchmarks y modelos.

## Contribuir

Se aceptan issues y pull requests. Crea una rama desde `main`, compila las configuraciones afectadas, ejecuta `ctest` y abre un PR enfocado.

## Licencia

Apache 2.0 — consulta [LICENSE](LICENSE).
