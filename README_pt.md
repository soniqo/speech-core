# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Infraestrutura de voz no dispositivo em **C++17** para **Linux, Windows e Android**: detecção de atividade de voz, transcrição em lote e streaming em tempo real, diarização, síntese de voz e o pipeline de agentes de voz que conecta tudo.

Executa localmente na CPU. Sem nuvem, sem Python na inferência e sem áudio saindo do dispositivo.

**[📚 Documentação completa →](https://soniqo.audio/pt/speech-core)** · **[🐧 Linux](https://soniqo.audio/pt/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/pt/getting-started/windows)** · **[⌨️ CLI desktop](docs/cli.md)** · **[🔊 HTTP TTS](docs/http-server.md)**

**[🤗 Modelos](https://huggingface.co/soniqo)** · **[🍎 Projeto irmão para Apple](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demonstração

<p align="center"><a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="Clonagem de voz com VoxCPM2 — demonstração do Speech Studio no YouTube"></a></p>
<p align="center"><em>Clonagem de voz com VoxCPM2 — demonstração do Speech Studio no YouTube</em></p>

## Por que speech-core

speech-core separa uma pequena camada de orquestração independente de modelo dos backends de inferência opcionais. O núcleo gerencia turnos, interrupções, utilitários de áudio, estado da conversa e chamadas de ferramentas; o aplicativo escolhe os modelos.

- **Local primeiro:** núcleo C++17 puro, buffers de áudio Float, sem dependência de rede ou áudio da plataforma.
- **Feito para agentes ao vivo:** turnos por VAD, STT antecipado, resultados parciais, barge-in, TTS streaming e ferramentas.
- **ASR streaming real:** RNN-T com cache, fim de enunciado, beam search e viés de frases contextuais.
- **Backend à escolha:** ONNX Runtime, LiteRT, ambos, nenhum ou implementações próprias das interfaces abstratas.
- **API portátil:** C++ nativo e APIs C para Kotlin/JNI, Swift/FFI, Linux embarcado e outros hosts.
- **Testes amplos:** Linux, Windows, macOS, builds arm64 voltadas ao Android, sanitizers e testes noturnos com modelos.

## Destaques da v0.0.11

- **TTS local compatível com OpenAI:** `speech-server` expõe `POST /v1/audio/speech` com aliases OpenAI, vozes nativas e genéricas, idioma e velocidade, saída WAV/PCM e autenticação Bearer opcional.
- **Pacote Windows:** ZIP x64 autocontido com servidor, ferramentas ONNX CLI, `speech.dll`, ONNX Runtime e downloader PowerShell; o CI extrai e testa o pacote.
- **Paridade DeepFilterNet3:** escala STFT compatível com libdf, normalização ERB/complexa, deep filtering, overlap-add e compensação de 480 amostras restauram o DSP de referência.
- **Pocket TTS em streaming:** o backend ONNX emite frames fixos de 80 ms, usa cache limitado e oferece validação round-trip opcional com o modelo.
- **Contexto correto do Silero v5:** cada inferência ONNX recebe agora as 64 amostras de contexto à esquerda exigidas.

## Modelos compatíveis

| Modelo | Tarefa | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/vad) | Detecção de atividade de voz | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/pt/guides/parakeet) | Voz para texto | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/whisper) | Voz para texto multilíngue | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/nemotron) | STT streaming | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/pt/guides/nemotron) | STT streaming condicionado por prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/pt/guides/dictate) | STT streaming + fim de enunciado | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/omnilingual) | STT multilíngue | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/diarize) | Segmentação para diarização | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/embed-speaker) | Embedding de locutor | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 kHz + clonagem | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/voxcpm2) | TTS 48 kHz + clonagem | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/cosyvoice) | TTS condicionado 24 kHz | em preparação | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/chatterbox) | Texto para voz 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/supertonic) | Texto para voz | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/indic-mio) | Clonagem hindi/índica + emoção | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/pt/guides/kokoro) | Texto para voz | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | TTS streaming (voz Alba fixa) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/denoise) | Aprimoramento de voz | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/restore) | Redução de ruído e reverberação (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/pt/guides/respond) | Voz para voz full-duplex (CUDA) | estrutural | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/pt/guides/function-calls) | Ferramentas estruturadas locais | — | LiteRT-LM |

Veja [docs/models.md](docs/models.md) para maturidade, layouts, pré-processamento, memória e exemplos completos.

## Plataformas e backends

| Backend | Target | Plataformas | Configuração |
|---|---|---|---|
| Apenas núcleo | `speech_core` | Linux, Windows, macOS, Android | nenhuma |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | release ONNX Runtime extraída via `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, caminho de build Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX pode usar CPU, NNAPI no Android, Qualcomm QNN no Linux ou um hook de Execution Provider fornecido pelo app. LiteRT usa atualmente CPU por sua API C.

## Início rápido

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Transcreva um buffer; Parakeet v3 detecta o idioma automaticamente:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Conecte qualquer VAD, STT, LLM e TTS ao pipeline ao vivo:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcrição, áudio de resposta, ferramenta ou erro
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Pacotes CLI Linux

Releases incluem `.deb` e `.tar.gz` para amd64 e arm64, com bibliotecas de runtime mas sem modelos.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 ou arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

O pacote amd64 também inclui clonagem VoxCPM2 com LiteRT. O bundle x86 tem cerca de 13 GB:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

A **[referência CLI Linux](docs/cli.md)** traz sintaxe, diretórios, binários e matriz amd64/arm64. [`soniqo.audio/cli`](https://soniqo.audio/pt/cli) documenta a CLI maior do speech-swift para Apple.

## Arquitetura

```text
áudio / eventos do aplicativo
             │
             ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · detecção de turnos   │
│ interrupções · ferramentas · áudio   │
│ APIs abstratas VAD/STT/LLM/TTS       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ modelos ref.    │ │ modelos ref.        │
      └─────────────────┘ └─────────────────────┘
```

O target de orquestração nunca depende de um modelo concreto. Trocar backend é uma escolha de construção e link, não uma reescrita do pipeline.

## Documentação

| Tema | Documentação |
|---|---|
| Visão geral e matriz | [soniqo.audio/pt/speech-core](https://soniqo.audio/pt/speech-core) |
| Configuração Linux | [soniqo.audio/pt/getting-started/linux](https://soniqo.audio/pt/getting-started/linux) |
| Configuração Windows | [soniqo.audio/pt/getting-started/windows](https://soniqo.audio/pt/getting-started/windows) |
| CLI Linux | [docs/cli.md](docs/cli.md) |
| Interfaces e backends próprios | [docs/interfaces.md](docs/interfaces.md) |
| Implementações de modelos | [docs/models.md](docs/models.md) |
| Pipeline e máquina de estados | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Chamadas de ferramentas | [docs/tools.md](docs/tools.md) |

## Variantes de build

```bash
# Apenas orquestração, sem runtime ML
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Modelos ONNX
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# Modelos LiteRT
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Testes

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Testes do núcleo não exigem modelos. Integrações são ignoradas corretamente sem suas variáveis. CI cobre Linux, Windows e macOS; workflows agendados validam bundles ONNX e LiteRT públicos.

## Projetos relacionados

- [speech-android](https://github.com/soniqo/speech-android) — SDK Kotlin e JNI sobre speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — stack MLX/CoreML nativo para macOS e iOS.
- [Documentação Soniqo](https://soniqo.audio/pt) — guias, arquitetura, benchmarks e modelos.

## Contribuição

Issues e pull requests são bem-vindos. Crie uma branch de `main`, compile as configurações afetadas, execute `ctest` e abra um PR focado.

## Licença

Apache 2.0 — veja [LICENSE](LICENSE).
