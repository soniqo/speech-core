# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

面向 **Linux、Windows 和 Android** 的 **C++17** 端侧语音基础设施：语音活动检测、批量与实时流式语音转文字、说话人分离、文字转语音，以及连接这些能力的语音智能体管线。

完全在本地 CPU 上运行。推理不依赖云端或 Python，音频不会离开设备。

**[📚 完整文档 →](https://soniqo.audio/zh/speech-core)** · **[🐧 Linux](https://soniqo.audio/zh/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/zh/getting-started/windows)** · **[⌨️ 桌面 CLI](docs/cli.md)** · **[🔊 HTTP 音频](docs/http-server.md)**

**[🤗 模型](https://huggingface.co/soniqo)** · **[🍎 Apple 端项目](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## 演示

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="使用 VoxCPM2 进行声音克隆 — 在 YouTube 观看 Speech Studio 演示">
  </a>
</p>
<p align="center"><em>使用 VoxCPM2 进行声音克隆 — 在 YouTube 观看 Speech Studio 演示</em></p>

## 为什么选择 speech-core

speech-core 将小巧、与模型无关的编排层和可选推理后端分离。核心负责话轮检测、打断处理、音频工具、对话状态与工具调用；应用自行选择模型。

- **本地优先：** 纯 C++17 核心，处理浮点音频缓冲，不依赖网络或平台音频 API。
- **为实时智能体设计：** VAD 话轮、提前 STT、部分转录、插话打断、流式 TTS 与工具调用。
- **真正的流式 ASR：** 带缓存的 RNN-T 解码、话语结束检测、束搜索与上下文短语偏置。
- **后端可选：** 可启用 ONNX Runtime、LiteRT、两者同时启用、完全不启用，或自行实现抽象接口。
- **可移植接口：** 原生 C++ API 以及适合 Kotlin/JNI、Swift/FFI、嵌入式 Linux 等宿主的 C API。
- **跨目标测试：** Linux、Windows、macOS、面向 Android 的 arm64 构建、Sanitizer 与模型夜间测试。

## v0.0.11 亮点

- **兼容 OpenAI 的本地 TTS：** `speech-server` 提供 `POST /v1/audio/speech`，支持 OpenAI 模型别名、原生与通用音色、语言和语速控制、WAV/PCM 输出以及可选 Bearer 认证。
- **Windows 发布包：** 自包含 x64 ZIP 内含服务器、ONNX CLI 工具、`speech.dll`、ONNX Runtime 和原生 PowerShell 模型下载器；CI 会构建、解压并冒烟测试该软件包。
- **DeepFilterNet3 对齐：** 与 libdf 一致的原生 STFT 缩放、ERB/复数归一化、深度滤波、重叠相加及 480 样本延迟补偿恢复参考 DSP 行为。
- **流式 Pocket TTS：** ONNX 后端输出固定 80 毫秒帧，使用有界解码器缓存，并提供可选的模型回环验证工具。
- **正确的 Silero v5 上下文：** 每次 ONNX 推理现在都会传入图所需的 64 样本左上下文。

## 支持的模型

| 模型 | 任务 | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/vad) | 语音活动检测 | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/zh/guides/parakeet) | 语音转文字 | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/whisper) | 多语言语音转文字 | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/nemotron) | 流式语音转文字 | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/zh/guides/nemotron) | 提示词控制的流式 STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/zh/guides/dictate) | 流式 STT + 话语结束检测 | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/omnilingual) | 多语言语音转文字 | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/diarize) | 说话人分离切分 | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/embed-speaker) | 说话人嵌入 | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + 声音克隆 | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/voxcpm2) | 48 kHz TTS + 声音克隆 | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/cosyvoice) | 24 kHz 条件式 TTS | 预备 | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/chatterbox) | 24 kHz 文字转语音 | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/supertonic) | 文字转语音 | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/indic-mio) | 印地语/印度语言声音克隆 + 情感 | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/zh/guides/kokoro) | 文字转语音 | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | 流式 TTS（固定 Alba 音色） | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/denoise) | 语音增强 | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/restore) | 降噪 + 去混响（16 → 48 kHz） | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/zh/guides/respond) | 全双工语音到语音（CUDA） | 结构已实现 | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/zh/guides/function-calls) | 端侧结构化工具调用 | — | LiteRT-LM |

有关成熟度、模型包布局、预处理、内存说明和完整示例，请参阅 [docs/models.md](docs/models.md)。

## 平台与后端

| 后端 | 目标 | 平台 | 运行时配置 |
|---|---|---|---|
| 仅核心 | `speech_core` | Linux、Windows、macOS、Android | 无 |
| ONNX Runtime | `speech_core_models` | Linux、Windows、macOS、Android | 通过 `ORT_DIR` 指向解压的 ONNX Runtime 发布包 |
| LiteRT | `speech_core_models_litert` | Linux x86_64、Windows x86_64、macOS arm64、Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS、Android 构建路径 | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX 可使用 CPU、Android NNAPI、Linux 上的 Qualcomm QNN，或由应用提供执行提供程序 hook。LiteRT 目前通过其 C API 使用 CPU。

## 快速开始

构建核心与 LiteRT 后端：

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

转录音频缓冲；Parakeet v3 会自动检测语言：

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite",
    "parakeet-decoder-joint.tflite",
    "vocab.json");

auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

将 VAD、STT、LLM 和 TTS 抽象接口的任意实现连接到实时管线：

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // 转录、响应音频、工具调用或错误
    });

pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

只链接应用实际使用的目标：

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI 软件包

每个 release 都提供 amd64 和 arm64 的 `.deb` 与 `.tar.gz` 软件包。软件包包含运行时库，但不包含模型。

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 或 arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

amd64 软件包还包含 LiteRT VoxCPM2 声音克隆命令。x86 模型包约 13 GB，需要显式下载：

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

准确语法、模型目录、独立二进制文件与 amd64/arm64 命令矩阵见 **[Linux CLI 参考](docs/cli.md)**。[`soniqo.audio/cli`](https://soniqo.audio/zh/cli) 介绍 Apple 平台上功能更完整的 speech-swift CLI。

## 架构

```text
应用音频 / 事件
       │
       ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · 话轮检测             │
│ 打断 · 工具 · 音频工具               │
│ 抽象 VAD / STT / LLM / TTS API       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ 参考模型        │ │ 参考模型            │
      └─────────────────┘ └─────────────────────┘
```

编排目标从不依赖具体模型。切换后端只是构造与链接选择，不需要重写管线。

## 文档

| 主题 | 文档 |
|---|---|
| 产品概览与模型矩阵 | [soniqo.audio/zh/speech-core](https://soniqo.audio/zh/speech-core) |
| Linux 配置 | [soniqo.audio/zh/getting-started/linux](https://soniqo.audio/zh/getting-started/linux) |
| Windows 配置 | [soniqo.audio/zh/getting-started/windows](https://soniqo.audio/zh/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| 接口与自定义后端 | [docs/interfaces.md](docs/interfaces.md) |
| 模型实现 | [docs/models.md](docs/models.md) |
| 语音管线与状态机 | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| 工具调用 | [docs/tools.md](docs/tools.md) |

## 构建变体

```bash
# 仅编排：无 ML 运行时
cmake -B build -DCMAKE_BUILD_TYPE=Release

# ONNX 模型
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime

# LiteRT 模型
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## 测试

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

核心测试无需模型文件。若未设置相应模型目录环境变量，后端集成测试会正常跳过。CI 覆盖 Linux、Windows 与 macOS；带模型的定时工作流覆盖公开的 ONNX 和 LiteRT 模型包。

## 相关项目

- [speech-android](https://github.com/soniqo/speech-android) — 基于 speech-core 的 Kotlin SDK 与 JNI 集成。
- [speech-swift](https://github.com/soniqo/speech-swift) — 面向 macOS 与 iOS 的原生 MLX/CoreML 语音栈。
- [Soniqo 文档](https://soniqo.audio/zh) — 指南、架构、基准测试与模型页面。

## 贡献

欢迎提交 issue 和 pull request。请从 `main` 创建分支，构建受影响的配置，运行 `ctest`，再开启聚焦单一主题的 PR。

## 许可证

Apache 2.0 — 参见 [LICENSE](LICENSE)。
