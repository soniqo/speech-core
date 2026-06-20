# Speech Core

[English](README.md) · **简体中文**

**C++17** 编写的语音 Agent 流水线引擎 —— 面向 **Linux、Windows 和 Android** 的设备端语音（Apple 平台见 [Swift 姊妹仓库](https://github.com/soniqo/speech-swift)）。

设备端语音活动检测、语音转文字（**批处理**和实时流式）、说话人分离、文字转语音。本地 CPU 即可运行 —— 不依赖云端、推理时无 Python、数据不出本机。

**[📖 文档](docs/)** · **[🤗 模型](https://huggingface.co/soniqo)** · **[🍎 Apple (Swift)](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## 演示

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="使用 VoxCPM2 进行声音克隆 —— 在 YouTube 观看 speech-studio 演示">
  </a>
</p>
<p align="center"><em>使用 VoxCPM2 进行声音克隆 —— 在 YouTube 观看 speech-studio 演示</em></p>

speech-core 是一个小型编排核心（状态机、轮次检测、打断处理、音频工具 —— 零 ML 依赖）加上一组抽象接口。模型推理通过两个可独立启用的可互换后端实现，**按需引入**：

- **ONNX Runtime**（`SPEECH_CORE_WITH_ONNX`）—— Silero VAD、Parakeet STT、Nemotron-3.5 多语言流式 STT、Kokoro TTS、DeepFilterNet3、Sidon 语音修复、**PersonaPlex 7B 全双工语音转语音**（CUDA 目标平台）。
- **LiteRT**（`SPEECH_CORE_WITH_LITERT`）—— Silero VAD、Parakeet STT、**Nemotron 流式 STT**、**Nemotron-3.5 多语言流式 STT**、Omnilingual STT、Pyannote 说话人分段、WeSpeaker 嵌入、VoxCPM2 TTS。底层依赖 Google 的 `ai-edge-litert`（`libLiteRt`）。

调用方可以启用任一、两个、或都不启用 —— 也可以自带接口实现（CPU、GPU、CoreML/MLX、远程 API 等）。

## 支持的模型

| 模型 | 任务 | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) | 语音活动检测 | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) | 语音转文字 | ✓ | ✓ |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) | 流式语音转文字 | ✓ | ✓ |
| [Nemotron-3.5 ASR Streaming Multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) | 流式语音转文字（多语言、Prompt 控制） | ✓ | ✓ |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) | 语音转文字（多语言） | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) | 说话人分离（分段） | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) | 说话人嵌入 | — | ✓ |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-LiteRT) | 文字转语音（48 kHz、声音克隆） | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-ONNX) | 文字转语音 | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) | 语音增强 | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) | 语音修复 —— 降噪 + 去混响（16 kHz → 48 kHz） | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) | 全双工语音转语音（CUDA）—— 4 个变体，7.6 GB → 17 GB | ✓ | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) | 设备端 LLM —— 结构化函数 / 工具调用 | — | ✓（`LiteRTFunctionGemmaLLM`，通过 `SPEECH_CORE_WITH_LITERT_LM` 启用） |

`LiteRTFunctionGemmaLLM` 通过 Google 的 `liblitert-lm` 运行时加载
[FunctionGemma 270M .litertlm 包](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM)
—— 它是比驱动 `.tflite` STT/VAD/TTS 模型的 `libLiteRt` 更高级的共享库。
使用 `scripts/fetch_litert_lm.sh` 提取一次，然后通过
`-DSPEECH_CORE_WITH_LITERT_LM=ON -DLITERT_LM_DIR=...` 构建。
同一模型的 Apple CoreML 构建位于 [speech-swift](https://github.com/soniqo/speech-swift) 中；
完整的 LLM 后端情况请参见 [docs/models.md](docs/models.md#llm-backends-llminterface)。

说话人分离（`DiarizationPipeline`）是纯 C++ 实现，将分段器和嵌入器组合成带说话人标签的片段 —— 自身不依赖任何 ML 运行时。

## 平台与后端

| 后端 | 静态库 | 运行时依赖 | 平台 | 配置 |
|---|---|---|---|---|
| ONNX | `speech_core_models` | `onnxruntime` | Linux、macOS、Windows、Android | `ORT_DIR` 指向 ONNX Runtime 发布包 |
| LiteRT | `speech_core_models_litert` | `libLiteRt` | Linux x86_64、Windows x86_64、Android、macOS arm64 | `scripts/fetch_litert.sh`（从 `ai-edge-litert` PyPI wheel 提取） |

**硬件加速。** ONNX：CPU、Android 上的 NNAPI、Qualcomm Linux 上的 QNN（将 `libQnnHtp.so` 放入库路径）。需要其它加速器时，可在 `OnnxEngine` 上安装 `SessionOptionsHook`，在 `CreateSession` 之前追加自定义 Execution Provider。LiteRT 目前仅支持 CPU；`libLiteRt` 中的 Hexagon / GPU delegate 已存在但尚未通过 C API 接入。

## 快速开始

构建核心 + LiteRT 后端（运行时库从 `ai-edge-litert` wheel 提取 —— 无需 TensorFlow 构建）：

```bash
git clone https://github.com/soniqo/speech-core && cd speech-core
scripts/fetch_litert.sh build/litert          # 如果 'python3' 版本过旧，可指定 PYTHON=python3.11
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert
cmake --build build
```

按需链接目标：

```cmake
target_link_libraries(my_app PRIVATE speech_core)                          # 仅编排
target_link_libraries(my_app PRIVATE speech_core speech_core_models)        # + ONNX 模型
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert) # + LiteRT 模型
```

**转录音频缓冲区：**

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");

auto r = stt.transcribe(audio, n_samples, 16000);   // r.text / r.language / r.confidence
```

**带部分结果的实时流式（CPU，~RTF 1.0）：**

```cpp
#include <speech_core/models/litert_nemotron_streaming_stt.h>

speech_core::LiteRTNemotronStreamingStt stt(
    "nemotron-streaming-encoder.tflite",
    "nemotron-streaming-decoder.tflite",
    "nemotron-streaming-joint.tflite", "vocab.json");

stt.begin_stream(16000);
for (const auto& chunk : mic_chunks) {              // 音频到达时按 ~80 ms 窗口送入
    auto partial = stt.push_chunk(chunk.data(), chunk.size());
    if (!partial.text.empty()) std::cout << partial.text << std::flush;
}
auto final = stt.end_stream();
```

**完整语音 Agent 流水线（VAD → STT → LLM → TTS）：**

```cpp
#include <speech_core/pipeline/voice_pipeline.h>

speech_core::AgentConfig cfg;
cfg.mode = speech_core::AgentConfig::Mode::Pipeline;   // 或 ::TranscribeOnly / ::Echo

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, cfg,
    [](const speech_core::PipelineEvent& ev) { /* 转录、音频输出、错误 */ });

pipeline.start();
pipeline.push_audio(mic_samples, count);               // 从你的音频线程调用
```

`VoicePipeline` 是实时语音 Agent 状态机 —— 基于 VAD 的轮次检测、打断处理、积极的 STT、对话跟踪、工具调用。它不拥有音频 I/O 或网络：平台层负责喂入音频并通过回调接收事件。传入 `Mode::TranscribeOnly`（同时 `llm = nullptr`）可得到纯转录流水线。

## 代码示例

### 语音活动检测

```cpp
#include <speech_core/models/litert_silero_vad.h>
speech_core::LiteRTSileroVad vad("silero-vad.tflite");
float p = vad.process_chunk(samples_512, 512);   // 语音概率在 [0, 1] 之间
```

将概率流送入 `StreamingVAD`（`speech_core/vad/streaming_vad.h`），以获得迟滞门控的 `SpeechStarted` / `SpeechEnded` 事件。

### 说话人分离

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

### 文字转语音

```cpp
#include <speech_core/models/litert_voxcpm2_tts.h>
speech_core::LiteRTVoxCPM2Tts tts(
    "voxcpm2-text-prefill.tflite", "voxcpm2-token-step.tflite",
    "voxcpm2-audio-encoder.tflite", "voxcpm2-audio-decoder.tflite", "tokenizer.json");

tts.synthesize("你好,世界", "zh", [](const float* samples, size_t len, bool is_final) {
    // 48 kHz Float32 PCM，按块流式输出
});
```

### 语音修复（Sidon）

降噪与去混响二合一。w2v-BERT 2.0 预测器（配 C++ 实现的 SeamlessM4T 对数梅尔前端）
驱动 DAC 声码器；输入会被重采样到 16 kHz，输出为 48 kHz 单声道。典型用途：在把
混响参考音频交给 TTS 声音克隆器之前先做清理。离线 / 整段处理，仅支持 ONNX。

```cpp
#include <speech_core/models/onnx_sidon_restorer.h>
speech_core::OnnxSidonRestorer rest("sidon-predictor.onnx", "sidon-vocoder.onnx");

// 任意输入采样率（内部重采样到 16 kHz）-> 48 kHz 单声道。
std::vector<float> clean = rest.restore(ref.data(), ref.size(), ref_rate);
```

命令行工具：`speech_sidon_restore <bundle_dir> <in.wav> <out.wav>`（在
`SPEECH_CORE_WITH_ONNX=ON` 下构建；读取任意采样率的 16 位 PCM WAV，输出 48 kHz）。

每个接口和模型的文档见 **[docs/interfaces.md](docs/interfaces.md)** 和 **[docs/models.md](docs/models.md)**（下载地址、大小、预处理）。

## 架构

```
┌──────────────────────────────────────────────┐
│            speech_core (始终构建)             │
│                                              │
│  VoicePipeline / TurnDetector / SpeechQueue  │  编排
│  StreamingVAD / AudioBuffer / Resampler      │
│  DiarizationPipeline                         │
│                                              │
│  STT / TTS / VAD / Enhancer / AEC / LLM      │  抽象接口
│  Segmentation / Embedding / Diarizer         │
└──────────────────────────────────────────────┘
              ▲                       ▲
              │ 实现（可选）           │
┌─────────────┴──────────┐  ┌─────────┴──────────────┐
│ speech_core_models     │  │ speech_core_models_litert │
│ (SPEECH_CORE_WITH_ONNX)│  │ (SPEECH_CORE_WITH_LITERT) │
│  ONNX Runtime          │  │  libLiteRt                │
└────────────────────────┘  └───────────────────────────┘
```

编排核心仅依赖接口 —— 从不依赖具体模型 —— 因此切换后端是链接时的选择，而非重写工作。设计原则：纯 C++17 核心、核心中无平台 API、无网络 I/O、无音频 I/O（基于 float 缓冲区运行）、回调驱动。

参考：**[接口](docs/interfaces.md)** · **[模型](docs/models.md)** · **[流水线 / 状态机](docs/pipeline.md)** · **[C API（FFI）](docs/c-api.md)** · **[工具调用](docs/tools.md)**

## 构建

```bash
# 仅编排（无 ML 依赖）
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# + ONNX 后端
cmake -B build -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime && cmake --build build

# + LiteRT 后端
scripts/fetch_litert.sh build/litert
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert && cmake --build build
```

LiteRT 头文件随仓库内置于 `third_party/litert/`（无需额外配置）。`LITERT_DIR` 指向放置 `libLiteRt.{so,dylib,dll}` 的目录（Windows 还需要 `LiteRt.lib`）。添加 `-DSPEECH_CORE_BUILD_EXAMPLES=ON` 可构建 Linux CLI 演示（`speech_transcribe`、`speech_synthesize`……）—— 见 [`examples/linux`](examples/linux)。声音克隆 CLI（`speech_voxcpm2_clone`）在 `SPEECH_CORE_WITH_LITERT=ON` 时自动构建 —— 见 [`examples/litert`](examples/litert)。

**设备端模型下载（可选）。** 添加 `-DSPEECH_CORE_WITH_HF_DOWNLOAD=ON` 可在首次使用时从 Hugging Face 拉取模型 bundle，免去手动配置。该选项链接 libcurl（`find_package(CURL)` —— Linux/macOS 使用系统 libcurl，Windows 使用 vcpkg），并为 [VoxCPM2 C ABI](include/speech_core/voxcpm2_c.h) 增加 `sc_voxcpm2_create_from_pretrained("soniqo/VoxCPM2-LiteRT", …)`：可恢复、可重试的下载（HTTP Range、原子重命名），容忍网络中断，缓存在系统缓存目录（`SPEECH_CORE_CACHE_DIR` 可覆盖；`HF_ENDPOINT` 可使用镜像）。默认关闭，嵌入式/离线构建不会引入 HTTP/TLS 依赖。`hf_fetch` 调试 CLI 可直接演练该流程。

### 在 Linux 上安装（预编译 `speech` 软件包）

每个 release 都附带 `.deb` 和 `.tar.gz` 软件包，内含 CLI 工具
（`speech_transcribe`、`speech_synthesize`、`speech_phonemize`、`speech_demo`，
amd64 还包含声音克隆 CLI `speech_voxcpm2_clone`），并将
`libonnxruntime.so` / `libLiteRt.so` 一并打包在 `/usr/lib/speech/` 下 ——
无需额外的运行时配置：

```bash
# amd64（arm64：speech_VERSION_arm64.deb —— 仅含 transcribe/synthesize/phonemize）
curl -LO https://github.com/soniqo/speech-core/releases/latest/download/speech_VERSION_amd64.deb
sudo apt install ./speech_VERSION_amd64.deb

speech download-models           # ONNX 模型集 → ~/.cache/speech-core/models（约 1.2 GB）
speech transcribe input.wav      # 与 speech-swift 的 `brew install speech` 相同的子命令式 CLI
speech speak "Hello world"
```

`speech` 命令会分发到独立的 `speech_<command>` 可执行文件
（`speech_transcribe`、`speech_synthesize` 等），后者提供更多选项。

软件包不含任何模型。工具按 `$SPEECH_MODEL_DIR`（LiteRT 为
`$SPEECH_LITERT_MODEL_DIR`）查找模型，未设置时回退到
`speech_download_models` / `speech_download_models_litert` 填充的用户缓存目录；
显式传入的模型目录参数始终优先。要求 Ubuntu 22.04+ / Debian 12+
（glibc ≥ 2.35）。

## 测试与 CI

```bash
cd build && ctest --output-on-failure        # 核心单元测试（无需模型）
```

编排 + 说话人分离的单元测试无需任何模型文件。集成测试加载真实的 `.tflite` / `.onnx` 文件，在 `SPEECH_LITERT_MODEL_DIR` / `SPEECH_MODEL_DIR` 未设置时**自动跳过**：

```bash
scripts/fetch_litert.sh build/litert
scripts/download_models_litert.sh            # 公开 soniqo/* 模型，无需 token
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert && cmake --build build
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

CI 在 **Linux、Windows、macOS** 上构建并测试（LiteRT 在 Linux + Windows，ONNX 在 Linux），外加一个 **aarch64** 交叉编译；**每日**任务针对公开模型文件运行模型集成测试。

## 贡献

欢迎提交 PR —— 模型集成、后端、文档、bug 修复。从 `main` 切出分支，构建 + `ctest`，开 PR。提交和 PR 中不要带营销话术。

## 许可证

Apache 2.0 —— 见 [LICENSE](LICENSE)。
