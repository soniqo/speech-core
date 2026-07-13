# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

**Linux, Windows, Android**용 **C++17** 온디바이스 음성 인프라입니다. 음성 활동 감지, 배치 및 실시간 스트리밍 음성 인식, 화자 분리, 음성 합성, 그리고 이 기능들을 연결하는 음성 에이전트 파이프라인을 제공합니다.

CPU에서 완전히 로컬로 실행됩니다. 추론 시 클라우드나 Python이 필요 없으며, 오디오가 기기 밖으로 나가지 않습니다.

**[📚 전체 문서 →](https://soniqo.audio/ko/speech-core)** · **[🐧 Linux](https://soniqo.audio/ko/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/ko/getting-started/windows)** · **[⌨️ Linux CLI](docs/cli.md)**

**[🤗 모델](https://huggingface.co/soniqo)** · **[🍎 Apple용 자매 프로젝트](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## 데모

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="VoxCPM2 음성 복제 — YouTube에서 Speech Studio 데모 보기">
  </a>
</p>
<p align="center"><em>VoxCPM2 음성 복제 — YouTube에서 Speech Studio 데모 보기</em></p>

## speech-core를 선택하는 이유

speech-core는 작고 모델에 독립적인 오케스트레이션 계층과 선택적 추론 백엔드를 분리합니다. 코어는 턴 감지, 끼어들기 처리, 오디오 유틸리티, 대화 상태, 도구 호출을 담당하며 애플리케이션이 모델을 선택합니다.

- **로컬 우선:** 순수 C++17 코어, Float 오디오 버퍼, 네트워크 및 플랫폼 오디오 종속성 없음.
- **실시간 에이전트용:** VAD 기반 턴, 조기 STT, 부분 전사, 바지인, 스트리밍 TTS, 도구 호출.
- **진정한 스트리밍 ASR:** 캐시 인식 RNN-T 디코더, 발화 종료 감지, 빔 검색, 문맥 구문 바이어싱.
- **백엔드 선택:** ONNX Runtime, LiteRT, 둘 다, 둘 다 사용하지 않는 구성, 또는 자체 추상 인터페이스 구현.
- **이식 가능한 표면:** 네이티브 C++ API와 Kotlin/JNI, Swift/FFI, 임베디드 Linux 등에 적합한 C API.
- **다중 타깃 검증:** Linux, Windows, macOS, Android 지향 arm64 빌드, Sanitizer, 모델 기반 nightly 레인.

## v0.0.9 주요 변경 사항

- **Parakeet-EOU 120M:** 발화 종료 토큰, 선택적 빔 검색, 문맥 구문 바이어싱, 과도한 바이어스 상한을 갖춘 저메모리 다국어 스트리밍 ASR.
- **네이티브 Whisper ONNX:** small부터 large-v3/turbo까지, 언어 감지 또는 고정 언어 프롬프트, 프로파일링, CPU 튜닝 제어.
- **확장된 TTS:** Kokoro와 함께 VoxCPM/VoxCPM2, CosyVoice3, Chatterbox, Supertonic, Indic-Mio 지원. 버퍼 후처리와 전사 유도 복제 포함.
- **더 빠른 대화:** Kokoro 짧은 턴 최적화, 긴 텍스트 문장 분할, 재생 전후에도 유지되는 프리 스피치 버퍼.
- **온디바이스 LLM 도구:** LiteRT-LM 기반 FunctionGemma, 기존 Ollama 어댑터, 파이프라인 도구 호출 루프.
- **릴리스 수준 Linux CLI:** amd64/arm64 패키지, 모델 다운로드 도우미, 아키텍처별 명령 가용성, 깨끗한 컨테이너 스모크 테스트.

## 지원 모델

| 모델 | 작업 | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/vad) | 음성 활동 감지 | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/ko/guides/parakeet) | 음성 인식 | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/whisper) | 다국어 음성 인식 | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/nemotron) | 스트리밍 음성 인식 | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/ko/guides/nemotron) | 프롬프트 조건부 스트리밍 STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/ko/guides/dictate) | 스트리밍 STT + 발화 종료 | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/omnilingual) | 다국어 음성 인식 | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/diarize) | 화자 분리 세그멘테이션 | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/embed-speaker) | 화자 임베딩 | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + 음성 복제 | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/voxcpm2) | 48 kHz TTS + 음성 복제 | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/cosyvoice) | 24 kHz 조건부 TTS | 단계적 | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/chatterbox) | 24 kHz 음성 합성 | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/supertonic) | 음성 합성 | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/ko/guides/indic-mio) | 힌디어/인도계 언어 음성 복제 + 감정 | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/kokoro) | 음성 합성 | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/denoise) | 음성 향상 | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/sidon) | 노이즈 제거 + 잔향 제거(16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/ko/guides/respond) | 전이중 음성 대 음성(CUDA) | 구조 구현 | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/ko/guides/functiongemma) | 온디바이스 구조화 도구 호출 | — | LiteRT-LM |

성숙도, 번들 구성, 전처리, 메모리 정보, 전체 예제는 [docs/models.md](docs/models.md)를 참고하세요.

## 플랫폼과 백엔드

| 백엔드 | 타깃 | 플랫폼 | 런타임 설정 |
|---|---|---|---|
| 코어만 | `speech_core` | Linux, Windows, macOS, Android | 없음 |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | 압축 해제한 ONNX Runtime을 `ORT_DIR`로 지정 |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android 빌드 경로 | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX는 CPU, Android NNAPI, Linux의 Qualcomm QNN 또는 애플리케이션 제공 실행 공급자 훅을 사용할 수 있습니다. LiteRT는 현재 C API를 통해 CPU를 사용합니다.

## 빠른 시작

코어와 LiteRT 백엔드를 빌드합니다.

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

오디오 버퍼를 전사합니다. Parakeet v3는 언어를 자동 감지합니다.

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite",
    "parakeet-decoder-joint.tflite",
    "vocab.json");

auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

추상 VAD, STT, LLM, TTS 인터페이스의 어떤 구현이든 실시간 파이프라인에 연결할 수 있습니다.

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // 전사, 응답 오디오, 도구 호출 또는 오류
    });

pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

애플리케이션이 사용하는 타깃만 링크합니다.

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI 패키지

릴리스에는 amd64 및 arm64용 `.deb`와 `.tar.gz` 패키지가 포함됩니다. 런타임 라이브러리는 번들되지만 모델은 포함되지 않습니다.

```bash
VERSION=0.0.9
ARCH="$(dpkg --print-architecture)"   # amd64 또는 arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
```

amd64 패키지에는 LiteRT VoxCPM2 음성 복제 명령도 포함됩니다. x86 번들은 약 13GB이므로 명시적으로 다운로드합니다.

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

정확한 문법, 모델 디렉터리, 독립 실행 바이너리, amd64/arm64 명령 표는 **[Linux CLI 참조](docs/cli.md)**를 확인하세요. [`soniqo.audio/cli`](https://soniqo.audio/ko/cli)는 Apple 플랫폼용 더 큰 speech-swift CLI를 설명합니다.

## 아키텍처

```text
애플리케이션 오디오 / 이벤트
              │
              ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · 턴 감지              │
│ 끼어들기 · 도구 · 오디오 유틸리티    │
│ 추상 VAD / STT / LLM / TTS API       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ 참조 모델       │ │ 참조 모델           │
      └─────────────────┘ └─────────────────────┘
```

오케스트레이션 타깃은 특정 모델에 의존하지 않습니다. 백엔드 교체는 생성 및 링크 선택이며 파이프라인을 다시 작성할 필요가 없습니다.

## 문서

| 주제 | 문서 |
|---|---|
| 제품 개요와 모델 표 | [soniqo.audio/ko/speech-core](https://soniqo.audio/ko/speech-core) |
| Linux 설정 | [soniqo.audio/ko/getting-started/linux](https://soniqo.audio/ko/getting-started/linux) |
| Windows 설정 | [soniqo.audio/ko/getting-started/windows](https://soniqo.audio/ko/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| 인터페이스와 사용자 백엔드 | [docs/interfaces.md](docs/interfaces.md) |
| 모델 구현 | [docs/models.md](docs/models.md) |
| 음성 파이프라인과 상태 머신 | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| 도구 호출 | [docs/tools.md](docs/tools.md) |

## 빌드 변형

```bash
# 오케스트레이션만: ML 런타임 없음
cmake -B build -DCMAKE_BUILD_TYPE=Release

# ONNX 모델
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime

# LiteRT 모델
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## 테스트

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

코어 테스트에는 모델 파일이 필요하지 않습니다. 관련 모델 디렉터리 환경 변수가 없으면 백엔드 통합 테스트는 정상적으로 건너뜁니다. CI는 Linux, Windows, macOS를 다루며 모델 기반 예약 워크플로가 공개 ONNX 및 LiteRT 번들을 검증합니다.

## 관련 프로젝트

- [speech-android](https://github.com/soniqo/speech-android) — speech-core 기반 Kotlin SDK와 JNI 통합.
- [speech-swift](https://github.com/soniqo/speech-swift) — macOS 및 iOS용 네이티브 MLX/CoreML 음성 스택.
- [Soniqo 문서](https://soniqo.audio/ko) — 가이드, 아키텍처, 벤치마크, 모델 페이지.

## 기여

Issue와 Pull Request를 환영합니다. `main`에서 브랜치를 만들고 영향받는 구성을 빌드한 뒤 `ctest`를 실행하고, 범위가 명확한 PR을 열어 주세요.

## 라이선스

Apache 2.0 — [LICENSE](LICENSE)를 참조하세요.
