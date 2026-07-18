# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Локальная речевая инфраструктура на **C++17** для **Linux, Windows и Android**: VAD, пакетное и потоковое распознавание речи, диаризация, синтез речи и объединяющий их конвейер голосового агента.

Работает локально на CPU. Без облака и Python во время инференса; аудио не покидает устройство.

**[📚 Полная документация →](https://soniqo.audio/ru/speech-core)** · **[🐧 Linux](https://soniqo.audio/ru/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/ru/getting-started/windows)** · **[⌨️ Настольный CLI](docs/cli.md)** · **[🔊 HTTP-аудио](docs/http-server.md)**

**[🤗 Модели](https://huggingface.co/soniqo)** · **[🍎 Проект для Apple](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Демонстрация

<p align="center"><a href="https://www.youtube.com/watch?v=7L7_Uvvxtv0"><img src="https://img.youtube.com/vi/7L7_Uvvxtv0/maxresdefault.jpg" width="640" alt="Полностью офлайновый голосовой агент в 1,2 ГБ на Android — смотреть демо на YouTube"></a></p>
<p align="center"><em>Полностью офлайновый голосовой агент в 1,2 ГБ на Android — control-demo из speech-android</em></p>

## Зачем нужен speech-core

speech-core отделяет компактный, независимый от моделей слой оркестрации от необязательных inference-бэкендов. Ядро отвечает за определение реплик, прерывания, аудиоутилиты, состояние диалога и вызовы инструментов; приложение выбирает модели.

- **Локальность:** чистое ядро C++17, Float-буферы аудио, без зависимости от сети и платформенного аудиоввода.
- **Для живых агентов:** реплики по VAD, ранний STT, частичные результаты, barge-in, потоковый TTS и инструменты.
- **Настоящий потоковый ASR:** RNN-T с кешем, определение конца высказывания, beam search и контекстное усиление фраз.
- **Выбор бэкенда:** ONNX Runtime, LiteRT, оба, ни одного либо собственная реализация абстрактных интерфейсов.
- **Переносимый API:** C++ и C API для Kotlin/JNI, Swift/FFI, встраиваемого Linux и других хостов.
- **Многоцелевые тесты:** Linux, Windows, macOS, arm64-сборки для Android, sanitizers и nightly с моделями.

## Основное в v0.0.11

- **Локальный TTS с API OpenAI:** `speech-server` предоставляет `POST /v1/audio/speech`, алиасы моделей OpenAI, встроенные и универсальные голоса, язык и скорость, WAV/PCM и опциональную Bearer-аутентификацию.
- **Пакет Windows:** автономный x64 ZIP включает сервер, ONNX CLI, `speech.dll`, ONNX Runtime и PowerShell-загрузчик моделей; CI распаковывает и проверяет пакет.
- **Паритет DeepFilterNet3:** совместимые с libdf STFT, ERB/комплексная нормализация, deep filtering, overlap-add и компенсация задержки 480 отсчётов восстанавливают эталонный DSP.
- **Потоковый Pocket TTS:** ONNX-бэкенд выдаёт фиксированные кадры по 80 мс, использует ограниченный кэш и поддерживает опциональную проверку round-trip с моделью.
- **Корректный контекст Silero v5:** каждая ONNX-инференция получает требуемые 64 отсчёта левого контекста.

## Поддерживаемые модели

| Модель | Задача | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/vad) | Детекция речи | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/ru/guides/parakeet) | Распознавание речи | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/whisper) | Многоязычное распознавание | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/nemotron) | Потоковое распознавание | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/ru/guides/nemotron) | Потоковый STT с prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/ru/guides/dictate) | Потоковый STT + конец реплики | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/omnilingual) | Многоязычный STT | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/diarize) | Сегментация для диаризации | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/embed-speaker) | Эмбеддинг говорящего | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 кГц + клонирование | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/voxcpm2) | TTS 48 кГц + клонирование | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/cosyvoice) | Условный TTS 24 кГц | поэтапно | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/chatterbox) | Синтез речи 24 кГц | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/supertonic) | Синтез речи | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/indic-mio) | Клонирование на языках Индии + эмоции | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/ru/guides/kokoro) | Синтез речи | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | Потоковый TTS (фиксированный голос Alba) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/denoise) | Улучшение речи | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/restore) | Шумоподавление + дереверберация (16 → 48 кГц) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/ru/guides/respond) | Полнодуплексная речь-в-речь (CUDA) | структура | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/ru/guides/function-calls) | Локальные структурированные tool calls | — | LiteRT-LM |

Статус готовности, устройство bundles, предобработку, память и полные примеры смотрите в [docs/models.md](docs/models.md).

## Платформы и бэкенды

| Бэкенд | Target | Платформы | Настройка runtime |
|---|---|---|---|
| Только ядро | `speech_core` | Linux, Windows, macOS, Android | не требуется |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | распакованный ONNX Runtime через `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android build path | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX поддерживает CPU, Android NNAPI, Qualcomm QNN в Linux и hook Execution Provider от приложения. LiteRT сейчас использует CPU через C API.

## Быстрый старт

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Распознавание аудиобуфера; Parakeet v3 определяет язык автоматически:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Подключите любые реализации VAD, STT, LLM и TTS к живому конвейеру:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // транскрипт, ответное аудио, tool call или ошибка
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Пакеты Linux CLI

Релизы содержат `.deb` и `.tar.gz` для amd64 и arm64. Runtime-библиотеки включены, модели — нет.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 или arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

В amd64-пакет также входит клонирование VoxCPM2 через LiteRT. x86-bundle занимает около 13 ГБ:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

Точный синтаксис, каталоги моделей, отдельные бинарники и матрица amd64/arm64 находятся в **[справочнике Linux CLI](docs/cli.md)**. [`soniqo.audio/cli`](https://soniqo.audio/ru/cli) описывает более крупный speech-swift CLI для Apple.

## Архитектура

```text
аудио / события приложения
            │
            ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · определение реплик   │
│ прерывания · tools · аудиоутилиты    │
│ абстрактные VAD/STT/LLM/TTS API      │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ эталонные модели│ │ эталонные модели   │
      └─────────────────┘ └─────────────────────┘
```

Оркестрация никогда не зависит от конкретной модели. Смена бэкенда — выбор конструкции и линковки, а не переписывание конвейера.

## Документация

| Тема | Документация |
|---|---|
| Обзор и матрица моделей | [soniqo.audio/ru/speech-core](https://soniqo.audio/ru/speech-core) |
| Настройка Linux | [soniqo.audio/ru/getting-started/linux](https://soniqo.audio/ru/getting-started/linux) |
| Настройка Windows | [soniqo.audio/ru/getting-started/windows](https://soniqo.audio/ru/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Интерфейсы и свои бэкенды | [docs/interfaces.md](docs/interfaces.md) |
| Реализации моделей | [docs/models.md](docs/models.md) |
| Голосовой конвейер и автомат | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Вызовы инструментов | [docs/tools.md](docs/tools.md) |

## Варианты сборки

```bash
# Только оркестрация, без ML runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
# ONNX-модели
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# LiteRT-модели
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Тестирование

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Тестам ядра модели не нужны. Интеграции корректно пропускаются без переменных каталогов. CI охватывает Linux, Windows и macOS; плановые workflows проверяют публичные ONNX- и LiteRT-bundles.

## Связанные проекты

- [speech-android](https://github.com/soniqo/speech-android) — Kotlin SDK и JNI поверх speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — нативный MLX/CoreML-стек для macOS и iOS.
- [Документация Soniqo](https://soniqo.audio/ru) — руководства, архитектура, бенчмарки и модели.

## Участие

Issues и pull requests приветствуются. Создайте ветку от `main`, соберите затронутые конфигурации, запустите `ctest` и откройте сфокусированный PR.

## Лицензия

Apache 2.0 — см. [LICENSE](LICENSE).
