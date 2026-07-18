# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

โครงสร้างพื้นฐานเสียงพูดแบบ on-device ด้วย **C++17** สำหรับ **Linux, Windows และ Android**: ตรวจจับกิจกรรมเสียง, แปลงเสียงเป็นข้อความทั้งแบบ batch และ streaming แบบเรียลไทม์, แยกผู้พูด, สังเคราะห์เสียง และ voice-agent pipeline ที่เชื่อมทุกส่วนเข้าด้วยกัน

ทำงานภายในเครื่องบน CPU ไม่ใช้คลาวด์หรือ Python ระหว่าง inference และเสียงไม่ออกจากอุปกรณ์

**[📚 เอกสารฉบับเต็ม →](https://soniqo.audio/th/speech-core)** · **[🐧 Linux](https://soniqo.audio/th/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/th/getting-started/windows)** · **[⌨️ Desktop CLI](docs/cli.md)** · **[🔊 เสียง HTTP](docs/http-server.md)**

**[🤗 โมเดล](https://huggingface.co/soniqo)** · **[🍎 โปรเจกต์สำหรับ Apple](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## เดโม

<p align="center"><a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="โคลนเสียงด้วย VoxCPM2 — เดโม Speech Studio บน YouTube"></a></p>
<p align="center"><em>โคลนเสียงด้วย VoxCPM2 — เดโม Speech Studio บน YouTube</em></p>

## เหตุผลที่ใช้ speech-core

speech-core แยกชั้น orchestration ขนาดเล็กที่ไม่ผูกกับโมเดลออกจาก inference backend ที่เลือกเปิดได้ core ดูแลการตรวจจับรอบสนทนา การขัดจังหวะ เครื่องมือเสียง สถานะบทสนทนา และ tool call ส่วนแอปเป็นผู้เลือกโมเดล

- **ทำงานในเครื่องเป็นหลัก:** core C++17 ล้วน ใช้ Float audio buffer ไม่พึ่งเครือข่ายหรือ audio API ของแพลตฟอร์ม
- **สร้างเพื่อเอเจนต์สด:** รอบสนทนาด้วย VAD, STT ล่วงหน้า, partial transcript, barge-in, streaming TTS และ tool calling
- **ASR streaming จริง:** RNN-T ที่มี cache, ตรวจจับจบคำพูด, beam search และ contextual phrase biasing
- **เลือก backend ได้:** ONNX Runtime, LiteRT, ทั้งคู่, ไม่ใช้เลย หรือ implement abstract interface เอง
- **API พกพาได้:** C++ native และ C API สำหรับ Kotlin/JNI, Swift/FFI, embedded Linux และ host อื่น
- **ทดสอบหลายเป้าหมาย:** Linux, Windows, macOS, build arm64 สำหรับ Android, sanitizer และ nightly ที่ใช้โมเดลจริง

## ไฮไลต์ v0.0.11

- **TTS local ที่เข้ากันได้กับ OpenAI:** `speech-server` ให้ `POST /v1/audio/speech` พร้อม alias โมเดล OpenAI, เสียง native และทั่วไป, การควบคุมภาษา/ความเร็ว, WAV/PCM และ Bearer authentication แบบเลือกใช้
- **แพ็กเกจ Windows:** ZIP x64 แบบ self-contained มี server, เครื่องมือ ONNX CLI, `speech.dll`, ONNX Runtime และตัวดาวน์โหลดโมเดล PowerShell โดย CI จะแตกไฟล์และ smoke test แพ็กเกจ
- **DeepFilterNet3 parity:** STFT scaling แบบ libdf, ERB/complex normalization, deep filtering, overlap-add และชดเชย delay 480 samples คืนพฤติกรรม DSP อ้างอิง
- **Pocket TTS แบบ streaming:** backend ONNX ส่ง frame คงที่ 80 ms ใช้ decoder cache แบบจำกัด และมี round-trip validation ด้วยโมเดลแบบเลือกใช้
- **Silero v5 context ที่ถูกต้อง:** ทุก ONNX inference ได้รับ left context 64 samples ตามที่ graph ต้องการ

## โมเดลที่รองรับ

| โมเดล | งาน | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/vad) | ตรวจจับกิจกรรมเสียง | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/th/guides/parakeet) | เสียงเป็นข้อความ | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/whisper) | เสียงเป็นข้อความหลายภาษา | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/nemotron) | STT streaming | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/th/guides/nemotron) | STT streaming แบบมี prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/th/guides/dictate) | STT streaming + จบคำพูด | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/omnilingual) | STT หลายภาษา | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/diarize) | segmentation สำหรับ diarization | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/embed-speaker) | speaker embedding | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 kHz + โคลนเสียง | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/voxcpm2) | TTS 48 kHz + โคลนเสียง | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/cosyvoice) | TTS แบบมี conditioning 24 kHz | เป็นขั้น | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/chatterbox) | สังเคราะห์เสียง 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/supertonic) | สังเคราะห์เสียง | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/indic-mio) | โคลนเสียงภาษาฮินดี/อินเดีย + อารมณ์ | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/th/guides/kokoro) | สังเคราะห์เสียง | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | TTS streaming (เสียง Alba คงที่) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/denoise) | ปรับปรุงเสียงพูด | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/restore) | ลดเสียงรบกวน + ลดเสียงก้อง (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/th/guides/respond) | speech-to-speech แบบ full-duplex (CUDA) | โครงสร้าง | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/th/guides/function-calls) | structured tool call บนอุปกรณ์ | — | LiteRT-LM |

ดูระดับความพร้อม โครงสร้าง bundle preprocessing หน่วยความจำ และตัวอย่างเต็มได้ที่ [docs/models.md](docs/models.md)

## แพลตฟอร์มและ backend

| Backend | Target | แพลตฟอร์ม | การตั้งค่า runtime |
|---|---|---|---|
| Core เท่านั้น | `speech_core` | Linux, Windows, macOS, Android | ไม่ต้องใช้ |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | ONNX Runtime ที่แตกไฟล์แล้วผ่าน `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, เส้นทาง build Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX ใช้ CPU, Android NNAPI, Qualcomm QNN บน Linux หรือ Execution Provider hook จากแอปได้ ส่วน LiteRT ตอนนี้ใช้ CPU ผ่าน C API

## เริ่มต้นอย่างรวดเร็ว

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

แปลง audio buffer เป็นข้อความ โดย Parakeet v3 ตรวจภาษาอัตโนมัติ:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

เชื่อม implementation ของ VAD, STT, LLM และ TTS ใด ๆ เข้ากับ live pipeline:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcript, เสียงตอบกลับ, tool call หรือ error
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## แพ็กเกจ Linux CLI

Release มี `.deb` และ `.tar.gz` สำหรับ amd64 และ arm64 พร้อม runtime library แต่ไม่รวมโมเดล

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 หรือ arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

แพ็กเกจ amd64 มีคำสั่งโคลนเสียง VoxCPM2 ด้วย LiteRT ด้วย โดย bundle x86 มีขนาดประมาณ 13 GB:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

**[คู่มือ Linux CLI](docs/cli.md)** มี syntax, model directory, binary และตาราง amd64/arm64 ส่วน [`soniqo.audio/cli`](https://soniqo.audio/th/cli) เป็น CLI ของ speech-swift สำหรับ Apple ที่มีคำสั่งมากกว่า

## สถาปัตยกรรม

```text
audio / event จากแอป
        │
        ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · ตรวจจับรอบสนทนา      │
│ การขัดจังหวะ · tools · audio utils   │
│ abstract VAD/STT/LLM/TTS APIs        │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ reference models│ │ reference models   │
      └─────────────────┘ └─────────────────────┘
```

target สำหรับ orchestration ไม่ขึ้นกับโมเดลใดโดยตรง การเปลี่ยน backend เป็นเพียงการเลือกตอนสร้างและ link ไม่ต้องเขียน pipeline ใหม่

## เอกสาร

| หัวข้อ | เอกสาร |
|---|---|
| ภาพรวมและตารางโมเดล | [soniqo.audio/th/speech-core](https://soniqo.audio/th/speech-core) |
| ตั้งค่า Linux | [soniqo.audio/th/getting-started/linux](https://soniqo.audio/th/getting-started/linux) |
| ตั้งค่า Windows | [soniqo.audio/th/getting-started/windows](https://soniqo.audio/th/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Interface และ backend ที่กำหนดเอง | [docs/interfaces.md](docs/interfaces.md) |
| การ implement โมเดล | [docs/models.md](docs/models.md) |
| Voice pipeline และ state machine | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Tool calling | [docs/tools.md](docs/tools.md) |

## รูปแบบการ build

```bash
# เฉพาะ orchestration ไม่มี ML runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
# โมเดล ONNX
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# โมเดล LiteRT
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## การทดสอบ

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

core test ไม่ต้องใช้โมเดล integration test จะ skip อย่างถูกต้องเมื่อไม่มีตัวแปร directory CI ครอบคลุม Linux, Windows และ macOS ส่วน workflow ตามตารางทดสอบ bundle ONNX และ LiteRT สาธารณะ

## โปรเจกต์ที่เกี่ยวข้อง

- [speech-android](https://github.com/soniqo/speech-android) — Kotlin SDK และ JNI บน speech-core
- [speech-swift](https://github.com/soniqo/speech-swift) — native MLX/CoreML speech stack สำหรับ macOS และ iOS
- [เอกสาร Soniqo](https://soniqo.audio/th) — คู่มือ สถาปัตยกรรม benchmark และหน้าโมเดล

## การมีส่วนร่วม

ยินดีรับ issue และ pull request สร้าง branch จาก `main`, build configuration ที่เกี่ยวข้อง, รัน `ctest` และเปิด PR ที่มีขอบเขตชัดเจน

## ใบอนุญาต

Apache 2.0 — ดู [LICENSE](LICENSE)
