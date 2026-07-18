# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Hạ tầng giọng nói chạy trên thiết bị bằng **C++17** cho **Linux, Windows và Android**: phát hiện hoạt động giọng nói, nhận dạng theo lô và streaming thời gian thực, phân tách người nói, tổng hợp giọng nói và pipeline tác tử giọng nói kết nối tất cả thành phần.

Chạy cục bộ trên CPU. Không đám mây, không Python khi suy luận và âm thanh không rời khỏi thiết bị.

**[📚 Tài liệu đầy đủ →](https://soniqo.audio/vi/speech-core)** · **[🐧 Linux](https://soniqo.audio/vi/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/vi/getting-started/windows)** · **[⌨️ Desktop CLI](docs/cli.md)** · **[🔊 Âm thanh HTTP](docs/http-server.md)**

**[🤗 Mô hình](https://huggingface.co/soniqo)** · **[🍎 Dự án Apple](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<table align="center">
  <tr>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/watch?v=7L7_Uvvxtv0"><img src="https://img.youtube.com/vi/7L7_Uvvxtv0/maxresdefault.jpg" alt="Tác nhân giọng nói hoàn toàn ngoại tuyến trong 1.2 GB trên Android — xem demo trên YouTube"></a>
      <br><em>Tác nhân giọng nói hoàn toàn ngoại tuyến trong 1.2 GB trên Android — control-demo của speech-android</em>
    </td>
    <td align="center" width="50%">
      <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" alt="Nhân bản giọng nói với VoxCPM2 — demo Speech Studio trên YouTube"></a>
      <br><em>Nhân bản giọng nói với VoxCPM2 — demo Speech Studio</em>
    </td>
  </tr>
</table>

## Vì sao chọn speech-core

speech-core tách lớp điều phối nhỏ, độc lập với mô hình khỏi các backend suy luận tùy chọn. Core quản lý lượt nói, ngắt lời, tiện ích âm thanh, trạng thái hội thoại và gọi công cụ; ứng dụng tự chọn mô hình.

- **Ưu tiên cục bộ:** core C++17 thuần, buffer âm thanh Float, không phụ thuộc mạng hoặc audio API của nền tảng.
- **Dành cho tác tử trực tiếp:** lượt nói theo VAD, STT sớm, transcript từng phần, barge-in, TTS streaming và gọi công cụ.
- **ASR streaming thực sự:** RNN-T có cache, phát hiện cuối phát ngôn, beam search và thiên lệch cụm từ theo ngữ cảnh.
- **Tự chọn backend:** ONNX Runtime, LiteRT, cả hai, không dùng backend nào, hoặc tự cài đặt interface trừu tượng.
- **API di động:** C++ native và C API cho Kotlin/JNI, Swift/FFI, Linux nhúng và các host khác.
- **Kiểm thử nhiều mục tiêu:** Linux, Windows, macOS, build arm64 hướng Android, sanitizer và nightly dùng mô hình.

## Điểm mới trong v0.0.11

- **TTS local tương thích OpenAI:** `speech-server` cung cấp `POST /v1/audio/speech` với alias model OpenAI, giọng native/phổ thông, điều khiển ngôn ngữ và tốc độ, đầu ra WAV/PCM và Bearer authentication tùy chọn.
- **Gói Windows:** ZIP x64 độc lập gồm server, công cụ ONNX CLI, `speech.dll`, ONNX Runtime và trình tải model PowerShell; CI giải nén và smoke test gói này.
- **DeepFilterNet3 parity:** STFT scaling tương thích libdf, chuẩn hóa ERB/phức, deep filtering, overlap-add và bù trễ 480 mẫu khôi phục DSP tham chiếu.
- **Pocket TTS streaming:** backend ONNX phát frame cố định 80 ms, dùng decoder cache giới hạn và có kiểm thử round-trip với model tùy chọn.
- **Silero v5 context chính xác:** mỗi lần ONNX inference giờ nhận đủ 64 mẫu left context mà graph yêu cầu.

## Mô hình được hỗ trợ

| Mô hình | Tác vụ | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/vad) | Phát hiện hoạt động giọng nói | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/vi/guides/parakeet) | Giọng nói thành văn bản | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/whisper) | STT đa ngôn ngữ | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/nemotron) | STT streaming | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/vi/guides/nemotron) | STT streaming theo prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/vi/guides/dictate) | STT streaming + cuối phát ngôn | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/omnilingual) | STT đa ngôn ngữ | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/diarize) | Phân đoạn diarization | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/embed-speaker) | Embedding người nói | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 kHz + nhân bản giọng | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/voxcpm2) | TTS 48 kHz + nhân bản giọng | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/cosyvoice) | TTS có điều kiện 24 kHz | từng phần | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/chatterbox) | Tổng hợp giọng nói 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/supertonic) | Tổng hợp giọng nói | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/indic-mio) | Nhân bản giọng Hindi/Ấn Độ + cảm xúc | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/vi/guides/kokoro) | Tổng hợp giọng nói | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | TTS streaming (giọng Alba cố định) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/denoise) | Tăng cường giọng nói | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/restore) | Khử nhiễu + khử vang (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/vi/guides/respond) | Giọng nói hai chiều full-duplex (CUDA) | cấu trúc | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/vi/guides/function-calls) | Gọi công cụ có cấu trúc trên thiết bị | — | LiteRT-LM |

Xem [docs/models.md](docs/models.md) để biết độ hoàn thiện, layout bundle, tiền xử lý, bộ nhớ và ví dụ đầy đủ.

## Nền tảng và backend

| Backend | Target | Nền tảng | Thiết lập runtime |
|---|---|---|---|
| Chỉ core | `speech_core` | Linux, Windows, macOS, Android | không cần |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | bản ONNX Runtime đã giải nén qua `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, đường build Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX có thể dùng CPU, NNAPI Android, Qualcomm QNN trên Linux hoặc hook Execution Provider từ ứng dụng. LiteRT hiện dùng CPU qua C API.

## Bắt đầu nhanh

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Chuyển buffer âm thanh thành văn bản; Parakeet v3 tự nhận diện ngôn ngữ:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Kết nối bất kỳ VAD, STT, LLM và TTS nào với pipeline trực tiếp:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcript, âm thanh phản hồi, gọi công cụ hoặc lỗi
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Gói Linux CLI

Release có `.deb` và `.tar.gz` cho amd64 và arm64. Thư viện runtime được đóng gói, mô hình thì không.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 hoặc arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

Gói amd64 còn có lệnh nhân bản VoxCPM2 bằng LiteRT. Bundle x86 khoảng 13 GB:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

**[Tham chiếu Linux CLI](docs/cli.md)** có cú pháp, thư mục, binary và ma trận amd64/arm64. [`soniqo.audio/cli`](https://soniqo.audio/vi/cli) dành cho CLI speech-swift lớn hơn trên Apple.

## Kiến trúc

```text
audio / sự kiện ứng dụng
          │
          ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · phát hiện lượt nói   │
│ ngắt lời · công cụ · tiện ích audio  │
│ API trừu tượng VAD/STT/LLM/TTS       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ mô hình tham chiếu│ │ mô hình tham chiếu │
      └─────────────────┘ └─────────────────────┘
```

Target điều phối không phụ thuộc mô hình cụ thể. Đổi backend là lựa chọn khởi tạo và liên kết, không phải viết lại pipeline.

## Tài liệu

| Chủ đề | Tài liệu |
|---|---|
| Tổng quan và ma trận | [soniqo.audio/vi/speech-core](https://soniqo.audio/vi/speech-core) |
| Thiết lập Linux | [soniqo.audio/vi/getting-started/linux](https://soniqo.audio/vi/getting-started/linux) |
| Thiết lập Windows | [soniqo.audio/vi/getting-started/windows](https://soniqo.audio/vi/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Interface và backend riêng | [docs/interfaces.md](docs/interfaces.md) |
| Cài đặt mô hình | [docs/models.md](docs/models.md) |
| Pipeline và state machine | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Gọi công cụ | [docs/tools.md](docs/tools.md) |

## Biến thể build

```bash
# Chỉ điều phối, không ML runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Mô hình ONNX
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# Mô hình LiteRT
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Kiểm thử

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Test core không cần mô hình. Test tích hợp được bỏ qua an toàn khi thiếu biến thư mục. CI phủ Linux, Windows và macOS; workflow định kỳ kiểm tra bundle ONNX và LiteRT công khai.

## Dự án liên quan

- [speech-android](https://github.com/soniqo/speech-android) — Kotlin SDK và JNI trên speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — stack MLX/CoreML native cho macOS và iOS.
- [Tài liệu Soniqo](https://soniqo.audio/vi) — hướng dẫn, kiến trúc, benchmark và mô hình.

## Đóng góp

Chúng tôi hoan nghênh issue và pull request. Tạo nhánh từ `main`, build cấu hình bị ảnh hưởng, chạy `ctest` và mở PR tập trung.

## Giấy phép

Apache 2.0 — xem [LICENSE](LICENSE).
