# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

**Linux, Windows ve Android** için **C++17** ile cihaz üzerinde konuşma altyapısı: ses etkinliği algılama, toplu ve gerçek zamanlı akışlı konuşma tanıma, konuşmacı ayrıştırma, metinden konuşma ve bunları birleştiren sesli ajan hattı.

CPU üzerinde yerel çalışır. Çıkarımda bulut veya Python yoktur; ses cihazdan ayrılmaz.

**[📚 Tam belgeler →](https://soniqo.audio/tr/speech-core)** · **[🐧 Linux](https://soniqo.audio/tr/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/tr/getting-started/windows)** · **[⌨️ Masaüstü CLI](docs/cli.md)** · **[🔊 HTTP ses](docs/http-server.md)**

**[🤗 Modeller](https://huggingface.co/soniqo)** · **[🍎 Apple kardeş projesi](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Demo

<p align="center"><a href="https://www.youtube.com/watch?v=7L7_Uvvxtv0"><img src="https://img.youtube.com/vi/7L7_Uvvxtv0/maxresdefault.jpg" width="640" alt="Android'de 1.2 GB'a sığan tamamen çevrimdışı sesli ajan — demoyu YouTube'da izleyin"></a></p>
<p align="center"><em>Android'de 1.2 GB içinde tamamen çevrimdışı sesli ajan — speech-android control-demo</em></p>

## Neden speech-core

speech-core küçük, modelden bağımsız bir orkestrasyon katmanını isteğe bağlı çıkarım backend'lerinden ayırır. Çekirdek konuşma turlarını, kesintileri, ses araçlarını, konuşma durumunu ve araç çağrılarını yönetir; modelleri uygulama seçer.

- **Önce yerel:** saf C++17 çekirdeği, Float ses tamponları, ağ veya platform ses bağımlılığı yok.
- **Canlı ajanlar için:** VAD tabanlı turlar, erken STT, kısmi metinler, araya girme, akışlı TTS ve araç çağrıları.
- **Gerçek akışlı ASR:** önbellekli RNN-T, söz sonu algılama, beam search ve bağlamsal ifade biasing.
- **Backend seçimi:** ONNX Runtime, LiteRT, ikisi, hiçbiri veya soyut arayüzlerin kendi uygulamanız.
- **Taşınabilir API:** yerel C++ ve Kotlin/JNI, Swift/FFI, gömülü Linux için C API'leri.
- **Çok hedefli test:** Linux, Windows, macOS, Android odaklı arm64 build'leri, sanitizer'lar ve modelli nightly testleri.

## v0.0.11 yenilikleri

- **OpenAI uyumlu yerel TTS:** `speech-server`, OpenAI model takma adları, yerel/genel sesler, dil ve hız kontrolü, WAV/PCM çıkışı ve isteğe bağlı Bearer kimlik doğrulamasıyla `POST /v1/audio/speech` sunar.
- **Windows paketi:** sunucu, ONNX CLI araçları, `speech.dll`, ONNX Runtime ve PowerShell model indiricisini içeren bağımsız x64 ZIP; CI paketi açıp smoke test uygular.
- **DeepFilterNet3 eşliği:** libdf uyumlu STFT ölçekleme, ERB/kompleks normalizasyon, deep filtering, overlap-add ve 480 örnek gecikme telafisi referans DSP davranışını geri getirir.
- **Akışlı Pocket TTS:** ONNX backend sabit 80 ms kareler, sınırlı decoder cache ve isteğe bağlı model destekli round-trip doğrulama sunar.
- **Doğru Silero v5 bağlamı:** her ONNX inference artık graph'ın istediği 64 örneklik sol bağlamı alır.

## Desteklenen modeller

| Model | Görev | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/vad) | Ses etkinliği algılama | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/tr/guides/parakeet) | Konuşmadan metne | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/whisper) | Çok dilli konuşmadan metne | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/nemotron) | Akışlı STT | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/tr/guides/nemotron) | Prompt koşullu akışlı STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/tr/guides/dictate) | Akışlı STT + söz sonu | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/omnilingual) | Çok dilli STT | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/diarize) | Diarization segmentasyonu | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/embed-speaker) | Konuşmacı embedding'i | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + ses klonlama | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/voxcpm2) | 48 kHz TTS + ses klonlama | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/cosyvoice) | Koşullu 24 kHz TTS | aşamalı | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/chatterbox) | 24 kHz metinden konuşma | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/supertonic) | Metinden konuşma | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/indic-mio) | Hint dilleri ses klonlama + duygu | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/tr/guides/kokoro) | Metinden konuşma | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | Akışlı TTS (sabit Alba sesi) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/denoise) | Konuşma iyileştirme | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/restore) | Gürültü ve yankı giderme (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/tr/guides/respond) | Full-duplex konuşmadan konuşmaya (CUDA) | yapısal | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/tr/guides/function-calls) | Cihaz içi yapılandırılmış araç çağrıları | — | LiteRT-LM |

Olgunluk, bundle yapısı, ön işleme, bellek ve tam örnekler için [docs/models.md](docs/models.md) belgesine bakın.

## Platformlar ve backend'ler

| Backend | Target | Platformlar | Runtime kurulumu |
|---|---|---|---|
| Yalnız çekirdek | `speech_core` | Linux, Windows, macOS, Android | yok |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | açılmış ONNX Runtime, `ORT_DIR` ile |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, Android build yolu | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX CPU, Android NNAPI, Linux'ta Qualcomm QNN veya uygulamanın verdiği Execution Provider hook'unu kullanabilir. LiteRT şu anda C API üzerinden CPU kullanır.

## Hızlı başlangıç

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Ses tamponunu yazıya çevirin; Parakeet v3 dili otomatik algılar:

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Herhangi bir VAD, STT, LLM ve TTS uygulamasını canlı pipeline'a bağlayın:

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transkript, yanıt sesi, araç çağrısı veya hata
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI paketleri

Yayınlar amd64 ve arm64 için `.deb` ve `.tar.gz` içerir. Runtime kütüphaneleri dahildir, modeller değildir.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 veya arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

amd64 paketi LiteRT VoxCPM2 ses klonlama komutunu da içerir. x86 bundle yaklaşık 13 GB'dir:

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

Kesin sözdizimi, model dizinleri, bağımsız binary'ler ve amd64/arm64 matrisi **[Linux CLI referansında](docs/cli.md)** yer alır. [`soniqo.audio/cli`](https://soniqo.audio/tr/cli), Apple için daha geniş speech-swift CLI'ını açıklar.

## Mimari

```text
uygulama sesi / olaylar
          │
          ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · tur algılama         │
│ kesinti · araçlar · ses yardımcıları │
│ soyut VAD/STT/LLM/TTS API'leri       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ referans modeller│ │ referans modeller  │
      └─────────────────┘ └─────────────────────┘
```

Orkestrasyon target'ı somut bir modele bağımlı değildir. Backend değiştirmek construction ve link seçimidir; pipeline'ı yeniden yazmak değildir.

## Belgeler

| Konu | Belge |
|---|---|
| Genel bakış ve model matrisi | [soniqo.audio/tr/speech-core](https://soniqo.audio/tr/speech-core) |
| Linux kurulumu | [soniqo.audio/tr/getting-started/linux](https://soniqo.audio/tr/getting-started/linux) |
| Windows kurulumu | [soniqo.audio/tr/getting-started/windows](https://soniqo.audio/tr/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| Arayüzler ve özel backend'ler | [docs/interfaces.md](docs/interfaces.md) |
| Model uygulamaları | [docs/models.md](docs/models.md) |
| Ses pipeline'ı ve durum makinesi | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Araç çağırma | [docs/tools.md](docs/tools.md) |

## Build çeşitleri

```bash
# Yalnız orkestrasyon, ML runtime yok
cmake -B build -DCMAKE_BUILD_TYPE=Release
# ONNX modelleri
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# LiteRT modelleri
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Test

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Çekirdek testleri model istemez. Backend entegrasyonları dizin değişkenleri yoksa temizce atlanır. CI Linux, Windows ve macOS'u kapsar; zamanlanmış iş akışları genel ONNX ve LiteRT bundle'larını doğrular.

## İlgili projeler

- [speech-android](https://github.com/soniqo/speech-android) — speech-core üzerinde Kotlin SDK ve JNI.
- [speech-swift](https://github.com/soniqo/speech-swift) — macOS ve iOS için yerel MLX/CoreML konuşma yığını.
- [Soniqo belgeleri](https://soniqo.audio/tr) — rehberler, mimari, benchmark ve model sayfaları.

## Katkı

Issue ve pull request'ler memnuniyetle karşılanır. `main` üzerinden branch açın, etkilenen yapıları derleyin, `ctest` çalıştırın ve odaklı bir PR açın.

## Lisans

Apache 2.0 — [LICENSE](LICENSE) dosyasına bakın.
