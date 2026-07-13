# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

<div dir="rtl">

بنية تحتية صوتية على الجهاز بلغة **C++17** لأنظمة **Linux وWindows وAndroid**: اكتشاف النشاط الصوتي، وتحويل الكلام إلى نص دفعة واحدة أو بالبث الفوري، وفصل المتحدثين، وتحويل النص إلى كلام، وخط أنابيب الوكيل الصوتي الذي يربط هذه المكوّنات.

تعمل محلياً على المعالج. لا سحابة ولا Python أثناء الاستدلال، ولا يغادر الصوت الجهاز.

**[📚 الوثائق الكاملة ←](https://soniqo.audio/ar/speech-core)** · **[🐧 Linux](https://soniqo.audio/ar/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/ar/getting-started/windows)** · **[⌨️ واجهة Linux](docs/cli.md)**

**[🤗 النماذج](https://huggingface.co/soniqo)** · **[🍎 مشروع Apple الشقيق](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## العرض التوضيحي

</div>

<p align="center"><a href="https://www.youtube.com/watch?v=EuIU8tOWyzg"><img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="استنساخ الصوت باستخدام VoxCPM2 — عرض Speech Studio على YouTube"></a></p>
<p align="center"><em>استنساخ الصوت باستخدام VoxCPM2 — عرض Speech Studio على YouTube</em></p>

<div dir="rtl">

## لماذا speech-core

يفصل speech-core طبقة تنسيق صغيرة مستقلة عن النموذج عن خلفيات الاستدلال الاختيارية. تدير النواة اكتشاف الأدوار والمقاطعات وأدوات الصوت وحالة المحادثة واستدعاءات الأدوات؛ ويختار التطبيق النماذج.

- **محلي أولاً:** نواة C++17 خالصة، ومخازن صوت Float، بلا اعتماد على الشبكة أو واجهات الصوت الخاصة بالمنصة.
- **مصمم للوكلاء المباشرين:** أدوار يقودها VAD، وSTT مبكر، ونتائج جزئية، ومقاطعة فورية، وTTS متدفق، واستدعاء أدوات.
- **ASR متدفق فعلياً:** مفككات RNN-T بذاكرة مخبأة، واكتشاف نهاية القول، وbeam search، وترجيح العبارات السياقية.
- **اختيار الخلفية:** ONNX Runtime أو LiteRT أو كلاهما أو لا شيء، أو تنفيذ الواجهات المجردة بنفسك.
- **واجهة قابلة للنقل:** C++ أصلية وواجهات C مناسبة لـ Kotlin/JNI وSwift/FFI وLinux المضمن.
- **اختبارات متعددة الأهداف:** Linux وWindows وmacOS وبناء arm64 الموجّه إلى Android وsanitizers واختبارات ليلية بالنماذج.

## أبرز ما في v0.0.10

- **Parakeet-EOU 120M:** ASR متدفق متعدد اللغات قليل الذاكرة، مع رموز نهاية القول وbeam search اختياري وترجيح سياقي وحد أعلى لمنع الإفراط في الترجيح.
- **Whisper ONNX أصلي:** من small إلى large-v3/turbo، مع اكتشاف اللغة أو prompt ثابت، وprofiling وضبط CPU.
- **TTS أوسع:** VoxCPM/VoxCPM2 وCosyVoice3 وChatterbox وSupertonic وIndic-Mio إلى جانب Kokoro، مع معالجة لاحقة مخزنة واستنساخ موجّه بالنص المرجعي.
- **محادثات أسرع:** تحسين Kokoro للأدوار القصيرة، وتقسيم النص الطويل إلى جمل، واستمرار مخزن ما قبل الكلام حول التشغيل.
- **أدوات LLM على الجهاز:** FunctionGemma عبر LiteRT-LM، ومحوّل Ollama، وحلقة الأدوات داخل خط الأنابيب.
- **CLI لـLinux بجودة إصدار:** حزم amd64/arm64، ومساعدات تنزيل النماذج، وتوافر الأوامر حسب المعمارية، واختبارات في حاويات نظيفة.

## النماذج المدعومة

| النموذج | المهمة | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/vad) | اكتشاف النشاط الصوتي | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/ar/guides/parakeet) | تحويل الكلام إلى نص | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/whisper) | تعرف متعدد اللغات | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/nemotron) | تعرف متدفق | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/ar/guides/nemotron) | STT متدفق مشروط بـprompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/ar/guides/dictate) | STT متدفق + نهاية القول | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/omnilingual) | تعرف متعدد اللغات | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/diarize) | تقسيم لفصل المتحدثين | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/embed-speaker) | تضمين المتحدث | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS ‏16 kHz + استنساخ | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/voxcpm2) | TTS ‏48 kHz + استنساخ | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/cosyvoice) | TTS مشروط 24 kHz | مرحلي | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/chatterbox) | تحويل النص إلى كلام 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/supertonic) | تحويل النص إلى كلام | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/indic-mio) | استنساخ هندي/لغات الهند + عاطفة | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/ar/guides/kokoro) | تحويل النص إلى كلام | ✓ | ✓ |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/denoise) | تحسين الكلام | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/sidon) | إزالة الضوضاء والصدى (16 ← 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/ar/guides/respond) | كلام إلى كلام مزدوج الاتجاه (CUDA) | هيكلي | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/ar/guides/functiongemma) | استدعاءات أدوات منظمة على الجهاز | — | LiteRT-LM |

راجع [docs/models.md](docs/models.md) للنضج وبنية الحزم والمعالجة المسبقة والذاكرة والأمثلة الكاملة.

## المنصات والخلفيات

| الخلفية | الهدف | المنصات | إعداد runtime |
|---|---|---|---|
| النواة فقط | `speech_core` | Linux وWindows وmacOS وAndroid | لا شيء |
| ONNX Runtime | `speech_core_models` | Linux وWindows وmacOS وAndroid | إصدار ONNX Runtime مفكوك عبر `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64 وWindows x86_64 وmacOS arm64 وAndroid | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS ومسار بناء Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

يمكن لـONNX استخدام CPU أو NNAPI على Android أو Qualcomm QNN على Linux أو hook لمزوّد تنفيذ يقدمه التطبيق. يستخدم LiteRT حالياً CPU عبر C API.

## البدء السريع

</div>

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

<div dir="rtl">

حوّل مخزن صوت إلى نص؛ يكتشف Parakeet v3 اللغة تلقائياً:

</div>

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

<div dir="rtl">

صل أي تنفيذ لـVAD وSTT وLLM وTTS بخط الأنابيب المباشر:

</div>

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcript, response audio, tool call, or error
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

<div dir="rtl">

## حزم CLI لنظام Linux

تتضمن الإصدارات حزم `.deb` و`.tar.gz` لـamd64 وarm64. مكتبات التشغيل مضمّنة، أما النماذج فلا.

</div>

```bash
VERSION=0.0.10
ARCH="$(dpkg --print-architecture)"   # amd64 or arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
```

<div dir="rtl">

تتضمن حزمة amd64 أيضاً استنساخ VoxCPM2 عبر LiteRT. يبلغ حجم حزمة x86 نحو 13 GB:

</div>

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

<div dir="rtl">

يوضح **[مرجع Linux CLI](docs/cli.md)** الصياغة والمجلدات والبرامج ومصفوفة amd64/arm64. أما [`soniqo.audio/cli`](https://soniqo.audio/ar/cli) فيوثق واجهة speech-swift الأوسع لمنصات Apple.

## المعمارية

</div>

```text
application audio / events
            │
            ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · turn detection       │
│ interruption · tools · audio utils   │
│ abstract VAD / STT / LLM / TTS APIs  │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ reference models│ │ reference models   │
      └─────────────────┘ └─────────────────────┘
```

<div dir="rtl">

لا يعتمد هدف التنسيق على نموذج محدد. تبديل الخلفية هو اختيار بناء وربط، وليس إعادة كتابة لخط الأنابيب.

## الوثائق

| الموضوع | الوثائق |
|---|---|
| النظرة العامة ومصفوفة النماذج | [soniqo.audio/ar/speech-core](https://soniqo.audio/ar/speech-core) |
| إعداد Linux | [soniqo.audio/ar/getting-started/linux](https://soniqo.audio/ar/getting-started/linux) |
| إعداد Windows | [soniqo.audio/ar/getting-started/windows](https://soniqo.audio/ar/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| الواجهات والخلفيات المخصصة | [docs/interfaces.md](docs/interfaces.md) |
| تنفيذات النماذج | [docs/models.md](docs/models.md) |
| خط الصوت وآلة الحالات | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| استدعاء الأدوات | [docs/tools.md](docs/tools.md) |

## خيارات البناء

</div>

```bash
# Orchestration only; no ML runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
# ONNX models
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# LiteRT models
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

<div dir="rtl">

## الاختبار

</div>

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

<div dir="rtl">

لا تحتاج اختبارات النواة إلى نماذج. تُتخطى اختبارات الخلفيات بشكل سليم عند غياب متغيرات مجلدات النماذج. يغطي CI أنظمة Linux وWindows وmacOS، وتتحقق المهام المجدولة من حزم ONNX وLiteRT العامة.

## مشاريع ذات صلة

- [speech-android](https://github.com/soniqo/speech-android) — حزمة Kotlin وتكامل JNI فوق speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — بنية MLX/CoreML أصلية لـmacOS وiOS.
- [وثائق Soniqo](https://soniqo.audio/ar) — أدلة ومعمارية واختبارات أداء وصفحات نماذج.

## المساهمة

نرحب بالمشكلات وطلبات السحب. أنشئ فرعاً من `main`، وابنِ الإعدادات المتأثرة، وشغّل `ctest`، ثم افتح PR محدد النطاق.

## الترخيص

Apache 2.0 — راجع [LICENSE](LICENSE).

</div>
