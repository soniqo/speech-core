# Reference Model Implementations

speech-core ships two parallel sets of model wrappers under `include/speech_core/models/`. Each implements one of the interfaces in `interfaces.h` and is compiled in via a backend-specific CMake flag. The two backends can be enabled simultaneously — consumers link only the targets they need.

### ONNX Runtime backend (`SPEECH_CORE_WITH_ONNX`)

| Model | Interface | Header |
|---|---|---|
| `SileroVad` | `VADInterface` | `speech_core/models/silero_vad.h` |
| `ParakeetStt` | `STTInterface` | `speech_core/models/parakeet_stt.h` |
| `KokoroTts` | `TTSInterface` | `speech_core/models/kokoro_tts.h` |
| `DeepFilterEnhancer` | `EnhancerInterface` | `speech_core/models/deepfilter.h` |

### LiteRT (TFLite) backend (`SPEECH_CORE_WITH_LITERT`)

| Model | Interface | Header |
|---|---|---|
| `LiteRTSileroVad` | `VADInterface` | `speech_core/models/litert_silero_vad.h` |
| `LiteRTParakeetStt` | `STTInterface` | `speech_core/models/litert_parakeet_stt.h` |

Kokoro 82M and DeepFilterNet3 do not yet have LiteRT exports — see `speech-models` for conversion status. When they land, wrappers will be added alongside the existing two.

All ORT wrappers share an internal ONNX Runtime singleton (`OnnxEngine` in `speech_core/models/onnx_engine.h`) that owns the `OrtEnv` and `OrtMemoryInfo`. All LiteRT wrappers share `LiteRTEngine` (`speech_core/models/litert_engine.h`) which currently configures CPU-only inference with a configurable thread count. NNAPI / GPU / Hexagon delegates are not yet wired through the C API in this version.

## Building with ONNX support

```bash
cmake -S . -B build \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DORT_DIR=/path/to/onnxruntime
cmake --build build
```

`ORT_DIR` must contain `include/onnxruntime_c_api.h` and a platform-appropriate shared library:

| Platform | Path |
|---|---|
| macOS | `lib/libonnxruntime.dylib` |
| Linux | `lib/libonnxruntime.so` |
| Android | `lib/${ANDROID_ABI}/libonnxruntime.so` |

Hardware-accelerated execution providers are picked automatically: NNAPI on Android, QNN on non-Android (if available), CPU fallback otherwise.

## Building with LiteRT support

```bash
cmake -S . -B build \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR=/path/to/litert
cmake --build build
```

`LITERT_DIR` must contain the TFLite C API headers and a platform-appropriate shared library:

| Platform | Header | Library |
|---|---|---|
| macOS | `include/tensorflow/lite/c/c_api.h` | `lib/libtensorflowlite_c.dylib` |
| Linux | `include/tensorflow/lite/c/c_api.h` | `lib/libtensorflowlite_c.so` |
| Android | `include/tensorflow/lite/c/c_api.h` | `lib/${ANDROID_ABI}/libtensorflowlite_c.so` |

If you don't have a TFLite C library handy, build one from TensorFlow source:

```bash
scripts/setup_litert.sh build/litert v2.18.0
# → build/litert/{include,lib} ready for use as LITERT_DIR
```

`SPEECH_CORE_WITH_ONNX` and `SPEECH_CORE_WITH_LITERT` are independent — enable either, both, or neither. Both flags produce separate static libraries (`speech_core_models`, `speech_core_models_litert`); consumers link only what they use.

## SileroVad

```cpp
#include <speech_core/models/silero_vad.h>

speech_core::SileroVad vad("/path/to/silero-vad.onnx");
float prob = vad.process_chunk(samples_512, 512);  // → [0, 1]
```

- 512 samples per chunk (32 ms @ 16 kHz)
- LSTM state carried across chunks; `reset()` clears it between sessions
- Returns speech probability; feed to `StreamingVAD` for start/end events
- Model files: [aufklarer/Silero-VAD-v5-ONNX](https://huggingface.co/aufklarer/Silero-VAD-v5-ONNX) — `silero-vad.onnx` (~2 MB)

## ParakeetStt

```cpp
#include <speech_core/models/parakeet_stt.h>

speech_core::ParakeetStt stt(
    "/models/parakeet_encoder.onnx",
    "/models/parakeet_decoder_joint.onnx",
    "/models/parakeet_vocab.json");

auto result = stt.transcribe(audio, length, 16000);
// result.text, result.language, result.confidence
```

- Parakeet TDT v3 (0.6B params), NeMo-exported as encoder + decoder_joint
- 128-bin mel spectrogram preprocessing
- Greedy TDT decoding with per-frame duration prediction
- Language detection via `<|xx|>` BPE tokens
- Streaming supported via `begin_stream` / `push_chunk` / `end_stream` (accumulates audio and re-transcribes each chunk; not a true streaming decoder)
- Model files: [aufklarer/Parakeet-TDT-v3-ONNX](https://huggingface.co/aufklarer/Parakeet-TDT-v3-ONNX) — `parakeet-encoder.onnx` (FP32, plus external `.onnx.data`) or `parakeet-encoder-int8.onnx` (~840 MB / ~100 MB INT8), `parakeet-decoder-joint.onnx` / `parakeet-decoder-joint-int8.onnx`, `vocab.json`

## LiteRTSileroVad

```cpp
#include <speech_core/models/litert_silero_vad.h>

speech_core::LiteRTSileroVad vad("/path/to/silero-vad.tflite");
float prob = vad.process_chunk(samples_512, 512);
```

- Same public contract as `SileroVad` (512-sample chunks, 16 kHz, [0,1] probability)
- The LiteRT model itself takes `[1, 576]` (64 left-context + 512 chunk); the wrapper hides the context buffer so callers see the same `VADInterface`
- Model files: [soniqo/Silero-VAD-v5-LiteRT](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) — `silero-vad.tflite` (~1.3 MB)

## LiteRTParakeetStt

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "/models/parakeet-encoder.tflite",
    "/models/parakeet-decoder-joint.tflite",
    "/models/vocab.json");

auto result = stt.transcribe(audio, length, 16000);
```

- Same public contract and TDT decode loop as `ParakeetStt`
- Encoder INT8 weight-quantized (~595 MB on disk vs ~840 MB ONNX FP32), decoder-joint stays FP32 to avoid LSTM drift
- Decoder-joint exposes `(encoder_out, target, h, c)` as four discrete tensors (ORT bundles `target_length` and uses suffix-`_1`/`_2` for h/c)
- Model files: [soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) — `parakeet-encoder.tflite`, `parakeet-decoder-joint.tflite`, `vocab.json`

## KokoroTts

```cpp
#include <speech_core/models/kokoro_tts.h>

speech_core::KokoroTts tts(
    "/models/kokoro.onnx",
    "/models/kokoro_voices",   // directory of .bin voice embeddings
    "/models/kokoro_data");    // directory of vocab + dictionaries

tts.synthesize("Hello world", "en",
    [](const float* samples, size_t len, bool is_final) {
        // append to playback buffer
    });
```

- Kokoro 82M, non-autoregressive single-pass synthesis
- 24 kHz Float32 output
- Single chunk per call (E2E model, not streaming-capable)
- Auto-switches voice on language change (en → af_heart, fr → ff_siwis, …)
- Phonemizer: GPL-free three-tier (dict + suffix stemming + rule-based G2P), no eSpeak dependency. See `kokoro_phonemizer.h` + `kokoro_multilingual.h`.
- Output post-processing: peak-clip detection (drops numerically unstable short prompts), trailing-silence trim, 5 ms fade-in / 10 ms fade-out at the speech boundary
- Model files: [aufklarer/Kokoro-82M-ONNX](https://huggingface.co/aufklarer/Kokoro-82M-ONNX) — `kokoro-e2e.onnx` + `kokoro-e2e.onnx.data` (~90 MB total), `vocab_index.json`, `us_gold.json`, `us_silver.json`, `dict_{fr,es,it,pt,hi}.json`, `voices/*.bin`

### Voice files

Voice embeddings are 256-float `.bin` files in `voices_dir`. Default voice is `af_heart`. Per-language defaults:

| Language | Voice |
|---|---|
| `en` | af_heart |
| `fr` | ff_siwis |
| `es` | ef_dora |
| `it` | if_sara |
| `pt` | pf_dora |
| `hi` | hf_alpha |
| `ja` | jf_alpha |
| `zh` | zf_xiaobei |
| `ko` | kf_somi |

### Data directory

`data_dir` must contain:

- `vocab_index.json` — IPA symbol → token ID map
- `us_gold.json`, `us_silver.json` — English pronunciation dictionaries (from misaki)
- `dict_<lang>.json` — optional per-language pronunciation dicts (fr, es, it, pt, hi)

## DeepFilterEnhancer

```cpp
#include <speech_core/models/deepfilter.h>

speech_core::DeepFilterEnhancer enh(
    "/models/deepfilter.onnx",
    "/models/deepfilter_aux.bin");

std::vector<float> clean(audio.size());
enh.enhance(audio.data(), audio.size(), 48000, clean.data());
```

- DeepFilterNet3 — ~2.1M params, real-time speech enhancement
- 48 kHz input (caller must resample if needed)
- STFT (960/480) → ERB filterbank → neural mask + deep filter coefficients → inverse STFT
- Auxiliary binary file holds precomputed ERB filterbanks and Vorbis window: `erb_fb [481*32] | erb_inv_fb [32*481] | window [960]` (float32)
- Model files: [aufklarer/DeepFilterNet3-ONNX](https://huggingface.co/aufklarer/DeepFilterNet3-ONNX) — `deepfilter.onnx` (~8 MB FP16), `deepfilter-auxiliary.bin`

## Testing

A `test_models` target is added when `SPEECH_CORE_WITH_ONNX=ON`. It loads each of the four model wrappers against real ONNX files and runs a smoke check (silence-vs-tone for VAD, silence transcribe for STT, "hello world" synth for TTS, noise enhancement for the enhancer).

```bash
# 1. Download model files (~1.2 GB total)
scripts/download_models.sh

# 2. Build with ONNX support
cmake -B build -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/ort
cmake --build build

# 3. Run
SPEECH_MODEL_DIR=scripts/models ctest --test-dir build --output-on-failure
```

`test_models` skips cleanly with exit code 0 when `SPEECH_MODEL_DIR` is unset or model files are missing — CI without model artifacts stays green.

A separate `test_litert_models` target is added when `SPEECH_CORE_WITH_LITERT=ON`, exercising the two LiteRT wrappers against `.tflite` artifacts:

```bash
scripts/download_models_litert.sh
scripts/setup_litert.sh                # builds libtensorflowlite_c locally
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert
cmake --build build
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

## Bring-your-own model

The interface contract is small. To use a non-ONNX backend (CoreML, MLX, Whisper.cpp, llama.cpp, remote API), inherit the relevant interface and pass an instance to `VoicePipeline`. The reference implementations live alongside the orchestration but are not required.

See speech-swift for a worked example with CoreML and MLX backends targeting the same conceptual interfaces.
