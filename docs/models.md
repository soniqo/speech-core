# Reference Model Implementations

speech-core ships two parallel sets of model wrappers under `include/speech_core/models/`. Each implements one of the interfaces in `interfaces.h` and is compiled in via a backend-specific CMake flag. The two backends can be enabled simultaneously — consumers link only the targets they need.

### ONNX Runtime backend (`SPEECH_CORE_WITH_ONNX`)

| Model | Interface | Header | Status |
|---|---|---|---|
| `SileroVad` | `VADInterface` | `speech_core/models/silero_vad.h` | full |
| `ParakeetStt` | `STTInterface` | `speech_core/models/parakeet_stt.h` | full |
| `KokoroTts` | `TTSInterface` | `speech_core/models/kokoro_tts.h` | full |
| `DeepFilterEnhancer` | `EnhancerInterface` | `speech_core/models/deepfilter.h` | full |
| `OnnxSidonRestorer` | (own API; `EnhancerInterface` adapter) | `speech_core/models/onnx_sidon_restorer.h` | full — see [OnnxSidonRestorer](#onnxsidonrestorer) |
| `OnnxVoxCPM2Tts` | `TTSInterface` | `speech_core/models/onnx_voxcpm2_tts.h` | full |
| `OnnxNemotronStreamingStt` | `STTInterface` | `speech_core/models/onnx_nemotron_streaming_stt.h` | full (streaming) |
| `OnnxPersonaPlex` | `FullDuplexSpeechInterface` | `speech_core/models/onnx_personaplex.h` | structural — see [OnnxPersonaPlex](#onnxpersonaplex) |

### LiteRT backend (`SPEECH_CORE_WITH_LITERT`)

| Model | Interface | Header | Status |
|---|---|---|---|
| `LiteRTSileroVad` | `VADInterface` | `speech_core/models/litert_silero_vad.h` | full |
| `LiteRTParakeetStt` | `STTInterface` | `speech_core/models/litert_parakeet_stt.h` | full |
| `LiteRTVoxCPM2Tts` | `TTSInterface` | `speech_core/models/litert_voxcpm2_tts.h` | full (text-only) |
| `LiteRTWeSpeakerEmbedding` | `EmbeddingInterface` | `speech_core/models/litert_wespeaker_embedding.h` | full |
| `LiteRTPyannoteSegmentation` | `SegmentationInterface` | `speech_core/models/litert_pyannote_segmentation.h` | full |
| `LiteRTOmnilingualStt` | `STTInterface` | `speech_core/models/litert_omnilingual_stt.h` | full |
| `LiteRTNemotronStreamingStt` | `STTInterface` | `speech_core/models/litert_nemotron_streaming_stt.h` | full (streaming) |

### LLM backends (`LLMInterface`)

speech-core provides the abstract `LLMInterface` (`interfaces.h`) plus a tool
registry (`tools/tool_registry.h`) and the pipeline integration that routes
LLM-emitted tool calls through `ToolExecutor`. Three reference implementations
ship today.

| Implementation | Header | Backend | Built when |
|---|---|---|---|
| `OllamaLLM` | `speech_core/llm/ollama_llm.h` | Local Ollama HTTP server (`/api/chat`) | `SPEECH_CORE_WITH_OLLAMA=ON` (in `speech_core_llm_ollama`) |
| `LiteRTFunctionGemmaLLM` | `speech_core/models/litert_functiongemma_llm.h` | Google's `liblitert-lm` runtime driving a `.litertlm` bundle (e.g. [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM)) | `SPEECH_CORE_WITH_LITERT_LM=ON` (in `speech_core_models_litert`) |
| **FunctionGemma 270M (CoreML)** | (platform) | Core ML on Apple via [speech-swift](https://github.com/soniqo/speech-swift) | platform-side |

`LiteRTFunctionGemmaLLM` loads `.litertlm` bundles via the higher-level
`liblitert-lm` shared library, NOT through the lower-level `libLiteRt` C API
that drives the `.tflite` models in this directory. Extract the runtime with
`scripts/fetch_litert_lm.sh` (mirrors `scripts/fetch_litert.sh` — pulls the
shared library out of the `litert-lm-api` PyPI wheel) and point CMake at it
with `-DSPEECH_CORE_WITH_LITERT_LM=ON -DLITERT_LM_DIR=...`. The class
implements `LLMInterface` directly, so it plugs into `VoicePipeline`
identically to `OllamaLLM`.

To wire your own on-device LLM into speech-core, subclass `LLMInterface`,
implement `set_tools()` (convert `ToolDefinition[]` into the model's prompt
format) and `chat()` (run prefill+decode, parse tool-call markers, populate
`LLMResponse.tool_calls`), then pass an instance to `VoicePipeline`. See
`OllamaLLM` or `LiteRTFunctionGemmaLLM` for reference shapes.

`DiarizationPipeline` (`speech_core/diarization/diarization_pipeline.h`, implements `DiarizerInterface`) composes a segmenter + embedder + constrained clustering. It is pure C++ and ships in the **core** library (built always, no LiteRT dependency); pair it with the LiteRT segmenter + embedder above.

Kokoro 82M and DeepFilterNet3 do not yet have LiteRT exports — see `speech-models` for conversion status. When they land, wrappers will be added alongside the existing two.

`LiteRTVoxCPM2Tts` runs the full 4-graph orchestration end-to-end: `text_prefill → token_step ×N → audio_decode` with explicit K/V cache handoff every step. Voice cloning via the `audio_encoder` is supported by the graph but not yet surfaced through `TTSInterface` — `synthesize()` always feeds zero audio_feats today; adding a `set_reference_audio()` method is a follow-up. The bundle is large (~8.7 GB fp16 `selective` for ARM at the repo root; ~13 GB fp32-token-step for x86 in the `fp32-p16/` subdir) and inference is slow on CPU, so end-to-end validation runs in the **weekly** workflow (`.github/workflows/weekly-voxcpm2.yml`) rather than the daily nightly.

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

The LiteRT backend uses Google's `ai-edge-litert` runtime (the modern LiteRT successor to the legacy TFLite C API — handles >2 GB models, ships prebuilt for all our target platforms).

Headers are vendored in `third_party/litert/` (no setup needed). The shared library `libLiteRt.{so,dylib,dll}` is extracted from the `ai-edge-litert` PyPI wheel:

```bash
scripts/fetch_litert.sh build/litert       # PYTHON=python3.11 if 'python3' is older
cmake -S . -B build \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR=$PWD/build/litert
cmake --build build
```

`LITERT_DIR` points at a directory containing the runtime library:

| Platform | File |
|---|---|
| macOS | `libLiteRt.dylib` |
| Linux | `libLiteRt.so` |
| Windows | `LiteRt.dll` + `LiteRt.lib` |
| Android | `${ANDROID_ABI}/libLiteRt.so` |

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
- Model files: [soniqo/Silero-VAD-v5-ONNX](https://huggingface.co/soniqo/Silero-VAD-v5-ONNX) — `silero-vad.onnx` (~2 MB)

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
- Model files: [soniqo/Parakeet-TDT-0.6B-ONNX](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-ONNX) — `parakeet-encoder.onnx` (FP32, plus external `.onnx.data`) or `parakeet-encoder-int8.onnx` (~840 MB / ~100 MB INT8), `parakeet-decoder-joint.onnx` / `parakeet-decoder-joint-int8.onnx`, `vocab.json`. Decoder-joint inputs `targets` + `target_length` are INT32; encoder length input stays INT64.

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

## LiteRTVoxCPM2Tts

```cpp
#include <speech_core/models/litert_voxcpm2_tts.h>

speech_core::LiteRTVoxCPM2Tts tts(
    "/models/voxcpm2-text-prefill.tflite",
    "/models/voxcpm2-token-step.tflite",
    "/models/voxcpm2-audio-encoder.tflite",
    "/models/voxcpm2-audio-decoder.tflite",
    "/models/tokenizer.json");

tts.synthesize("Hello world", "en", [](const float* samples, size_t length, bool is_final) {
    // 48 kHz Float32 PCM, streamed in 64-step chunks (10.24 s each).
    // is_final marks the last chunk of the utterance.
});
```

- 2B-parameter multilingual TTS, 48 kHz studio-quality output. Voice cloning and instruction-driven voice design supported by the upstream model.
- Ships as **four** LiteRT graphs plus an HF BPE tokenizer:
  - `text-prefill`: text + (optional) reference-audio prefix → LM hidden + initial K/V cache
  - `token-step`: one autoregressive step (called up to 2048 times per generation), consumes and emits the K/V cache explicitly
  - `audio-encoder`: 16 kHz PCM reference clip → conditioning features
  - `audio-decoder`: latent → 48 kHz PCM output
- **Constructor** loads all four graphs via `LiteRTEngine` and verifies the tokenizer file exists; `synthesize()` runs the full pipeline (text-prefill → token-step ×N → audio-decoder) with the hand-rolled BPE tokenizer in [`voxcpm2_tokenizer.h`](../include/speech_core/models/voxcpm2_tokenizer.h). Reference-audio voice cloning isn't yet surfaced through `TTSInterface` (see the paragraph above).
- **Precision variants**: the repo *root* holds the `selective` bundle — fp16 weights except the **fp32 LocDiT** diffusion estimator, which needs full precision for clean cloned-voice sibilants (~8.7 GB; the ARM default). On **x86_64** the `fp32-p16/` subdir holds an **fp32 token-step** bundle (fp16 prefill, fp32 token-step + audio, ~13 GB): the fp16 token-step *over-generates* on x86 — its stop-margin (`stop_logits[1] > stop_logits[0]`) rounds the wrong way under x86 XNNPACK so the stop token never fires and the AR loop runs to the cap; the fp32 token-step computes that margin precisely and stops cleanly. `sc_voxcpm2_create_from_pretrained` picks the variant by architecture automatically. Download with `scripts/download_voxcpm2_litert.sh` (arch-aware); kept out of `download_models_litert.sh` because the bundle blows the standard nightly's `actions/cache` budget.
- Model files: [soniqo/VoxCPM2-LiteRT](https://huggingface.co/soniqo/VoxCPM2-LiteRT) — root (ARM / `selective`) and the `fp32-p16/` subdir (x86), each with `voxcpm2-{text-prefill,token-step,audio-encoder,audio-decoder}.tflite`, `tokenizer.json`, `config.json`

## LiteRTWeSpeakerEmbedding

```cpp
#include <speech_core/models/litert_wespeaker_embedding.h>

speech_core::LiteRTWeSpeakerEmbedding emb("/models/wespeaker-resnet34.tflite");
auto vec = emb.embed(audio, length, 16000);   // 256-float L2-normalised
```

- WeSpeaker ResNet34-LM. Computes a kaldi-style 80-bin log-mel fbank (25 ms / 10 ms, Hamming, per-frame DC removal) from raw 16 kHz audio internally; pads/tiles to the fixed 298-frame (~3 s) input.
- Model files: [soniqo/WeSpeaker-ResNet34-LM-LiteRT](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) — `wespeaker-resnet34.tflite`

## LiteRTPyannoteSegmentation

```cpp
#include <speech_core/models/litert_pyannote_segmentation.h>

speech_core::LiteRTPyannoteSegmentation seg("/models/pyannote-segmentation.tflite");
auto windows = seg.segment(audio, length, 16000);   // per-10 s window posteriors + speaker_activity
```

- Pyannote Segmentation 3.0, streaming mode: 1-s chunks with LSTM state carried across a 10-s window (reset per window, slide by 5 s). Powerset-decodes 7 classes → per-speaker activity for up to 3 local speakers.
- Model files: [soniqo/Pyannote-Segmentation-LiteRT](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) — `pyannote-segmentation.tflite`

## LiteRTOmnilingualStt

```cpp
#include <speech_core/models/litert_omnilingual_stt.h>

speech_core::LiteRTOmnilingualStt stt("/models/omnilingual-ctc-300m.tflite",
                                      "/models/tokenizer.model");
auto result = stt.transcribe(audio, length, 16000);
```

- Meta Omnilingual ASR CTC-300M. Single-model CTC: z-score-normalised waveform → logits @ 50 Hz; greedy CTC decode (collapse repeats + blanks) with a minimal SentencePiece `.model` tokenizer. Fixed chunk length + vocab size are read from the model output layout at load.
- Model files: [soniqo/Omnilingual-ASR-CTC-300M-LiteRT](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) — `omnilingual-ctc-300m.tflite`, `tokenizer.model`

## LiteRTNemotronStreamingStt

```cpp
#include <speech_core/models/litert_nemotron_streaming_stt.h>

speech_core::LiteRTNemotronStreamingStt stt(
    "/models/nemotron-streaming-encoder.tflite",
    "/models/nemotron-streaming-decoder.tflite",
    "/models/nemotron-streaming-joint.tflite",
    "/models/vocab.json");

stt.begin_stream(16000);
auto partial = stt.push_chunk(audio_chunk, chunk_len);   // partial.text grows as windows fill
auto final   = stt.end_stream();
```

- Nemotron Speech Streaming 0.6B — **true** cache-aware streaming RNN-T (three graphs: encoder-with-cache, decoder LSTM, joint). `push_chunk` drains fixed ~80 ms windows, advancing the encoder cache + decoder state across calls and greedily decoding the first encoder frame. One instance == one stream.
- Config defaults match the export; vocab size auto-derives from `vocab.json`.
- Model files: [soniqo/Nemotron-Speech-Streaming-LiteRT](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) — `nemotron-streaming-{encoder,decoder,joint}.tflite`, `vocab.json`, `config.json`

## OnnxPersonaPlex

> **Status: shipped on HuggingFace.** All four ONNX graphs exported, parity-verified, quantized through multiple recipes. Four production bundle variants live at [**soniqo/PersonaPlex-7B-ONNX**](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX): `fp16`, `mixed`, `int8-nb-dep_gint8` ⭐ (recommended ship default), and `int4-nb-dep_gint8`. All run end-to-end with voice prompt + system prompt + silence spacer + embedding prefix prefill, producing semantically appropriate English responses to real user audio.

### Quick start — the customer-service round-trip

```bash
# Build (Windows, ORT 1.26):
cmake -B build-ort -DSPEECH_CORE_WITH_ONNX=ON \
    -DORT_DIR=path/to/onnxruntime-win-x64-1.26.0
cmake --build build-ort --config Release --target run_personaplex

# Download the recommended ship default bundle from HF (9.4 GB):
scripts/download_personaplex_onnx.sh int8-nb-dep_gint8

# Run on the speech-swift test fixture ("Can you guarantee the replacement
# will be shipped tomorrow?"):
SPEECH_CORE_PARAKEET_DIR=scripts/models \
SPEECH_CORE_PP_PROMPT=helpful \
run_personaplex.exe scripts/personaplex-int8-nb-dep_gint8 50 \
    tests/data/test_audio.wav VARF2

# Expected: Parakeet transcribes the agent audio as coherent English.
```

### Configuration env vars

| Env | Default | Effect |
|---|---|---|
| `SPEECH_CORE_PP_PROMPT` | `helpful` | System prompt name: `helpful` / `expert` / `warm` / `direct` (pre-tokenized in `system_prompts.bin`) |
| `SPEECH_CORE_PP_EMB_SCALE` | `10` | Voice embedding prefix scale factor; `0` disables the embedding prefix |
| `SPEECH_CORE_PP_GPU_KV` | on | Keep KV cache OrtValues GPU-resident across calls (auto-on when CUDA EP resolves); `0` falls back to host mirror |
| `SPEECH_MODEL_DIR` | — | Where `test_models` looks for the bundle (see test harness section) |

NVIDIA's PersonaPlex 7B — a full-duplex speech-to-speech model on Kyutai's Moshi architecture. Listens and speaks simultaneously at 12.5 Hz, conditioned on a voice preset and text system prompt. Already shipped in [speech-swift](https://github.com/) as native Swift/MLX (8-bit / 4-bit). This ONNX path targets CUDA on Linux/Windows servers via `SPEECH_CORE_WITH_ONNX=ON`.

### Architecture (from the upstream model)

```
[User audio 24 kHz]
        ↓
[Mimi encoder: SEANet conv + 8L transformer + RVQ] → 16 codebooks @ 12.5 Hz
        ↓
17 streams summed: text (vocab 32001) + 8 user audio + 8 agent audio (vocab 2049)
        ↓
[Temporal transformer: 32L, dim=4096, 7B params, RoPE, RMSNorm, SwiGLU, KV-cache ctx=3000]
        ↓
[Depformer: 6L, dim=1024, MultiLinear ×16 codebook steps] → 16 agent audio tokens
        ↓
[Mimi decoder] → 24 kHz agent audio
```

### Status by piece

| # | Piece | State | Where |
|---|---|---|---|
| 1 | PyTorch reference + Mimi encoder/decoder ONNX | ✅ shipped | `convert_onnx.py --stage mimi` |
| 2 | `temporal_step` ONNX with externalized KV cache | ✅ shipped, multi-precision via `--stage temporal` (FP16/FP32) + `--stage quantize` (INT8 dynamic) + `--stage int8-matmulnbits` + `--stage int4` | `convert_onnx.py` |
| 3 | `depformer_step` ONNX with per-step weight Gather | ✅ shipped, with custom 3D-Gather INT8 quant in `quantize_depformer_gather.py` | |
| 4 | `FullDuplexSpeechInterface` | ✅ shipped | `include/speech_core/interfaces.h` |
| 5 | `OnnxPersonaPlex` wrapper | ✅ shipped end-to-end, multi-turn KV cache, auto-detect dtypes, GPU-resident KV | `include/speech_core/models/onnx_personaplex.h`, `src/models/personaplex/onnx_personaplex.cpp` |
| 6 | Custom EP routing | ✅ shipped via `OnnxEngine::SessionOptionsHook`; alternate input-binding paths remain runtime-gated by `has_gpu_provider()` | `include/speech_core/models/onnx_engine.h` |
| 7 | 4 bundle variants on HuggingFace + downloader + tests | ✅ shipped | `soniqo/PersonaPlex-7B-ONNX`, `scripts/download_personaplex_onnx.sh`, `tests/test_models.cpp::test_onnx_personaplex_load` |

### Bundle variants on HuggingFace

The C++ wrapper auto-detects three dtypes independently at session load: temporal KV, temporal hidden, and depformer KV. This lets the same wrapper handle every shipped variant without code changes.

All four variants on [soniqo/PersonaPlex-7B-ONNX](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX), measured on RTX 5090 with `SPEECH_CORE_USE_ENV_ALLOCATORS=0`:

| Variant | Disk | Host RAM | VRAM | RTF | hidden cos vs FP32 | Notes |
|---|---|---|---|---|---|---|
| [**`int8-nb-dep_gint8`**](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX/tree/main/int8-nb-dep_gint8) ⭐ | 9.4 GB | **1.4 GB** | 12.1 GB | **1.12×** | 0.998 | **Ship default** — INT8 MatMulNBits (b=128) temporal + custom INT8 depformer Gather quant + FP32 mimi |
| [`mixed`](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX/tree/main/mixed) | 11 GB | 7.9 GB | **6.6 GB** | 3.5× | 0.990 | **Quality + VRAM Pareto winner** — INT8 dynamic temporal + FP16 depformer + FP32 mimi |
| [`int4-nb-dep_gint8`](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX/tree/main/int4-nb-dep_gint8) | **7.6 GB** | 1.4 GB | **9.6 GB** | 1.12× | 0.877 | Smallest disk + lowest VRAM. Coherent but visibly degraded |
| [`fp16`](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX/tree/main/fp16) | 17 GB | 1.5 GB | 18.3 GB | 5.3× | 0.9999 | Maximum quality, near-perfect parity |

**RTF (real-time factor)** is per-frame latency / frame interval at 12.5 Hz. **1.0× = realtime**, < 1.0× = faster than realtime.

#### Which variant to pick

- **Most deployments → `int8-nb-dep_gint8`**: best balance. Sub-realtime RTF, low host RAM, excellent quality, fits comfortably in 16 GB VRAM.
- **VRAM-constrained (≤8 GB GPU) → `mixed`**: only 6.6 GB VRAM. Trades host RAM and RTF for the lowest VRAM, with the best topical responses on our customer-service benchmark ("We're concerned about it.").
- **Disk-constrained → `int4-nb-dep_gint8`**: smallest at 7.6 GB. Accept some quality drift.
- **Maximum quality → `fp16`**: near-perfect parity vs the FP32 reference.

#### Custom INT8 depformer (3D-Gather quant) — the `dep_gint8` suffix

The depformer's bulk (5.46 GB of 5.59 GB FP32) lives in 24 large 3D tensors `[16, K, N]` accessed via `Gather(step_idx)` — these are per-step weight tables for the 16 codebook steps. `MatMulNBitsQuantizer` skips them because they're Gather inputs, not MatMul inputs. The `dep_gint8` variants use a custom Python pass (`quantize_depformer_gather.py`) that:

1. Replaces each FP32 `[16, K, N]` weight with INT8 `[16, K, N]` + per-step per-output-channel FP32 scale `[16, N]`
2. Inserts a `DequantizeLinear(axis=-1)` after the original `Gather` so the downstream `Squeeze → MatMul` chain still sees FP32

Net: depformer disk drops **5.59 GB → 1.50 GB (-73%)** with single-frame parity cos = 0.9999 vs the FP32 reference.

Memory protections on top:
- **GPU-resident KV cache** (auto-on when CUDA EP resolves): the `temporal_step` output `OrtValue`s for K/V are kept alive across calls and passed directly as the next call's inputs. No host mirror, no per-frame `GetTensorMutableData`. Opt out with `SPEECH_CORE_PP_GPU_KV=0`.
- `reset_session()` swaps the KV vectors with empty ones (`std::vector<>().swap(...)`) to release capacity, not just `.clear()`. Also releases the GPU-resident `OrtValue`s.
- `temporal_forward()` soft-caps `t_past` at `kMaxContext=3000` and ring-shifts the oldest column on each step past the cap — bounded memory + stable quality (positions beyond training ctx are attention-poison anyway). Host-path only — GPU mode trusts ORT's dynamic shape support.

### Full prefill sequence — coherent multi-voice responses

The wrapper now mirrors speech-swift's MLX prefill layout end-to-end:

1. **Voice prompt replay** — replays the voice `.bin`'s 4-token cache tail through `temporal_step` to populate KV with speaker-conditioned state
2. **Silence spacer** — 6 frames (~0.5 s) of PAD text + zero audio, clean transition boundary
3. **System prompt prefill** — N frames of pre-tokenized text (from `system_prompts.bin`, produced by `tokenize_system_prompts.py` — avoids the SentencePiece C++ dep)
4. **Silence spacer** — another 6 frames
5. **`respond_stream`** — user audio frames begin

`set_system_prompt("helpful" | "direct" | "expert" | "warm")` picks a pre-tokenized prompt. `set_voice("<name>")` selects the speaker.

INT8 bundle results on RTX 5090, real user audio: *"Can you guarantee the replacement will be shipped tomorrow?"*, 50 frames of agent generation:

| Voice | Prompt | Peak | Parakeet transcript |
|---|---|---|---|
| NATM0 | helpful | 1.19 | **"I'm not sure"** |
| NATF0 | helpful | 0.94 | **"I have to find it."** |
| VARF2 | helpful | 0.81 | **"Never been finished."** |
| VARF2 | direct | 0.76 | **"That's what I'm gonna do."** |
| VARF4 | helpful | 0.91 | **"Yeah."** |
| VARM0 | direct | 0.82 | "And" |

**Five voices produce semantically appropriate English responses.** Audio peaks mostly < 1.0 (no clipping). First PersonaPlex ONNX/CUDA INT8 implementation producing actual conversational responses end-to-end.

### Voice embedding prefix

The voice `.bin`'s 50-frame embedding prefix IS used by default (`SPEECH_CORE_PP_EMB_SCALE=10`). Stored embeddings are at ~0.03 std but depformer was trained on temporal output at ~1.5 std — scaling 10× brings them into distribution. Empirically measured across NATM0 / NATM1 / VARM0 / VARF2 / VARF4 voices on the customer-service fixture. Set `SPEECH_CORE_PP_EMB_SCALE=0` to disable (slightly less topical but still coherent).

### Multi-voice quality across bundles

Same fixture *"Can you guarantee the replacement will be shipped tomorrow?"*, prompts `helpful` / `direct`, sampled across voices. Each bundle's Parakeet round-trip:

| Bundle | Sample transcripts |
|---|---|
| `fp16` | "We can do" (VARF2) |
| `mixed` | **"We're concerned about it." (VARF2)** — best topical match, most variants produce English |
| `int8-nb-dep_gint8` | "I don't think I'm" (VARF2) |
| `int4-nb-dep_gint8` | "I'm gonna function." (VARF2), "I like it." (VARM0/direct) |

### Memory tuning — `OnnxEngine` env knobs

12 env knobs available for ORT session + CUDA EP memory tuning. The two with the largest measured impact:

| Env | Effect |
|---|---|
| `SPEECH_CORE_USE_ENV_ALLOCATORS=0` | **Biggest win.** Disables ORT's shared CPU arena which was hoarding 6–13 GB of allocated-but-unused space across PersonaPlex's 4 sessions. Drops host RAM by 50–89% depending on bundle. Strongly recommended for PersonaPlex |
| `SPEECH_CORE_DEVICE_INITIALIZERS=1` (default-on) | Sets `session.use_device_allocator_for_initializers=1` so quantized weights load directly from the device allocator without a CPU shadow buffer |

Opt-in knobs (all measured but with smaller impact): `SPEECH_CORE_DISABLE_MEM_ARENA`, `SPEECH_CORE_DISABLE_MEM_PATTERN`, `SPEECH_CORE_DISABLE_PREPACKING`, `SPEECH_CORE_DQ_MATMULNBITS`, `SPEECH_CORE_DISABLE_QDQ_FOLD`, `SPEECH_CORE_ORT_OPT_LEVEL`, `SPEECH_CORE_GPU_MEM_LIMIT_GB`, `SPEECH_CORE_CUDA_ARENA_STRATEGY`, `SPEECH_CORE_CUDNN_NO_MAX_WS`, `SPEECH_CORE_CUDNN_HEURISTIC`. See `include/speech_core/models/onnx_engine.h` for the full inline documentation per knob.

### TensorRT EP — measured and rejected

TensorRT EP looked promising on paper (compiles INT8 to native GPU kernels, no CPU dequantize staging). Measured against the mixed bundle on RTX 5090:

| Metric | CUDA EP | TensorRT EP |
|---|---|---|
| Load time | 17 s | **392 s** (per-shape engine compilation storm) |
| Peak host RAM | 15.9 GB | **24.3 GB** (worse — engines cached on host + ORT staging) |
| RTF (50 frames) | 3.5× | **265.9×** (TRT rebuilds engines for each T_past) |
| Output | "We're concerned about it." | **""** (broken — TRT can't reconcile Q/DQ with dynamic shapes) |

Making it work would need shape profiles (min/opt/max on the dynamic KV axis), an INT8 calibration table, validated op support, and pre-allocated max-shape KV buffers — together ~10-15 hours of dedicated engineering. The shipping `int8-nb-dep_gint8` variant achieves the "small disk + small host RAM" target without TRT — `MatMulNBits` is a single fused op running entirely on the GPU.

### Per-frame parity vs FP32 reference

Measured by `compare_bundle_quality.py` on a single temporal_step forward call with empty KV (CPU EP):

| Recipe | hidden cos |
|---|---|
| FP16 export | 0.999999 |
| INT8-NB-b32 | 0.998901 |
| INT8-NB-b128 (ship `int8-nb-dep_gint8`) | 0.997706 |
| INT8 dynamic per-channel (mixed) | 0.990448 |
| INT4-NB-b32 | 0.877393 |

### PyTorch CUDA reference (upper-bound target)

PyTorch FP16 reference benchmarked via `bench_pytorch_cuda.py` on the same RTX 5090:

| Dtype | Load | VRAM | Per-frame | RTF |
|---|---|---|---|---|
| **FP16** | 84 s | 17.6 GB | 3.75 ms (batched 50 frames) | 0.047× |
| BF16 | 58 s | 17.6 GB | 92 ms | 1.15× (no fast BF16 kernels on RTX 5090) |
| FP32 | 95 s | 34.4 GB | 712 ms | 8.9× |

PyTorch FP16's 3.75 ms/frame is the **batched** LM forward (50 frames at once); our ONNX wrapper runs **autoregressive** streaming, so the per-call ORT dispatch overhead dominates. The wrapper's autoregressive RTF for `int8-nb-dep_gint8` is 1.12× — within 24× of the batched PyTorch baseline, with 5 GB less VRAM.

`bench_pytorch_cuda.py --dump-logits` writes per-frame text + audio logits to a `.npz` for downstream comparison via `compare_bundle_quality.py`.

### Multi-turn dialogue

`SPEECH_CORE_PP_MULTITURN=N` (in `tests/run_personaplex.cpp`) runs `respond_stream` N times **without** `reset_session` between calls, validating that the KV cache survives across turns. Measured 5-turn dialogue continuation:

| Turn | Time | Per-frame | RTF | Cumulative frames |
|---|---|---|---|---|
| 1 | 4.4 s | 87.6 ms | 1.10× | 50 |
| 2 | 4.7 s | 93.3 ms | 1.17× | 100 |
| 3 | 5.0 s | 100.7 ms | 1.26× | 150 |
| 4 | 5.5 s | 110.4 ms | 1.38× | 200 |
| 5 | 5.8 s | 116.9 ms | 1.46× | 250 |

Cumulative transcript: "I want to in my land. It's been a good one." (coherent across 5 turns). RTF degrades linearly with cumulative T_past — expected attention scaling.

### Files in each bundle

Each variant at [soniqo/PersonaPlex-7B-ONNX](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) ships:

| File | Purpose |
|---|---|
| `mimi_encoder.onnx`(+`.data`) | 24 kHz PCM → 16 audio codebooks @ 12.5 Hz |
| `mimi_decoder.onnx`(+`.data`) | 16 audio codebooks @ 12.5 Hz → 24 kHz PCM |
| `temporal_step.onnx`(+`.data`) | One frame of the 32-layer 7B temporal transformer, explicit KV-cache I/O |
| `depformer_step.onnx`(+`.data`) | One inner step of the 6-layer depformer, 16 codebook steps per frame |
| `tokenizer_spm_32k_3.model` | SentencePiece text tokenizer |
| `voices/<name>.bin` | 18 voice prompts (NATF0-3, NATM0-3, VARF0-4, VARM0-4) |
| `system_prompts.bin` | Pre-tokenized "helpful" / "expert" / "warm" / "direct" prompts |
| `config.json` | Architecture + precision + measured metrics per bundle |
| `README.md` | Per-variant model card |

Total per variant: 30 files. Repo root also has a top-level model card with the variant matrix.

### Validation pattern

End-to-end: **generate speech, transcribe with Parakeet STT, assert content words**. The wrapper's per-frame parity is gated by `compare_bundle_quality.py` (single-frame cosine similarity vs the FP32 reference). The PyTorch reference is gated by `bench_pytorch_cuda.py`.

The ctest `test_onnx_personaplex_load` exercises the full prefill + 4-frame generation loop on whatever bundle is at `SPEECH_MODEL_DIR`, asserting `chunks > 0`, `got_final`, `total_emitted > 0`. Validated against all four HF bundles end-to-end.

## DiarizationPipeline

```cpp
#include <speech_core/diarization/diarization_pipeline.h>

speech_core::LiteRTPyannoteSegmentation seg("/models/pyannote-segmentation.tflite");
speech_core::LiteRTWeSpeakerEmbedding   emb("/models/wespeaker-resnet34.tflite");
speech_core::DiarizationPipeline        diar(seg, emb);

speech_core::DiarizerConfig cfg;   // onset/offset/min_speech/clustering_threshold/min,max_speakers
auto segments = diar.diarize(audio, length, 16000, cfg);  // [{start,end,speaker}, …]
```

- Pure-C++ orchestration (no LiteRT dependency): segmentation → per-speaker embedding → constrained agglomerative clustering (cosine distance, window-uniqueness constraint) → merged speaker-labelled segments. Lives in the **core** library; inject any `SegmentationInterface` + `EmbeddingInterface`.

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
- Model files: [soniqo/Kokoro-82M-ONNX](https://huggingface.co/soniqo/Kokoro-82M-ONNX) — `kokoro-e2e.onnx` + `kokoro-e2e.onnx.data` (~90 MB total), `vocab_index.json`, `us_gold.json`, `us_silver.json`, `dict_{fr,es,it,pt,hi}.json`, `voices/*.bin`

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
- Model files: [soniqo/DeepFilterNet3-ONNX](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) — `deepfilter.onnx` (~8 MB FP16), `deepfilter-auxiliary.bin`

## OnnxSidonRestorer

Sidon — combined denoise + dereverb speech restoration. Two ONNX graphs (a
w2v-BERT 2.0 predictor truncated to 8 layers with a merged LoRA, and a DAC
decoder vocoder) plus a C++ SeamlessM4T log-mel front-end. Input is 16 kHz;
output is **48 kHz**. Primary use case: clean a reverberant voice-cloning
reference before a TTS voice-cloner.

```cpp
#include <speech_core/models/onnx_sidon_restorer.h>

speech_core::OnnxSidonRestorer rest(
    "/models/sidon-predictor.onnx",
    "/models/sidon-vocoder.onnx");

// restore() takes audio at any rate (resampled to 16 kHz internally) and
// returns 48 kHz mono.
std::vector<float> clean = rest.restore(ref.data(), ref.size(), ref_rate);
```

- Pipeline: `audio (16 kHz)` → SeamlessM4T log-mel front-end → `input_features [1, T, 160]` → predictor → `features [1, T, 1024]` → DAC vocoder → `audio (48 kHz)`.
- Front-end (`speech_core/audio/seamless_fbank.h`, pure C++17, in the core lib): Kaldi-compatible 80-bin log-mel (povey window, pre-emphasis 0.97, per-frame DC removal, power spectrum, kaldi mel scale, natural log) + per-mel-bin CMVN + ×2 frame stacking (80 → 160). Validated against `transformers.SeamlessM4TFeatureExtractor` (`facebook/w2v-bert-2.0`): cosine ≈ 1.0, golden-value parity in `test_seamless_fbank`.
- **Not an `EnhancerInterface`** natively: restoration changes both sample rate (16 → 48 kHz) and length, while `EnhancerInterface` is fixed-rate, equal-length, in-place. Use `restore()`. `as_enhancer()` provides an adapter (resamples 48 kHz back to the input rate/length) for callers that only hold the abstract handle.
- **ONNX-only.** The DAC decoder's `ConvTranspose1d` does not legalise to TFLite, so there is no LiteRT vocoder (see speech-models `models/sidon/export/NOTES.md`). ONNX FP32 is bit-exact vs PyTorch; FP16 is near-lossless at half the size.
- CLI: `speech_sidon_restore <bundle_dir> <in.wav> <out.wav>` (`examples/onnx/sidon_restore.cpp`).
- Model files (provisional): `aufklarer/Sidon-ONNX` — `sidon-predictor.onnx` + `sidon-vocoder.onnx` (FP32 ~939 MB, FP16 ~470 MB).

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

A separate `test_litert_models` target is added when `SPEECH_CORE_WITH_LITERT=ON`, exercising the LiteRT wrappers (Silero VAD, Parakeet STT, VoxCPM2 TTS, WeSpeaker embedding, Pyannote segmentation, Omnilingual STT, Nemotron streaming STT) + the `DiarizationPipeline` + the VoxCPM2 tokenizer against `.tflite` artifacts:

```bash
scripts/fetch_litert.sh build/litert        # extracts libLiteRt from ai-edge-litert wheel
scripts/download_models_litert.sh           # Silero + Parakeet
scripts/download_voxcpm2_litert.sh          # VoxCPM2 bundle (mixed int8/fp16, ~6.4 GB, optional)
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert
cmake --build build
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

Each per-model test skips cleanly when its files aren't in `SPEECH_LITERT_MODEL_DIR`. The weekly CI workflow (`.github/workflows/weekly-voxcpm2.yml`) downloads the VoxCPM2 bundle and runs an end-to-end synth → Parakeet STT round-trip; the daily nightly skips VoxCPM2 to keep the cache budget reasonable.

## Bring-your-own model

The interface contract is small. To use a non-ONNX backend (CoreML, MLX, Whisper.cpp, llama.cpp, remote API), inherit the relevant interface and pass an instance to `VoicePipeline`. The reference implementations live alongside the orchestration but are not required.

See speech-swift for a worked example with CoreML and MLX backends targeting the same conceptual interfaces.
