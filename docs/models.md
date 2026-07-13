# Reference Model Implementations

speech-core ships two parallel sets of model wrappers under `include/speech_core/models/`. Each implements one of the interfaces in `interfaces.h` and is compiled in via a backend-specific CMake flag. The two backends can be enabled simultaneously — consumers link only the targets they need.

### ONNX Runtime backend (`SPEECH_CORE_WITH_ONNX`)

| Model | Interface | Header | Status |
|---|---|---|---|
| `SileroVad` | `VADInterface` | `speech_core/models/silero_vad.h` | full |
| `ParakeetStt` | `STTInterface` | `speech_core/models/parakeet_stt.h` | full |
| `OnnxWhisperStt` | `STTInterface` | `speech_core/models/onnx_whisper_stt.h` | full |
| `KokoroTts` | `TTSInterface` | `speech_core/models/kokoro_tts.h` | full |
| `DeepFilterEnhancer` | `EnhancerInterface` | `speech_core/models/deepfilter.h` | full |
| `OnnxSidonRestorer` | (own API; `EnhancerInterface` adapter) | `speech_core/models/onnx_sidon_restorer.h` | full — see [OnnxSidonRestorer](#onnxsidonrestorer) |
| `OnnxCosyVoice3Tts` | `TTSInterface` | `speech_core/models/onnx_cosyvoice3_tts.h` | staged — ONNX runtime + cached conditioning |
| `OnnxVoxCPMTts` | `TTSInterface` | `speech_core/models/onnx_voxcpm_tts.h` | full |
| `OnnxVoxCPM2Tts` | `TTSInterface` | `speech_core/models/onnx_voxcpm2_tts.h` | full |
| `OnnxNemotronStreamingStt` | `STTInterface` | `speech_core/models/onnx_nemotron_streaming_stt.h` | full (streaming) |
| `NemotronMultilingualStt` | `STTInterface` | `speech_core/models/nemotron_multilingual_stt.h` | full (streaming, prompt-conditioned) |
| `OnnxPersonaPlex` | `FullDuplexSpeechInterface` | `speech_core/models/onnx_personaplex.h` | structural — see [OnnxPersonaPlex](#onnxpersonaplex) |

### LiteRT backend (`SPEECH_CORE_WITH_LITERT`)

| Model | Interface | Header | Status |
|---|---|---|---|
| `LiteRTSileroVad` | `VADInterface` | `speech_core/models/litert_silero_vad.h` | full |
| `LiteRTParakeetStt` | `STTInterface` | `speech_core/models/litert_parakeet_stt.h` | full |
| `LiteRTKokoroTts` | `TTSInterface` | `speech_core/models/litert_kokoro_tts.h` | full (staged 60-frame FP32) |
| `LiteRTVoxCPM2Tts` | `TTSInterface` | `speech_core/models/litert_voxcpm2_tts.h` | full (text-only) |
| `LiteRTChatterboxTts` | `TTSInterface` | `speech_core/models/litert_chatterbox_tts.h` | full |
| `LiteRTSupertonicTts` | `TTSInterface` | `speech_core/models/litert_supertonic_tts.h` | full |
| `LiteRTIndicMioTts` | `TTSInterface` | `speech_core/models/litert_indic_mio_tts.h` | full |
| `LiteRTWeSpeakerEmbedding` | `EmbeddingInterface` | `speech_core/models/litert_wespeaker_embedding.h` | full |
| `LiteRTPyannoteSegmentation` | `SegmentationInterface` | `speech_core/models/litert_pyannote_segmentation.h` | full |
| `LiteRTOmnilingualStt` | `STTInterface` | `speech_core/models/litert_omnilingual_stt.h` | full |
| `LiteRTNemotronStreamingStt` | `STTInterface` | `speech_core/models/litert_nemotron_streaming_stt.h` | full (streaming) |
| `LiteRTNemotronMultilingualStt` | `STTInterface` | `speech_core/models/litert_nemotron_multilingual_stt.h` | full (streaming, prompt-conditioned) |

### LLM backends (`LLMInterface`)

speech-core provides the abstract `LLMInterface` (`interfaces.h`) plus a tool
registry (`tools/tool_registry.h`) and the pipeline integration that routes
LLM-emitted tool calls through `ToolExecutor`. Three reference implementations
ship today.

| Implementation | Header | Backend | Built when |
|---|---|---|---|
| `OllamaLLM` | `speech_core/llm/ollama_llm.h` | Local Ollama HTTP server (`/api/chat`) | `SPEECH_CORE_WITH_OLLAMA=ON` (in `speech_core_llm_ollama`) |
| `LiteRTFunctionGemmaLLM` | `speech_core/models/litert_functiongemma_llm.h` | Google's `liblitert-lm` runtime driving a `.litertlm` bundle (e.g. [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM)) | `SPEECH_CORE_WITH_LITERT_LM=ON` (in `speech_core_models_litert_lm` — a standalone target, independent of `speech_core_models_litert`) |
| **FunctionGemma 270M (CoreML)** | (platform) | Core ML on Apple via [speech-swift](https://github.com/soniqo/speech-swift) | platform-side |

`LiteRTFunctionGemmaLLM` loads `.litertlm` bundles via the higher-level
`liblitert-lm` shared library, NOT through the lower-level `libLiteRt` C API
that drives the `.tflite` models in this directory. It builds into its own
static library (`speech_core_models_litert_lm`) so Android consumers that need
only the LLM can link it without dragging in `libLiteRt`. Point CMake at the
runtime with `-DSPEECH_CORE_WITH_LITERT_LM=ON -DLITERT_LM_DIR=...`. The class
implements `LLMInterface` directly, so it plugs into `VoicePipeline`
identically to `OllamaLLM`.

On **macOS**, `scripts/fetch_litert_lm.sh` extracts the shared library out of
the `litert-lm-api` PyPI wheel (mirrors `scripts/fetch_litert.sh`). On
**Android**, Google publishes neither `libLiteRt.so` nor `liblitert-lm.so` —
the PyPI wheel is macOS-only and the Maven artifact only exposes the Kotlin
facade. Use `scripts/build_litert_lm_android.sh` to cross-compile
`liblitert-lm.so` from the pinned `google-ai-edge/LiteRT-LM` v0.13.1 source
tree; outputs land in the same `${LITERT_LM_DIR}/${ANDROID_ABI}/` layout the
top-level CMake expects. Set `EMULATOR_SAFE=1` when targeting the
Apple-Silicon-hosted Android emulator: HVF passthrough mistraps KleidiAI's
SME `rdsvl` instruction (the guest CPU presents as implementer 0x61 Apple),
so the script disables XNNPack assembly + KleidiAI SME kernels. Costs 3–5×
decode rate but unblocks the emulator dev loop; real devices should leave it
off.

To wire your own on-device LLM into speech-core, subclass `LLMInterface`,
implement `set_tools()` (convert `ToolDefinition[]` into the model's prompt
format) and `chat()` (run prefill+decode, parse tool-call markers, populate
`LLMResponse.tool_calls`), then pass an instance to `VoicePipeline`. See
`OllamaLLM` or `LiteRTFunctionGemmaLLM` for reference shapes.

`DiarizationPipeline` (`speech_core/diarization/diarization_pipeline.h`, implements `DiarizerInterface`) composes a segmenter + embedder + constrained clustering. It is pure C++ and ships in the **core** library (built always, no LiteRT dependency); pair it with the LiteRT segmenter + embedder above.

DeepFilterNet3 does not yet have a LiteRT export; see `speech-models` for conversion status.

`OnnxVoxCPMTts` is the smaller VoxCPM 0.5B serving wrapper used by the CPU cloud synth path. It loads split prefill/token-step decoder graphs when `voxcpm-text-prefill*.onnx` and `voxcpm-token-step*.onnx` sit beside the requested `voxcpm-decoder*.onnx`, with automatic fallback to the legacy unified decoder graph when split files are absent. It outputs 16 kHz PCM and supports prompt-audio cloning via `set_reference()`. For best clone fidelity, call `set_reference_transcript()` with the exact text spoken in the reference clip before `synthesize()`.

`OnnxVoxCPM2Tts` runs the VoxCPM2 2B ONNX deployment bundle and outputs 48 kHz
PCM. It supports two decoder layouts: a unified `voxcpm2-decoder.onnx` graph,
or split `voxcpm2-text-prefill.onnx` and `voxcpm2-token-step.onnx` graphs. The
unified export keeps shared transformer weights resident once, which is useful
for GPU deployments; the split export is useful for CPU serving where unified
peak RSS can exceed the pod memory budget. Both layouts use the same
`voxcpm2-audio-encoder.onnx`, `voxcpm2-audio-decoder.onnx`, and
`tokenizer.json` files, and both support reference-audio cloning through
`set_reference()`. When the exact words spoken in the reference clip are known,
call `set_reference_transcript()` after `set_reference()` to use VoxCPM2's
combined reference + continuation clone mode.

`OnnxCosyVoice3Tts` runs the CosyVoice3 0.5B ONNX deployment bundle (`llm_prefill`, `llm_step`, `flow_frontend`, flow estimator, `hift`) and outputs 24 kHz PCM. The current C++ contract expects zero-shot voice conditioning to be supplied explicitly through `set_conditioning()`: prompt text token IDs, prompt speech tokens, prompt mel features, and a 192-dim speaker embedding. This matches the cloud serving shape: compute conditioning once when a voice is created, persist the binary blob with the voice, then reuse it for synthesis without re-running the prompt frontend per request. The bundled `encode_conditioning_blob()` / `decode_conditioning_blob()` helpers define that persistence format. If the bundle also contains `hift_128.onnx` or `hift_256.onnx`, the wrapper uses the smallest fitting HiFT bucket for shorter clips and falls back to `hift.onnx`.

`LiteRTVoxCPM2Tts` runs the full 4-graph orchestration end-to-end: `text_prefill → token_step ×N → audio_decode` with explicit K/V cache handoff every step. Voice cloning is surfaced through `set_reference()`. Supplying `set_reference_transcript()` additionally uses VoxCPM2's upstream combined reference + continuation prompt layout. The bundle is large (~8.7 GB fp16 `selective` for ARM at the repo root; ~13 GB fp32-token-step for x86 in the `fp32-p16/` subdir) and inference is slow on CPU, so end-to-end validation runs in the **weekly** workflow (`.github/workflows/weekly-voxcpm2.yml`) rather than the daily nightly.

All ORT wrappers share an internal ONNX Runtime singleton (`OnnxEngine` in `speech_core/models/onnx_engine.h`) that owns the `OrtEnv` and `OrtMemoryInfo`. Most LiteRT wrappers share `LiteRTEngine` (`speech_core/models/litert_engine.h`). Kokoro uses the stable TFLite Interpreter ABI exported by `libLiteRt` so its constructor can set the XNNPACK thread count directly. The current reference wrappers are CPU-only; NNAPI / GPU / Hexagon delegates are not wired here.

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

`SPEECH_CORE_WITH_HF_DOWNLOAD=ON` adds first-run Hugging Face bundle downloads
for LiteRT models exposed through the C ABI, including
`sc_voxcpm2_create_from_pretrained("soniqo/VoxCPM2-LiteRT", ...)` and
`sc_indic_mio_create_from_pretrained("soniqo/Indic-Mio-LiteRT", ...)`.
Downloads resume from `.part` files, retry on low-speed stalls, and use
parallel HTTP ranges for large files by default. Tune with
`SPEECH_CORE_DOWNLOAD_CONNECTIONS` (`4` default, clamped `1`–`16`; set `1` for
single-stream), `SPEECH_CORE_CACHE_DIR`, and `HF_ENDPOINT`.

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
- Language selection is automatic. Parakeet TDT has no decoder prompt channel,
  and the published v3 exports do not emit a language token in greedy decode,
  so the wrapper deliberately does not expose a language-forcing API.
- Streaming supported via `begin_stream` / `push_chunk` / `end_stream` (accumulates audio and re-transcribes each chunk; not a true streaming decoder)
- Model files: [soniqo/Parakeet-TDT-0.6B-ONNX](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-ONNX) — `parakeet-encoder.onnx` (FP32, plus external `.onnx.data`) or `parakeet-encoder-int8.onnx` (~840 MB / ~100 MB INT8), `parakeet-decoder-joint.onnx` / `parakeet-decoder-joint-int8.onnx`, `vocab.json`. Decoder-joint inputs `targets` + `target_length` are INT32; encoder length input stays INT64.

## OnnxWhisperStt

```cpp
#include <speech_core/models/onnx_whisper_stt.h>

speech_core::OnnxWhisperStt stt(
    "/models/whisper-turbo/turbo-encoder.int8.onnx",
    "/models/whisper-turbo/turbo-decoder.int8.onnx",
    "/models/whisper-turbo/turbo-tokens.txt");

auto result = stt.transcribe(audio, length, 16000);
auto profile = stt.last_profile();
```

- OpenAI Whisper small, medium, large-v3, and large-v3-turbo exported through the sherpa-onnx encoder/decoder graph contract.
- Native ONNX Runtime implementation in `speech_core_models`: Whisper log-mel frontend, encoder cross-attention KV, greedy decoder self-KV cache, metadata-driven language detection, and base64 token-table decoding.
- Fixed language prompts are available with `Config.language` or `set_language("de")`; empty language auto-detects on multilingual models.
- The default chunk size leaves room for sherpa-style tail padding, so long audio is processed in approximately 29.5 second windows.
- `last_profile()` reports the most recent transcription's total, feature, encoder, language-detection, first-token, and decoder timings.
- CPU Whisper uses larger encoder matmuls than the short-call ONNX models. Keep the global `SPEECH_CORE_ORT_THREADS` default conservative for those models, and tune Whisper separately with `Config.intra_threads` or `SPEECH_CORE_WHISPER_ORT_THREADS`.
- Low-latency fixed-language usage typically sets `Config.language`, lowers `Config.tail_padding_frames` after WER validation, and raises `Config.intra_threads` on CPU:

```cpp
speech_core::OnnxWhisperStt::Config cfg;
cfg.language = "en";
cfg.tail_padding_frames = 50;
cfg.intra_threads = 16;
speech_core::OnnxWhisperStt stt(enc, dec, tok, cfg);
```

- Whisper benchmark:

```bash
SPEECH_MODEL_DIR=/models/whisper-turbo \
SPEECH_WHISPER_ONNX_DIR=/models/whisper-turbo \
SPEECH_BENCH_ONLY=whisper \
SPEECH_WHISPER_BENCH_CONFIG=en-tail50 \
SPEECH_CORE_WHISPER_ORT_THREADS=16 \
./build/bench_ort_models
```

- Download helper:

```bash
scripts/download_whisper_onnx.sh turbo int8
scripts/download_whisper_onnx.sh medium fp16
```

- Model files:
  [soniqo/Whisper-Small-ONNX](https://huggingface.co/soniqo/Whisper-Small-ONNX),
  [soniqo/Whisper-Medium-ONNX](https://huggingface.co/soniqo/Whisper-Medium-ONNX),
  [soniqo/Whisper-Large-v3-ONNX](https://huggingface.co/soniqo/Whisper-Large-v3-ONNX),
  [soniqo/Whisper-Large-v3-Turbo-ONNX](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX).

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
- Language selection is automatic; like the ONNX wrapper, it does not expose a
  language-forcing API.
- Encoder INT8 weight-quantized (~595 MB on disk vs ~840 MB ONNX FP32), decoder-joint stays FP32 to avoid LSTM drift
- Decoder-joint exposes `(encoder_out, target, h, c)` as four discrete tensors (ORT bundles `target_length` and uses suffix-`_1`/`_2` for h/c)
- Model files: [soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) — `parakeet-encoder.tflite`, `parakeet-decoder-joint.tflite`, `vocab.json`

## LiteRTKokoroTts

```cpp
#include <speech_core/models/litert_kokoro_tts.h>

speech_core::LiteRTKokoroTts tts(
    "/models/kokoro-encoder.tflite",
    "/models/kokoro-recurrent-equivalent32.tflite",
    "/models/kokoro-vocoder.tflite",
    "/models/voices",
    "/models",
    /*hw_accel=*/false,
    /*num_threads=*/4);

tts.set_seed(1234);  // optional deterministic harmonic phases
tts.set_speed(1.0f);
tts.synthesize("Hello world.", "en",
    [](const float* samples, size_t length, bool is_final) {
        // 24 kHz Float32 PCM; one final callback for the submitted text.
    });
```

- The release is a fixed-shape, three-stage FP32 bundle: `kokoro-encoder.tflite`, `kokoro-recurrent-equivalent32.tflite`, and `kokoro-vocoder.tflite` (333,529,644 bytes total). The recurrent graph preserves the accepted 128-slot result by precomputing the state of the guaranteed 96-slot zero tail, then evaluating only 32 recurrent slots.
- The public tensors retain 128 token slots, while the optimized ALBERT path evaluates 32 active slots. The wrapper proactively chunks around 14 active tokens at speed 1.0.
- The final convolutional stack needs unused right context. A model-reported duration above 56 frames (33,600 samples) is discarded and retried as smaller text; frames 57–60 are never accepted merely because they fit the 36,000-sample tensor.
- Internal text chunks are combined before the callback. Output is 24 kHz Float32 with finite/peak checks, trailing-noise trim, and short fades at chunk boundaries.
- `num_threads` configures each XNNPACK interpreter directly. On a physical Galaxy S23 Ultra (`SM-S918B`), 8 threads was the best tested setting: 0.5502 warmed p50 RTF, 0.5807 p90 RTF, and 1,017.5 MiB peak RSS for `Hello world.` (one warm-up + ten measured runs). Android and Windows wrapper outputs correlated at 0.999795. Use `speech_kokoro_litert_bench <bundle-dir> --variant equivalent32 --threads 8 --warmup 1 --runs 10` to reproduce it.
- The physical-device GPU audit was rejected; this wrapper is CPU/XNNPACK-only.
- Use the staged bundle when `.tflite` format compatibility is required; the ONNX backend has separate full-graph and guarded short-turn profiles documented below.
- Model files: [soniqo/Kokoro-82M-LiteRT](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) — the three graphs, `vocab_index.json`, `us_gold.json`, `us_silver.json`, and `voices/af_heart.bin`.

## OnnxVoxCPMTts

```cpp
#include <speech_core/models/onnx_voxcpm_tts.h>

speech_core::OnnxVoxCPMTts tts(
    "/models/voxcpm-decoder.fp16w.onnx",
    "/models/voxcpm-audio-encoder.onnx",
    "/models/voxcpm-audio-decoder.onnx",
    "/models/tokenizer.json");

tts.set_reference(reference_pcm_16khz.data(), reference_pcm_16khz.size(), 16000);
tts.set_reference_transcript("This is the exact sentence in the reference clip.");
tts.synthesize("Hello world", "en", [](const float* samples, size_t length, bool is_final) {
    // 16 kHz Float32 PCM, streamed in decoder chunks.
});
tts.clear_reference();
```

For offline post-processing, use buffered delivery:

```cpp
speech_core::TtsSynthesisOptions options;
options.mode = speech_core::TtsSynthesisMode::Buffered;
options.postprocess_flags = speech_core::kTtsPostProcessDeEsser;

tts.synthesize_with_options("Hello world", "en", options,
    [](const float* samples, size_t length, bool is_final) {
        // One final callback with the full post-processed utterance.
    });
```

- VoxCPM 0.5B bilingual TTS, ONNX Runtime backend.
- `synthesize()` is the streaming path: the callback receives each decoder flush chunk.
- `synthesize_with_options()` accepts `TtsSynthesisOptions`: `Streaming` preserves chunked delivery, while `Buffered` accumulates all PCM produced for the single submitted text input before invoking the callback once with `is_final=true`.
- Offline post-processing requires `Buffered` mode so it runs on the complete synthesized result, not on decoder flush chunks.
- Serving bundle: [soniqo/VoxCPM-0.5B-ONNX](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX).
- Loads four graphs when split decoder artifacts are present:
  `voxcpm-text-prefill*.onnx`, `voxcpm-token-step*.onnx`,
  `voxcpm-audio-encoder.onnx`, and `voxcpm-audio-decoder.onnx`.
  Older bundles that only ship `voxcpm-decoder*.onnx` still work via the
  legacy unified fallback.
- Default cloud CPU deployment uses `voxcpm-decoder.fp16w.onnx`: FP16 external weights with FP32 compute tensors, keeping CPU RSS lower while preserving graph I/O shape.
- Voice cloning is prompt-audio based. `set_reference()` encodes the 16 kHz prompt clip into latent frames; `set_reference_transcript()` is optional but recommended, and should be the exact transcript of that clip. Existing callers that only set audio still work.
- For latency canaries, `SPEECH_CORE_VOXCPM_REF_MAX_FRAMES` (or the shorter
  `VOXCPM_REF_MAX_FRAMES`) caps the prompt-audio frames consumed by
  `set_reference()`. The default uses the full model cap.
- End-to-end audio-quality validation is intentionally heavy because it downloads a multi-GB bundle and runs autoregressive synthesis. Keep ordinary CI to compile/smoke checks; run synth → ASR round-trips manually or from a weekly workflow when promoting a new bundle.

## OnnxCosyVoice3Tts

```cpp
#include <speech_core/models/onnx_cosyvoice3_tts.h>

speech_core::OnnxCosyVoice3Tts tts("/models/cosyvoice3", /*hw_accel=*/false);

auto conditioning =
    speech_core::OnnxCosyVoice3Tts::decode_conditioning_blob(blob.data(), blob.size());
tts.set_conditioning(std::move(conditioning));
tts.set_seed(1986);
tts.set_max_steps(80);
tts.synthesize("Hello from CosyVoice3.", "", [](const float* samples,
                                                 size_t length,
                                                 bool is_final) {
    // 24 kHz Float32 PCM.
});
tts.clear_conditioning();
```

- Serving bundle: [soniqo/CosyVoice3-0.5B-ONNX](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX).
- Loads five required inference graphs: `llm_prefill.onnx`, `llm_step.onnx`, `flow_frontend.onnx`, `flow.decoder.estimator.fp32.onnx`, and `hift.onnx`, plus `CosyVoice-BlankEN/{vocab.json,merges.txt}` for the Qwen text tokenizer. Optional `hift_128.onnx` and `hift_256.onnx` vocoder buckets reduce fixed HiFT latency for shorter outputs.
- Zero-shot voice cloning is intentionally a two-stage contract. This wrapper consumes cached conditioning tensors; it does not yet compute the frontend tensors from raw reference audio. Cloud and app callers should compute that conditioning at voice-create time and persist `encode_conditioning_blob()` output with the voice.
- The conditioning blob contains prompt text token IDs, LLM prompt speech tokens, flow prompt speech tokens, prompt mel features `[frames,80]`, and a 192-dim speaker embedding. `prompt_text_from_transcript()` prepends the helper prompt prefix when the transcript does not already include `<|endofprompt|>`.
- After each `synthesize()` call, `prefill_ms()`, `ar_ms()`, and `audio_decode_ms()` expose coarse stage timing. `flow_frontend_ms()`, `flow_estimator_ms()`, and `hift_ms()` split `audio_decode_ms()` into the Flow/HiFT sub-stages used by cloud observability.
- `tests/test_models.cpp` includes a load/blob smoke. Set `SPEECH_COSYVOICE3_ONNX_DIR=/path/to/CosyVoice3-0.5B-ONNX` to run it locally; add `SPEECH_COSYVOICE3_E2E=1` to synthesize a short clip and assert populated stage timings. Without the bundle, it skips like the other heavyweight model tests.

## LiteRTVoxCPM2Tts

```cpp
#include <speech_core/models/litert_voxcpm2_tts.h>

speech_core::LiteRTVoxCPM2Tts tts(
    "/models/voxcpm2-text-prefill.tflite",
    "/models/voxcpm2-token-step.tflite",
    "/models/voxcpm2-audio-encoder.tflite",
    "/models/voxcpm2-audio-decoder.tflite",
    "/models/tokenizer.json");

tts.set_reference(reference_pcm_16khz.data(), reference_pcm_16khz.size(), 16000);
tts.set_reference_transcript("Exact words spoken in the reference clip.");
tts.synthesize("Hello world", "en", [](const float* samples, size_t length, bool is_final) {
    // 48 kHz Float32 PCM, streamed in 64-step chunks (10.24 s each).
    // is_final marks the last chunk of the utterance.
});
```

For offline post-processing, such as spectral de-essing, use buffered delivery:

```cpp
speech_core::TtsSynthesisOptions options;
options.mode = speech_core::TtsSynthesisMode::Buffered;
options.postprocess_flags = speech_core::kTtsPostProcessDeEsser;

tts.synthesize_with_options("Hello world", "en", options,
    [](const float* samples, size_t length, bool is_final) {
        // One final callback with the full post-processed utterance for this text input.
    });
```

- 2B-parameter multilingual TTS, 48 kHz studio-quality output. Voice cloning and instruction-driven voice design supported by the upstream model.
- `set_reference()` enables reference-audio cloning. `set_reference_transcript()` is optional but recommended when the exact reference words are known; it switches the prompt layout to the upstream combined reference + continuation clone mode.
- `synthesize()` is the streaming path: the callback receives each decoder flush chunk, up to 64 AR steps / 10.24 s per chunk.
- `synthesize_with_options()` accepts the same `TtsSynthesisOptions` contract as the other TTS wrappers: `Streaming` preserves chunked delivery, while `Buffered` accumulates all PCM produced for the single submitted text input before invoking the callback once with `is_final=true`. `VoxCPM2SynthesisOptions` remains as a compatibility alias.
- Post-process flags currently include `kTtsPostProcessDeEsser`. Offline post-processing requires `Buffered` mode so it runs on the complete synthesized result, not on decoder flush chunks. If an app splits long text before calling VoxCPM2, buffering is scoped to each submitted text input.
- Ships as **four** LiteRT graphs plus an HF BPE tokenizer:
  - `text-prefill`: text + optional reference-audio prefix + optional prompt transcript/audio continuation → LM hidden + initial K/V cache
  - `token-step`: one autoregressive step (called up to 2048 times per generation), consumes and emits the K/V cache explicitly
  - `audio-encoder`: 16 kHz PCM reference clip → conditioning features
  - `audio-decoder`: latent → 48 kHz PCM output
- **Constructor** loads all four graphs via `LiteRTEngine` and verifies the tokenizer file exists; `synthesize()` runs the full pipeline (text-prefill → token-step ×N → audio-decoder) with the hand-rolled BPE tokenizer in [`voxcpm2_tokenizer.h`](../include/speech_core/models/voxcpm2_tokenizer.h).
- **Precision variants**: the repo *root* holds the `selective` bundle — fp16 weights except the **fp32 LocDiT** diffusion estimator, which needs full precision for clean cloned-voice sibilants (~8.7 GB; the ARM default). On **x86_64** the `fp32-p16/` subdir holds an **fp32 token-step** bundle (fp16 prefill, fp32 token-step + audio, ~13 GB): the fp16 token-step *over-generates* on x86 — its stop-margin (`stop_logits[1] > stop_logits[0]`) rounds the wrong way under x86 XNNPACK so the stop token never fires and the AR loop runs to the cap; the fp32 token-step computes that margin precisely and stops cleanly. `sc_voxcpm2_create_from_pretrained` picks the variant by architecture automatically. Download with `scripts/download_voxcpm2_litert.sh` (arch-aware); kept out of `download_models_litert.sh` because the bundle blows the standard nightly's `actions/cache` budget.
- Model files: [soniqo/VoxCPM2-LiteRT](https://huggingface.co/soniqo/VoxCPM2-LiteRT) — root (ARM / `selective`) and the `fp32-p16/` subdir (x86), each with `voxcpm2-{text-prefill,token-step,audio-encoder,audio-decoder}.tflite`, `tokenizer.json`, `config.json`

## LiteRTIndicMioTts

```cpp
#include <speech_core/models/litert_indic_mio_tts.h>

speech_core::LiteRTIndicMioTts tts(
    "/models/indicmio-text-prefill.tflite",
    "/models/indicmio-token-step.tflite",
    "/models/indicmio-audio-decoder.tflite",
    "/models/indicmio-ref-encoder.tflite",
    "/models/tokenizer.json");

tts.set_reference(reference_pcm.data(), reference_pcm.size(), 24000);  // cloning (optional)
tts.set_seed(1234);                                                    // reproducible take
tts.synthesize("नमस्ते, आज मौसम बहुत अच्छा है। <happy>", "", 
    [](const float* samples, size_t length, bool is_final) {
        // 24 kHz Float32 PCM, one final buffered chunk per utterance.
    });
```

- Indic-Mio: Hindi/Indic emotion TTS — a Qwen3-0.6B speech-token LM plus the
  MioCodec wave decoder. Emotion is controlled **inline in the text** with
  end-of-utterance suffix tags (`<happy> <sad> <angry> <disgust> <fear>
  <surprise>`); there is no separate style API, and the language is implicit
  in the text (the `language` argument is ignored).
- Voice cloning: `set_reference()` resamples the clip to 24 kHz, center-crops
  (preferred — zero padding dilutes the pooled embedding) or pads to the
  encoder's 10 s window, and caches the 128-dim global speaker embedding on
  the handle. Without a reference the model's default voice is used.
- Sampling defaults follow the upstream reference (temperature 0.9, top-k 50,
  top-p 0.9); EOS is suppressed until the first speech token so a take can
  never be empty. `set_seed(0)` draws a fresh seed per call (see
  `seed_used()`), a fixed seed reproduces a take — the host's regenerate
  contract.
- Ships as **four** LiteRT graphs plus the Qwen tokenizer:
  - `text-prefill`: 64-token chat prompt (right-padded) + last index → logits + K/V `[28,1,8,512,128]`
  - `token-step`: one AR step with an explicit functional K/V cache (ping-ponged host-side, no per-step copies)
  - `audio-decoder`: 384-token bucket + `valid_tokens` masking (pad codes cannot contaminate the real region) → STFT real/imag frames; the **host ISTFT** (`indic_mio_istft.h`, kissfft) reconstructs 24 kHz PCM
  - `ref-encoder`: 10 s @ 24 kHz reference → 128-dim global embedding
- The tokenizer is byte-level BPE (Qwen3 family) — different from every other
  tokenizer in this tree; `indic_mio_tokenizer.h` reimplements the Split-regex
  pretokenizer with utf8proc and is pinned by golden HF fixtures
  (`tests/data/indic_mio_tokenizer_fixtures.json`), as is the ISTFT
  (`tests/data/indic_mio_istft_*`). e2e smoke: `test_litert_indic_mio`
  (gated on `SPEECH_CORE_INDIC_MIO_BUNDLE`).
- Memory policy mirrors VoxCPM2: token-step/decoder/ref-encoder stay resident;
  the 1.1 GB prefill graph loads for its single call per synthesis and is
  released after. Bundle ≈ 2.6 GB total.
- `sc_indic_mio_create_from_pretrained("soniqo/Indic-Mio-LiteRT", ...)`
  downloads and caches the same bundle for C ABI users; set
  `SPEECH_CORE_DOWNLOAD_CONNECTIONS=1` only when a network or proxy breaks
  parallel HTTP ranges.
- Model files: [soniqo/Indic-Mio-LiteRT](https://huggingface.co/soniqo/Indic-Mio-LiteRT)
  — `indicmio-{text-prefill,token-step,audio-decoder,ref-encoder}.tflite`,
  `tokenizer.json`, `config.json` (manifest: token offsets, stop ids, prompt
  template, bucket sizes); the model card documents the full host contract.

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

## Parakeet-EOU (streaming ONNX)

```cpp
#include <speech_core/models/onnx_nemotron_streaming_stt.h>

// Parakeet-EOU-120M is wire-identical to Nemotron streaming (the same
// three-graph cache-aware RNN-T contract), so it runs through the same wrapper.
// Point it at the model directory; the wrapper self-configures dims, vocab, and
// the <EOU>/<EOB> token ids from config.json.
speech_core::OnnxNemotronStreamingStt stt(
    "/models/parakeet-eou-encoder.onnx",
    "/models/parakeet-eou-decoder.onnx",
    "/models/parakeet-eou-joint.onnx",
    "/models/vocab.json");

stt.begin_stream(16000);
stt.push_chunk(audio_chunk, chunk_len);
if (stt.end_of_utterance()) { /* turn ended — the model emitted <EOU> */ }
auto final = stt.end_stream();
```

- Parakeet-EOU-120M — multilingual (25 European) streaming RNN-T with inline
  end-of-utterance detection. **~231 MB peak RSS on a Galaxy S23 Ultra** (arm64 CPU,
  native; ~377 MB on desktop) — 5–6× lighter than Parakeet-TDT 0.6B (~1.1–1.3 GB)
  and comfortably real-time on a phone (RTF 0.21 on the S23 Ultra).
- Streaming windowing mirrors the reference session: a `melFrames * hop` window
  advanced by `outputFrames * subsamplingFactor * hop` samples (overlapping),
  committing `outputFrames` encoder frames per step. Pre-emphasis (config
  `preEmphasis`) is applied to the waveform with cross-window carry.
- Same runtime as Nemotron streaming: when `config.json` declares `eouTokenId` /
  `eobTokenId`, the decoder treats `<EOU>` as end-of-turn (surfaced through
  `end_of_utterance()`, not written into the transcript) and `<EOB>` as a soft
  boundary. A config without those ids is plain Nemotron and is byte-identical to
  before.
- **Decoding.** Greedy by default and byte-identical to prior releases. Set
  `Config.beam_size > 1` for modified RNN-T beam search: N hypotheses, each with
  its own predictor state and context position, carried across streaming windows.
  Greedy stays the default; beam is opt-in.
- **Under-emission correction (optional).** `Config.beam_emit_bonus` adds a
  fixed reward for each emitted text token, countering the blank bias of models
  whose beam search truncates. It defaults to `0`: the published Parakeet-EOU
  bundle does not under-emit at beam width 4 in measured recordings, while a
  positive reward can over-emit or hallucinate. Raise it only after observing
  truncation on a different export; the generic beam algorithm and reward
  threshold are covered by model-free unit tests.
- **Contextual biasing (shallow fusion).** `set_context_phrases()` biases beam
  search toward a caller-supplied phrase list — command words, a brand name, the
  contact / track names currently on the device. It rides on a tokenizer-agnostic
  character automaton (`ContextGraph`, Aho-Corasick) that matches on the *decoded
  surface text* of tokens, so it works even though the published EOU bundle ships
  only a decode vocabulary (no encoder tokenizer). Matches are anchored to word
  starts and a broken partial match refunds its bonus via the fail arc, so
  unrelated speech nets ~zero. Rebuild the phrase list per utterance to inject
  live entities; no effect unless `beam_size > 1`, and an empty list is a no-op.
- **Over-biasing guardrail (optional).** Uncapped, longer phrases accumulate a
  larger boost — the same behavior as sherpa-onnx / k2, where the score is tuned
  by hand and an over-wide beam or long list can let a phrase override clear
  audio (e.g. every segment collapsing to "what can you do"). Pass a positive
  `max_bonus` to `set_context_phrases()` to cap each phrase's per-character
  boost: sufficiently long phrases then all reach the same ceiling, so no single
  phrase dominates. Default 0 keeps the uncapped, tune-by-hand behavior. With a
  cap set, a wider beam and a longer bias list stay safe.

```cpp
speech_core::OnnxNemotronStreamingStt::Config cfg;
cfg.beam_size = 4;                        // opt into beam search
speech_core::OnnxNemotronStreamingStt stt(
    "/models/parakeet-eou-encoder.onnx", "/models/parakeet-eou-decoder.onnx",
    "/models/parakeet-eou-joint.onnx",   "/models/vocab.json", cfg);
// Per-utterance: the fixed command grammar + whatever is on the device now.
// The trailing max_bonus (6.0) caps each phrase's boost so a wide beam / long
// list can't override clear audio; pass 0 (default) for the uncapped behavior.
stt.set_context_phrases({"Soniqo", "set volume", "play music", "stop playing",
                         /* live contact + track names */},
                        /*per_char=*/1.5f, /*completion=*/3.0f, /*max_bonus=*/6.0f);
```

- Encoder INT8, decoder + joint FP32; 320 ms streaming chunks.
- Model files: [soniqo/Parakeet-EOU-120M-ONNX-INT8](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) — `parakeet-eou-{encoder,decoder,joint}.onnx`, `vocab.json`, `config.json`

## On-device benchmarks

Measured on a Samsung Galaxy S23 Ultra (SM-S918B, arm64), CPU only, INT8 where noted.
RTF is wall-seconds ÷ audio-seconds (lower is faster; <1.0 is faster than real time); RSS is peak
resident set. STT rows use a 20 s clip; TTS reports RTF or time-to-first-audio (TTFA);
the LLM row reports tool-call decode throughput in tokens/s.

| Model | Task | Backend | Peak RSS | Speed |
|---|---|---|---|---|
| Parakeet-EOU-120M | streaming STT + EOU | ONNX INT8 | **~232 MB** | 0.21 RTF |
| Omnilingual CTC-300M | multilingual STT | LiteRT | ~831 MB | 0.15 RTF |
| Nemotron streaming 0.6B | streaming STT | LiteRT | ~1.30 GB | 0.67 RTF |
| Parakeet-TDT 0.6B | STT (batch) | ONNX INT8 | ~1.15 GB | 0.082 RTF |
| SupertonicTTS-3 (99M) | TTS (preset voice) | LiteRT | ~832 MB | 0.34 RTF · ~1.1 s TTFA |
| Kokoro-82M (full graph, published two-thread CPU default) | TTS (preset voice) | ONNX FP32 | ~604 MiB | 1.81 RTF |
| Kokoro-82M (full graph, four-thread local tuning) | TTS (preset voice) | ONNX FP32 | ~604 MiB | 1.16 RTF |
| Kokoro-82M (realtime 3.0 s short-turn graph) | TTS (preset voice) | ONNX FP32 | ~527 MiB | **0.75–0.88 RTF** |
| Kokoro-82M (staged 60-frame reference) | TTS (preset voice) | LiteRT FP32 / XNNPACK | ~1.02 GiB | 0.55 p50 RTF |
| FunctionGemma-270M | LLM (tool calls) | LiteRT-LM | ~611 MB | ~118 tok/s |

- Supertonic runs 3 diffusion steps and streams (first audio ~1.1 s while the rest
  synthesizes). Each Kokoro text chunk is one non-autoregressive graph run, so its
  first audio for that chunk equals its synthesis time. The published two-thread
  full graph measured ~4.16 s for a 2.3 s line; four-thread local tuning measured
  ~2.66 s. The published 120-frame realtime graph measured a 1.73 s p50 for that line.
  Physical-device runs vary with thermal state (0.75–0.88 RTF here). The short-turn
  result describes an in-profile English reply, not a worst-case long-form result;
  output beyond the 2.8 s guarded limit is split and retried safely.
- Disabling the ONNX CPU memory arena (now the default) cut Parakeet-TDT peak RSS
  from ~1.34 GB to ~1.15 GB (−15%) for ~1% throughput.
- Parakeet-EOU is the lightest STT here while staying multilingual + streaming —
  its 120M size vs the 600M of the 0.6B models is the difference.
- FunctionGemma-270M runs the constrained function-call grammar via `liblitert-lm`
  (`LiteRTFunctionGemmaLLM`, CPU); ~118 tok/s is end-to-end for a tool call
  (prefill + decode). The Apple CoreML build reaches 128 tok/s on iPhone 16 Pro
  (see [speech-swift](https://github.com/soniqo/speech-swift/blob/main/docs/benchmarks/ios-coreml.md)).

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
    "/models/kokoro-e2e.onnx",
    "/models/kokoro_voices",   // directory of .bin voice embeddings
    "/models/kokoro_data");    // directory of vocab + dictionaries

tts.synthesize("Hello world", "en",
    [](const float* samples, size_t len, bool is_final) {
        // append to playback buffer
    });
```

For tightly bounded voice-agent replies, select a compatible 120-frame
short-turn graph. The official `kokoro-e2e-realtime.onnx` filename selects the
matching runtime profile automatically. The graph is published in the linked
public model repository and reuses the full graph's external weight file.

```cpp
auto profile = speech_core::KokoroTts::Config::short_turn_3s();
speech_core::KokoroTts tts(
    "/models/kokoro-e2e-realtime.onnx",
    "/models/voices", "/models", profile);
```

The explicit profile remains useful for a renamed graph. With the official
filename, the existing bool constructor selects it automatically:

```cpp
speech_core::KokoroTts tts(
    "/models/kokoro-e2e-realtime.onnx",
    "/models/voices", "/models", /*hw_accel=*/false);
```

`short_turn_3p5s()` remains available for a compatible 140-frame graph when
fewer split/retries are more important than the shorter graph's latency.

- Kokoro 82M, non-autoregressive synthesis; long input is sentence/chunk split
- 24 kHz Float32 output
- One callback per safe internal text chunk; `is_final=true` marks the final one
- Auto-switches voice on language change (en → af_heart, fr → ff_siwis, …)
- Phonemizer: GPL-free three-tier (dict + suffix stemming + rule-based G2P), no eSpeak dependency. See `kokoro_phonemizer.h` + `kokoro_multilingual.h`.
- Unsafe length, non-finite PCM, or numerical instability triggers bounded split/retry; output is never clamped or silently dropped
- Output post-processing: trailing-silence trim, 5 ms fade-in / 10 ms fade-out at the speech boundary
- Kokoro CPU sessions use four intra-op threads by default for its large
  inference run in each internal chunk. `SPEECH_CORE_KOKORO_ORT_THREADS` overrides the
  global `SPEECH_CORE_ORT_THREADS` setting when a device needs separate tuning.
- Default public bundle files: [soniqo/Kokoro-82M-ONNX](https://huggingface.co/soniqo/Kokoro-82M-ONNX) — `kokoro-e2e-realtime.onnx` (the Android default) or `kokoro-e2e.onnx`, one shared `kokoro-e2e.onnx.data` weight file (~310 MiB), plus `vocab_index.json`, `us_gold.json`, `us_silver.json`, `dict_{fr,es,it,pt,hi}.json`, and `voices/*.bin`.

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

A separate `test_litert_models` target is added when `SPEECH_CORE_WITH_LITERT=ON`, exercising the LiteRT wrappers (Silero VAD, Parakeet STT, Kokoro TTS, VoxCPM2 TTS, Indic-Mio TTS fixtures, WeSpeaker embedding, Pyannote segmentation, Omnilingual STT, Nemotron streaming STT) + the `DiarizationPipeline` + the VoxCPM2 tokenizer against `.tflite` artifacts:

```bash
scripts/fetch_litert.sh build/litert        # extracts libLiteRt from ai-edge-litert wheel
scripts/download_models_litert.sh           # standard public LiteRT bundles, including Kokoro
scripts/download_voxcpm2_litert.sh          # VoxCPM2 bundle (mixed int8/fp16, ~6.4 GB, optional)
cmake -B build -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR=$PWD/build/litert
cmake --build build
SPEECH_LITERT_MODEL_DIR=scripts/models-litert ctest --test-dir build --output-on-failure
```

Each per-model test skips cleanly when its files aren't in `SPEECH_LITERT_MODEL_DIR`. The weekly CI workflow (`.github/workflows/weekly-voxcpm2.yml`) downloads the VoxCPM2 bundle and runs an end-to-end synth → Parakeet STT round-trip; the daily nightly skips VoxCPM2 to keep the cache budget reasonable.

## Bring-your-own model

The interface contract is small. To use a non-ONNX backend (CoreML, MLX, Whisper.cpp, llama.cpp, remote API), inherit the relevant interface and pass an instance to `VoicePipeline`. The reference implementations live alongside the orchestration but are not required.

See speech-swift for a worked example with CoreML and MLX backends targeting the same conceptual interfaces.
