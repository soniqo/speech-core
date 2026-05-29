# CUDA inference for speech-core — ONNX export plan for the LiteRT-only models

## Why this doc exists

NVIDIA GPU inference in speech-core runs through the **ONNX Runtime backend**
(`speech_core_models`, `SPEECH_CORE_WITH_ONNX`). The new `SPEECH_CORE_WITH_CUDA`
build option wires `CUDAExecutionProvider` / `TensorrtExecutionProvider` into
`onnx_engine.h`. LiteRT (`speech_core_models_litert`) **cannot** target CUDA —
its runtime is edge-only (OpenCL / Metal / WebGPU / NNAPI). So "CUDA for all
models" literally means **"every model needs an ONNX export"**.

Four models already ship ONNX (Silero VAD, Parakeet STT, Kokoro TTS,
DeepFilterNet3) and run on CUDA today via the engine change. The five below are
LiteRT-only and need an ONNX export before they can use the GPU.

All export tooling lives in the sibling **speech-models** repo under
`models/{name}/export/`. The conversion engine is PyTorch → `torch.onnx.export`
(opset 18 is the repo standard) → optional `onnxruntime.quantization.quantize_dynamic`
INT8 on MatMul/Gemm. Two of the five already have a `convert_onnx.py`.

## Ground truth found in speech-models

| Model | ONNX script today? | Closest working ONNX precedent in-repo |
|---|---|---|
| Pyannote segmentation | **Yes** — `pyannote-vad/export/convert_onnx.py` (validated) | itself |
| WeSpeaker embedding | **Yes** — `wespeaker/export/convert_onnx.py` (validated) | itself |
| Nemotron streaming STT | No (LiteRT only) | `parakeet-asr/export/convert_onnx.py` — same RNN-T family |
| Omnilingual STT | No (LiteRT only) | `wespeaker`/`pyannote` opset-18 single-graph pattern |
| VoxCPM2 TTS | No (LiteRT only) | none — novel 4-graph AR loop |

Key precedent: **Parakeet TDT already exports to ONNX** with a FastConformer
encoder graph (INT8) + a *fused decoder-joint* graph carrying explicit LSTM
state (`input_states_1/2` → `output_states_1/2`). Nemotron streaming is the same
NeMo RNN-T machinery, so the encoder/decoder/joint → ONNX recipe is proven; only
the cache tensors are new. Also notable: **Pyannote's LiteRT export already goes
*through* ONNX** (`onnx2tf` + `sng4onnx` in its `pyproject.toml`), i.e. a valid
Pyannote ONNX graph is produced as an intermediate today — the standalone
`convert_onnx.py` just stops at that intermediate.

---

## Ranked path to "CUDA for all models"

Ordered easiest/lowest-risk first. Rank = do-this-first ordering.

### 1. WeSpeaker ResNet34-LM embedding — **TRIVIAL, already done**

- **Export status:** `convert_onnx.py` exists and self-validates (runs a CPU
  ORT session, checks embedding shape + L2 norm). Opset 18, single graph,
  dynamic time axis (`dynamic_axes={"audio": {2: "samples"}}`).
- **Architecture:** plain ResNet34 (Conv2d stack) → stats pooling → FC →
  L2-normalized 256-d embedding. The one quirk — pyannote's fbank uses
  `torch.vmap`, untraceable — is *already patched* (`_patch_wespeaker_vmap`
  replaces it with a batch loop). Pure conv/matmul; no control flow, no state.
- **ORT EP:** **TensorRT.** Static-ish CNN, fixed op set, the textbook TRT case;
  builds one engine, runs fastest steady-state. CUDA EP is the safe fallback and
  is plenty fast already. Dynamic length axis → set a TRT optimization profile
  (min/opt/max samples) or pad to a fixed window.
- **Effort:** ~0 (export) + ~0.5 day (verify on a real GPU, wire the C++
  `WeSpeakerEmbedding` ONNX wrapper analogous to the existing Silero/Parakeet
  ONNX wrappers). **Risk: very low.**

### 2. Pyannote Segmentation 3.0 — **EASY, already done**

- **Export status:** `convert_onnx.py` exists and self-validates (checks the
  `[1,293,7]` powerset posteriors sum). Opset 18, single graph, **fully static**
  shapes (fixed 10 s / 160000-sample window → 293 frames).
- **Architecture:** SincNet/Conv frontend → **bi-LSTM** → linear → powerset
  classifier. The LSTM is the only thing to watch: `torch.onnx.export` lowers
  LSTM to the standard ONNX `LSTM` op, which both CUDA and TensorRT support, but
  fixed-shape export sidesteps every dynamic-loop hazard.
- **ORT EP:** **CUDA.** ONNX `LSTM` runs natively and reliably on the CUDA EP.
  TensorRT *can* take LSTM but is fussier about recurrent ops; given the model is
  tiny (~6 MB) the TRT engine-build cost isn't worth it — CUDA is the right call.
  TensorRT remains available as a layered fallback (the engine appends CUDA
  beneath TRT automatically).
- **Effort:** ~0 (export) + ~0.5 day (GPU verify, C++ ONNX wrapper for the
  segmentation model used by `diarization_pipeline`). **Risk: low** (LSTM-on-TRT
  is the only thing to confirm; CUDA path is safe regardless).

### 3. Omnilingual ASR CTC-300M — **EASY-MODERATE, new script needed**

- **Export status:** LiteRT only, but the LiteRT path is the green light: the
  `Wav2Vec2AsrTraceable` wrapper was explicitly written to be valid for **both
  `torch.jit.trace` and `torch.export`**, rebuilding fairseq2's `BatchLayout`
  from the tensor shape inside `forward`. `torch.onnx.export` (the dynamo=False
  TorchScript path the repo uses everywhere) is exactly that trace path, so the
  same wrapper drops straight into a new `convert_onnx.py`.
- **Architecture:** Wav2Vec2 CNN frontend → Transformer encoder → CTC linear
  head. Single forward pass, no autoregression, no recurrent state — the
  simplest possible "audio in → logits out" shape. Output `T = ceil(S/320)`.
- **The one quirk:** fairseq2's `BatchLayout` object. Already solved by the
  traceable wrapper. Watch for fairseq2 custom ops that lack ONNX symbolics —
  if any appear, they're isolated in the frontend and can be replaced with
  stock ops (same class of fix as the WeSpeaker vmap patch).
- **ORT EP:** **TensorRT** for the encoder (Transformer + Conv, static or
  profiled dynamic length → ideal TRT workload, big steady-state win at 300M
  params). CUDA fallback fine. Quantize INT8 (MatMul/Gemm only) as the LiteRT
  path already does.
- **Effort:** ~1–1.5 days (clone `convert_litert.py`, swap
  `litert_torch.convert` → `torch.onnx.export` with the same wrapper + dynamic
  length axis, add a CTC-decode validation like Parakeet's `test_onnx.py`).
  **Risk: low-moderate** (only unknown is a stray fairseq2 op without a
  symbolic; the traceable wrapper makes this unlikely).

### 4. Nemotron Speech Streaming 0.6B (RNN-T) — **MODERATE, new script, strong precedent**

- **Export status:** LiteRT only, but **Parakeet TDT already proves the recipe.**
  Nemotron's `convert.py` wrappers (`StreamingEncoderWrapper`, `DecoderWrapper`,
  `JointWrapper`) already produce clean static-shaped tensor I/O and already pass
  `torch.jit.trace` (the CoreML path). `torch.onnx.export` consumes the same
  traced modules.
- **Architecture:** 3 graphs — cache-aware FastConformer encoder, RNN-T LSTM
  prediction net, RNN-T joint. The streaming-specific complexity is the
  **explicit cache I/O**: `cache_last_channel` `[24,1,70,H]`, `cache_last_time`
  `[24,1,H,conv_cache]`, `cache_last_channel_len`, plus a rolling mel `pre_cache`.
  These become additional ONNX graph inputs *and* outputs — mechanically the
  same trick Parakeet's decoder-joint uses for `input_states_*`/`output_states_*`,
  just more tensors. The C++ worker already owns this state (it does for LiteRT),
  so the wire contract carries over.
- **Per-graph EP split (mirror the quantization split):**
  - **Encoder → CUDA** (not TensorRT initially). Cache-aware attention with
    per-step changing cache shapes + the `cache_aware_stream_step` gather/scatter
    is exactly what makes TRT engine-building painful (TRT wants stable shapes;
    streaming caches fight that). CUDA EP runs it correctly out of the box. INT8
    weight quantization as in the LiteRT export.
  - **Decoder (LSTM) → CUDA.** ONNX `LSTM`, keep **FP32** (the LiteRT/CoreML
    notes are explicit: the RNN-T LSTM hidden state drifts under INT8, corrupting
    logits — same lesson as Parakeet). CUDA, not TRT, for the recurrent op.
  - **Joint → CUDA or TensorRT.** Tiny FP32 MatMul; either works. Not worth a
    separate TRT engine.
  - Revisit TensorRT for the encoder *after* CUDA works, by fixing the chunk
    shape (the 80 ms config is single-frame and most static) and adding a TRT
    optimization profile. Treat as a perf follow-up, not a blocker.
- **Effort:** ~2–3 days (new `convert_onnx.py` reusing the existing wrappers;
  the work is declaring ~6 encoder inputs/outputs with correct `dynamic_axes`,
  plus a streaming-decode validation harness like the existing
  `_smoke_litert.py` / `_realtime_replay.py`). **Risk: moderate** — RNN-T export
  itself is proven by Parakeet; the risk is purely getting the cache-tensor
  input/output names + dynamic axes right and confirming the encoder's
  gather/scatter lowers cleanly to ONNX (it does for TorchScript→CoreML, so the
  ops exist).

### 5. VoxCPM2 TTS — **HARD, new multi-graph script, real unknowns**

- **Export status:** LiteRT only, and the **most complex model by far.** Not a
  single graph: the runtime is `text/audio prefill → repeated token/acoustic
  step → AudioVAE decode`, exported as **4 graphs** (`text_prefill`,
  `token_step`, `audio_encoder`, `audio_decoder`) plus a manifest the C++ worker
  uses to wire them. ONNX export must reproduce this same 4-graph split — ONNX is
  a static dataflow graph and likewise cannot represent the AR loop; the loop
  stays in C++.
- **Architecture quirks, per graph:**
  - **`text_prefill`** — MiniCPM base-LM + residual-LM over the full context;
    emits packed KV caches `[2, L, B, KVH, T, D]`. **This is the >2 GB graph.**
    FP32 it was ~10 GB / ~13 GiB RAM; the LiteRT export only fit by INT8-quantizing
    it. **ONNX has a hard 2 GB protobuf limit** → this graph *must* be exported
    with `torch.onnx.export(..., use_external_data_format=True)` so weights spill
    to a sidecar `.onnx_data` file. This is the single biggest gotcha. INT8 it as
    well (the recipe matches: weight-only INT8 / MatMul-Gemm).
  - **`token_step`** — one acoustic step with **explicit in/out KV cache tensors**
    and a hand-rolled functional attention (the converter already rewrote the
    upstream `StaticKVCache` `copy_`/in-place path into a functional
    write-mask + gather, precisely *because* the graph compilers reject in-place
    cache mutation). That same functional rewrite is what makes it ONNX-able —
    but it also contains `scaled_dot_product_attention` with `enable_gqa=True`,
    a CFM Euler solver (`solve_euler` with `torch.linspace`/`cos`), and rotary
    embeddings. SDPA + GQA need opset 18+ and a recent exporter; the Euler solve
    is plain math and should trace. **Marked "experimental" even for LiteRT** —
    expect ONNX to surface the same edges.
  - **`audio_encoder` / `audio_decoder`** (AudioVAE) — Conv stacks, kept **FP32**
    in LiteRT because the INT8 recipe rejected Conv ops. For ONNX, dynamic INT8
    on MatMul/Gemm-only would skip Conv anyway, so keep them FP32. The decoder
    has SR-conditioned conv layers (`sr_cond_model`) — straightforward conv,
    should export.
- **ORT EP:**
  - `text_prefill` + `token_step` → **CUDA.** Transformer-with-KV-cache and SDPA
    are well-supported on the CUDA EP; TensorRT struggles with the
    external-data >2 GB graph and the per-step dynamic cache, so do **not** start
    with TRT here. INT8 weights, FP32 activations.
  - `audio_encoder` / `audio_decoder` → **TensorRT** candidate (pure conv, FP32,
    static-friendly) once CUDA end-to-end works; CUDA fallback fine.
- **Effort:** ~5–8 days. Sub-tasks: (a) external-data ONNX export for the >2 GB
  prefill graph and confirm ORT loads the sidecar on GPU; (b) get `token_step`'s
  functional-cache + SDPA-GQA + Euler-solver graph through the exporter (the
  highest-risk item — may need an opset bump, an SDPA decomposition, or an
  exporter patch); (c) AudioVAE encoder/decoder graphs; (d) a 4-graph
  round-trip validation harness analogous to `smoke_litert_roundtrip.py`.
  **Risk: high** — the >2 GB external-data path and the experimental
  functional-KV-cache + GQA-SDPA lowering are both genuine unknowns until run on
  a GPU box.

---

## Summary table

| Rank | Model | ONNX script | Architecture | Primary EP | Effort | Risk |
|---|---|---|---|---|---|---|
| 1 | WeSpeaker ResNet34 | exists, validated | ResNet CNN + pool | **TensorRT** | ~0.5 d | very low |
| 2 | Pyannote Seg 3.0 | exists, validated | Conv + bi-LSTM | **CUDA** | ~0.5 d | low |
| 3 | Omnilingual CTC-300M | new (wrapper reusable) | Wav2Vec2 + CTC | **TensorRT** | ~1–1.5 d | low-mod |
| 4 | Nemotron streaming | new (Parakeet precedent) | RNN-T + cache I/O | **CUDA** (per-graph) | ~2–3 d | moderate |
| 5 | VoxCPM2 | new (4-graph, novel) | AR LM + AudioVAE | **CUDA** (per-graph) | ~5–8 d | high |

**Recommended sequence:** ship 1+2 immediately (export already exists — only GPU
verification + C++ ONNX wrappers remain), do 3 next (one reusable wrapper), then
4 (lean on Parakeet's RNN-T ONNX recipe), and treat 5 as its own mini-project
gated on the external-data + GQA-SDPA spikes.

**Cross-cutting EP rule of thumb observed across all five:** TensorRT for static,
conv-heavy, single-pass graphs (WeSpeaker, Omnilingual encoder, AudioVAE);
CUDA for anything with recurrent state, per-step dynamic KV caches, or >2 GB
external-data (Pyannote LSTM, every RNN-T/LM graph). The engine layers CUDA
beneath TensorRT automatically, and falls back to CPU if neither EP is in the
linked ORT build — so a wrong EP guess degrades, never crashes.
