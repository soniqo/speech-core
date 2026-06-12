# fdb_bench — Full-Duplex Bench driver

`fdb_bench` runs each Full-Duplex Bench (FDB v1.0) audio sample through
`VoicePipeline` and writes a `<sample_id>.wav` + `<sample_id>.json` per
sample into the chosen output directory. The driver is the C++ side of
the FDB workflow; scoring still lives in upstream Python (M4, future).

## What M1 is, and what it isn't

In:

- Corpus iterator that walks the real FDB v1.0 layout
  (`v1_0/{candor_pause_handling,synthetic_pause_handling,
  candor_turn_taking,synthetic_user_interruption,icc_backchannel}/<id>/`).
- Tiny WAV reader/writer vendored under `examples/fdb_bench/`.
- Per-sample driver loop with mock STT + mock TTS by default so the
  binary builds and runs without ORT or model files.
- `OllamaLLM` adapter (PR #61) as the only LLM backend.
- Smoke `ctest` (`test_fdb_bench_smoke`) that runs the whole loop
  against a checked-in 4-sample fixture and an in-process Ollama mock.

Out:

- M2 — real STT/TTS (`--stt parakeet`, `--tts kokoro`) is wired but
  requires `SPEECH_CORE_WITH_ONNX=ON` and model files; treated as a
  follow-up.
- M3 — per-run summary CSV.
- M4 — bridge to the upstream Python eval scripts (CrisperWhisper +
  JSD / pause-handling metrics).

## What FDB is

[Full-Duplex Bench v1.0](https://github.com/DanielLin94144/Full-Duplex-Bench)
(arXiv 2503.04721, ASRU 2025) is a 727-sample public benchmark for
evaluating voice agents on **turn-taking behaviour** — when to wait, when
to respond, when to yield, and (for full-duplex models) when to
backchannel. Four scoring categories:

| Category | What it tests | Fair for cascades? |
|---|---|---|
| Pause Handling | User pauses mid-utterance — agent must wait | yes |
| Smooth Turn-Taking | User finishes — agent must respond promptly | yes |
| User Interruption | User barges in — agent must stop and re-respond | yes |
| Backchannel | Agent emits "mhm" while user talks | **no** — cascades can't vocalize concurrently |

speech-core is a cascaded STT → LLM → TTS pipeline, so the backchannel
category is architecturally N/A; the other three are fair to report.

## Fetching the corpus

The full corpus is hosted on Google Drive (no single tarball; browse
category-by-category):

  https://drive.google.com/drive/folders/1DtoxMVO9_Y_nDs2peZtx3pw-U2qYgpd3

Expected on-disk layout after download:

    v1_0/
    ├── candor_pause_handling/<id>/        (216 samples; 48 kHz)
    ├── synthetic_pause_handling/<id>/     (137 samples; 16 kHz)
    ├── candor_turn_taking/<id>/           (119 samples; 16 kHz)
    ├── synthetic_user_interruption/<id>/  (200 samples; 16 kHz)
    └── icc_backchannel/<id>/              ( 55 samples; 16 kHz)

Each `<id>/` directory contains `input.wav`, a category-specific
annotation JSON (`pause.json`, `turn_taking.json`, or `interrupt.json`;
icc_backchannel has none), and `transcription.json` (a CrisperWhisper
word-level transcript). Candor pause-handling clips are 48 kHz; the
driver resamples to 16 kHz at load time via `speech_core::Resampler`.

For smoke runs without the full download you can also point at the
single-sample-per-category subset upstream ships under
[`v1_v1.5/evaluation/example_data`](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/main/v1_v1.5/evaluation/example_data),
or at this repo's tiny `tests/data/fdb_mini/v1_0/`.

## Building

Minimal — mock STT + mock TTS + real Ollama:

    cmake -B build -DSPEECH_CORE_WITH_OLLAMA=ON
    cmake --build build --target fdb_bench

Full — Parakeet STT + Kokoro TTS via ONNX Runtime:

    cmake -B build \
        -DSPEECH_CORE_WITH_OLLAMA=ON \
        -DSPEECH_CORE_WITH_ONNX=ON \
        -DORT_DIR=/path/to/onnxruntime
    cmake --build build --target fdb_bench

The driver only builds when `SPEECH_CORE_WITH_OLLAMA=ON`. Without ONNX
the `--stt parakeet` and `--tts kokoro` switches produce a clean error
at startup; the mock backends work either way.

## Running

CI-style mock run against the fixture:

    ./build/fdb_bench \
        --corpus-dir tests/data/fdb_mini/v1_0 \
        --out-dir /tmp/fdb_out \
        --llm-model llama3.2:1b

One category through the full real pipeline:

    ./build/fdb_bench \
        --corpus-dir /path/to/v1_0 \
        --out-dir /tmp/fdb_out \
        --llm-model qwen2.5:7b \
        --stt parakeet --parakeet-dir /path/to/parakeet \
        --tts kokoro   --kokoro-dir   /path/to/kokoro \
        --category candor_pause_handling \
        --limit 50

Run `./build/fdb_bench --help` for the full flag list.

## Output layout

For each input sample `<id>`, the driver writes:

- `<out-dir>/<id>.wav` — the agent's TTS response audio (PCM16 mono at
  the TTS backend's native rate; mock TTS = 24 kHz).
- `<out-dir>/<id>.json` — per-sample record:

      {
        "sample_id": "1",
        "category": "candor_pause_handling",
        "category_dir": "candor_pause_handling",
        "input_wav": "...",
        "input_duration_sec": 1.0,
        "ground_truth_transcript": "hello world",
        "agent_transcript_input": "hello world",
        "output_wav": "1.wav",
        "output_duration_sec": 0.21,
        "output_sample_rate": 24000,
        "timings_ms": {
          "stt": 0,
          "llm": 12,
          "tts": 3,
          "ttft_first_audio_from_speech_end": 18,
          "total_wall": 25
        },
        "stt_backend": "mock",
        "tts_backend": "mock",
        "llm_model": "llama3.2:1b",
        "error": ""
      }

When a sample fails (e.g. LLM transport error), the JSON is still
written with the `error` field populated so the eventual M3 summary
roll-up has a complete record per sample.

Final stdout line:

    fdb_bench: samples_ok=N errors=M avg_ttft_ms=X out_dir=...

## How M2-M4 layer on top

- **M2** — drop `--stt parakeet` / `--tts kokoro` defaults once latency
  and quality are validated against a small corpus subset. No CLI break.
- **M3** — post-process `<out-dir>/*.json` into a single summary CSV
  (`fdb_summary.csv`) suitable for one-line CI assertions and trend
  tracking across runs.
- **M4** — bridge `<out-dir>/<id>.wav` files to the upstream
  `eval_*.py` scripts (CrisperWhisper output transcripts + JSD /
  pause-handling metrics) and a weekly workflow that pulls the corpus
  to a runner and asserts no regression vs a checked-in baseline.

## Regenerating the smoke fixture

`tests/data/fdb_mini/` is checked in. To regenerate (e.g. after
extending the iterator to cover a new category):

    python3 scripts/make_fdb_mini.py

Stdlib-only — no numpy/scipy.
