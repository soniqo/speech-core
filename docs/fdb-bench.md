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

Out: nothing — M1-M4 are done.

M2 (done) — real STT/TTS (`--stt parakeet`, `--tts kokoro`) is now
usable when built with `-DSPEECH_CORE_WITH_ONNX=ON`. Point at a flat
models directory (the layout produced by `scripts/download_models.sh`)
via `--models-dir`, or override per-family with `--parakeet-dir` /
`--kokoro-dir`.

M3 (done) — `fdb_summary` reads an out-dir of `<category>__<id>.json`
files and writes a single CSV with per-bucket sample / error counts
and p50/p90/p99 of stt / llm / tts / ttft / total_wall.

M4 (done) — `scripts/fdb_score.py` consumes an out-dir and emits a
`fdb_report.md` + `fdb_score.json` per FDB v1.0's scoring rules (TOR +
latency per category). `tests/data/fdb_baseline.json` is the
regression gate; `.github/workflows/weekly-fdb.yml` runs the cascade
against `tests/data/fdb_mini/` and diffs against the baseline.

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

One category through the full real pipeline (single flat models dir
matching `scripts/download_models.sh`):

    ./build/fdb_bench \
        --corpus-dir /path/to/v1_0 \
        --out-dir /tmp/fdb_out \
        --llm-model qwen2.5:7b \
        --stt parakeet --tts kokoro \
        --models-dir ./scripts/models \
        --category candor_pause_handling \
        --limit 50

Or with separate per-family directories:

    ./build/fdb_bench \
        --corpus-dir /path/to/v1_0 \
        --out-dir /tmp/fdb_out \
        --llm-model qwen2.5:7b \
        --stt parakeet --parakeet-dir /path/to/parakeet \
        --tts kokoro   --kokoro-dir   /path/to/kokoro \
        --category candor_pause_handling \
        --limit 50

Run `./build/fdb_bench --help` for the full flag list.

## Smoke vs. real-models integration test

`test_fdb_bench_smoke` always runs the mock-backend smoke (no model
files, no ORT). When built with `-DSPEECH_CORE_WITH_ONNX=ON` it
additionally compiles an opt-in `real_models_integration` block that
loads Parakeet + Kokoro + Silero from `$SPEECH_MODEL_DIR` and runs a
single real-speech sample (`tests/data/test_audio.wav`) through the
same per-sample driver path. It is **skipped silently** unless both env
vars are set:

    SPEECH_FDB_BENCH_INTEGRATION=1 \
    SPEECH_MODEL_DIR=$PWD/scripts/models \
    ctest --test-dir build -R test_fdb_bench_smoke --output-on-failure

Per-file skips also kick in if specific `.onnx` files are missing, so
partial model installs don't fail the test.

## Output layout

For each input sample `<id>`, the driver writes:

- `<out-dir>/<category_dir>__<id>.wav` — the agent's TTS response
  audio (PCM16 mono at the TTS backend's native rate; mock TTS =
  24 kHz). The `<category_dir>__` prefix prevents collisions because
  FDB v1.0 reuses `1/`, `2/`, … under every category directory.
- `<out-dir>/<category_dir>__<id>.json` — per-sample record:

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

## Summarizing a run

After `fdb_bench` writes per-sample JSONs, run `fdb_summary` to roll
them up into a single CSV with per-bucket percentiles:

    ./build/fdb_bench   --corpus-dir /path/to/v1_0 \
                        --out-dir /tmp/fdb_out \
                        --llm-model llama3.2:3b
    ./build/fdb_summary --in-dir   /tmp/fdb_out \
                        --out-csv  /tmp/fdb_summary.csv

The CSV is sorted by `(category, stt_backend, tts_backend, llm_model)`
so two runs against different model configurations diff cleanly. Errored
samples are counted in `samples` and `errors` but excluded from the
timing percentiles. Pass `-` as `--out-csv` to stream to stdout.

## Scoring a run

After `fdb_summary` extracts timing percentiles, `scripts/fdb_score.py`
applies the FDB v1.0 metric rules (TOR + first-word latency) per
category and emits a markdown report + machine-readable JSON:

    pip install -r scripts/requirements-fdb-score.txt
    python scripts/fdb_score.py \
        --in-dir     /tmp/fdb_out \
        --report-md  /tmp/fdb_report.md \
        --score-json /tmp/fdb_score.json \
        --asr-mode   metadata

Three of FDB's four categories are scored:

| Category | TOR direction | Notes |
|---|---|---|
| pause_handling      | lower better  | agent should stay silent during user's mid-utterance pause |
| smooth_turn_taking  | higher better | agent should take the turn after user finishes |
| user_interruption   | higher better | agent should yield + re-respond on barge-in |
| backchannel         | **N/A**       | cascaded STT→LLM→TTS cannot vocalize while user is speaking |

ASR modes:

- `--asr-mode metadata` (default) — no Whisper. Uses
  `output_duration_sec` + `agent_transcript_input` from the JSON as a
  deterministic proxy. Best for hermetic CI gates.
- `--asr-mode whisper` — `distil-whisper/distil-small.en` via
  faster-whisper (~150 MB cached download).

Pass `--baseline tests/data/fdb_baseline.json` to gate on a baseline.
Direction-aware: a TOR drop on smooth/interruption (higher better)
regresses; a TOR rise on pause_handling (lower better) does; any
latency rise does. Quality improvements never fail the gate.

The GPT-4-turbo LLM-judge for user_interruption is opt-in (requires
`OPENAI_API_KEY`); CI runs without it.

## Weekly regression gate

`.github/workflows/weekly-fdb.yml` runs the chain against
`tests/data/fdb_mini/v1_0` and diffs against
`tests/data/fdb_baseline.json`. Currently manual-trigger only
(`workflow_dispatch`); flip the cron block on once the baseline
reflects a meaningful corpus run. Baseline regen is manual: rerun
against a known-good commit, copy the `categories` block out of the
artifact's `fdb_score.json` into `tests/data/fdb_baseline.json`
(preserving the `_comment` + `tolerances`), re-commit.

## How everything fits together

- **M2 (done)** — `--stt parakeet` / `--tts kokoro` build out when
  `SPEECH_CORE_WITH_ONNX=ON`; `--models-dir` shortcut matches the
  `scripts/download_models.sh` layout; opt-in `real_models_integration`
  smoke test verifies the path.
- **M3 (done)** — `fdb_summary` reads `<out-dir>/*.json` and emits a
  single CSV with per-bucket sample / error counts and p50/p90/p99 of
  stt / llm / tts / ttft / total_wall.
- **M4 (done)** — `scripts/fdb_score.py` + `tests/data/fdb_baseline.json`
  + `.github/workflows/weekly-fdb.yml` close the loop from raw
  fdb_bench output to a regression-gated FDB metric.

## Regenerating the smoke fixture

`tests/data/fdb_mini/` is checked in. To regenerate (e.g. after
extending the iterator to cover a new category):

    python3 scripts/make_fdb_mini.py

Stdlib-only — no numpy/scipy.
