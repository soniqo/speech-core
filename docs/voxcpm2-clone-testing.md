# Testing VoxCPM2 voice cloning — quiet-reference rescue

VoxCPM2's `audio_encoder` conditions on both timbre **and** amplitude of the
reference clip, so output loudness tracks reference loudness within ±10%. A
reference recorded quietly (RMS ~0.002, typical of un-mastered phone-mic
captures) used to clone at sub-audible level — perceived as "broken speech" /
"missing audio".

Both wrappers (`LiteRTVoxCPM2Tts`, `OnnxVoxCPM2Tts`) now rescue quiet
references in `set_reference()`: when the clip's RMS (after the leading-silence
trim) is below **0.04**, it is scaled to **~0.08 RMS** with a **0.95 peak cap**.
Clips already at healthy loudness pass through unchanged — bit-exact with the
previous behaviour and with upstream `openbmb/VoxCPM2` (whose canonical
fixture `examples/reference_speaker.wav` measures RMS 0.0896; our 0.08 target
matches that distribution).

This doc shows how to verify the behaviour on any machine, two ways: via the
clone CLI (listening test) and via the regression test (automated).

## Prerequisites

- CMake ≥ 3.22, a C++17 toolchain (MSVC 2022 / clang / gcc), git, Python 3.10+
  with `numpy` and `scipy`.
- `libLiteRt` for your platform — on Windows the repo vendors it under
  `litert/`; on Linux/macOS run `scripts/fetch_litert.sh` to extract it from
  the `ai-edge-litert` PyPI wheel.
- The VoxCPM2 LiteRT bundle (~4.6 GB): `scripts/download_models_litert.sh`
  fetches it into `scripts/models-litert/`.
- Two reference clips:
  - **Quiet** (the case being fixed): any mono speech WAV with RMS below ~0.01.
    To fabricate one from a normal clip:
    ```bash
    python - <<'EOF'
    import scipy.io.wavfile as wav, numpy as np
    sr, x = wav.read("tests/data/test_audio.wav")
    wav.write("quiet_ref.wav", sr, (x * 0.04).astype(np.int16))  # ~25x quieter
    EOF
    ```
  - **Healthy** (the control): `tests/data/test_audio.wav` as-is (RMS ~0.05).

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR=path/to/litert        # repo's litert/ dir on Windows
cmake --build build --config Release --target speech_voxcpm2_clone test_litert_models -j
```

On Windows, copy `libLiteRt.dll` next to the executables in `build\Release\`.

## Way 1 — clone CLI (listening test)

Run the same text through both references:

`bundle_dir` is optional: omit it and the CLI reads `$SPEECH_LITERT_MODEL_DIR`
(else the per-user cache dir). An explicit first-arg directory still wins.

```bash
export SPEECH_LITERT_MODEL_DIR=scripts/models-litert

./build/speech_voxcpm2_clone quiet_ref.wav \
    "The quick brown fox jumps over the lazy dog" out_quiet.wav

./build/speech_voxcpm2_clone tests/data/test_audio.wav \
    "The quick brown fox jumps over the lazy dog" out_healthy.wav
```

Each run takes ~3–5 min on a modern CPU. Measure:

```bash
python - <<'EOF'
import scipy.io.wavfile as wav, numpy as np
for name in ("out_quiet.wav", "out_healthy.wav"):
    sr, x = wav.read(name); x = x.astype(np.float32) / 32768.0
    print(f"{name:18s} dur={len(x)/sr:5.2f}s "
          f"peak={float(np.max(np.abs(x))):.4f} "
          f"rms={float(np.sqrt((x**2).mean())):.4f}")
EOF
```

**Pass criteria:**

| Output | Expected RMS | Meaning |
|---|---|---|
| `out_quiet.wav` | **0.04 – 0.10** | rescue engaged; intelligible cloned speech |
| `out_healthy.wav` | 0.04 – 0.12 | unchanged from pre-fix behaviour (no-op path) |

On a pre-fix build (any commit before this fix), `out_quiet.wav` lands at
RMS ≈ 0.001–0.005 — audibly a whisper. That is the regression this guards.

Also listen to both in a DAW: the quiet-ref output must be intelligible
speech in the cloned voice (not amplified noise), and there must be no new
artifacts (clicks, distortion, EQ shifts) — the rescue is pure scalar gain.

## Way 2 — regression test (automated)

`tests/test_litert_models.cpp` includes `test_litert_voxcpm2_hindi_cloning`,
which clones five Hindi phrases and prints per-phrase `rms`/`peak`/`dur` so
loudness regressions are visible in stdout. It prefers
`tests/data/test_hindi_ref.wav` as the reference when that file exists —
drop in any quiet clip (it is intentionally not committed):

```bash
cp quiet_ref.wav tests/data/test_hindi_ref.wav

SPEECH_LITERT_MODEL_DIR=scripts/models-litert \
SPEECH_VOXCPM2_WAV_DUMP=/tmp/voxcpm2-wavs \
./build/test_litert_models
```

Look for the per-phrase lines:

```
test_litert_voxcpm2_hindi_cloning ... (stt=off)
  [stt-lang=... reference=test_hindi_ref ref_rms=0.0019 ...]
  [0] tokens=38 dur=6.08s rms=0.0674 peak=0.4911  ref="..."
```

Every phrase should report `rms` ≥ 0.02. On a pre-fix build with a quiet
reference, these lines show `rms=0.0012`-class values.

Optionally run the dumped WAVs through the spectral checker:

```bash
python scripts/spectrum_check.py /tmp/voxcpm2-wavs
```

Note: the `BASS DROPOUTS` heuristic in `spectrum_check.py` is noisy on clips
shorter than ~3 s — judge short phrases by the `rms` column, not the verdict.

## Tuning knobs (in both wrappers' `set_reference()`)

| Constant | Value | Rationale |
|---|---|---|
| `kQuietThreshold` | 0.04 | upstream-healthy references measure 0.09+; below 0.04 is outside the training distribution |
| `kTargetRms` | 0.08 | matches upstream `reference_speaker.wav` (RMS 0.0896) |
| `kPeakCap` | 0.95 | prevents clipping on clips with high peak-to-RMS ratio |
