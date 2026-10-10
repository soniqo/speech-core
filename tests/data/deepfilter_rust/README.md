These are mono 48 kHz little-endian Float32 PCM goldens from the actual upstream
Rust `DfTract::process()` at revision `d375b2d8309e0935d165700c91da9de862a99c31`,
using its official `DeepFilterNet3_onnx.tar.gz` bundle and tract 0.21.4.
`manifest.json` records model options, checksums, and observed stage counts.

The input combines speech from the existing `../test_audio.wav` fixture with
deterministic noise, startup silence, a long silent gap, quiet nonzero audio,
resumed speech, and a partial final 480-sample hop. Each output starts with the
C++ packet adapter's 480 zero samples; all subsequent samples come directly from
Rust, including the delayed tail. Numerical tests compare every sample within
`2e-5`; they also require exact C++ output invariance across packet boundaries.

To regenerate:

```bash
git clone https://github.com/Rikorose/DeepFilterNet.git /tmp/deepfilter-reference
git -C /tmp/deepfilter-reference checkout d375b2d8309e0935d165700c91da9de862a99c31
cp scripts/deepfilter_rust_reference.rs /tmp/deepfilter-reference/libDF/src/bin/speech-core-reference.rs
cargo build --manifest-path /tmp/deepfilter-reference/Cargo.toml -p deep_filter \
  --no-default-features --features capi --release --bin speech-core-reference
python3 scripts/generate_deepfilter_rust_fixtures.py /tmp/deepfilter-reference
```

The generator uses NumPy and Python's standard library. Rust and its dependencies
are needed only to regenerate these reference files, not to run speech-core or
its C++ tests. The default case targets the library's `RuntimeParams::default()`;
the `cli` case targets the CLI's different `-15 / 35 / 35` dB thresholds. The
forced `erb_only`, `clean`, and `zero_mask` cases exercise policies that may be
rare on a particular speech sample. `passthrough` tests Rust's zero-dB attenuation
bypass and its shorter alignment delay. Separate short CLI-reference recordings
exercise flushes at 1, 479, 480, and 481 samples.
