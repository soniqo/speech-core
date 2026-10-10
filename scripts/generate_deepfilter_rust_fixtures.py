#!/usr/bin/env python3
"""Regenerate waveform goldens using the actual pinned upstream Rust runtime.

In an upstream checkout at UPSTREAM_COMMIT, copy deepfilter_rust_reference.rs
into libDF/src/bin/speech-core-reference.rs, then build:
  cargo build -p deep_filter --no-default-features --features capi --release \
    --bin speech-core-reference
Pass that checkout's path below. Rust is a fixture-generation dependency only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT = "d375b2d8309e0935d165700c91da9de862a99c31"
CASES = {
    "default": (-10, 30, 20, 0, 100),
    "cli": (-15, 35, 35, 0, 100),
    "post_filter": (-10, 30, 20, 0.02, 100),
    "limited": (-10, 30, 20, 0.02, 12),
    "passthrough": (-10, 30, 20, 0, 0),
    "erb_only": (-100, 100, -100, 0, 100),
    "clean": (-100, -100, -100, 0.02, 100),
    "zero_mask": (100, 100, 100, 0.02, 100),
}


def generate(upstream, destination):
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError("Use the pinned upstream Rust revision")
    subprocess.run(["git", "-C", str(upstream), "diff", "--exit-code", "--",
                    "libDF/src/lib.rs", "libDF/src/tract.rs"], check=True)
    root = Path(__file__).resolve().parent.parent
    fixture = root / "tests/data/test_audio.wav"
    with wave.open(str(fixture), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError("Expected the existing 24 kHz mono PCM16 speech fixture")
        speech = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    def segment(offset, count):
        positions = np.arange(count) / 2 + offset * 24000
        return np.interp(positions, np.arange(speech.size), speech).astype(np.float32)
    rng = np.random.default_rng(147)
    first = segment(5, 19200) + rng.uniform(-0.02, 0.02, 19200).astype(np.float32)
    second = segment(6.5, 24000) + rng.uniform(-0.01, 0.01, 24000).astype(np.float32)
    # Include a gap long enough to exercise Rust's quiet-frame policy, a quiet
    # nonzero input, a resumed utterance, and a partial final hop.
    audio = np.concatenate((np.zeros(1440, np.float32), first, np.zeros(4800, np.float32),
                            np.full(960, 0.0001, np.float32), second, first[:77])).astype("<f4")
    destination.mkdir(parents=True, exist_ok=True)
    input_path = destination / "input.f32"
    input_path.write_bytes(audio.tobytes())
    executable = upstream / "target/release/speech-core-reference"
    bundle = upstream / "models/DeepFilterNet3_onnx.tar.gz"
    metadata = {"upstream_commit": commit, "source": "DfTract::process, mono Float32 at 48000 Hz",
                "input_samples": int(audio.size), "input_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "tract_version": "0.21.4",
                "cargo_lock_sha256": hashlib.sha256((upstream / "Cargo.lock").read_bytes()).hexdigest(),
                "cases": {}}
    for name, options in CASES.items():
        output_path = destination / (name + ".f32")
        result = subprocess.check_output([str(executable), str(bundle), str(input_path), str(output_path),
                                          *map(str, options)], text=True).strip()
        output = np.fromfile(output_path, dtype="<f4")
        delay = 480 if abs(options[-1]) < 0.01 else 1920
        if output.size != audio.size + delay or not np.isfinite(output).all():
            raise ValueError("Invalid Rust reference output")
        metadata["cases"][name] = {"options": options, "samples": int(output.size), "stages": result,
                                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}
        print(name, result)
    short_input = first[:481].astype("<f4")
    (destination / "short_input.f32").write_bytes(short_input.tobytes())
    metadata["short_cases"] = {}
    for count in (1, 479, 480, 481):
        partial_path = destination / "short_source.f32"
        partial_path.write_bytes(short_input[:count].tobytes())
        output_path = destination / f"short_{count}.f32"
        subprocess.run([str(executable), str(bundle), str(partial_path), str(output_path),
                        *map(str, CASES["cli"])], check=True, stdout=subprocess.DEVNULL)
        output = np.fromfile(output_path, dtype="<f4")
        if output.size != count + 1920 or not np.isfinite(output).all():
            raise ValueError("Invalid short Rust reference output")
        metadata["short_cases"][str(count)] = {"options": CASES["cli"],
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()}
    partial_path.unlink()
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent /
                        "tests/data/deepfilter_rust")
    args = parser.parse_args()
    generate(args.upstream.resolve(), args.output.resolve())
