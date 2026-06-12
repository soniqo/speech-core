#!/usr/bin/env python3
"""Generate the tests/data/fdb_mini/ fixture for test_fdb_bench_smoke.

Synthesises one 1.0 s 16 kHz mono PCM16 WAV per FDB category along with
the per-category annotation JSON. Uses only the Python standard library
so any contributor can regenerate without numpy / scipy / pip installs.

Re-running is idempotent and overwrites in place. Not run by CI — the
output is committed into tests/data/fdb_mini/ so the C++ smoke test has
something to point at without any runtime tooling.
"""

import json
import math
import os
import struct
import wave

ROOT = os.path.join(os.path.dirname(__file__), "..",
                    "tests", "data", "fdb_mini", "v1_0")
SR = 16000
DUR_SEC = 1.0
N = int(SR * DUR_SEC)


def synthesize(seed_freq_hz: float) -> bytes:
    """One second of decaying sine at the given frequency, PCM16 LE."""
    out = bytearray(2 * N)
    amp = 0.25
    for i in range(N):
        # Mild decay so the waveform is non-stationary; this gives the
        # mock VAD a believable envelope when probabilities are scripted.
        env = 0.6 + 0.4 * math.exp(-3.0 * (i / N))
        v = amp * env * math.sin(2 * math.pi * seed_freq_hz * (i / SR))
        s = int(max(-1.0, min(1.0, v)) * 32767)
        struct.pack_into("<h", out, 2 * i, s)
    return bytes(out)


def write_wav(path: str, frames: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(frames)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def transcription(words):
    """Shape mirrors FDB v1.0's transcription.json — list of speakers,
    each with a list of timed word items."""
    return [{
        "speaker": 0,
        "item": [
            {"text": w, "timestamp": [round(i * 0.3, 2),
                                      round(i * 0.3 + 0.2, 2)]}
            for i, w in enumerate(words)
        ]
    }]


def main():
    samples = [
        ("candor_pause_handling",        220.0, ["hello", "world"],
         "pause.json",
         [{"text": "[PAUSE]", "timestamp": [0.4, 0.6]}]),
        ("candor_turn_taking",           330.0, ["how", "are", "you"],
         "turn_taking.json",
         [{"text": "[TURN-TAKING]", "timestamp": [0.8, 1.0]}]),
        ("synthetic_user_interruption",  440.0, ["tell", "me", "a", "long", "story"],
         "interrupt.json",
         [{"context": "tell me a long story",
           "interrupt": "wait stop",
           "timestamp": [0.4, 0.7]}]),
        ("icc_backchannel",              165.0, ["mhm"],
         None, None),
    ]

    for cat_dir, freq, words, ann_name, ann_body in samples:
        base = os.path.join(ROOT, cat_dir, "1")
        write_wav(os.path.join(base, "input.wav"), synthesize(freq))
        write_json(os.path.join(base, "transcription.json"),
                   transcription(words))
        if ann_name:
            write_json(os.path.join(base, ann_name), ann_body)

    readme = os.path.join(ROOT, "..", "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "# fdb_mini\n"
            "\n"
            "Synthetic 4-sample fixture mirroring the FDB v1.0 directory\n"
            "layout — one sample per category — consumed by\n"
            "`test_fdb_bench_smoke` so the FDB driver's smoke test has\n"
            "something to point at without downloading the real 727-\n"
            "sample corpus. Not real FDB data; do not benchmark against\n"
            "this. Regenerate with:\n"
            "\n"
            "    python3 scripts/make_fdb_mini.py\n"
        )

    print(f"wrote fixture to {os.path.normpath(ROOT)}")


if __name__ == "__main__":
    main()
