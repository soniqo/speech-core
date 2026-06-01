#!/usr/bin/env python3
"""Compare Nemotron streaming STT engines (ORT-CUDA / ORT-CPU / LiteRT) on WER and RTF.

Reads a LibriSpeech-style manifest plus up to three engine CSVs produced by the
benchmark harness, and prints a fixed-width comparison table.

CSV row format (with leading '#' header line):
    uid,provider,audio_s,wall_ms,transcript

Manifest row format (no header):
    uid,wav_path,reference
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from jiwer import wer as _jiwer_wer  # type: ignore
    _HAVE_JIWER = True
except Exception:
    _HAVE_JIWER = False


_PUNCT_RE = re.compile(r"[\.,\!\?;:\'\"\-—]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip ASCII punctuation (.,!?;:'\"-—), collapse whitespace."""
    if text is None:
        return ""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def _levenshtein_words(ref: List[str], hyp: List[str]) -> int:
    """Classic word-level Levenshtein distance, O(len(ref) * len(hyp)) time, O(min) memory."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = curr
    return prev[-1]


def utterance_wer(reference: str, hypothesis: str) -> float:
    """Per-utterance WER in [0, +inf). Empty reference -> 0.0 if hyp also empty, else 1.0."""
    ref_n = normalize(reference)
    hyp_n = normalize(hypothesis)
    if _HAVE_JIWER:
        if not ref_n:
            return 0.0 if not hyp_n else 1.0
        try:
            return float(_jiwer_wer(ref_n, hyp_n))
        except Exception:
            pass
    ref_words = ref_n.split()
    hyp_words = hyp_n.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein_words(ref_words, hyp_words)
    return dist / len(ref_words)


def load_manifest(path: str) -> Dict[str, str]:
    """Returns {uid: reference}. Reference is taken verbatim from column 2 (already lowercase)."""
    refs: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue
            uid = row[0].strip()
            reference = row[2]
            # Some manifests may embed commas in the reference; rejoin trailing fields.
            if len(row) > 3:
                reference = ",".join(row[2:])
            refs[uid] = reference
    return refs


@dataclass
class EngineRow:
    uid: str
    provider: str
    audio_s: float
    wall_ms: float
    transcript: str


def load_engine_csv(path: str) -> List[EngineRow]:
    rows: List[EngineRow] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if len(row) < 5:
                continue
            uid = row[0].strip()
            provider = row[1].strip()
            try:
                audio_s = float(row[2])
                wall_ms = float(row[3])
            except ValueError:
                continue
            transcript = row[4] if len(row) == 5 else ",".join(row[4:])
            rows.append(EngineRow(uid, provider, audio_s, wall_ms, transcript))
    return rows


@dataclass
class EngineStats:
    label: str
    provider: str
    n: int
    wer_mean: float
    wer_p50: float
    wer_p95: float
    rtf: float
    wall_p50: float
    wall_p95: float


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    # Linear interpolation between closest ranks.
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_stats(label: str, rows: List[EngineRow], refs: Dict[str, str]) -> EngineStats:
    wers: List[float] = []
    walls: List[float] = []
    audio_total = 0.0
    wall_total = 0.0
    provider = ""
    for r in rows:
        if r.uid not in refs:
            continue
        if not provider:
            provider = r.provider
        wers.append(utterance_wer(refs[r.uid], r.transcript))
        walls.append(r.wall_ms)
        audio_total += r.audio_s
        wall_total += r.wall_ms

    n = len(wers)
    wer_mean = statistics.fmean(wers) if wers else 0.0
    wer_p50 = _percentile(wers, 50.0)
    wer_p95 = _percentile(wers, 95.0)
    wall_p50 = _percentile(walls, 50.0)
    wall_p95 = _percentile(walls, 95.0)
    rtf = (audio_total / (wall_total / 1000.0)) if wall_total > 0 else 0.0

    return EngineStats(
        label=label,
        provider=provider or label,
        n=n,
        wer_mean=wer_mean,
        wer_p50=wer_p50,
        wer_p95=wer_p95,
        rtf=rtf,
        wall_p50=wall_p50,
        wall_p95=wall_p95,
    )


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:.2f}%"


def _fmt_rtf(v: float) -> str:
    return f"{v:.2f}x"


def _fmt_ms(v: float) -> str:
    return f"{int(round(v))} ms"


def print_table(stats: List[EngineStats]) -> None:
    headers = ["provider", "n", "WER_mean", "WER_p50", "WER_p95", "RTF", "wall_p50", "wall_p95"]
    widths = [16, 2, 8, 7, 7, 5, 8, 8]

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for s in stats:
        cells = [
            s.label,
            str(s.n),
            _fmt_pct(s.wer_mean),
            _fmt_pct(s.wer_p50),
            _fmt_pct(s.wer_p95),
            _fmt_rtf(s.rtf),
            _fmt_ms(s.wall_p50),
            _fmt_ms(s.wall_p95),
        ]
        print(fmt_row(cells))


def print_verdict(stats: List[EngineStats]) -> None:
    by_label = {s.label: s for s in stats}
    baseline = by_label.get("litert")
    candidate = by_label.get("ort-cuda")
    if baseline is None or candidate is None:
        return

    delta = candidate.wer_mean - baseline.wer_mean
    abs_delta_pct = abs(delta) * 100.0
    threshold_pct = 1.0

    if abs_delta_pct <= threshold_pct:
        print(
            f"[OK]  ort-cuda WER within {threshold_pct:.1f}% absolute of litert-cpu baseline "
            f"-- safe to publish."
        )
    else:
        worse_or_better = "worse" if delta > 0 else "better"
        print(
            f"[WARN] ort-cuda WER is {abs_delta_pct:.1f}% absolute {worse_or_better} than "
            f"litert-cpu -- investigate before publishing."
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Nemotron streaming STT engines on WER/RTF.")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV (uid,wav_path,reference).")
    parser.add_argument("--ort-cuda", dest="ort_cuda", default=None, help="Path to ORT CUDA results CSV.")
    parser.add_argument("--ort-cpu", dest="ort_cpu", default=None, help="Path to ORT CPU results CSV.")
    parser.add_argument("--litert", dest="litert", default=None, help="Path to LiteRT results CSV.")
    args = parser.parse_args(argv)

    refs = load_manifest(args.manifest)
    if not refs:
        print(f"error: no references loaded from manifest {args.manifest!r}", file=sys.stderr)
        return 2

    sources: List[Tuple[str, Optional[str]]] = [
        ("ort-cuda", args.ort_cuda),
        ("ort-cpu", args.ort_cpu),
        ("litert", args.litert),
    ]

    stats: List[EngineStats] = []
    for label, path in sources:
        if not path:
            continue
        try:
            rows = load_engine_csv(path)
        except FileNotFoundError:
            print(f"warning: {label} CSV not found at {path!r} -- skipping.", file=sys.stderr)
            continue
        if not rows:
            print(f"warning: {label} CSV {path!r} contained no usable rows -- skipping.", file=sys.stderr)
            continue
        stats.append(compute_stats(label, rows, refs))

    if not stats:
        print("error: no engine CSVs provided (or all empty).", file=sys.stderr)
        return 2

    print_table(stats)
    print()
    print_verdict(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
