#!/usr/bin/env python3
"""fdb_score — bridges fdb_bench output to upstream FDB v1.0 metrics.

Reads a directory of fdb_bench output JSONs + WAVs and produces:

  - fdb_report.md     human-readable per-category summary
  - fdb_score.json    machine-readable score for trend tracking / CI gates

Three of FDB's four categories are scored using the upstream rule
(duration < 1s AND num_words <= 3 => not a turn). Backchannel is
reported as "N/A (architectural)" — a cascaded STT->LLM->TTS pipeline
cannot vocalize concurrently with user audio, so JSD against the GT
distribution is structurally undefined for us.

Two modes:

  --asr-mode whisper    Transcribe output.wav with faster-whisper to
                        derive (word_count, duration) per sample.
                        Default model: distil-whisper/distil-small.en.
                        Caches results under <in-dir>/.asr_cache/.

  --asr-mode metadata   Skip ASR. Use output_duration_sec from the JSON
                        and estimate word count from
                        agent_transcript_input (the prompt fed to the
                        LLM) when the response itself isn't recorded.
                        Faster and deterministic; suitable for
                        baseline-gated CI when --tts mock is used.

Pass --baseline <path> to gate on a checked-in fdb_baseline.json; the
script prints a per-metric diff table and exits 1 on regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Constants ---------------------------------------------------------------
# Exact thresholds upstream eval_pause_handling.py / eval_smooth_turn_taking.py
# / eval_user_interruption.py all share.
TURN_DURATION_THRESHOLD_S = 1.0
TURN_NUM_WORDS_THRESHOLD = 3

CASCADE_FAIR_CATEGORIES = {
    "candor_pause_handling",
    "synthetic_pause_handling",
    "smooth_turn_taking",
    "user_interruption",
}
BACKCHANNEL_CATEGORY = "backchannel"

# Categories whose lower TOR is better (the agent SHOULD stay silent
# during pauses).
LOWER_IS_BETTER = {"pause_handling"}


@dataclass
class Sample:
    sample_id: str
    category: str           # the fdb_bench-emitted category name
    category_dir: str       # FDB v1.0 dir, used for grouping
    output_wav: str         # resolved absolute path
    input_wav: str
    input_duration_sec: float
    output_duration_sec: float
    agent_transcript_input: str
    ground_truth_transcript: str
    stt_backend: str
    tts_backend: str
    llm_model: str
    error: str
    timings_ms: dict
    # Filled in by transcribe_sample(); may be None if ASR is skipped.
    asr_text: Optional[str] = None
    asr_chunks: list = field(default_factory=list)
    asr_duration_sec: Optional[float] = None
    asr_first_word_start_s: Optional[float] = None


# --- ASR backends -----------------------------------------------------------
# A single faster-whisper model is loaded once and reused across samples.

_whisper_model = None


def _load_whisper(model_id: str):
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "fdb_score: faster-whisper not installed. "
            "Install with `pip install -r scripts/requirements-fdb-score.txt` "
            "or run with --asr-mode metadata.",
            file=sys.stderr,
        )
        sys.exit(2)
    _whisper_model = WhisperModel(model_id, device="cpu", compute_type="int8")
    return _whisper_model


def _asr_cache_path(in_dir: Path, wav: Path, model_id: str) -> Path:
    h = hashlib.sha1(
        f"{wav.resolve()}|{wav.stat().st_mtime}|{model_id}".encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = in_dir / ".asr_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{wav.stem}.{h}.json"


def transcribe_wav(wav: Path, in_dir: Path, model_id: str) -> dict:
    cache = _asr_cache_path(in_dir, wav, model_id)
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            pass  # regenerate

    model = _load_whisper(model_id)
    segments, _info = model.transcribe(
        str(wav), word_timestamps=True, vad_filter=False
    )
    chunks = []
    text_parts = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                chunks.append(
                    {"text": w.word, "timestamp": [w.start or 0.0, w.end or 0.0]}
                )
                text_parts.append(w.word)
        else:
            text_parts.append(seg.text)
    out = {"text": "".join(text_parts).strip(), "chunks": chunks}
    cache.write_text(json.dumps(out))
    return out


# --- Discovery + transcription orchestration --------------------------------


def discover_samples(in_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for js in sorted(in_dir.glob("*.json")):
        if js.parent.name == ".asr_cache":
            continue
        try:
            data = json.loads(js.read_text())
        except json.JSONDecodeError as ex:
            print(f"fdb_score: bad json in {js}: {ex}", file=sys.stderr)
            continue
        wav_name = data.get("output_wav", js.stem + ".wav")
        out_wav = (js.parent / wav_name).resolve()
        samples.append(
            Sample(
                sample_id=str(data.get("sample_id", js.stem)),
                category=str(data.get("category", "")),
                category_dir=str(data.get("category_dir", "")),
                output_wav=str(out_wav),
                input_wav=str(data.get("input_wav", "")),
                input_duration_sec=float(data.get("input_duration_sec", 0.0)),
                output_duration_sec=float(data.get("output_duration_sec", 0.0)),
                agent_transcript_input=str(data.get("agent_transcript_input", "")),
                ground_truth_transcript=str(data.get("ground_truth_transcript", "")),
                stt_backend=str(data.get("stt_backend", "")),
                tts_backend=str(data.get("tts_backend", "")),
                llm_model=str(data.get("llm_model", "")),
                error=str(data.get("error", "")),
                timings_ms=dict(data.get("timings_ms", {})),
            )
        )
    return samples


def fill_from_asr(s: Sample, in_dir: Path, model_id: str) -> None:
    wav = Path(s.output_wav)
    if not wav.exists():
        return
    asr = transcribe_wav(wav, in_dir, model_id)
    s.asr_text = asr.get("text", "")
    s.asr_chunks = asr.get("chunks", [])
    if s.asr_chunks:
        first = s.asr_chunks[0]["timestamp"][0]
        last = s.asr_chunks[-1]["timestamp"][1]
        s.asr_duration_sec = max(0.0, last - first)
        s.asr_first_word_start_s = max(0.0, first)


def fill_from_metadata(s: Sample) -> None:
    """Stand-in for ASR when --asr-mode metadata is used."""
    # We don't capture the LLM response text in the JSON yet; use the
    # STT input transcript as a length proxy. Word count from whitespace
    # split; duration from output_duration_sec. These are deterministic
    # under mock LLM + mock TTS, so the baseline gate stays stable.
    text = s.agent_transcript_input or ""
    word_count = max(1, len(text.split())) if text else 0
    s.asr_text = text
    s.asr_chunks = [
        {"text": w, "timestamp": [0.0, 0.0]} for w in text.split()
    ] if word_count else []
    s.asr_duration_sec = s.output_duration_sec
    s.asr_first_word_start_s = 0.0


# --- Scoring rules ----------------------------------------------------------


def classify_turn(num_words: int, duration_s: float) -> bool:
    """Upstream rule: a 'turn' is anything not (short AND few words)."""
    return not (duration_s < TURN_DURATION_THRESHOLD_S
                and num_words <= TURN_NUM_WORDS_THRESHOLD)


def _tor_for_sample(s: Sample) -> Optional[bool]:
    """True/False if scoreable; None if no audio was captured (error)."""
    if s.error:
        return None
    chunks = s.asr_chunks or []
    duration = s.asr_duration_sec or 0.0
    if not chunks and duration <= 0.0:
        return None
    return classify_turn(len(chunks), duration)


def _latency_for_sample(s: Sample) -> Optional[float]:
    """FDB latency = wall-clock time between user-turn-end and the agent's
    first audio chunk. We use fdb_bench's recorded
    timings_ms.ttft_first_audio_from_speech_end (the SpeechEnded ->
    first ResponseAudioDelta delta), NOT the Whisper-derived first-word
    timestamp inside output.wav — that WAV only contains the agent's
    response and starts at sample 0, so the asr first-word timestamp is
    always ~0 s and useless as a latency measure. This also makes the
    metric available in --asr-mode metadata (no Whisper required), which
    is what the weekly CI gate uses.
    """
    if s.error:
        return None
    ttft_ms = s.timings_ms.get("ttft_first_audio_from_speech_end")
    if ttft_ms is None:
        return None
    return float(ttft_ms) / 1000.0


def _aggregate(samples: list[Sample], higher_is_better: bool,
               metric: str) -> dict:
    n = len(samples)
    errors = sum(1 for s in samples if s.error)
    tor_flags = [v for v in (_tor_for_sample(s) for s in samples)
                 if v is not None]
    tor_mean = (sum(tor_flags) / len(tor_flags)) if tor_flags else 0.0
    latencies = [v for v in (_latency_for_sample(s) for s in samples)
                 if v is not None]
    latency_mean = (sum(latencies) / len(latencies)) if latencies else 0.0
    return {
        "category": metric,
        "n": n,
        "errors": errors,
        "tor_mean": round(tor_mean, 4),
        "latency_mean_s": round(latency_mean, 4),
        "higher_tor_is_better": higher_is_better,
    }


def score_categories(samples_by_cat: dict[str, list[Sample]]) -> list[dict]:
    results = []
    pause = samples_by_cat.get("pause_handling", [])
    smooth = samples_by_cat.get("smooth_turn_taking", [])
    intr = samples_by_cat.get("user_interruption", [])
    back = samples_by_cat.get(BACKCHANNEL_CATEGORY, [])
    results.append(_aggregate(pause, higher_is_better=False,
                              metric="pause_handling"))
    results.append(_aggregate(smooth, higher_is_better=True,
                              metric="smooth_turn_taking"))
    results.append(_aggregate(intr, higher_is_better=True,
                              metric="user_interruption"))
    results.append({
        "category": BACKCHANNEL_CATEGORY,
        "n": len(back),
        "errors": sum(1 for s in back if s.error),
        "tor_mean": None,
        "latency_mean_s": None,
        "note": ("Cascade architecture cannot vocalize concurrently "
                 "with user audio; backchannel score N/A."),
    })
    return results


def bucket_samples(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Group fdb_bench's emitted categories into the four scoring
    families: pause_handling (candor + synthetic), smooth_turn_taking,
    user_interruption, backchannel."""
    out: dict[str, list[Sample]] = {
        "pause_handling": [],
        "smooth_turn_taking": [],
        "user_interruption": [],
        "backchannel": [],
    }
    for s in samples:
        cat = s.category
        if cat in ("candor_pause_handling", "synthetic_pause_handling"):
            out["pause_handling"].append(s)
        elif cat == "smooth_turn_taking":
            out["smooth_turn_taking"].append(s)
        elif cat == "user_interruption":
            out["user_interruption"].append(s)
        elif cat == "backchannel":
            out["backchannel"].append(s)
    return out


# --- Output emitters --------------------------------------------------------


def metadata_from_samples(samples: list[Sample], asr_mode: str,
                          asr_model: str) -> dict:
    if not samples:
        return {}
    s = samples[0]
    return {
        "stt_backend": s.stt_backend,
        "tts_backend": s.tts_backend,
        "llm_model": s.llm_model,
        "asr_mode": asr_mode,
        "asr_model": asr_model,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def emit_report_md(path: Path, results: list[dict], meta: dict) -> None:
    lines = [
        "# FDB scoring report",
        "",
        f"- Generated: {meta.get('generated_at', '?')}",
        f"- STT backend: `{meta.get('stt_backend', '?')}`",
        f"- TTS backend: `{meta.get('tts_backend', '?')}`",
        f"- LLM model:   `{meta.get('llm_model', '?')}`",
        f"- ASR mode:    `{meta.get('asr_mode', '?')}` "
        f"(model: `{meta.get('asr_model', '-')}`)",
        "",
        "## Per-category metrics",
        "",
        "| Category | n | errors | TOR | Latency (s) | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        tor = "—" if r.get("tor_mean") is None else f"{r['tor_mean']:.3f}"
        lat = ("—" if r.get("latency_mean_s") is None
               else f"{r['latency_mean_s']:.3f}")
        direction = ""
        if "higher_tor_is_better" in r:
            direction = (" (higher better)"
                         if r["higher_tor_is_better"]
                         else " (lower better)")
        note = r.get("note", "")
        lines.append(
            f"| {r['category']}{direction} | {r['n']} | "
            f"{r['errors']} | {tor} | {lat} | {note} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def emit_score_json(path: Path, results: list[dict], meta: dict) -> None:
    payload = {"metadata": meta, "categories": results}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# --- Baseline gate ----------------------------------------------------------


def compare_against_baseline(score_json_path: Path,
                             baseline_path: Path) -> int:
    score = json.loads(score_json_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    tol = baseline.get("tolerances", {})
    tor_tol = float(tol.get("tor_mean", 0.05))
    lat_tol = float(tol.get("latency_mean_s", 0.05))

    base_by_cat = {c["category"]: c for c in baseline.get("categories", [])}
    new_by_cat = {c["category"]: c for c in score.get("categories", [])}

    print(f"{'category':<22} {'metric':<18} {'baseline':>10} "
          f"{'observed':>10} {'delta':>10} {'verdict':>10}")
    print("-" * 90)

    failed = False
    for cat, new in new_by_cat.items():
        base = base_by_cat.get(cat)
        if not base:
            print(f"{cat:<22} {'(no baseline)':<18}")
            continue
        for metric in ("tor_mean", "latency_mean_s"):
            b_val = base.get(metric)
            n_val = new.get(metric)
            if b_val is None or n_val is None:
                continue
            delta = n_val - b_val
            this_tol = tor_tol if metric == "tor_mean" else lat_tol
            # Direction-aware: for TOR on pause_handling, higher is worse.
            # For TOR on smooth/interruption, lower is worse. For latency,
            # higher is worse universally.
            regressed = False
            if metric == "tor_mean":
                if new.get("higher_tor_is_better", True):
                    regressed = delta < -this_tol
                else:
                    regressed = delta > this_tol
            else:  # latency
                regressed = delta > this_tol
            verdict = "REGRESSED" if regressed else "ok"
            if regressed:
                failed = True
            print(f"{cat:<22} {metric:<18} {b_val:>10.4f} "
                  f"{n_val:>10.4f} {delta:>+10.4f} {verdict:>10}")
    return 1 if failed else 0


# --- Self-test --------------------------------------------------------------


def self_test() -> int:
    """Smoke test driven by canned in-memory samples — no I/O, no ASR."""
    samples = [
        Sample(sample_id="1", category="candor_pause_handling",
               category_dir="candor_pause_handling", output_wav="",
               input_wav="", input_duration_sec=1.0, output_duration_sec=0.2,
               agent_transcript_input="", ground_truth_transcript="hi",
               stt_backend="mock", tts_backend="mock", llm_model="test",
               error="", timings_ms={"stt": 1, "llm": 2, "tts": 3,
                                     "ttft_first_audio_from_speech_end": 5,
                                     "total_wall": 11}),
        Sample(sample_id="2", category="smooth_turn_taking",
               category_dir="candor_turn_taking", output_wav="",
               input_wav="", input_duration_sec=1.0, output_duration_sec=2.5,
               agent_transcript_input="hello there friend",
               ground_truth_transcript="", stt_backend="mock", tts_backend="mock",
               llm_model="test", error="", timings_ms={}),
    ]
    for s in samples:
        fill_from_metadata(s)
    buckets = bucket_samples(samples)
    results = score_categories(buckets)
    by_cat = {r["category"]: r for r in results}

    # candor_pause_handling: 1 word, 0.2s -> NOT a turn (lower TOR is good)
    assert by_cat["pause_handling"]["tor_mean"] == 0.0, by_cat
    # smooth_turn_taking: 3 words, 2.5s -> IS a turn
    assert by_cat["smooth_turn_taking"]["tor_mean"] == 1.0, by_cat
    # user_interruption: no samples
    assert by_cat["user_interruption"]["n"] == 0
    # backchannel: no samples, N/A
    assert by_cat[BACKCHANNEL_CATEGORY]["tor_mean"] is None

    # Direct classify_turn checks.
    assert classify_turn(0, 0.0) is False
    assert classify_turn(2, 0.5) is False  # short + few words
    assert classify_turn(4, 0.5) is True   # many words wins over short
    assert classify_turn(2, 1.5) is True   # long wins over few words

    print("fdb_score self-test: ok")
    return 0


# --- Main -------------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Score an fdb_bench output directory.")
    p.add_argument("--in-dir", help="fdb_bench out-dir of *.json + *.wav")
    p.add_argument("--report-md", help="output markdown summary")
    p.add_argument("--score-json", help="output machine-readable score")
    p.add_argument("--baseline", default=None,
                   help="checked-in baseline JSON; regression exits 1")
    p.add_argument("--asr-mode", default="metadata",
                   choices=["metadata", "whisper"],
                   help="ASR strategy (default: metadata — no ASR)")
    p.add_argument("--asr-model", default="distil-whisper/distil-small.en",
                   help="faster-whisper model id when --asr-mode whisper")
    p.add_argument("--self-test", action="store_true",
                   help="run the inline scoring sanity check and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.in_dir or not args.report_md or not args.score_json:
        p.error("--in-dir, --report-md, --score-json are required "
                "(or pass --self-test)")
        return 2

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"fdb_score: not a directory: {in_dir}", file=sys.stderr)
        return 1
    report_md = Path(args.report_md)
    score_json = Path(args.score_json)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    score_json.parent.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(in_dir)
    if not samples:
        print(f"fdb_score: no *.json under {in_dir}", file=sys.stderr)
        return 1
    print(f"fdb_score: {len(samples)} samples, asr_mode={args.asr_mode}",
          file=sys.stderr)

    for s in samples:
        if args.asr_mode == "whisper":
            fill_from_asr(s, in_dir, args.asr_model)
        else:
            fill_from_metadata(s)

    buckets = bucket_samples(samples)
    results = score_categories(buckets)
    meta = metadata_from_samples(samples, args.asr_mode, args.asr_model)
    emit_report_md(report_md, results, meta)
    emit_score_json(score_json, results, meta)
    print(f"fdb_score: wrote {report_md} + {score_json}", file=sys.stderr)

    if args.baseline:
        rc = compare_against_baseline(score_json, Path(args.baseline))
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
