"""Run NeMo PyTorch FP32 reference on a corpus manifest, emit the same CSV
format as the C++ bench corpus modes:
  uid,nemo-fp32,audio_s,wall_ms,transcript

This is the GOLD-STANDARD reference. Compare ORT/LiteRT against it to
quantify (a) export fidelity (ORT vs NeMo) and (b) quantization cost
(LiteRT INT8 vs NeMo).

Uses NeMo's model.transcribe() — offline inference path; the cache-aware
streaming encoder still runs, but with the full sequence in one shot
(no chunk boundaries discarded), which is the model's quality ceiling.
"""

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

# Silence NeMo noise — same env knobs the convert scripts use.
warnings.filterwarnings("ignore")
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--nemo-id", default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace model id NeMo restores from. Cache lives in "
             "$HF_HOME/hub/models--nvidia--... and is reused across runs.",
    )
    args = ap.parse_args()

    import torch
    import soundfile as sf
    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo {args.nemo_id} ...", file=sys.stderr, flush=True)
    t_load = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(args.nemo_id)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"
    print(f"  loaded in {time.perf_counter()-t_load:.1f}s on {device}",
          file=sys.stderr, flush=True)

    rows = []
    with open(args.manifest, "r", encoding="utf-8", newline="") as m:
        reader = csv.reader(m)
        for r in reader:
            if not r or r[0].startswith("#"):
                continue
            if len(r) < 3:
                continue
            rows.append((r[0].strip(), r[1].strip()))

    print(f"Transcribing {len(rows)} utterances...", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("#uid,provider,audio_s,wall_ms,transcript\n")
        for i, (uid, wav_path) in enumerate(rows):
            try:
                info = sf.info(wav_path)
                audio_s = info.duration
            except Exception as e:
                print(f"  skip {uid}: {e}", file=sys.stderr)
                continue

            t0 = time.perf_counter()
            with torch.no_grad():
                # NeMo's transcribe returns a list; element may be a str
                # (old API) or a Hypothesis (newer API). Handle both.
                results = model.transcribe([wav_path], verbose=False)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            if results and hasattr(results[0], "text"):
                text = results[0].text
            elif results:
                text = str(results[0])
            else:
                text = ""
            text = text.replace(",", " ")

            out_f.write(f"{uid},nemo-fp32,{audio_s:.2f},{wall_ms:.1f},{text}\n")
            out_f.flush()

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(rows)} done", file=sys.stderr, flush=True)

    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
