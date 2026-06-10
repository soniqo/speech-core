import numpy as np, scipy.io.wavfile as wav, os, glob, math, sys

dump = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/voxcpm2-hi-wavs"
files = sorted(glob.glob(os.path.join(dump, "voxcpm2_hindi_*.wav")))
hdr = f"{'file':30s}  {'dur':>5s}  {'rms':>6s}  {'low_frac':>8s}  {'tstab':>6s}  {'spec_cv':>7s}  {'verdict':>20s}"
print(hdr); print("-" * len(hdr))

for f in files:
    sr, x = wav.read(f); x = x.astype(np.float32) / 32768.0
    N, H = 4096, 1024
    win = np.hanning(N)
    n = max(0, (len(x) - N) // H + 1)
    sp = np.empty((n, N // 2 + 1), dtype=np.float64)
    for t in range(n):
        X = np.fft.rfft(x[t*H:t*H+N] * win); sp[t] = X.real**2 + X.imag**2
    def b(hz): return int(round(hz * N / sr))
    tot = sp.sum(axis=1)
    lo  = sp[:, b(50):b(500)+1].sum(axis=1)
    act = tot > 0.05 * tot.max()
    low_frac = ((lo > 0.05 * tot) & act).sum() / max(1, act.sum())
    hb = sp[act, b(4000):b(12000)+1]
    if hb.shape[0] < 3:
        print(f"{os.path.basename(f):30s}  too short to analyse")
        continue
    bm = hb.mean(axis=0); bs = hb.std(axis=0); cv = bs / (bm + 1e-20)
    thr = np.quantile(bm, 0.95)
    tall = bm > thr
    stationary = tall & (cv < 0.5)
    tstab = stationary.sum() / max(1, tall.sum())
    top3 = np.argsort(bm)[-3:]
    spec_cv = float(cv[top3].mean())
    rms = float(np.sqrt(x.astype(np.float64).mean() ** 2 + (x.astype(np.float64) ** 2).mean()))
    flags = []
    if low_frac < 0.95:                       flags.append("BASS DROPOUTS")
    if tstab   > 0.30 and spec_cv < 0.50:     flags.append("STRIATIONS")
    verdict = ", ".join(flags) if flags else "clean"
    print(f"{os.path.basename(f):30s}  {len(x)/sr:5.2f}  {rms:6.4f}  {low_frac:8.2f}  {tstab:6.2f}  {spec_cv:7.3f}  {verdict:>20s}")
