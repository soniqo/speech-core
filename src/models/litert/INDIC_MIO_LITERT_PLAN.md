# Indic-Mio LiteRT runtime — implementation plan

Runtime for the published `soniqo/Indic-Mio-LiteRT` bundle (all four graphs
fidelity-gated ≥0.995 against upstream; Hindi ASR roundtrip green). The model
card on that repo is the authoritative host contract; this file maps it onto
speech-core.

## Shape

`litert_indic_mio_tts.cpp` + `indic_mio_c.cpp` behind
`include/speech_core/indic_mio_c.h`, patterned on the VoxCPM2 pair
(`litert_voxcpm2_tts.cpp` / `voxcpm2_c.cpp`). Reuse as-is: `LiteRTEngine`
graph loading, `hf_download`, the resampler, and `voxcpm2_tokenizer.cpp`'s HF
tokenizer.json loader (the bundle ships the standard Qwen tokenizer.json; the
added speech tokens are never produced by text encoding, so encode-only use is
safe).

## Graph I/O (from the bundle's config.json)

| Graph | In | Out |
|---|---|---|
| text-prefill | ids[1,64] i64 (pad 151643), last_index scalar | logits[1,164480], k/v[28,1,8,512,128] |
| token-step | id[1,1], pos[1,1], write_index scalar, k/v | logits[1,164480], k/v |
| audio-decoder | codes[1,384] i64 (zero-pad), global[1,128], valid_tokens scalar | STFT real/imag [1,961,768] |
| ref-encoder | audio24k[1,240000] | global[1,128] |

## Host algorithm

1. Prompt `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n` →
   tokenizer → right-pad to 64 with 151643, `last_index = n-1`. Reject >64.
2. Prefill → logits + KV. AR loop: sample (temperature/top-k/top-p/repetition
   penalty; suppress EOS 151645/151643 until the first speech token; seeded
   std::mt19937) → token-step with `pos = write_index = current length` →
   repeat. Stop on EOS or max_new_tokens (≤448 cache budget, ≤384 decoder
   bucket). Speech code = id − 151669 (valid range 0..12799).
3. Decode: zero-pad codes to 384, `valid_tokens = n` → real/imag frames.
4. **Host ISTFT** ("same" padding, n_fft 1920, hop 480, Hann(1920)):
   per-frame `irfft(real+i·imag, 1920)` × window; overlap-add at hop 480 into
   `(T−1)·480 + 1920` samples; divide by the window envelope (sum of squared
   windows OLA — constant per T, precompute once); trim `(1920−480)/2 = 720`
   from both ends; keep `valid_tokens·960` samples and drop the last ~8 hops
   (conv boundary guard, matching the fidelity gate).
   FFT NOTE: n_fft 1920 is not a power of two — Ooura (third_party/fftooura)
   is radix-2 only. Options, in preference order: (a) vendor kissfft (BSD,
   arbitrary-N real FFT, header-only-ish); (b) Bluestein wrapped over Ooura
   at 4096. Decide at implementation; (a) is simpler and battle-tested.
5. Reference path: resample to 24 kHz, center-crop to 240000 samples when
   longer (crop, don't pad — zero-padding dilutes the pooled embedding,
   measured cosine 0.97), zero-pad symmetric when shorter; run ref-encoder
   once per set_reference() and cache the 128 floats on the handle. No
   reference → zeros[128] (upstream default-voice path).

## Config / registry

- Bundle files: 4 .tflite + config.json (manifest: offsets, stop ids, prompt
  template, buckets) + tokenizer.json. Parse the manifest instead of
  hardcoding ids where practical.
- CMake: add the two .cpp to `speech_core_models_litert`; no new deps besides
  the FFT decision above.
- CLI hook (optional, follow voxcpm2): `speech_core_tts_cli` engine flag for
  local smoke testing.

## Tests

- Unit (ctest, no model): prompt build + padding/last_index; ISTFT vs a
  precomputed reference vector (generate with numpy istft of a small random
  spec — commit the fixture); envelope precompute; sampling determinism for a
  fixed seed over a synthetic logits sequence; speech-code mapping bounds.
- Gated e2e (needs download, opt-in like the voxcpm2 ones): synthesize a short
  Hindi line with `<happy>`, assert non-silent 24 kHz PCM of plausible length
  and `stopped_on_eos`, then a cloned variant with a bundled reference WAV.

## Order

1. kissfft (or Bluestein) + ISTFT unit-tested against a numpy fixture.
2. Runner: bundle load, tokenizer, prefill/step loop (greedy first), decode.
3. Sampling (temp/top-k/top-p/rep-penalty + EOS suppression) + seed plumbing.
4. Reference path + C ABI surface + CLI hook.
5. e2e gate + docs/models.md entry.

Then core-sidecar (`synthesize_indic_mio` NDJSON command) and the Studio
registry flip (`macosOnly: false` + Win/Linux platformOverrides) land in
speech-studio — one PR per repo, speech-core first.
