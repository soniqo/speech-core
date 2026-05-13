# Agent Instructions

## Project

speech-core — voice agent pipeline engine in C++17. Provides:

1. **Orchestration core** — state machine, turn detection, interruption handling, speech queue, conversation context, streaming VAD state machine, audio utilities. Zero ML dependencies, pure C++17.
2. **Abstract interfaces** — `STTInterface`, `TTSInterface`, `VADInterface`, `EnhancerInterface`, `EchoCancellerInterface`, `LLMInterface` in `include/speech_core/interfaces.h`. Consumers (speech-swift, speech-android, …) implement these with their own model backends.
3. **Optional ONNX reference implementations** — `SileroVad`, `ParakeetStt`, `KokoroTts`, `DeepFilterEnhancer` in `include/speech_core/models/`. Compiled in only when `SPEECH_CORE_WITH_ONNX=ON`.

## Structure

- `include/speech_core/` — public headers (`pipeline/`, `vad/`, `audio/`, `tools/`, `models/`, `util/`, `interfaces.h`, `speech_core_c.h`)
- `src/` — implementations matching the header layout
- `tests/` — unit and integration tests (`test_*.cpp`, picked up via `file(GLOB)`)
- `docs/` — `pipeline.md`, `interfaces.md`, `models.md`, `c-api.md`, `tools.md`
- `scripts/` — utility scripts

Sibling repos under `~/repos/`:

- **speech-android** — Android SDK + JNI bridge to speech-core
- **speech-swift** — Swift Package consuming speech-core as a prebuilt xcframework binary target; CoreML/MLX models implement the same conceptual interfaces (see speech-swift's `docs/shared-protocols.md`)
- **speech-models** — ONNX/CoreML model artifacts hosted under `aufklarer/` on HuggingFace
- **speech-cloud** — server-side counterpart

## Build

### Default (orchestration only, no ML deps)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest
```

### With ONNX Runtime reference models

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DORT_DIR=/path/to/onnxruntime
cmake --build build
```

`ORT_DIR` must contain `include/onnxruntime_c_api.h` and a platform shared library:
- macOS: `lib/libonnxruntime.dylib`
- Linux: `lib/libonnxruntime.so`
- Android: `lib/${ANDROID_ABI}/libonnxruntime.so`

## CMake targets

- **`speech_core`** — static library, orchestration + interfaces + audio utilities. No ORT. Always built.
- **`speech_core_models`** — static library, ONNX Runtime reference implementations. Links `speech_core` + imported `onnxruntime`. Only built when `SPEECH_CORE_WITH_ONNX=ON`.

Consumers link the targets they need:

```cmake
target_link_libraries(my_app PRIVATE speech_core)                  # orchestration only
target_link_libraries(my_app PRIVATE speech_core speech_core_models)  # + ONNX models
```

## Key files

| File | Purpose |
|---|---|
| `include/speech_core/interfaces.h` | Abstract STT / TTS / VAD / Enhancer / AEC / LLM interfaces |
| `include/speech_core/pipeline/voice_pipeline.h` | Main orchestrator |
| `include/speech_core/pipeline/turn_detector.h` | VAD-driven turn boundaries + interruption logic |
| `include/speech_core/pipeline/speech_queue.h` | Priority queue with cancel/resume |
| `include/speech_core/vad/streaming_vad.h` | 4-state hysteresis state machine (consumes VADInterface probabilities) |
| `include/speech_core/speech_core_c.h` | C ABI for FFI (Swift, Kotlin) — vtable-based interface bridging |
| `include/speech_core/models/silero_vad.h` | Silero VAD v5 (ORT) — implements `VADInterface` |
| `include/speech_core/models/parakeet_stt.h` | Parakeet TDT v3 (ORT) — implements `STTInterface` |
| `include/speech_core/models/kokoro_tts.h` | Kokoro 82M (ORT) — implements `TTSInterface` |
| `include/speech_core/models/deepfilter.h` | DeepFilterNet3 (ORT) — implements `EnhancerInterface` |
| `include/speech_core/models/onnx_engine.h` | ORT singleton with NNAPI/QNN/CPU EP selection |

## Tests

```bash
cmake -B build && cmake --build build && cd build && ctest --output-on-failure
```

9 test executables: `test_audio_buffer`, `test_c_api`, `test_conversation_context`, `test_pcm_codec`, `test_pipeline_e2e`, `test_resampler`, `test_speech_queue`, `test_streaming_vad`, `test_tools`.

Known: `test_pipeline_e2e` is intermittently flaky (SIGTRAP under load) — not a regression, pre-existing on main. Re-run if it fails once.

## Workflow

- **Never push directly to main.** Create a feature branch, open a PR, merge after review.
- Branch naming: `feat/description`, `fix/description`, `chore/description`, `docs/description`
- PR description: summary, what changed, test plan. No marketing fluff.
- Tag releases from main after merge: `git tag v0.0.X && git push origin v0.0.X`

## Guidelines

- **C++17 only.** No external dependencies in the orchestration core. ONNX Runtime is the only third-party dep, and only in `speech_core_models`.
- **No platform-specific code in `speech_core`.** Platform features (NNAPI, QNN, `__system_property_get`) live in `models/` and are gated by `__ANDROID__` ifdefs inside `onnx_engine.h`.
- **Interfaces are the contract.** Public headers in `include/speech_core/` are the API surface — minimize churn. Add to `interfaces.h` only when the abstraction is needed; don't anticipate.
- **No Claude attribution** in commits, PRs, or any externally-visible artifact. Strip both the `🤖 Generated with [Claude Code]` footer and the `Co-Authored-By: Claude …` trailer from defaults.
- **Always ask for confirmation before externally-visible actions** — pushing to any branch, opening / commenting on / merging PRs, posting to any external service. Local commits are fine without asking; pushes and PRs are not.
- **Don't commit unless explicitly asked.** Likewise for `git push`.
- **Run tests after code changes** — `ctest --output-on-failure` in `build/`.
- **Build both configurations after CMake changes** — default and `SPEECH_CORE_WITH_ONNX=ON`.

## Related repos

When working across repos, prefer one PR per repo (don't bundle Android + core changes into a single PR straddling submodules). The merge sequence usually matters:

1. speech-core PR lands first (changes the API or models).
2. speech-android PR bumps the submodule pointer and updates consumer code.
3. speech-swift PR re-pins the xcframework binary target to the new release.
4. soniqo-web docs PR updates link targets last (after the source code paths are finalized).
