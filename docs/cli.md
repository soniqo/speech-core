# speech-core CLI (Linux)

The Linux release package exposes a small verb-style `speech` command backed
by speech-core's ONNX and LiteRT examples. It is intentionally narrower than
the Apple CLI documented at [soniqo.audio/cli](https://soniqo.audio/cli).

**[Speech Core overview](https://soniqo.audio/speech-core)** ·
**[Linux guide](https://soniqo.audio/getting-started/linux)** ·
**[Windows build guide](https://soniqo.audio/getting-started/windows)**

## Install a release package

Ubuntu 22.04+, Ubuntu 24.04+, and Debian 12+ are supported (glibc 2.35 or
newer). Releases contain `.deb` and `.tar.gz` packages for `amd64` and `arm64`.

```bash
VERSION=0.0.10
ARCH="$(dpkg --print-architecture)"   # amd64 or arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"
speech --help
```

Models are downloaded separately and remain in the user's cache. The package
does not contact a cloud service during inference.

## Commands by package

| Command | Runtime/model | amd64 | arm64 |
|---|---|:---:|:---:|
| `speech transcribe` | ONNX Runtime, Silero VAD + Parakeet STT | ✓ | ✓ |
| `speech speak` / `synthesize` | ONNX Runtime, Kokoro TTS | ✓ | ✓ |
| `speech phonemize` | Kokoro phonemizer | ✓ | ✓ |
| `speech clone` | LiteRT, VoxCPM2 voice cloning | ✓ | — |
| `speech demo` | ALSA microphone voice pipeline | ✓ | — |
| `speech download-models` | ONNX model set | ✓ | ✓ |
| `speech download-models voxcpm2` | LiteRT VoxCPM2 bundle | ✓ | — |

Commands omitted from an architecture's package fail with a clear
`not available in this package` message. The arm64 package is ONNX-only because
the upstream LiteRT Linux wheel is host-architecture specific.

## Quick examples

Download the ONNX models needed by transcription, Kokoro speech synthesis,
phonemization, and the live demo:

```bash
speech download-models
```

The default destination is `~/.cache/speech-core/models`. Then run:

```bash
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech demo --transcribe-only
```

The input to `transcribe` must be a WAV file. Mono and stereo PCM16, PCM24,
and Float32 WAV inputs are accepted and resampled to 16 kHz mono internally.
The `demo` command uses the default ALSA capture device unless `--device` is
provided.

VoxCPM2 is an optional, much larger download (about 13 GB for the x86 bundle):

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

Use a 16-bit PCM WAV reference. Stereo is downmixed, any sample rate is
resampled to 16 kHz, and the clip is trimmed or padded to 6.4 seconds.

## Command syntax

```text
speech transcribe <input.wav>
speech speak "<text>" [output.wav] [language]
speech synthesize "<text>" [output.wav] [language]
speech phonemize "<text>" [language]
speech clone [bundle_dir] <ref.wav> "<text>" <out.wav> [instruction] [max_steps] [seed]
speech demo [--model-dir <path>] [--qnn] [--transcribe-only] [--device <alsa_device>]
speech download-models [output_dir]
speech download-models voxcpm2 [output_dir]
```

Each verb dispatches to a standalone executable. Run a standalone tool with no
arguments for its complete positional syntax:

```text
speech_transcribe [model_dir] <input.wav>
speech_synthesize [model_dir] <output.wav> "<text>" [language]
speech_phonemize [model_dir] "<text>" [language]
speech_voxcpm2_clone [bundle_dir] <ref.wav> "<text>" <out.wav> [instruction] [max_steps] [seed]
```

## Model directories

| Variable | Used by | Default |
|---|---|---|
| `SPEECH_MODEL_DIR` | ONNX transcribe, speak, phonemize, demo | `~/.cache/speech-core/models` |
| `SPEECH_LITERT_MODEL_DIR` | VoxCPM2 clone | architecture-specific VoxCPM2 cache directory |
| `SPEECH_CORE_CACHE_DIR` | Overrides the speech-core cache root | `$XDG_CACHE_HOME/speech-core` or `~/.cache/speech-core` |

An explicit model or bundle directory in a standalone command takes
precedence over the environment.

## Build the CLI from source

```bash
./examples/linux/setup_linux.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON \
    -DSPEECH_CORE_BUILD_EXAMPLES=ON \
    -DORT_DIR="$PWD/ort-linux"
cmake --build build --parallel
```

This produces the ONNX tools under `build/examples/linux/`. See the
[Linux example](../examples/linux/README.md) for Docker, C API, ALSA, and
cross-compilation instructions. Enable LiteRT as shown in the
[Linux guide](https://soniqo.audio/getting-started/linux) to also build the
VoxCPM2 tool on amd64.

## Test the command surface

The model-free dispatcher contract is part of the default CTest suite. It
checks command aliases, positional argument forwarding, model downloader
routing, and unavailable-command diagnostics.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R speech_dispatch
```

The release workflow additionally installs each generated `.deb` in clean
Ubuntu 22.04 and 24.04 containers, checks every executable's runtime-library
resolution, and exercises the installed dispatcher before publishing assets.
