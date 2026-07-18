# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

Infrastructure vocale embarquée en **C++17** pour **Linux, Windows et Android** : détection d'activité vocale, transcription par lots et en streaming temps réel, diarisation, synthèse vocale et pipeline d'agent vocal.

Tout s'exécute localement sur CPU. Aucun cloud, aucun Python à l'inférence et aucun son ne quitte l'appareil.

**[📚 Documentation complète →](https://soniqo.audio/fr/speech-core)** · **[🐧 Linux](https://soniqo.audio/fr/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/fr/getting-started/windows)** · **[⌨️ CLI desktop](docs/cli.md)** · **[🔊 Audio HTTP](docs/http-server.md)**

**[🤗 Modèles](https://huggingface.co/soniqo)** · **[🍎 Projet Apple associé](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## Démonstration

<p align="center"><a href="https://www.youtube.com/watch?v=7L7_Uvvxtv0"><img src="https://img.youtube.com/vi/7L7_Uvvxtv0/maxresdefault.jpg" width="640" alt="Un agent vocal entièrement hors ligne dans 1,2 Go sur Android — voir la démo sur YouTube"></a></p>
<p align="center"><em>Un agent vocal entièrement hors ligne dans 1,2 Go sur Android — la control-demo de speech-android</em></p>

## Pourquoi speech-core

speech-core sépare une petite couche d'orchestration indépendante des modèles des backends d'inférence optionnels. Le cœur gère les tours de parole, interruptions, outils audio, état de conversation et appels d'outils ; l'application choisit les modèles.

- **Local avant tout :** cœur C++17 pur, buffers audio Float, sans dépendance réseau ni audio de plateforme.
- **Conçu pour les agents en direct :** tours pilotés par VAD, STT anticipé, résultats partiels, interruption, TTS streaming et outils.
- **Véritable ASR streaming :** RNN-T avec cache, détection de fin d'énoncé, beam search et biais de phrases contextuelles.
- **Backend au choix :** ONNX Runtime, LiteRT, les deux, aucun, ou vos implémentations des interfaces abstraites.
- **API portable :** C++ natif et API C pour Kotlin/JNI, Swift/FFI, Linux embarqué et autres hôtes.
- **Tests multi-cibles :** Linux, Windows, macOS, builds arm64 orientés Android, sanitizers et nightly avec modèles.

## Nouveautés de la v0.0.11

- **TTS local compatible OpenAI :** `speech-server` expose `POST /v1/audio/speech` avec alias de modèles OpenAI, voix natives et génériques, langue et vitesse, sortie WAV/PCM et authentification Bearer facultative.
- **Paquet Windows :** un ZIP x64 autonome contient le serveur, les outils CLI ONNX, `speech.dll`, ONNX Runtime et un téléchargeur PowerShell ; CI extrait et teste le paquet.
- **Parité DeepFilterNet3 :** mise à l'échelle STFT compatible libdf, normalisation ERB/complexe, filtrage profond, overlap-add et compensation de 480 échantillons restaurent le DSP de référence.
- **Pocket TTS en streaming :** le backend ONNX émet des trames fixes de 80 ms, utilise un cache borné et propose une validation aller-retour facultative avec le modèle.
- **Contexte Silero v5 correct :** chaque inférence ONNX reçoit désormais les 64 échantillons de contexte gauche requis.

## Modèles pris en charge

| Modèle | Tâche | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/vad) | Détection d'activité vocale | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/fr/guides/parakeet) | Parole vers texte | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/whisper) | Parole vers texte multilingue | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/nemotron) | STT en streaming | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/fr/guides/nemotron) | STT streaming conditionné par prompt | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/fr/guides/dictate) | STT streaming + fin d'énoncé | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/omnilingual) | STT multilingue | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/diarize) | Segmentation de diarisation | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/embed-speaker) | Embedding de locuteur | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | TTS 16 kHz + clonage | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/voxcpm2) | TTS 48 kHz + clonage | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/cosyvoice) | TTS conditionné 24 kHz | en préparation | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/chatterbox) | Synthèse vocale 24 kHz | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/supertonic) | Synthèse vocale | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/indic-mio) | Clonage hindi/langues indiennes + émotion | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/fr/guides/kokoro) | Synthèse vocale | ✓ | ✓ |
| [Pocket TTS 100M](https://huggingface.co/soniqo/Pocket-TTS-100M-ONNX-INT8) | TTS en streaming (voix Alba fixe) | ✓ | — |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/denoise) | Amélioration de la parole | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/restore) | Débruitage + déréverbération (16 → 48 kHz) | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/fr/guides/respond) | Parole-à-parole full-duplex (CUDA) | structurel | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/fr/guides/function-calls) | Appels d'outils structurés embarqués | — | LiteRT-LM |

Consultez [docs/models.md](docs/models.md) pour la maturité, les bundles, le prétraitement, la mémoire et les exemples complets.

## Plateformes et backends

| Backend | Cible | Plateformes | Configuration |
|---|---|---|---|
| Cœur seul | `speech_core` | Linux, Windows, macOS, Android | aucune |
| ONNX Runtime | `speech_core_models` | Linux, Windows, macOS, Android | archive ONNX Runtime extraite via `ORT_DIR` |
| LiteRT | `speech_core_models_litert` | Linux x86_64, Windows x86_64, macOS arm64, Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS, chemin de build Android | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX peut utiliser le CPU, NNAPI sur Android, Qualcomm QNN sous Linux ou un hook d'Execution Provider fourni par l'application. LiteRT utilise actuellement le CPU via son API C.

## Démarrage rapide

Compiler le cœur et LiteRT :

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

Transcrire un buffer ; Parakeet v3 détecte automatiquement la langue :

```cpp
#include <speech_core/models/litert_parakeet_stt.h>
speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite", "parakeet-decoder-joint.tflite", "vocab.json");
auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

Connecter toute implémentation VAD, STT, LLM et TTS au pipeline temps réel :

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;
speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // transcription, audio de réponse, outil ou erreur
    });
pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

Ne lier que les cibles nécessaires :

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Paquets CLI Linux

Les releases fournissent des paquets `.deb` et `.tar.gz` pour amd64 et arm64. Les bibliothèques runtime sont incluses, pas les modèles.

```bash
VERSION=0.0.11
ARCH="$(dpkg --print-architecture)"   # amd64 ou arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
speech serve
```

Le paquet amd64 inclut aussi le clonage VoxCPM2 avec LiteRT. Son bundle x86 fait environ 13 Go et se télécharge explicitement :

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

La **[référence CLI Linux](docs/cli.md)** décrit la syntaxe exacte, les répertoires, les binaires et la matrice amd64/arm64. [`soniqo.audio/cli`](https://soniqo.audio/fr/cli) documente la CLI speech-swift plus large pour Apple.

## Architecture

```text
audio / événements de l'application
                │
                ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · gestion des tours    │
│ interruptions · outils · audio       │
│ API abstraites VAD/STT/LLM/TTS       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ modèles réf.    │ │ modèles réf.        │
      └─────────────────┘ └─────────────────────┘
```

La cible d'orchestration ne dépend jamais d'un modèle concret. Changer de backend est un choix de construction et de liaison, pas une réécriture du pipeline.

## Documentation

| Sujet | Documentation |
|---|---|
| Vue d'ensemble et matrice | [soniqo.audio/fr/speech-core](https://soniqo.audio/fr/speech-core) |
| Installation Linux | [soniqo.audio/fr/getting-started/linux](https://soniqo.audio/fr/getting-started/linux) |
| Installation Windows | [soniqo.audio/fr/getting-started/windows](https://soniqo.audio/fr/getting-started/windows) |
| CLI Linux | [docs/cli.md](docs/cli.md) |
| Interfaces et backends personnalisés | [docs/interfaces.md](docs/interfaces.md) |
| Implémentations de modèles | [docs/models.md](docs/models.md) |
| Pipeline vocal et machine d'états | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| Appels d'outils | [docs/tools.md](docs/tools.md) |

## Variantes de build

```bash
# Orchestration seule, sans runtime ML
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Modèles ONNX
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime
# Modèles LiteRT
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## Tests

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Les tests du cœur n'ont besoin d'aucun modèle. Les intégrations sont ignorées proprement sans leurs variables de répertoire. La CI couvre Linux, Windows et macOS ; les workflows planifiés valident les bundles ONNX et LiteRT publics.

## Projets liés

- [speech-android](https://github.com/soniqo/speech-android) — SDK Kotlin et intégration JNI sur speech-core.
- [speech-swift](https://github.com/soniqo/speech-swift) — pile MLX/CoreML native pour macOS et iOS.
- [Documentation Soniqo](https://soniqo.audio/fr) — guides, architecture, benchmarks et modèles.

## Contribuer

Issues et pull requests sont bienvenus. Créez une branche depuis `main`, compilez les configurations concernées, exécutez `ctest` puis ouvrez une PR ciblée.

## Licence

Apache 2.0 — voir [LICENSE](LICENSE).
