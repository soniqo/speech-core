# Speech Core

📖 Read in: [English](README.md) · [中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [Tiếng Việt](README_vi.md) · [Türkçe](README_tr.md) · [ไทย](README_th.md)

[![CI](https://github.com/soniqo/speech-core/actions/workflows/ci.yml/badge.svg)](https://github.com/soniqo/speech-core/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/soniqo/speech-core)](https://github.com/soniqo/speech-core/releases/latest)
[![License](https://img.shields.io/github/license/soniqo/speech-core)](LICENSE)

**Linux、Windows、Android** 向けの **C++17** オンデバイス音声基盤です。音声区間検出、バッチおよびリアルタイムのストリーミング音声認識、話者ダイアライゼーション、音声合成、そしてそれらを接続する音声エージェントパイプラインを提供します。

CPU 上でローカルに動作します。推論時にクラウドや Python は不要で、音声が端末の外へ送信されることもありません。

**[📚 完全なドキュメント →](https://soniqo.audio/ja/speech-core)** · **[🐧 Linux](https://soniqo.audio/ja/getting-started/linux)** · **[🪟 Windows](https://soniqo.audio/ja/getting-started/windows)** · **[⌨️ Linux CLI](docs/cli.md)**

**[🤗 モデル](https://huggingface.co/soniqo)** · **[🍎 Apple 向け兄弟プロジェクト](https://github.com/soniqo/speech-swift)** · **[💬 Discord](https://discord.gg/TnCryqEMgu)**

## デモ

<p align="center">
  <a href="https://www.youtube.com/watch?v=EuIU8tOWyzg">
    <img src="https://img.youtube.com/vi/EuIU8tOWyzg/maxresdefault.jpg" width="640" alt="VoxCPM2 による音声クローン — YouTube で Speech Studio のデモを見る">
  </a>
</p>
<p align="center"><em>VoxCPM2 による音声クローン — YouTube で Speech Studio のデモを見る</em></p>

## speech-core を選ぶ理由

speech-core は、小さくモデル非依存のオーケストレーション層と、任意で有効化する推論バックエンドを分離しています。コアはターン検出、割り込み処理、音声ユーティリティ、会話状態、ツール呼び出しを担い、アプリケーション側がモデルを選択します。

- **ローカルファースト：** 純粋な C++17 コア。Float 音声バッファを扱い、ネットワークやプラットフォーム固有の音声 API に依存しません。
- **ライブエージェント向け：** VAD ベースのターン、先行 STT、部分認識、バージイン、ストリーミング TTS、ツール呼び出し。
- **真のストリーミング ASR：** キャッシュ対応 RNN-T デコーダー、発話終了検出、ビームサーチ、コンテキスト語句バイアス。
- **バックエンドを選択可能：** ONNX Runtime、LiteRT、両方、どちらも使わない構成、または独自の抽象インターフェース実装。
- **移植可能な API：** ネイティブ C++ API と、Kotlin/JNI、Swift/FFI、組み込み Linux などに適した C API。
- **複数ターゲットで検証：** Linux、Windows、macOS、Android 向け arm64 ビルド、Sanitizer、モデル使用の nightly レーン。

## v0.0.10 の主な変更

- **Parakeet-EOU 120M：** 低メモリの多言語ストリーミング ASR。発話終了トークン、任意のビームサーチ、コンテキスト語句バイアス、過剰バイアス上限を搭載。
- **ネイティブ Whisper ONNX：** small から large-v3/turbo まで対応。言語検出または固定言語プロンプト、プロファイリング、CPU 調整機能を提供。
- **TTS を大幅拡張：** Kokoro に加え、VoxCPM/VoxCPM2、CosyVoice3、Chatterbox、Supertonic、Indic-Mio。バッファ型後処理と書き起こし誘導クローンにも対応。
- **会話を高速化：** Kokoro の短いターンの最適化、長文の文単位分割、再生前後も維持されるプリスピーチバッファ。
- **オンデバイス LLM ツール：** LiteRT-LM の FunctionGemma、既存の Ollama アダプター、パイプラインのツール呼び出しループ。
- **リリース品質の Linux CLI：** amd64/arm64 パッケージ、モデル取得ヘルパー、アーキテクチャ別コマンド判定、クリーンコンテナでのスモークテスト。

## 対応モデル

| モデル | タスク | ONNX | LiteRT |
|---|---|:---:|:---:|
| [Silero VAD v5](https://huggingface.co/soniqo/Silero-VAD-v5-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/vad) | 音声区間検出 | ✓ | ✓ |
| [Parakeet TDT v3 (0.6B)](https://huggingface.co/soniqo/Parakeet-TDT-0.6B-v3-LiteRT-INT8) · [soniqo.audio](https://soniqo.audio/ja/guides/parakeet) | 音声認識 | ✓ | ✓ |
| [Whisper v3 / turbo](https://huggingface.co/soniqo/Whisper-Large-v3-Turbo-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/whisper) | 多言語音声認識 | ✓ | — |
| [Nemotron Speech Streaming (0.6B)](https://huggingface.co/soniqo/Nemotron-Speech-Streaming-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/nemotron) | ストリーミング音声認識 | ✓ | ✓ |
| [Nemotron-3.5 multilingual (0.6B)](https://huggingface.co/soniqo/Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-FP16) · [soniqo.audio](https://soniqo.audio/ja/guides/nemotron) | プロンプト条件付きストリーミング STT | ✓ | ✓ |
| [Parakeet-EOU (120M)](https://huggingface.co/soniqo/Parakeet-EOU-120M-ONNX-INT8) · [soniqo.audio](https://soniqo.audio/ja/guides/dictate) | ストリーミング STT + 発話終了 | ✓ | — |
| [Omnilingual ASR CTC (300M)](https://huggingface.co/soniqo/Omnilingual-ASR-CTC-300M-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/omnilingual) | 多言語音声認識 | — | ✓ |
| [Pyannote Segmentation 3.0](https://huggingface.co/soniqo/Pyannote-Segmentation-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/diarize) | ダイアライゼーション区間分割 | — | ✓ |
| [WeSpeaker ResNet34-LM](https://huggingface.co/soniqo/WeSpeaker-ResNet34-LM-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/embed-speaker) | 話者埋め込み | — | ✓ |
| [VoxCPM 0.5B](https://huggingface.co/soniqo/VoxCPM-0.5B-ONNX) | 16 kHz TTS + 音声クローン | ✓ | — |
| [VoxCPM2 (2B)](https://huggingface.co/soniqo/VoxCPM2-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/voxcpm2) | 48 kHz TTS + 音声クローン | ✓ | ✓ |
| [CosyVoice3 0.5B](https://huggingface.co/soniqo/CosyVoice3-0.5B-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/cosyvoice) | 24 kHz 条件付き TTS | 段階導入 | — |
| [Chatterbox](https://huggingface.co/soniqo/Chatterbox-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/chatterbox) | 24 kHz 音声合成 | — | ✓ |
| [Supertonic 3](https://huggingface.co/soniqo/Supertonic-3-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/supertonic) | 音声合成 | — | ✓ |
| [Indic-Mio](https://huggingface.co/soniqo/Indic-Mio-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/indic-mio) | ヒンディー/インド諸語の音声クローン + 感情 | — | ✓ |
| [Kokoro 82M](https://huggingface.co/soniqo/Kokoro-82M-LiteRT) · [soniqo.audio](https://soniqo.audio/ja/guides/kokoro) | 音声合成 | ✓ | ✓ |
| [DeepFilterNet3](https://huggingface.co/soniqo/DeepFilterNet3-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/denoise) | 音声強調 | ✓ | — |
| [Sidon](https://huggingface.co/aufklarer/Sidon-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/sidon) | ノイズ除去 + 残響除去（16 → 48 kHz） | ✓ | — |
| [PersonaPlex 7B](https://huggingface.co/soniqo/PersonaPlex-7B-ONNX) · [soniqo.audio](https://soniqo.audio/ja/guides/respond) | 全二重 Speech-to-Speech（CUDA） | 構造実装 | — |
| [FunctionGemma 270M](https://huggingface.co/soniqo/FunctionGemma-270M-LiteRT-LM) · [soniqo.audio](https://soniqo.audio/ja/guides/functiongemma) | オンデバイスの構造化ツール呼び出し | — | LiteRT-LM |

成熟度、バンドル構成、前処理、メモリ情報、完全な例は [docs/models.md](docs/models.md) を参照してください。

## プラットフォームとバックエンド

| バックエンド | ターゲット | プラットフォーム | ランタイム設定 |
|---|---|---|---|
| コアのみ | `speech_core` | Linux、Windows、macOS、Android | 不要 |
| ONNX Runtime | `speech_core_models` | Linux、Windows、macOS、Android | 展開した ONNX Runtime を `ORT_DIR` で指定 |
| LiteRT | `speech_core_models_litert` | Linux x86_64、Windows x86_64、macOS arm64、Android | `scripts/fetch_litert.sh` / `LITERT_DIR` |
| LiteRT-LM | `speech_core_models_litert_lm` | macOS、Android ビルドパス | `scripts/fetch_litert_lm.sh` / `LITERT_LM_DIR` |

ONNX は CPU、Android NNAPI、Linux の Qualcomm QNN、またはアプリ提供の Execution Provider フックを利用できます。LiteRT は現在 C API 経由で CPU を使用します。

## クイックスタート

コアと LiteRT バックエンドをビルドします。

```bash
git clone https://github.com/soniqo/speech-core.git
cd speech-core
scripts/fetch_litert.sh build/litert
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON \
    -DLITERT_DIR="$PWD/build/litert"
cmake --build build --parallel
```

音声バッファを認識します。Parakeet v3 は言語を自動検出します。

```cpp
#include <speech_core/models/litert_parakeet_stt.h>

speech_core::LiteRTParakeetStt stt(
    "parakeet-encoder.tflite",
    "parakeet-decoder-joint.tflite",
    "vocab.json");

auto result = stt.transcribe(audio, sample_count, 16000);
std::cout << result.text << "\n";
```

抽象 VAD、STT、LLM、TTS インターフェースの任意の実装をライブパイプラインへ接続できます。

```cpp
speech_core::AgentConfig config;
config.mode = speech_core::AgentConfig::Mode::Pipeline;

speech_core::VoicePipeline pipeline(
    stt, tts, &llm, vad, config,
    [](const speech_core::PipelineEvent& event) {
        // 認識結果、応答音声、ツール呼び出し、またはエラー
    });

pipeline.start();
pipeline.push_audio(mic_samples, sample_count);
```

アプリケーションで使うターゲットだけをリンクします。

```cmake
target_link_libraries(my_app PRIVATE speech_core)
target_link_libraries(my_app PRIVATE speech_core speech_core_models)
target_link_libraries(my_app PRIVATE speech_core speech_core_models_litert)
```

## Linux CLI パッケージ

リリースには amd64 と arm64 向けの `.deb` / `.tar.gz` が含まれます。ランタイムライブラリは同梱されますが、モデルは含まれません。

```bash
VERSION=0.0.10
ARCH="$(dpkg --print-architecture)"   # amd64 または arm64
curl -fLO "https://github.com/soniqo/speech-core/releases/download/v${VERSION}/speech_${VERSION}_${ARCH}.deb"
sudo apt install "./speech_${VERSION}_${ARCH}.deb"

speech download-models
speech transcribe recording.wav
speech speak "Hello world" hello.wav
speech phonemize "Bonjour le monde" fr
```

amd64 パッケージには LiteRT VoxCPM2 音声クローンコマンドも含まれます。x86 バンドルは約 13 GB のため、明示的にダウンロードします。

```bash
speech download-models voxcpm2
speech clone reference.wav "This is my cloned voice." cloned.wav
```

正確な構文、モデルディレクトリ、単体バイナリ、amd64/arm64 のコマンド対応表は **[Linux CLI リファレンス](docs/cli.md)** を参照してください。[`soniqo.audio/cli`](https://soniqo.audio/ja/cli) は Apple プラットフォーム向けの高機能な speech-swift CLI のドキュメントです。

## アーキテクチャ

```text
アプリケーションの音声 / イベント
                │
                ▼
┌──────────────────────────────────────┐
│ speech_core                          │
│ VoicePipeline · ターン検出           │
│ 割り込み · ツール · 音声処理         │
│ 抽象 VAD / STT / LLM / TTS API       │
└──────────────┬───────────────┬───────┘
               │               │
      ┌────────▼────────┐ ┌────▼────────────┐
      │ ONNX Runtime    │ │ LiteRT / LiteRT-LM │
      │ リファレンスモデル│ │ リファレンスモデル │
      └─────────────────┘ └─────────────────────┘
```

オーケストレーションターゲットは具体的なモデルに依存しません。バックエンドの変更は構築とリンクの選択であり、パイプラインを書き直す必要はありません。

## ドキュメント

| トピック | ドキュメント |
|---|---|
| 製品概要とモデル一覧 | [soniqo.audio/ja/speech-core](https://soniqo.audio/ja/speech-core) |
| Linux セットアップ | [soniqo.audio/ja/getting-started/linux](https://soniqo.audio/ja/getting-started/linux) |
| Windows セットアップ | [soniqo.audio/ja/getting-started/windows](https://soniqo.audio/ja/getting-started/windows) |
| Linux CLI | [docs/cli.md](docs/cli.md) |
| インターフェースと独自バックエンド | [docs/interfaces.md](docs/interfaces.md) |
| モデル実装 | [docs/models.md](docs/models.md) |
| 音声パイプラインと状態機械 | [docs/pipeline.md](docs/pipeline.md) |
| C API / FFI | [docs/c-api.md](docs/c-api.md) |
| ツール呼び出し | [docs/tools.md](docs/tools.md) |

## ビルド構成

```bash
# オーケストレーションのみ：ML ランタイム不要
cmake -B build -DCMAKE_BUILD_TYPE=Release

# ONNX モデル
cmake -B build-onnx -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR=/path/to/onnxruntime

# LiteRT モデル
scripts/fetch_litert.sh build/litert
cmake -B build-litert -DCMAKE_BUILD_TYPE=Release \
    -DSPEECH_CORE_WITH_LITERT=ON -DLITERT_DIR="$PWD/build/litert"
```

## テスト

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

コアテストにモデルファイルは不要です。対応するモデルディレクトリ環境変数が未設定の場合、バックエンド統合テストは正常にスキップされます。CI は Linux、Windows、macOS を対象とし、モデルを使う定期ワークフローが公開 ONNX/LiteRT バンドルを検証します。

## 関連プロジェクト

- [speech-android](https://github.com/soniqo/speech-android) — speech-core 上の Kotlin SDK と JNI 統合。
- [speech-swift](https://github.com/soniqo/speech-swift) — macOS/iOS 向けネイティブ MLX/CoreML 音声スタック。
- [Soniqo ドキュメント](https://soniqo.audio/ja) — ガイド、アーキテクチャ、ベンチマーク、モデルページ。

## コントリビューション

Issue と Pull Request を歓迎します。`main` からブランチを作成し、影響する構成をビルドして `ctest` を実行したうえで、焦点を絞った PR を開いてください。

## ライセンス

Apache 2.0 — [LICENSE](LICENSE) を参照してください。
