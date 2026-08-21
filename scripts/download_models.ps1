param(
    [Parameter(Position = 0)]
    [string]$OutputDirectory,

    [ValidateSet("default", "stenograf")]
    [string]$ModelSet = "default"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProgressPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    if (-not [string]::IsNullOrWhiteSpace($env:SPEECH_MODEL_DIR)) {
        $OutputDirectory = $env:SPEECH_MODEL_DIR
    } elseif (-not [string]::IsNullOrWhiteSpace($env:SPEECH_CORE_CACHE_DIR)) {
        $OutputDirectory = Join-Path $env:SPEECH_CORE_CACHE_DIR "models"
    } elseif (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $OutputDirectory = Join-Path $env:LOCALAPPDATA "speech-core\models"
    } else {
        $OutputDirectory = Join-Path (Get-Location) "speech-core-models"
    }
}

$outputRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
$voicesDirectory = Join-Path $outputRoot "voices"
New-Item -ItemType Directory -Force -Path $voicesDirectory | Out-Null

$parakeetRevision = if ([string]::IsNullOrWhiteSpace($env:SPEECH_PARAKEET_ONNX_REVISION)) {
    "c1652ab21826e26dfc2be5273fe69edc7f9cc938"
} else {
    $env:SPEECH_PARAKEET_ONNX_REVISION
}
$redimnetRevision = if ([string]::IsNullOrWhiteSpace($env:SPEECH_REDIMNET_ONNX_REVISION)) {
    "e911e3f063899805f3d94ee5d1db53fff8e9f3e8"
} else {
    $env:SPEECH_REDIMNET_ONNX_REVISION
}
$sortformerRevision = if ([string]::IsNullOrWhiteSpace($env:SPEECH_SORTFORMER_ONNX_REVISION)) {
    "a7176b247fb7df5588414c20632f584d4f8562c8"
} else {
    $env:SPEECH_SORTFORMER_ONNX_REVISION
}

$defaultFiles = @(
    "Silero-VAD-v5-ONNX/silero-vad.onnx",
    "Parakeet-TDT-0.6B-ONNX/parakeet-encoder-int8.onnx.data",
    "Parakeet-TDT-0.6B-ONNX/parakeet-encoder-int8.onnx",
    "Parakeet-TDT-0.6B-ONNX/parakeet-decoder-joint-int8.onnx.data",
    "Parakeet-TDT-0.6B-ONNX/parakeet-decoder-joint-int8.onnx",
    "Parakeet-TDT-0.6B-ONNX/vocab.json",
    "Kokoro-82M-ONNX/kokoro-e2e.onnx",
    "Kokoro-82M-ONNX/kokoro-e2e.onnx.data",
    "Kokoro-82M-ONNX/vocab_index.json",
    "Kokoro-82M-ONNX/us_gold.json",
    "Kokoro-82M-ONNX/us_silver.json",
    "Kokoro-82M-ONNX/dict_fr.json",
    "Kokoro-82M-ONNX/dict_es.json",
    "Kokoro-82M-ONNX/dict_it.json",
    "Kokoro-82M-ONNX/dict_pt.json",
    "Kokoro-82M-ONNX/dict_hi.json",
    "Kokoro-82M-ONNX/voices/af_alloy.bin",
    "Kokoro-82M-ONNX/voices/af_bella.bin",
    "Kokoro-82M-ONNX/voices/af_heart.bin",
    "Kokoro-82M-ONNX/voices/af_nicole.bin",
    "Kokoro-82M-ONNX/voices/af_sky.bin",
    "Kokoro-82M-ONNX/voices/am_adam.bin",
    "Kokoro-82M-ONNX/voices/am_michael.bin",
    "Kokoro-82M-ONNX/voices/bf_emma.bin",
    "Kokoro-82M-ONNX/voices/bm_george.bin",
    "DeepFilterNet3-ONNX/deepfilter.onnx"
)

$stenografFiles = @(
    "Silero-VAD-v5-ONNX/silero-vad.onnx",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/encoder.onnx",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/decoder.onnx",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/decoder.onnx.data",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/joint.onnx",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/joint.onnx.data",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/vocab.json",
    "Nemotron-3.5-ASR-Streaming-Multilingual-0.6B-ONNX-INT8/languages.json",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/audio_encoder.onnx",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/decoder.onnx",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/decoder.onnx.data",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/config.json",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/processor_config.json",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/preprocessor_config.json",
    "MOSS-Transcribe-Diarize-0.9B-ONNX-INT8-ENC/vocab.json",
    "ReDimNet2-B6-ONNX-FP32/ReDimNet2B6.onnx.data",
    "ReDimNet2-B6-ONNX-FP32/ReDimNet2B6.onnx",
    "ReDimNet2-B6-ONNX-FP32/config.json",
    # Only the light transcription pipeline loads this, and which pipeline runs
    # is not known until a session has tried the CUDA provider. It is listed
    # here because the application asks for the manifest a second time when it
    # turns out to need it, and already-present files are skipped.
    "Sortformer-Diarization-4spk-ONNX/sortformer-default.onnx.data",
    "Sortformer-Diarization-4spk-ONNX/sortformer-default.onnx",
    "Sortformer-Diarization-4spk-ONNX/config.json",
    "LocalVQE-v1.4-AEC-200K-ONNX-FP32/LocalVQEAECResidualMask.onnx",
    "LocalVQE-v1.4-AEC-200K-ONNX-FP32/LocalVQEAECFrontend.json",
    "LocalVQE-v1.4-AEC-200K-ONNX-FP32/config.json"
)

$files = if ($ModelSet -eq "stenograf") {
    $stenografFiles
} else {
    $defaultFiles
}

foreach ($entry in $files) {
    $slash = $entry.IndexOf('/')
    $repository = $entry.Substring(0, $slash)
    $relativePath = $entry.Substring($slash + 1)
    $remoteRelativePath = $relativePath
    $revision = "main"
    $forceGraph = $false
    if ($repository -eq "Parakeet-TDT-0.6B-ONNX") {
        $revision = $parakeetRevision
    } elseif ($repository -eq "ReDimNet2-B6-ONNX-FP32") {
        $revision = $redimnetRevision
    } elseif ($repository -eq "Sortformer-Diarization-4spk-ONNX") {
        $revision = $sortformerRevision
    }
    if ($repository -eq "Parakeet-TDT-0.6B-ONNX" -and
        $relativePath -like "parakeet-*-int8.onnx*") {
        $remoteRelativePath = "external-v2/$relativePath"
        $forceGraph = $relativePath.EndsWith(".onnx")
    } elseif ($repository -eq "ReDimNet2-B6-ONNX-FP32" -and
              $relativePath -like "ReDimNet2B6.onnx*") {
        $remoteRelativePath = "external-v2/$relativePath"
        $forceGraph = $relativePath.EndsWith(".onnx")
    } elseif ($repository -eq "Sortformer-Diarization-4spk-ONNX" -and
              $relativePath -like "sortformer-default.onnx*") {
        $remoteRelativePath = "external-v2/$relativePath"
        $forceGraph = $relativePath.EndsWith(".onnx")
    }
    if ($ModelSet -eq "stenograf") {
        # Keep each application model in its repository directory. Nemotron
        # external-data references and common config/vocab filenames require
        # this layout.
        $destination = Join-Path (Join-Path $outputRoot $repository) ($relativePath.Replace('/', '\'))
    } elseif ($relativePath.StartsWith("voices/")) {
        $destination = Join-Path $outputRoot ($relativePath.Replace('/', '\'))
    } else {
        $destination = Join-Path $outputRoot ([System.IO.Path]::GetFileName($relativePath))
    }

    if (-not $forceGraph -and
        (Test-Path -LiteralPath $destination) -and
        (Get-Item -LiteralPath $destination).Length -gt 0) {
        Write-Host "[skip] $relativePath (already exists)"
        continue
    }

    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$destination.part"
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $temporary
    $url = "https://huggingface.co/soniqo/$repository/resolve/$revision/$remoteRelativePath"
    Write-Host "[fetch] $repository@$revision/$remoteRelativePath"

    & curl.exe --fail --location --retry 3 --output $temporary $url
    if ($LASTEXITCODE -ne 0) {
        Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $temporary
        if ($relativePath -eq "deepfilter.onnx") {
            Write-Warning "$relativePath is not available (HTTP error)"
            continue
        }
        throw "Required model file could not be downloaded: $relativePath"
    }
    if ((Get-Item -LiteralPath $temporary).Length -le 0) {
        Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $temporary
        throw "Downloaded model file is empty: $relativePath"
    }
    Move-Item -Force -LiteralPath $temporary -Destination $destination
}

Write-Host ""
Write-Host "$ModelSet models downloaded to: $outputRoot"
Write-Host "Start the server with: speech-server.exe --model-dir `"$outputRoot`""
