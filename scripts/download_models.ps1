param(
    [Parameter(Position = 0)]
    [string]$OutputDirectory
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

$files = @(
    "Silero-VAD-v5-ONNX/silero-vad.onnx",
    "Parakeet-TDT-0.6B-ONNX/parakeet-encoder-int8.onnx",
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

foreach ($entry in $files) {
    $slash = $entry.IndexOf('/')
    $repository = $entry.Substring(0, $slash)
    $relativePath = $entry.Substring($slash + 1)
    if ($relativePath.StartsWith("voices/")) {
        $destination = Join-Path $outputRoot ($relativePath.Replace('/', '\'))
    } else {
        $destination = Join-Path $outputRoot ([System.IO.Path]::GetFileName($relativePath))
    }

    if ((Test-Path -LiteralPath $destination) -and
        (Get-Item -LiteralPath $destination).Length -gt 0) {
        Write-Host "[skip] $relativePath (already exists)"
        continue
    }

    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$destination.part"
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $temporary
    $url = "https://huggingface.co/soniqo/$repository/resolve/main/$relativePath"
    Write-Host "[fetch] $relativePath"

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
Write-Host "Models downloaded to: $outputRoot"
Write-Host "Start the server with: speech-server.exe --model-dir `"$outputRoot`""
