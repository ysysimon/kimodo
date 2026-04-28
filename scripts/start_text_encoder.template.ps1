param(
    [string]$HfHome = "",
    [string]$Llm2VecBaseModelPath = "",
    [string]$UvCacheDir = "",
    [string]$TextEncoderDevice = "",
    [switch]$Fp32
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if (-not $HfHome) {
    throw "Set -HfHome or edit this template after copying it to scripts/start_text_encoder.ps1."
}

if (-not $Llm2VecBaseModelPath) {
    throw "Set -Llm2VecBaseModelPath or edit this template after copying it to scripts/start_text_encoder.ps1."
}

if (-not $UvCacheDir) {
    $UvCacheDir = Join-Path $repoRoot ".uv-cache"
}

$env:UV_CACHE_DIR = $UvCacheDir
$env:HF_HOME = $HfHome
$env:LLM2VEC_BASE_MODEL_PATH = $Llm2VecBaseModelPath

if ($TextEncoderDevice) {
    $env:TEXT_ENCODER_DEVICE = $TextEncoderDevice
}

$argsList = @("run", "--no-sync", "kimodo_textencoder")
if ($Fp32) {
    $argsList += "--fp32"
}

Write-Host "Starting Kimodo text encoder..."
Write-Host "Repo: $repoRoot"
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "LLM2VEC_BASE_MODEL_PATH: $env:LLM2VEC_BASE_MODEL_PATH"
Write-Host "UV_CACHE_DIR: $env:UV_CACHE_DIR"
if ($env:TEXT_ENCODER_DEVICE) {
    Write-Host "TEXT_ENCODER_DEVICE: $env:TEXT_ENCODER_DEVICE"
}

uv @argsList
