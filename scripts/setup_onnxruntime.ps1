# Download official Microsoft ONNX Runtime release for Windows x64
param(
    [string]$OrtVersion = "1.19.2",
    [switch]$Gpu = $false,
    [string]$TargetDir = ".cache_reto3d/onnxruntime"
)

$ErrorActionPreference = "Stop"

if ($env:ORT_VERSION) {
    $OrtVersion = $env:ORT_VERSION
}

$ZipName = if ($Gpu) {
    "onnxruntime-win-x64-gpu-$OrtVersion.zip"
} else {
    "onnxruntime-win-x64-$OrtVersion.zip"
}

$ExtractFolderName = if ($Gpu) {
    "onnxruntime-win-x64-gpu-$OrtVersion"
} else {
    "onnxruntime-win-x64-$OrtVersion"
}

$DownloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/$ZipName"
$ResolvedTargetDir = [System.IO.Path]::GetFullPath($TargetDir)
New-Item -ItemType Directory -Force -Path $ResolvedTargetDir | Out-Null

$ExtractDir = Join-Path $ResolvedTargetDir $ExtractFolderName
$LibDir = Join-Path $ExtractDir "lib"
$DllPath = Join-Path $LibDir "onnxruntime.dll"

if (-not (Test-Path $DllPath)) {
    $ZipPath = Join-Path $ResolvedTargetDir $ZipName
    Write-Host "Downloading ONNX Runtime v$OrtVersion from $DownloadUrl..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath
    Write-Host "Extracting $ZipName to $ResolvedTargetDir..."
    Expand-Archive -Path $ZipPath -DestinationPath $ResolvedTargetDir -Force
    Remove-Item -Force $ZipPath
}

$CacheLibDir = [System.IO.Path]::GetFullPath(".cache_reto3d/lib")
New-Item -ItemType Directory -Force -Path (Split-Path $CacheLibDir) | Out-Null
if (-not (Test-Path $CacheLibDir)) {
    Copy-Item -Recurse -Force $LibDir $CacheLibDir
}

Write-Host "ONNX Runtime shared library ready at: $DllPath"
Write-Host "ORT_LIB_LOCATION: $LibDir"

if ($env:GITHUB_ENV) {
    Add-Content -Path $env:GITHUB_ENV -Value "ORT_LIB_LOCATION=$LibDir"
    Add-Content -Path $env:GITHUB_ENV -Value "ORT_PREFER_DYNAMIC_LINK=1"
}
if ($env:GITHUB_PATH) {
    Add-Content -Path $env:GITHUB_PATH -Value "$LibDir"
}
