<#
.SYNOPSIS
    Wiggle-3D Standalone 1-Line Installer for Windows (PowerShell)
.DESCRIPTION
    Installs pre-built Wiggle-3D CLI (reto-cli.exe) and provisions ONNX Runtime dynamic libraries.
.EXAMPLE
    irm https://raw.githubusercontent.com/phd-eokho/wiggle-3d/main/install.ps1 | iex
.EXAMPLE
    .\install.ps1 -Uninstall
#>

[CmdletBinding(DefaultParameterSetName = "Install")]
param(
    [Parameter(Position = 0, ParameterSetName = "Install")]
    [ValidateSet("install", "Install", "uninstall", "Uninstall", "--uninstall", "-u", "help", "Help", "--help", "-h")]
    [string]$Action = "install",

    [Parameter(ParameterSetName = "UninstallDirect")]
    [Alias("u")]
    [switch]$Uninstall,

    [Parameter(ParameterSetName = "HelpDirect")]
    [Alias("h")]
    [switch]$Help,

    [string]$InstallDir = $env:WIGGLE3D_INSTALL_DIR,
    [string]$LibDir = $env:WIGGLE3D_LIB_DIR,
    [string]$Version = $env:WIGGLE3D_VERSION,
    [string]$Repo = $env:WIGGLE3D_REPO,
    [Alias("Gpu")]
    [switch]$Cuda,
    [switch]$AutoOrt
)

$ErrorActionPreference = "Stop"

# Resolve Environment Overrides & Defaults
if (-not $Repo) {
    $Repo = "phd-eokho/wiggle-3d"
}

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:USERPROFILE ".local\bin"
}

if (-not $LibDir) {
    $InstallParent = Split-Path $InstallDir -Parent
    if ($InstallParent) {
        $LibDir = Join-Path $InstallParent "lib"
    } else {
        $LibDir = Join-Path $InstallDir "lib"
    }
}

if (-not $Version) {
    $Version = "latest"
}

if ($env:WIGGLE3D_CUDA -eq "1" -or $env:WIGGLE3D_CUDA -eq "true") {
    $Cuda = $true
}

if ($env:WIGGLE3D_AUTO_ORT -eq "1" -or $env:WIGGLE3D_AUTO_ORT -eq "true") {
    $AutoOrt = $true
}

$StateDir = if ($env:WIGGLE3D_STATE_DIR) {
    $env:WIGGLE3D_STATE_DIR
} elseif ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "reto3d"
} else {
    Join-Path $env:USERPROFILE ".local\share\reto3d"
}

$OrtManifest = Join-Path $StateDir "onnxruntime_files.txt"

$CacheDir = if ($env:LOCALAPPDATA) {
    Join-Path $env:LOCALAPPDATA "reto3d"
} else {
    Join-Path $env:USERPROFILE ".cache\reto3d"
}

# Standardized Logging
function Log-Debug {
    param([string]$Message)
    Write-Host "[DEBUG] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Log-Info {
    param([string]$Message)
    Write-Host "[INFO] " -ForegroundColor Green -NoNewline
    Write-Host $Message
}

function Log-Warn {
    param([string]$Message)
    Write-Host "[WARN] " -ForegroundColor Yellow -NoNewline
    Write-Host $Message
}

function Log-Error {
    param([string]$Message)
    Write-Host "[ERROR] " -ForegroundColor Red -NoNewline
    Write-Host $Message
}

# 1. Parse Arguments & Handle Uninstallation / Help
$IsUninstall = $Uninstall -or ($Action -in @("uninstall", "Uninstall", "--uninstall", "-u"))
$IsHelp = $Help -or ($Action -in @("help", "Help", "--help", "-h"))

if ($IsHelp) {
    Write-Host "Wiggle-3D Standalone Installer / Uninstaller (Windows PowerShell)`n"
    Write-Host "Usage:"
    Write-Host "  install.ps1 [ACTION] [OPTIONS]`n"
    Write-Host "Actions:"
    Write-Host "  install                     Install Wiggle-3D CLI (default)"
    Write-Host "  -Uninstall, -u              Uninstall Wiggle-3D binary and model caches"
    Write-Host "  -Help, -h                   Show this help message`n"
    Write-Host "Options / Environment Overrides:"
    Write-Host "  -InstallDir <DIR>           Target install directory (default: $InstallDir)"
    Write-Host "                              Env: WIGGLE3D_INSTALL_DIR"
    Write-Host "  -LibDir <DIR>               Target library directory (default: $LibDir)"
    Write-Host "                              Env: WIGGLE3D_LIB_DIR"
    Write-Host "  -Version <VER>              Target release version (default: $Version)"
    Write-Host "                              Env: WIGGLE3D_VERSION"
    Write-Host "  -Repo <USER/REPO>           GitHub repository (default: $Repo)"
    Write-Host "                              Env: WIGGLE3D_REPO"
    Write-Host "  -Cuda                       Force CUDA GPU runtime"
    Write-Host "                              Env: WIGGLE3D_CUDA=1"
    Write-Host "  -AutoOrt                    Auto-confirm ONNX Runtime download"
    Write-Host "                              Env: WIGGLE3D_AUTO_ORT=1"
    return
}

if ($IsUninstall) {
    Log-Info "Uninstalling Wiggle-3D..."
    $CliExe = Join-Path $InstallDir "reto-cli.exe"
    if (Test-Path $CliExe) {
        Remove-Item -Force $CliExe
        Log-Info "Removed binary: $CliExe"
    } else {
        Log-Warn "Binary $CliExe was not found."
    }

    # Remove ONNX Runtime shared libraries only if provisioned by this script
    if (Test-Path $OrtManifest) {
        $ManifestLines = Get-Content $OrtManifest
        foreach ($File in $ManifestLines) {
            $CleanFile = $File.Trim()
            if ($CleanFile -and (Test-Path $CleanFile)) {
                Remove-Item -Force $CleanFile -ErrorAction SilentlyContinue
                Log-Info "Removed library: $CleanFile"
            }
        }
        Remove-Item -Force $OrtManifest -ErrorAction SilentlyContinue
        Log-Info "Removed installer-provisioned ONNX Runtime libraries."
    }

    if (Test-Path $CacheDir) {
        Remove-Item -Recurse -Force $CacheDir -ErrorAction SilentlyContinue
        Log-Info "Cleaned model cache: $CacheDir"
    }

    if (Test-Path $StateDir) {
        Remove-Item -Recurse -Force $StateDir -ErrorAction SilentlyContinue
    }

    Log-Info "Wiggle-3D has been successfully uninstalled."
    return
}

# 2. Detect Operating System & Architecture
if ($PSVersionTable.PSEdition -eq "Core") {
    if ($IsLinux -or $IsMacOS) {
        Log-Error "This script is designed for Windows. Please use install.sh on Linux or macOS."
        exit 1
    }
}

$Arch = if ([Environment]::Is64BitOperatingSystem) {
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64" -or $env:PROCESSOR_ARCHITEW6432 -eq "ARM64") {
        "arm64"
    } else {
        "x86_64"
    }
} else {
    "x86"
}

if ($Arch -ne "x86_64") {
    Log-Error "Unsupported Windows architecture: $Arch. Currently only 64-bit x86_64 Windows is supported for pre-built binaries. Please build from source via cargo."
    exit 1
}

$TargetTriple = "windows-x86_64"
Log-Info "Detected target platform: $TargetTriple"

# 3. Resolve Release Tag
$ReleaseTag = $Version
if ($Version -eq "latest") {
    try {
        $LatestApi = "https://api.github.com/repos/$Repo/releases/latest"
        $WebReqParams = @{
            Uri = $LatestApi
            Headers = @{ "User-Agent" = "Wiggle3D-Installer" }
            UseBasicParsing = $true
            ErrorAction = "Stop"
        }
        $ReleaseData = Invoke-RestMethod @WebReqParams
        if ($ReleaseData.tag_name) {
            $ReleaseTag = $ReleaseData.tag_name
        } else {
            Log-Warn "Could not parse latest release tag via GitHub API; defaulting to v0.1.0."
            $ReleaseTag = "v0.1.0"
        }
    } catch {
        Log-Warn "Could not fetch latest release tag via GitHub API; defaulting to v0.1.0."
        $ReleaseTag = "v0.1.0"
    }
}

$ZipName = "wiggle-3d-${ReleaseTag}-${TargetTriple}.zip"
$DownloadUrl = "https://github.com/${Repo}/releases/download/${ReleaseTag}/${ZipName}"
$ChecksumUrl = "${DownloadUrl}.sha256"

# 4. Create Temporary Workspace
$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("wiggle3d_" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    Log-Info "Downloading Wiggle-3D (${ReleaseTag}) from ${DownloadUrl}..."
    $TempZip = Join-Path $TempDir $ZipName

    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $TempZip -UseBasicParsing -ErrorAction Stop
    } catch {
        # Fallback to alternate naming if needed
        $AltZipName = "wiggle-3d-${ReleaseTag}-x86_64-pc-windows-msvc.zip"
        $AltDownloadUrl = "https://github.com/${Repo}/releases/download/${ReleaseTag}/${AltZipName}"
        try {
            Log-Info "Trying alternate archive name: $AltDownloadUrl..."
            Invoke-WebRequest -Uri $AltDownloadUrl -OutFile $TempZip -UseBasicParsing -ErrorAction Stop
            $ZipName = $AltZipName
            $ChecksumUrl = "${AltDownloadUrl}.sha256"
        } catch {
            Log-Error "Failed to download release archive from $DownloadUrl"
            Log-Error "Please check available releases at https://github.com/${Repo}/releases"
            exit 1
        }
    }

    # Optional Checksum verification
    $TempSha = Join-Path $TempDir "${ZipName}.sha256"
    try {
        Invoke-WebRequest -Uri $ChecksumUrl -OutFile $TempSha -UseBasicParsing -ErrorAction Stop
        if (Test-Path $TempSha) {
            Log-Info "Verifying SHA-256 checksum..."
            $ExpectedHash = (Get-Content $TempSha -Raw).Trim().Split(" ")[0].ToLower()
            $ActualHash = (Get-FileHash -Path $TempZip -Algorithm SHA256).Hash.ToLower()
            if ($ExpectedHash -and ($ActualHash -ne $ExpectedHash)) {
                Log-Error "SHA-256 checksum verification failed! Expected: $ExpectedHash, Actual: $ActualHash"
                exit 1
            }
        }
    } catch {
        # Checksum file not available, proceed
    }

    # 5. Extract Archive
    $ExtractDir = Join-Path $TempDir "extracted"
    Expand-Archive -Path $TempZip -DestinationPath $ExtractDir -Force

    $BinSrc = Get-ChildItem -Path $ExtractDir -Recurse -Filter "reto-cli.exe" | Select-Object -First 1
    if (-not $BinSrc -or -not (Test-Path $BinSrc.FullName)) {
        Log-Error "Binary 'reto-cli.exe' was not found in the downloaded archive."
        exit 1
    }

    # 6. Install Binary
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $TargetExe = Join-Path $InstallDir "reto-cli.exe"
    Copy-Item -Path $BinSrc.FullName -Destination $TargetExe -Force

    Log-Info "Successfully installed 'reto-cli.exe' to $TargetExe"

    # 7. Check & Configure PATH
    $InSessionPath = ($env:PATH -split [System.IO.Path]::PathSeparator) -contains $InstallDir
    $UserPath = [Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
    $InUserPath = ($UserPath -split [System.IO.Path]::PathSeparator) -contains $InstallDir

    if (-not $InUserPath) {
        try {
            $NewUserPath = if ([string]::IsNullOrEmpty($UserPath)) { $InstallDir } else { "$UserPath;$InstallDir" }
            [Environment]::SetEnvironmentVariable("Path", $NewUserPath, [System.EnvironmentVariableTarget]::User)
            $env:PATH = "$env:PATH;$InstallDir"
            Log-Info "Added $InstallDir to user PATH."
        } catch {
            Log-Warn "$InstallDir is not in your PATH."
            Write-Host "  Add it to your user PATH with:"
            Write-Host "    [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';$InstallDir', 'User')`n"
        }
    } elseif (-not $InSessionPath) {
        $env:PATH = "$env:PATH;$InstallDir"
    }

    # 8. Verify ONNX Runtime Dependency & Setup
    $OnnxVersion = "1.19.2"
    $CliHelpSuccess = $false
    try {
        $CliTest = & $TargetExe --help 2>&1
        if ($LASTEXITCODE -eq 0) {
            $CliHelpSuccess = $true
        }
    } catch {
        $CliHelpSuccess = $false
    }

    if (-not $CliHelpSuccess) {
        Log-Warn "ONNX Runtime (v$OnnxVersion) was not detected on your system."
        Write-Host "  'reto-cli' requires ONNX Runtime dynamic libraries for neural vision alignment."
        Write-Host "  Placing the libraries in '$InstallDir' satisfies this dependency on Windows.`n"

        # Detect CUDA GPU availability
        $HasCuda = $false
        if ($Cuda) {
            $HasCuda = $true
        } elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $HasCuda = $true
        } elseif ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
            $HasCuda = $true
        }

        $OrtZip = if ($HasCuda) {
            "onnxruntime-win-x64-gpu-$OnnxVersion.zip"
        } else {
            "onnxruntime-win-x64-$OnnxVersion.zip"
        }

        if ($HasCuda) {
            Log-Info "Detected NVIDIA CUDA GPU. Selected runtime: GPU package ($OrtZip)"
        } else {
            Log-Info "Selected runtime: CPU package ($OrtZip)"
        }

        $OrtDownloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OnnxVersion/$OrtZip"

        $DoDownload = $false
        if ($AutoOrt) {
            $DoDownload = $true
        } else {
            try {
                if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
                    $Reply = Read-Host "`n  Would you like to download and install ONNX Runtime now? [Y/n]"
                    if ([string]::IsNullOrWhiteSpace($Reply) -or $Reply -match '^(y|yes)$') {
                        $DoDownload = $true
                    }
                }
            } catch {
                $DoDownload = $false
            }
        }

        if ($DoDownload) {
            Log-Info "Downloading ONNX Runtime v$OnnxVersion from $OrtDownloadUrl..."
            $OrtTempZip = Join-Path $TempDir $OrtZip
            $OrtExtractDir = Join-Path $TempDir "ort"

            try {
                Invoke-WebRequest -Uri $OrtDownloadUrl -OutFile $OrtTempZip -UseBasicParsing -ErrorAction Stop
                Expand-Archive -Path $OrtTempZip -DestinationPath $OrtExtractDir -Force

                New-Item -ItemType Directory -Path $StateDir -Force | Out-Null
                New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
                if ($LibDir -ne $InstallDir) {
                    New-Item -ItemType Directory -Path $LibDir -Force | Out-Null
                }

                $Dlls = Get-ChildItem -Path $OrtExtractDir -Recurse -Filter "*.dll"
                $InstalledFiles = @()

                foreach ($Dll in $Dlls) {
                    $DestInstall = Join-Path $InstallDir $Dll.Name
                    Copy-Item -Path $Dll.FullName -Destination $DestInstall -Force
                    $InstalledFiles += $DestInstall

                    if ($LibDir -ne $InstallDir) {
                        $DestLib = Join-Path $LibDir $Dll.Name
                        Copy-Item -Path $Dll.FullName -Destination $DestLib -Force
                        $InstalledFiles += $DestLib
                    }
                }

                $InstalledFiles | Out-File -FilePath $OrtManifest -Encoding utf8 -Force
                Log-Info "Successfully installed ONNX Runtime libraries to $InstallDir."
            } catch {
                Log-Error "Failed to download or install ONNX Runtime: $_"
            }
        } else {
            Write-Host "`n  Manual Installation Command (PowerShell):"
            Write-Host "    Invoke-WebRequest -Uri `"$OrtDownloadUrl`" -OutFile `"`$env:TEMP\$OrtZip`""
            Write-Host "    Expand-Archive -Path `"`$env:TEMP\$OrtZip`" -DestinationPath `"`$env:TEMP\ort`" -Force"
            Write-Host "    Copy-Item `"`$env:TEMP\ort\*\lib\*.dll`" `"$InstallDir\`" -Force"
            Write-Host "    Remove-Item -Recurse -Force `"`$env:TEMP\ort`", `"`$env:TEMP\$OrtZip`"`n"
        }
    }

    Log-Info "Wiggle-3D installation completed! Run 'reto-cli --help' to get started."

} finally {
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
}
