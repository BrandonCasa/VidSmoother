param(
    [switch]$SkipVenv,
    [switch]$SkipTorch,
    [switch]$SkipModelDownload,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $RepoRoot "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $Body
}

Set-Location $RepoRoot

Invoke-Step "Initialize git submodules" {
    git submodule sync --recursive
    git submodule update --init --recursive extern/vs-rife
}

if (-not $SkipVenv) {
    Invoke-Step "Create Python virtual environment" {
        if (-not (Test-Path $VenvPython)) {
            & $Python -m venv $VenvDir
        }
    }
}

if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment Python not found at $VenvPython. Re-run without -SkipVenv or pass -Python."
}

Invoke-Step "Upgrade packaging tools" {
    & $VenvPython -m pip install -U pip packaging setuptools wheel
}

if (-not $SkipTorch) {
    Invoke-Step "Install PyTorch, Torch-TensorRT, and TensorRT packages" {
        & $VenvPython -m pip install -U torch torchvision torch_tensorrt --extra-index-url $TorchIndexUrl --extra-index-url "https://pypi.nvidia.com"
    }
}

Invoke-Step "Install VapourSynth Python packages and vs-rife" {
    $VsRifePath = Join-Path $RepoRoot "extern\vs-rife"
    if (Test-Path $VsRifePath) {
        & $VenvPython -m pip install -U vapoursynth "$VsRifePath"
    } else {
        & $VenvPython -m pip install -U vapoursynth vsrife
    }
}

Invoke-Step "Install VapourSynth source plugins with vsrepo when available" {
    $VsRepo = Get-Command vsrepo -ErrorAction SilentlyContinue
    if ($null -eq $VsRepo) {
        Write-Warning "vsrepo was not found. Install VapourSynth R69+ and run: vsrepo install lsmas miscfilters_obsolete"
    } else {
        vsrepo install lsmas
        vsrepo install miscfilters_obsolete
    }
}

if (-not $SkipModelDownload) {
    Invoke-Step "Download vs-rife models" {
        & $VenvPython -m vsrife
    }
}

Invoke-Step "Verify command availability" {
    $Vspipe = Get-Command vspipe -ErrorAction SilentlyContinue
    if ($null -eq $Vspipe) {
        Write-Warning "vspipe was not found on PATH. Install VapourSynth R69+ and make sure vspipe.exe is available."
    } else {
        Write-Host "vspipe: $($Vspipe.Source)"
    }

    $Ffmpeg = Join-Path $RepoRoot "libs\ffmpeg\ffmpeg.exe"
    if (Test-Path $Ffmpeg) {
        Write-Host "ffmpeg: $Ffmpeg"
    } else {
        Write-Warning "Bundled ffmpeg.exe was not found at $Ffmpeg. Install ffmpeg or pass --ffmpeg."
    }
}

Write-Host ""
Write-Host "Setup complete. Try:"
Write-Host "  .\venv\Scripts\python.exe main.py --dry-run"
Write-Host "  .\venv\Scripts\python.exe main.py --overwrite"
