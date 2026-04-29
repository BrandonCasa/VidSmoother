# VidSmoother

VidSmoother interpolates video with VapourSynth, RIFE TensorRT, vspipe, and FFmpeg NVENC. It is currently aimed at Windows 11 systems with NVIDIA GPUs.

## Features

- RIFE frame interpolation through `vs-rife` with TensorRT enabled.
- NVIDIA NVENC output through FFmpeg, defaulting to HEVC 10-bit.
- VapourSynth script generation per input video.
- Batch processing from an input folder, with optional recursive scanning.
- Dry-run mode that prints the generated `vspipe` and FFmpeg commands.
- Configurable RIFE model, interpolation factor, CUDA device, TensorRT shape settings, and NVENC options.
- Manual-only GitHub Actions release build for Windows 11/NVIDIA.
- Release workflow can select `latest` or a specific release tag for FFmpeg, VapourSynth, and L-SMASH-Works.
- Release artifacts include a PyInstaller-built `VidSmoother.exe` intended to run without Python installed on the target machine.

## How to Build From Source

The repository does not include binary tools under `libs/`. For local source runs, install the required tools yourself or pass explicit paths with CLI options.

Requirements:

- Windows 11
- NVIDIA GPU with a current NVIDIA driver
- Python 3.12 or newer
- FFmpeg with `ffmpeg.exe` and `ffprobe.exe`
- VapourSynth with `vspipe.exe`
- A VapourSynth source plugin for your selected source filter, such as L-SMASH-Works for the default `lsmas`

Clone with submodules:

```powershell
git clone --recurse-submodules <repo-url>
cd VidSmoother
```

Create a virtual environment and install Python packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip packaging setuptools wheel
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torch_tensorrt --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe -m pip install --upgrade vapoursynth .\extern\vs-rife
```

Run from source:

```powershell
.\.venv\Scripts\python.exe main.py --dry-run --ffmpeg C:\path\to\ffmpeg.exe --ffprobe C:\path\to\ffprobe.exe --vspipe C:\path\to\vspipe.exe
```

To build the release executable, use the GitHub Actions workflow instead of a local setup script:

1. Open GitHub Actions.
2. Select `Windows NVIDIA Release`.
3. Click `Run workflow`.
4. Set `version`, `python_version`, and release tags for `ffmpeg_release`, `vapoursynth_release`, and `lsmash_release`.
5. Leave a component input as `latest` to use that project's latest GitHub release.
6. Set `publish_release` to `true` only when you want the workflow to create a GitHub release.

The workflow is manual-only. It does not run on pushes, pull requests, tags, schedules, or release events.

## How to Use the Releases

Download either the `VidSmoother.exe` artifact or the release zip from a completed `Windows NVIDIA Release` workflow run.

Target machine requirements:

- Windows 11
- NVIDIA GPU with a current NVIDIA driver

Python, FFmpeg, VapourSynth, L-SMASH-Works, and Python package dependencies are intended to be bundled into the release executable by the workflow.

Basic usage:

```powershell
mkdir input
copy C:\videos\example.mp4 .\input\
.\VidSmoother.exe --dry-run
.\VidSmoother.exe --overwrite
```

By default, VidSmoother reads from `input`, writes processed videos to `output`, and stores temporary work files under `output\_work` beside the executable.

Useful options:

```powershell
.\VidSmoother.exe --input-dir C:\videos\in --output-dir C:\videos\out --recursive --overwrite
.\VidSmoother.exe --factor-num 2 --factor-den 1 --rife-model 4.26
.\VidSmoother.exe --nvenc-codec hevc_nvenc --cq 18 --pix-fmt yuv420p10le
.\VidSmoother.exe --device-index 0 --trt-opt-shape 1920x1080 --trt-max-shape 3840x2160
```

Run help for the complete option list:

```powershell
.\VidSmoother.exe --help
```
