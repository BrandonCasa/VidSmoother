# VidSmoother

<p align="center">
  <img src="docs/assets/vidsmoother-banner.png" alt="VidSmoother banner: Butter Your Frames, smooth videos with frame interpolation" width="100%">
</p>

<p align="center">
  <a href="https://github.com/BrandonCasa/VidSmoother/actions/workflows/windows-release.yml">
    <img alt="Windows NVIDIA Release" src="https://github.com/BrandonCasa/VidSmoother/actions/workflows/windows-release.yml/badge.svg?event=workflow_dispatch">
  </a>
  <a href="https://github.com/BrandonCasa/VidSmoother/releases/latest">
    <img alt="Latest release" src="https://img.shields.io/github/v/release/BrandonCasa/VidSmoother?include_prereleases&sort=semver">
  </a>
  <img alt="Target platform" src="https://img.shields.io/badge/platform-Windows%2011%20%7C%20NVIDIA-0078D4">
</p>

VidSmoother interpolates videos and animated GIFs with VapourSynth, RIFE TensorRT, vspipe, and FFmpeg. It is currently aimed at Windows 11 systems with NVIDIA GPUs.

## Features

- RIFE frame interpolation through `vs-rife` with TensorRT enabled.
- FFmpeg output that follows the input codec and pixel format by default, using matching NVIDIA NVENC encoders for H.264, HEVC, or AV1 when available.
- Animated GIF input and output, including the normal scene-change guarded RIFE interpolation path and palette-based GIF encoding.
- VapourSynth script generation per input video.
- Batch processing from an input folder, with optional recursive scanning.
- Dry-run mode that prints the generated `vspipe` and FFmpeg commands.
- Configurable RIFE model, interpolation factor, CUDA device, TensorRT shape settings, and NVENC options.
- Manual-only GitHub Actions release build for Windows 11/NVIDIA.
- Release workflow defaults to fixed FFmpeg, VapourSynth, L-SMASH-Works, and Python package versions, while still allowing `latest` to be typed for component inputs.
- Release artifacts include a PyInstaller-built `VidSmoother.exe` intended to run without Python installed on the target machine.
- Release builds bundle FFmpeg, VapourSynth, L-SMASH-Works, MiscFilters, Python runtime files, and required Python packages.

## How to Use the Releases

Download the release archive from a completed `Windows NVIDIA Release` workflow run or from the GitHub Releases page.

Large NVIDIA builds may be split into multiple `.7z` parts because GitHub release assets have a per-file size limit. Download every part before extracting.

Example release files:

```text
VidSmoother-windows11-nvidia-v0.1.0.7z.001
VidSmoother-windows11-nvidia-v0.1.0.7z.002
VidSmoother-windows11-nvidia-v0.1.0.7z.003
```

Install 7-Zip if you do not already have it:

```powershell
winget install 7zip.7zip
```

Put all `.7z.xxx` parts in the same folder, then extract only the first file:

```text
VidSmoother-windows11-nvidia-v0.1.0.7z.001
```

7-Zip will automatically read the remaining parts.

Target machine requirements:

- Windows 11
- NVIDIA GPU
- Current NVIDIA driver
- Enough free disk space for the extracted app, temporary work files, and output videos

Python does not need to be installed on the target machine when using the release build. FFmpeg, VapourSynth, L-SMASH-Works, MiscFilters, and Python package dependencies are intended to be bundled by the workflow.

Basic usage:

```powershell
mkdir input
copy C:\videos\example.mp4 .\input\
.\VidSmoother.exe --overwrite
```

By default, VidSmoother reads from `input`, writes processed videos or GIFs to `output`, and stores temporary work files under `output\_work` beside the executable.

Animated GIFs are written back as `.gif` files. By default, GIF inputs use a timeline-aware path that extracts source frame delays, preserves unusually long holds, runs RIFE on shorter frame transitions, and reassembles a variable-delay GIF with a generated palette. Use `--no-gif-timeline-smoothing` to fall back to the older constant-rate GIF path when deduplication is disabled.
GIF timeline output uses a 50 fps timing quantum and 720 px width cap by default to keep interpolated GIFs from growing excessively. Use `--gif-max-fps 0` or `--gif-max-width 0` to disable either cap. Unusually long delays at or above the 85th percentile are treated as hard holds by default; tune this with `--gif-hard-hold-percentile`.
Optional frame deduplication removes repeated or nearly repeated source frames before interpolation. VidSmoother records the timestamp gap left by each removed frame and renders that interval with a larger per-transition RIFE factor, so duplicate cleanup is not run after processing.
For non-GIF deduplication, `--workers` can render multiple transition pairs from the same video in parallel. Keep this modest on a single GPU because each worker starts its own RIFE/VapourSynth pipeline.

Useful options:

```powershell
.\VidSmoother.exe --input-dir C:\videos\in --output-dir C:\videos\out --recursive --overwrite
.\VidSmoother.exe --factor-num 2 --factor-den 1 --rife-model 4.26
.\VidSmoother.exe --gif-max-fps 30 --gif-max-width 640
.\VidSmoother.exe --gif-hard-hold-percentile 90
.\VidSmoother.exe --dedup-strength 50 --dedup-algorithm cuda-mpdecimate
.\VidSmoother.exe --dedup-strength 50 --workers 2
.\VidSmoother.exe --nvenc-codec auto --pix-fmt auto
.\VidSmoother.exe --nvenc-codec hevc_nvenc --cq 18 --pix-fmt yuv420p10le
.\VidSmoother.exe --device-index 0 --trt-opt-shape 1920x1080 --trt-max-shape 3840x2160
```

Run help for the complete option list:

```powershell
.\VidSmoother.exe --help
```

## How to Build From Source

The repository does not include binary tools under `libs/`. For local source runs, install the required tools yourself or pass explicit paths with CLI options.

Requirements:

- Windows 11
- NVIDIA GPU with a current NVIDIA driver
- Python 3.12 or newer
- FFmpeg with `ffmpeg.exe` and `ffprobe.exe`
- VapourSynth R75 or newer with `vspipe.exe`
- L-SMASH-Works VapourSynth plugin for the default `lsmas` source filter
- MiscFilters VapourSynth plugin for `misc.SCDetect`, which is used by `vs-rife` scene-change detection
- 7-Zip, required by `vsrepo` when installing some VapourSynth plugins

Clone with submodules:

```powershell
git clone --recurse-submodules <repo-url>
cd VidSmoother
```

Create a virtual environment and install Python packages:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade -c requirements-release-lock.txt pip packaging setuptools wheel
.\.venv\Scripts\python.exe -m pip install --upgrade -c requirements-release-lock.txt torch torchvision torch_tensorrt --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe -m pip install --upgrade -c requirements-release-lock.txt vapoursynth .\extern\vs-rife
```

Configure VapourSynth and confirm the active plugin directory:

```powershell
.\.venv\Scripts\vapoursynth.exe config
.\.venv\Scripts\vapoursynth.exe get-plugin-dir
```

Install 7-Zip if needed:

```powershell
winget install 7zip.7zip
```

Install the VapourSynth `misc` plugin used by `vs-rife` scene-change detection:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -c requirements-release-lock.txt vsrepo
.\.venv\Scripts\vsrepo.exe update
.\.venv\Scripts\vsrepo.exe install misc
```

Verify that `misc` loads:

```powershell
.\.venv\Scripts\python.exe -c "import vapoursynth as vs; print(vs.core.misc)"
```

If that command fails with `There is no attribute or namespace named misc`, then `MiscFilters.dll` is not installed into the VapourSynth plugin directory being used by this environment.

Install L-SMASH-Works for the default source filter. The release workflow downloads and bundles this automatically, but local source runs need it installed manually. Download the x64 VapourSynth plugin from the L-SMASH-Works release archive, then copy `LSMASHSource.dll` into the folder printed by:

```powershell
.\.venv\Scripts\vapoursynth.exe get-plugin-dir
```

Verify available VapourSynth plugins:

```powershell
.\.venv\Scripts\python.exe -c "import vapoursynth as vs; print(vs.core.plugins())"
```

Run from source:

```powershell
.\.venv\Scripts\python.exe main.py --dry-run --ffmpeg C:\path\to\ffmpeg.exe --ffprobe C:\path\to\ffprobe.exe --vspipe C:\path\to\vspipe.exe
```

Example using local project tools:

```powershell
.\.venv\Scripts\python.exe main.py --ffmpeg "E:\VidSmoother\ffmpeg\ffmpeg.exe" --ffprobe "E:\VidSmoother\ffmpeg\ffprobe.exe" --vspipe ".\.venv\Scripts\vspipe.exe"
```

## ReactPy Interface

Install the optional UI dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -c requirements-release-lock.txt -r requirements-ui.txt
```

Start the local ReactPy interface:

```powershell
.\.venv\Scripts\python.exe ui_main.py --host 127.0.0.1 --port 8764
```

Open `http://127.0.0.1:8764`. The interface provides settings for the same pipeline options as the CLI, a media list populated from the input folder, per-file selection, and a processing job panel.

## Common Local Setup Errors

### `There is no attribute or namespace named misc`

This means `vs-rife` tried to call `misc.SCDetect`, but VapourSynth could not find the MiscFilters plugin.

Fix:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -c requirements-release-lock.txt vsrepo
.\.venv\Scripts\vsrepo.exe update
.\.venv\Scripts\vsrepo.exe install misc
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -c "import vapoursynth as vs; print(vs.core.misc)"
```

### `No module named vsrepo.__main__`

Use the `vsrepo.exe` console script instead of `python -m vsrepo`.

Correct:

```powershell
.\.venv\Scripts\vsrepo.exe update
.\.venv\Scripts\vsrepo.exe install misc
```

Incorrect:

```powershell
.\.venv\Scripts\python.exe -m vsrepo update
```

### `FileNotFoundError: [WinError 2]`

If this happens during `vsrepo install misc`, 7-Zip is probably missing or not on `PATH`.

Fix:

```powershell
winget install 7zip.7zip
```

Then reopen the terminal and rerun:

```powershell
.\.venv\Scripts\vsrepo.exe install misc
```

### L-SMASH or `core.lsmas` is missing

The default source filter needs L-SMASH-Works. Copy `LSMASHSource.dll` into the VapourSynth plugin directory returned by:

```powershell
.\.venv\Scripts\vapoursynth.exe get-plugin-dir
```

Then verify:

```powershell
.\.venv\Scripts\python.exe -c "import vapoursynth as vs; print(vs.core.plugins())"
```

## Building the Release Executable

Use the GitHub Actions workflow instead of a local setup script:

1. Open GitHub Actions.
2. Select `Windows NVIDIA Release`.
3. Click `Run workflow`.
4. Set `version`, for example `v0.1.0`.
5. Set `python_version`, usually `3.12`.
6. Leave the fixed defaults for reproducible builds, or set `ffmpeg_release`, `vapoursynth_release`, and `lsmash_release` explicitly.
7. Type `latest` into a component input only when you intentionally want the latest matching release or package for that run.
8. Leave `ignore_workflow_changes_for_caching` off for normal runs so edits to the workflow file invalidate the Python build/runtime cache.
9. Turn on `ignore_workflow_changes_for_caching` only when you intentionally want to reuse that cache after changing the workflow.
10. Bump `python_package_cache_version` when you need to force-refresh the cached Python build/runtime packages.
11. Set `publish_release` to `true` only when you want the workflow to create a GitHub release.

The workflow is manual-only. It does not run on pushes, pull requests, tags, schedules, or release events.

Version inputs and Python dependencies are pinned in the repository:

- `.github/workflows/windows-release.yml` defines the manual release workflow, fixed component defaults, cache keys, the `ignore_workflow_changes_for_caching` cache override, and `latest` behavior.
- `requirements-ui.txt` pins the direct ReactPy interface dependencies.
- `requirements-trt.txt` pins the direct VapourSynth, RIFE, PyTorch, Torch-TensorRT, TensorRT, and NVIDIA runtime dependencies.
- `requirements-release-lock.txt` pins the full transitive Python package set resolved for the Windows NVIDIA release workflow on Python 3.12.

When changing Python package versions, update the direct requirement file first, then refresh `requirements-release-lock.txt` from pip's resolver using the workflow indexes:

```powershell
.\.venv\Scripts\python.exe -m pip install --dry-run --ignore-installed --report pip-report.json --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com -r requirements-ui.txt -r requirements-trt.txt vsrepo
```

If the resolved set changes, update `requirements-release-lock.txt` and bump `python_package_cache_version` on the next manual release run.

The release workflow is expected to:

- Download FFmpeg.
- Install VapourSynth.
- Install `vs-rife`.
- Download and bundle L-SMASH-Works.
- Install and bundle MiscFilters for `misc.SCDetect`.
- Bundle the Python runtime and runtime packages.
- Build a PyInstaller `onedir` app.
- Smoke-test the executable.
- Create compressed release archive parts.
- Upload the archive parts as workflow artifacts.
- Optionally publish the archive parts to a GitHub release.
