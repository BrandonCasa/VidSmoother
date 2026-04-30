# VidSmoother

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
- Release workflow can select `latest` or a specific release tag for FFmpeg, VapourSynth, and L-SMASH-Works.
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

Animated GIFs are written back as `.gif` files. During processing, GIF inputs are first converted to a temporary lossless video so the same VapourSynth and RIFE scene-change detection pipeline can be used. FFmpeg then encodes the final GIF with a generated palette.

Useful options:

```powershell
.\VidSmoother.exe --input-dir C:\videos\in --output-dir C:\videos\out --recursive --overwrite
.\VidSmoother.exe --factor-num 2 --factor-den 1 --rife-model 4.26
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
- VapourSynth R74 or newer with `vspipe.exe`
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
.\.venv\Scripts\python.exe -m pip install --upgrade pip packaging setuptools wheel
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torch_tensorrt --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.nvidia.com
.\.venv\Scripts\python.exe -m pip install --upgrade vapoursynth .\extern\vs-rife
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
.\.venv\Scripts\python.exe -m pip install --upgrade vsrepo
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
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
```

Start the local ReactPy interface:

```powershell
.\.venv\Scripts\python.exe -m vidsmoother.ui --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The interface provides settings for the same pipeline options as the CLI, a media list populated from the input folder, per-file selection, and a processing job panel.

## Common Local Setup Errors

### `There is no attribute or namespace named misc`

This means `vs-rife` tried to call `misc.SCDetect`, but VapourSynth could not find the MiscFilters plugin.

Fix:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade vsrepo
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
6. Set `ffmpeg_release`, `vapoursynth_release`, and `lsmash_release`.
7. Leave a component input as `latest` to use the latest matching release or package.
8. Set `publish_release` to `true` only when you want the workflow to create a GitHub release.

The workflow is manual-only. It does not run on pushes, pull requests, tags, schedules, or release events.

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
