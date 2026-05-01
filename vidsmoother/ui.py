from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cli import config_from_args
from .errors import VidSmootherError
from .media import iter_videos
from .pipeline import process_video
from .tools import default_ffmpeg_paths, repo_root

APP_TITLE = "VidSmoother"


@dataclass
class UiJob:
    status: str = "idle"
    log: list[str] = field(default_factory=list)
    running: bool = False


_JOB = UiJob()
_JOB_LOCK = threading.Lock()
_SHUTDOWN_LOCK = threading.Lock()
_SHUTDOWN_REQUESTED = False


def default_settings() -> dict[str, Any]:
    root = repo_root()
    ffmpeg, ffprobe = default_ffmpeg_paths()
    return {
        "input_dir": str(root / "input"),
        "output_dir": str(root / "output"),
        "work_dir": str(root / "output" / "_work"),
        "recursive": False,
        "overwrite": False,
        "dry_run": False,
        "workers": "1",
        "ffmpeg": str(ffmpeg) if ffmpeg else "",
        "ffprobe": str(ffprobe) if ffprobe else "",
        "vspipe": "",
        "source_filter": "lsmas",
        "matrix": "709",
        "fp32": False,
        "vs_cache_mb": "",
        "rife_model": "4.26",
        "device_index": "0",
        "factor_num": "2",
        "factor_den": "1",
        "scale": "1.0",
        "ensemble": False,
        "no_scene_change": False,
        "scene_threshold": "0.15",
        "no_auto_download": False,
        "trt_dynamic_shape": False,
        "trt_cache_dir": str(root / "output" / "_trt_cache"),
        "trt_workspace_size": "0",
        "trt_optimization_level": "",
        "trt_max_aux_streams": "",
        "trt_min_shape": "128x128",
        "trt_opt_shape": "1920x1080",
        "trt_max_shape": "1920x1080",
        "nvenc_codec": "auto",
        "nvenc_preset": "p7",
        "nvenc_rc": "vbr",
        "cq": "18",
        "qp": "",
        "bitrate": "",
        "maxrate": "",
        "bufsize": "",
        "pix_fmt": "auto",
        "audio_codec": "copy",
        "gif_max_fps": "50",
        "gif_max_width": "720",
        "gif_timeline_smoothing": False,
        "gif_hard_hold_percentile": "85",
        "dedup_preset": "none",
        "dedup_strength": "0",
        "dedup_algorithm": "cuda-mpdecimate",
        "subtitle_mode": "none",
    }


def scan_media(settings: dict[str, Any]) -> tuple[list[Path], str | None]:
    input_dir = Path(settings["input_dir"]).expanduser()
    if not input_dir.exists():
        return [], f"Input folder does not exist: {input_dir}"
    if not input_dir.is_dir():
        return [], f"Input path is not a folder: {input_dir}"
    return iter_videos(input_dir, recursive=bool(settings["recursive"])), None


def _optional_int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def _optional_str(value: str) -> str | None:
    value = value.strip()
    return value or None


def _namespace_from_settings(settings: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=Path(settings["input_dir"]),
        output_dir=Path(settings["output_dir"]),
        work_dir=Path(settings["work_dir"]),
        recursive=bool(settings["recursive"]),
        overwrite=bool(settings["overwrite"]),
        dry_run=bool(settings["dry_run"]),
        workers=int(settings["workers"] or 1),
        ffmpeg=_optional_str(settings["ffmpeg"]),
        ffprobe=_optional_str(settings["ffprobe"]),
        vspipe=_optional_str(settings["vspipe"]),
        source_filter=settings["source_filter"],
        matrix=settings["matrix"],
        fp32=bool(settings["fp32"]),
        vs_cache_mb=_optional_int(settings["vs_cache_mb"]),
        rife_model=settings["rife_model"],
        device_index=int(settings["device_index"] or 0),
        factor_num=int(settings["factor_num"] or 2),
        factor_den=int(settings["factor_den"] or 1),
        scale=float(settings["scale"] or 1.0),
        ensemble=bool(settings["ensemble"]),
        no_scene_change=bool(settings["no_scene_change"]),
        scene_threshold=float(settings["scene_threshold"] or 0.15),
        no_auto_download=bool(settings["no_auto_download"]),
        trt_dynamic_shape=bool(settings["trt_dynamic_shape"]),
        trt_cache_dir=Path(settings["trt_cache_dir"]),
        trt_workspace_size=int(settings["trt_workspace_size"] or 0),
        trt_optimization_level=_optional_int(settings["trt_optimization_level"]),
        trt_max_aux_streams=_optional_int(settings["trt_max_aux_streams"]),
        trt_min_shape=settings["trt_min_shape"],
        trt_opt_shape=settings["trt_opt_shape"],
        trt_max_shape=settings["trt_max_shape"],
        nvenc_codec=settings["nvenc_codec"],
        nvenc_preset=settings["nvenc_preset"],
        nvenc_rc=settings["nvenc_rc"],
        cq=_optional_int(settings["cq"]),
        qp=_optional_int(settings["qp"]),
        bitrate=_optional_str(settings["bitrate"]),
        maxrate=_optional_str(settings["maxrate"]),
        bufsize=_optional_str(settings["bufsize"]),
        pix_fmt=settings["pix_fmt"],
        audio_codec=settings["audio_codec"],
        gif_max_fps=float(settings["gif_max_fps"] or 0),
        gif_max_width=int(settings["gif_max_width"] or 0),
        no_gif_timeline_smoothing=not bool(settings["gif_timeline_smoothing"]),
        gif_hard_hold_percentile=float(settings["gif_hard_hold_percentile"] or 85),
        dedup_strength=float(settings["dedup_strength"] or 0),
        dedup_algorithm=settings["dedup_algorithm"],
        subtitle_mode=settings["subtitle_mode"],
    )


def _job_snapshot() -> UiJob:
    with _JOB_LOCK:
        return UiJob(status=_JOB.status, log=list(_JOB.log), running=_JOB.running)


def _set_job(
    *,
    status: str | None = None,
    running: bool | None = None,
    message: str | None = None,
) -> None:
    with _JOB_LOCK:
        if status is not None:
            _JOB.status = status
        if running is not None:
            _JOB.running = running
        if message:
            _JOB.log.append(message)


def _run_job(settings: dict[str, Any], selected_media: list[str]) -> None:
    try:
        config = config_from_args(_namespace_from_settings(settings))
        videos = [Path(path) for path in selected_media] or iter_videos(
            config.input_dir, recursive=config.recursive
        )
        if not videos:
            _set_job(status="idle", running=False, message="No media files selected.")
            return

        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.work_dir.mkdir(parents=True, exist_ok=True)
        config.rife.trt_cache_dir.mkdir(parents=True, exist_ok=True)
        _set_job(status="running", message=f"Processing {len(videos)} media file(s).")

        for index, video in enumerate(videos, start=1):
            _set_job(message=f"[{index}/{len(videos)}] {video.name}")
            output = process_video(video, config)
            _set_job(message=f"Output: {output}")

        _set_job(status="complete", running=False, message="Processing complete.")
    except (OSError, ValueError, VidSmootherError) as exc:
        _set_job(status="error", running=False, message=f"Error: {exc}")
    except Exception:
        _set_job(status="error", running=False, message=traceback.format_exc())


def start_job(settings: dict[str, Any], selected_media: list[str]) -> bool:
    with _JOB_LOCK:
        if _JOB.running:
            return False
        _JOB.status = "starting"
        _JOB.running = True
        _JOB.log = ["Starting processing job."]

    thread = threading.Thread(
        target=_run_job, args=(dict(settings), list(selected_media)), daemon=True
    )
    thread.start()
    return True


def _posix_child_pids(parent_pid: int) -> list[int]:
    children: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return children

    for status_path in proc_root.glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent = None
            for line in status_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue

        if parent == parent_pid:
            children.append(pid)
            children.extend(_posix_child_pids(pid))

    return children


def _shutdown_process_tree() -> None:
    pid = os.getpid()
    time.sleep(0.25)

    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        time.sleep(2.0)
        os._exit(0)

    child_pids = _posix_child_pids(pid)
    for child_pid in reversed(child_pids):
        try:
            os.kill(child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            continue

    time.sleep(0.75)

    for child_pid in reversed(child_pids):
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            continue

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        os._exit(0)


def request_app_shutdown() -> bool:
    global _SHUTDOWN_REQUESTED

    with _SHUTDOWN_LOCK:
        if _SHUTDOWN_REQUESTED:
            return False
        _SHUTDOWN_REQUESTED = True

    _set_job(
        status="shutting down",
        running=False,
        message="Shutting down VidSmoother and child processes.",
    )
    threading.Thread(target=_shutdown_process_tree, daemon=True).start()
    return True


def make_app_component():
    from reactpy import component, hooks, html

    styles = {
        "page": {
            "fontFamily": "Inter, Segoe UI, Arial, sans-serif",
            "color": "#1f2933",
            "background": "#f5f7fa",
            "minHeight": "100vh",
            "padding": "24px",
        },
        "shell": {
            "maxWidth": "1180px",
            "margin": "0 auto",
            "display": "grid",
            "gridTemplateColumns": "minmax(320px, 430px) minmax(420px, 1fr)",
            "gap": "20px",
        },
        "panel": {
            "background": "#ffffff",
            "border": "1px solid #d9e2ec",
            "borderRadius": "8px",
            "padding": "18px",
            "boxShadow": "0 1px 2px rgba(16, 24, 40, 0.05)",
        },
        "field": {"display": "grid", "gap": "6px", "marginBottom": "12px"},
        "label_row": {
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "gap": "8px",
        },
        "label": {
            "fontSize": "12px",
            "fontWeight": "700",
            "color": "#52606d",
            "textTransform": "uppercase",
        },
        "help_button": {
            "width": "22px",
            "height": "22px",
            "border": "1px solid #9fb3c8",
            "borderRadius": "50%",
            "background": "#ffffff",
            "color": "#334e68",
            "fontWeight": "800",
            "fontSize": "13px",
            "lineHeight": "20px",
            "cursor": "pointer",
            "padding": "0",
            "flex": "0 0 auto",
        },
        "input": {
            "height": "34px",
            "border": "1px solid #bcccdc",
            "borderRadius": "6px",
            "padding": "0 10px",
            "fontSize": "14px",
        },
        "row": {"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
        "segmented": {
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "6px",
            "marginBottom": "16px",
            "background": "#edf1f5",
            "borderRadius": "6px",
            "padding": "4px",
        },
        "segment": {
            "height": "34px",
            "border": "0",
            "borderRadius": "5px",
            "background": "transparent",
            "color": "#52606d",
            "fontWeight": "700",
            "cursor": "pointer",
        },
        "segment_active": {
            "height": "34px",
            "border": "0",
            "borderRadius": "5px",
            "background": "#ffffff",
            "color": "#1f2933",
            "fontWeight": "700",
            "cursor": "pointer",
            "boxShadow": "0 1px 2px rgba(16, 24, 40, 0.08)",
        },
        "group_title": {
            "fontSize": "14px",
            "fontWeight": "800",
            "margin": "18px 0 10px",
            "color": "#334e68",
        },
        "toggle_row": {
            "display": "grid",
            "gridTemplateColumns": "auto 1fr auto",
            "alignItems": "center",
            "gap": "8px",
            "marginBottom": "10px",
        },
        "button": {
            "height": "36px",
            "border": "1px solid #186faf",
            "borderRadius": "6px",
            "background": "#1d7bbf",
            "color": "#ffffff",
            "fontWeight": "700",
            "cursor": "pointer",
            "padding": "0 12px",
        },
        "secondary": {
            "height": "36px",
            "border": "1px solid #bcccdc",
            "borderRadius": "6px",
            "background": "#ffffff",
            "color": "#1f2933",
            "fontWeight": "700",
            "cursor": "pointer",
            "padding": "0 12px",
        },
        "danger": {
            "height": "36px",
            "border": "1px solid #b42318",
            "borderRadius": "6px",
            "background": "#c9352b",
            "color": "#ffffff",
            "fontWeight": "700",
            "cursor": "pointer",
            "padding": "0 12px",
        },
        "media": {
            "display": "grid",
            "gridTemplateColumns": "24px 1fr auto",
            "gap": "10px",
            "alignItems": "center",
            "padding": "10px 0",
            "borderBottom": "1px solid #edf1f5",
        },
        "muted": {"color": "#7b8794", "fontSize": "13px"},
        "log": {
            "background": "#102a43",
            "color": "#e6f6ff",
            "borderRadius": "6px",
            "padding": "12px",
            "minHeight": "120px",
            "maxHeight": "260px",
            "overflow": "auto",
            "whiteSpace": "pre-wrap",
            "fontFamily": "Consolas, monospace",
            "fontSize": "12px",
        },
        "modal_backdrop": {
            "position": "fixed",
            "inset": "0",
            "background": "rgba(16, 24, 40, 0.42)",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "20px",
            "zIndex": "20",
        },
        "modal": {
            "width": "min(720px, 100%)",
            "maxHeight": "86vh",
            "overflow": "auto",
            "background": "#ffffff",
            "border": "1px solid #bcccdc",
            "borderRadius": "8px",
            "boxShadow": "0 20px 40px rgba(16, 24, 40, 0.22)",
            "padding": "20px",
        },
        "modal_header": {
            "display": "flex",
            "justifyContent": "space-between",
            "gap": "16px",
            "alignItems": "flex-start",
            "marginBottom": "12px",
        },
        "help_section": {"margin": "12px 0"},
        "help_heading": {
            "fontSize": "12px",
            "fontWeight": "800",
            "color": "#52606d",
            "textTransform": "uppercase",
        },
        "help_text": {
            "fontSize": "14px",
            "lineHeight": "1.48",
            "margin": "6px 0",
            "color": "#243b53",
        },
    }

    HELP_CONTENT = {
        "input_dir": {
            "beginner": {
                "title": "Videos folder",
                "what": "This is the folder VidSmoother scans when you press Scan.",
                "how": "The app looks for supported video files in this folder. If Include videos inside subfolders is enabled, it also walks through nested folders.",
                "implications": "Changing this only changes what appears in the Media list. It does not move or edit your original videos.",
                "related": "Works with Include videos inside subfolders and with the selected checkboxes in the Media panel.",
            },
            "advanced": {
                "title": "Input directory",
                "what": "Maps to --input-dir and becomes PipelineConfig.input_dir.",
                "how": "Scanning calls iter_videos(input_dir, recursive=recursive). Processing uses the checked media list, or all videos from this directory when no explicit selection is passed.",
                "implications": "Relative paths are resolved later by the config builder. The original file path is also passed to FFmpeg as the secondary input for audio mapping.",
                "related": "Interacts with recursive scan, media selection, ffprobe probing, and output name generation.",
            },
        },
        "output_dir": {
            "beginner": {
                "title": "Finished videos folder",
                "what": "This is where completed smoothed videos are saved.",
                "how": "Each output gets a new name based on the source video plus the smoothing factor, such as name_trt_rife_2x.mp4.",
                "implications": "If a finished file with the same generated name already exists, VidSmoother skips it unless Replace finished videos with the same name is enabled.",
                "related": "Works with Replace finished videos with the same name and Smoothness boost.",
            },
            "advanced": {
                "title": "Output directory",
                "what": "Maps to --output-dir and is created before processing.",
                "how": "output_path_for() writes stem_trt_rife_<factor>x.mp4, or .gif for GIF inputs.",
                "implications": "The factor numerator is part of the filename; denominator is not. Changing factors can create separate outputs, but unusual ratios may still produce names that only show the numerator.",
                "related": "Interacts with overwrite, factor_num, and the output encoder settings.",
            },
        },
        "work_dir": {
            "beginner": {
                "title": "Temporary files folder",
                "what": "This stores scripts and logs created while a video is being processed.",
                "how": "VidSmoother creates a folder per video, writes a VapourSynth script there, and stores FFmpeg/VapourSynth logs.",
                "implications": "Use a disk with enough free space. GIF inputs may also create a temporary lossless video here.",
                "related": "Separate from the finished videos folder and TensorRT model cache.",
            },
            "advanced": {
                "title": "Work directory",
                "what": "Maps to --work-dir. Per-input .vpy files and logs are written below this path.",
                "how": "process_video() creates work_dir/video_stem/logs and writes the generated VapourSynth script before piping vspipe into FFmpeg.",
                "implications": "Keeping this on fast storage helps with logs and GIF preparation, but most interpolation work happens on the GPU.",
                "related": "Different from trt_cache_dir, which stores TensorRT engines.",
            },
        },
        "recursive": {
            "beginner": {
                "title": "Include subfolders",
                "what": "When on, Scan finds videos inside folders under your Videos folder.",
                "how": "VidSmoother walks the folder tree instead of only checking the top-level folder.",
                "implications": "This can add many more files to the Media list, so check the selection before processing.",
                "related": "Only affects scanning and automatic video discovery. It does not change output folder structure.",
            },
            "advanced": {
                "title": "Recursive scan",
                "what": "Maps to --recursive.",
                "how": "Passed to iter_videos() for scan_media() and process_all-style discovery.",
                "implications": "The UI processes checked absolute paths; after scanning, changing recursive does not alter the selected list until you scan again.",
                "related": "Interacts with input_dir and selected media state.",
            },
        },
        "overwrite": {
            "beginner": {
                "title": "Replace same-name results",
                "what": "When on, VidSmoother can replace an already finished output file.",
                "how": "The app checks the generated output path before processing. If the file exists and this is off, that video is skipped.",
                "implications": "Leave it off when comparing settings. Turn it on when you intentionally want to rerun the same output name.",
                "related": "Output names are affected by the source filename and smoothness factor.",
            },
            "advanced": {
                "title": "Overwrite outputs",
                "what": "Maps to --overwrite.",
                "how": "process_video() bypasses the existing-output skip when this is true. FFmpeg is also invoked with -y.",
                "implications": "Existing files at the generated path can be replaced without another prompt.",
                "related": "Most relevant when output_dir and factor_num are unchanged between runs.",
            },
        },
        "dry_run": {
            "beginner": {
                "title": "Preview commands only",
                "what": "When on, VidSmoother prepares the job but does not actually run the video conversion.",
                "how": "It builds the same scripts and commands it would use for a real run, then stops before executing external processing commands.",
                "implications": "Use this to check paths, settings, and command generation without spending GPU time.",
                "related": "Good before changing advanced tool paths or TensorRT settings.",
            },
            "advanced": {
                "title": "Dry run",
                "what": "Maps to --dry-run.",
                "how": "The runner receives dry_run=True for command execution. Config construction and script generation still happen.",
                "implications": "Validation can still fail for malformed numeric fields or missing required paths.",
                "related": "Useful with FFmpeg, vspipe, source filter, and TensorRT shape changes.",
            },
        },
        "factor": {
            "beginner": {
                "title": "Smoothness boost",
                "what": "This controls how many in-between frames RIFE creates.",
                "how": "2x makes one new frame between each original pair, 3x and 4x create more in-between timing steps, and Keep original frame rate disables the frame-rate increase.",
                "implications": "Higher values look smoother but take longer, increase output frame rate, and can make artifacts more noticeable around fast motion.",
                "related": "Scene cut detection helps prevent fake transition frames at hard cuts. Scale and model version affect quality and speed.",
            },
            "advanced": {
                "title": "Frame-rate factor",
                "what": "Maps to factor_num and factor_den in vs-rife.",
                "how": "The output frame rate is multiplied by factor_num / factor_den before vspipe streams frames into FFmpeg.",
                "implications": "Non-integer ratios are allowed by the config path, but output naming only includes the numerator. Test unusual ratios with a short clip.",
                "related": "Interacts with scene_change, RIFE model, GPU load, and encoder bitrate needs.",
            },
        },
        "factor_num": {
            "advanced": {
                "title": "Factor numerator",
                "what": "Top half of the frame-rate multiplier passed to RIFE.",
                "how": "A numerator of 2 with denominator 1 doubles the frame rate; 3 with 2 produces a 1.5x rate.",
                "implications": "Higher values require more generated frames, more GPU work, and more encoded video data.",
                "related": "Pair with Factor denominator. Scene detection becomes more important as more frames are synthesized.",
            },
        },
        "factor_den": {
            "advanced": {
                "title": "Factor denominator",
                "what": "Bottom half of the frame-rate multiplier passed to RIFE.",
                "how": "RIFE uses factor_num / factor_den, so increasing this reduces the multiplier.",
                "implications": "Use this for fractional targets. Avoid zero, which will fail before or during processing.",
                "related": "Pair with Factor numerator and check output naming if you use fractional rates.",
            },
        },
        "rife_model": {
            "beginner": {
                "title": "Model version",
                "what": "This chooses which RIFE motion model estimates the missing frames.",
                "how": "The default 4.26 is a recent bundled vs-rife model. Other versions may behave differently on animation, live action, fine texture, or fast motion.",
                "implications": "A newer or heavier model is not automatically better for every clip. Changing models can also require a separate TensorRT engine build the first time.",
                "related": "Works with Scale, Extra smoothing passes, and TensorRT cache.",
            },
            "advanced": {
                "title": "RIFE model",
                "what": "Maps to --rife-model and vsrife.rife(model=...).",
                "how": "The value selects the vs-rife network weights. With auto-download enabled, vs-rife can fetch missing weights if supported.",
                "implications": "Changing model invalidates prior assumptions about speed, artifacts, and cached TensorRT engines.",
                "related": "Interacts with no_auto_download, trt_cache_dir, scale, ensemble, and GPU memory.",
            },
        },
        "scale": {
            "beginner": {
                "title": "Processing scale",
                "what": "This changes the size RIFE uses internally while estimating motion.",
                "how": "1.0 processes at normal scale. Lower values are faster and use less memory but can miss fine detail. Higher values can be slower and heavier.",
                "implications": "If you run out of GPU memory, try lowering this before changing quality settings. Very low values can soften or distort motion.",
                "related": "Works with Model version, GPU number, and Extra smoothing passes.",
            },
            "advanced": {
                "title": "RIFE scale",
                "what": "Maps to vsrife.rife(scale=...).",
                "how": "Controls RIFE's internal processing scale, commonly 0.25, 0.5, 1.0, 2.0, or 4.0.",
                "implications": "Affects TensorRT engine characteristics, VRAM pressure, throughput, and motion-estimation detail.",
                "related": "Tune with trt shapes, ensemble, model, and workers.",
            },
        },
        "device_index": {
            "beginner": {
                "title": "GPU number",
                "what": "This chooses which NVIDIA GPU does the interpolation.",
                "how": "Most single-GPU computers use 0. Multi-GPU systems usually count from 0, then 1, and so on.",
                "implications": "Choosing the wrong number can fail or run on a GPU you did not intend to use.",
                "related": "Videos at once increases load on the chosen GPU.",
            },
            "advanced": {
                "title": "CUDA device index",
                "what": "Maps to vsrife.rife(device_index=...).",
                "how": "The selected CUDA device runs TensorRT inference for RIFE.",
                "implications": "Multiple UI workers with the same device index compete for VRAM and encoder resources.",
                "related": "Coordinate with workers, NVENC sessions, and TensorRT cache engines.",
            },
        },
        "workers": {
            "beginner": {
                "title": "Videos at once",
                "what": "This controls how many videos VidSmoother tries to process in parallel.",
                "how": "1 processes one video at a time. Higher values start multiple jobs at once.",
                "implications": "More is not always faster. Interpolation and NVENC already use the GPU heavily, so multiple videos can cause slowdowns or out-of-memory errors.",
                "related": "Affected by GPU number, Scale, Extra smoothing passes, and output codec.",
            },
            "advanced": {
                "title": "Workers",
                "what": "Maps to --workers and is clamped to at least 1.",
                "how": "process_all() uses a ThreadPoolExecutor when workers > 1. The UI currently starts selected files through the same per-video processing path.",
                "implications": "Parallel jobs can contend for CUDA, NVENC, disk, and TensorRT build resources.",
                "related": "Tune alongside device_index, trt_cache_dir, and encoder limits.",
            },
        },
        "ensemble": {
            "beginner": {
                "title": "Extra smoothing passes",
                "what": "This asks RIFE to do extra inference work to reduce interpolation artifacts.",
                "how": "Ensemble mode combines more than one prediction path instead of relying on a single pass.",
                "implications": "It may improve difficult motion, but it is slower and uses more GPU resources.",
                "related": "Higher smoothness boosts and higher scale make this more expensive.",
            },
            "advanced": {
                "title": "Ensemble mode",
                "what": "Maps to vsrife.rife(ensemble=True).",
                "how": "Enables RIFE ensemble inference for generated frames.",
                "implications": "Expect lower throughput and more VRAM pressure. Quality changes are content-dependent.",
                "related": "Interacts with scale, model, factor, and workers.",
            },
        },
        "no_scene_change": {
            "beginner": {
                "title": "Ignore hard scene cuts",
                "what": "When on, VidSmoother stops protecting hard cuts between unrelated shots.",
                "how": "Normally scene change detection tells RIFE not to invent frames between two very different scenes. This option disables that guard.",
                "implications": "Leaving it off is safer for most videos. Turning it on can create strange blended frames at edits, flashes, or cuts.",
                "related": "The higher your smoothness boost, the more visible bad cut interpolation can become.",
            },
            "advanced": {
                "title": "Disable scene change detection",
                "what": "Maps to --no-scene-change and passes sc=False to vs-rife.",
                "how": "When disabled, scene_threshold is ignored by the RIFE call.",
                "implications": "Useful only when the detector is too aggressive. Otherwise it can synthesize transition frames across cuts.",
                "related": "Interacts with scene_threshold and factor_num/factor_den.",
            },
        },
        "source_filter": {
            "advanced": {
                "title": "Source filter",
                "what": "Selects the VapourSynth source plugin used in the generated .vpy script.",
                "how": "lsmas uses LWLibavSource, ffms2 uses FFMS2 Source, and bestsource uses BestSource VideoSource.",
                "implications": "Source filters can differ in indexing behavior, format support, seeking, and timestamp handling.",
                "related": "Affects the frames fed into resize, RIFE, and then FFmpeg. It does not control audio; audio is mapped from the original file by FFmpeg.",
            },
        },
        "nvenc_codec": {
            "beginner": {
                "title": "Video codec",
                "what": "This chooses the format used to compress the finished video.",
                "how": "Auto tries to match the source codec and use NVIDIA encoding when available. H.264 is widely compatible, H.265 is efficient, and AV1 can be efficient but requires newer hardware and players.",
                "implications": "Codec choice changes file size, compatibility, encoding speed, and whether your GPU supports the job.",
                "related": "Quality, bitrate, pixel format, and audio settings all affect the final file.",
            },
            "advanced": {
                "title": "NVENC codec",
                "what": "Maps to --nvenc-codec. auto becomes None in config.",
                "how": "auto tries to resolve an NVENC encoder matching the input codec, then falls back to the input codec string. Explicit choices pass hevc_nvenc, h264_nvenc, or av1_nvenc to FFmpeg.",
                "implications": "Explicit NVENC choices require FFmpeg support and compatible NVIDIA hardware.",
                "related": "Interacts with nvenc_preset, nvenc_rc, cq, qp, bitrate, pix_fmt, and audio mapping.",
            },
        },
        "cq": {
            "beginner": {
                "title": "Quality number",
                "what": "This controls visual quality for NVIDIA encoding when bitrate is not the main target.",
                "how": "Lower numbers keep more detail and make larger files. Higher numbers compress more and make smaller files.",
                "implications": "18 is a high-quality starting point. If files are too large, raise it gradually.",
                "related": "Exact bitrate can override the feel of quality-based encoding. Codec choice also changes file size.",
            },
            "advanced": {
                "title": "CQ",
                "what": "Maps to FFmpeg -cq when the selected video encoder is NVENC.",
                "how": "Used with the configured NVENC rate-control mode, currently vbr by default.",
                "implications": "Ignored for non-NVENC fallback encoders. When bitrate/maxrate are set, those constraints also shape output size and quality.",
                "related": "Interacts with qp, bitrate, maxrate, bufsize, nvenc_rc, and codec.",
            },
        },
        "pix_fmt": {
            "beginner": {
                "title": "Pixel format",
                "what": "This controls how color data is stored in the finished video.",
                "how": "Auto keeps the source pixel format when possible. Manual values like yuv420p or yuv420p10le force a specific output format.",
                "implications": "yuv420p is broadly compatible. 10-bit formats can preserve gradients better but may not play everywhere.",
                "related": "The chosen pixel format also determines the final VapourSynth resize output format.",
            },
            "advanced": {
                "title": "Pixel format",
                "what": "Maps to FFmpeg -pix_fmt and to the final VapourSynth output format.",
                "how": "auto uses the probed source pix_fmt. Known YUV formats map to matching VapourSynth formats; unknown values currently fall back to YUV420P10 in the script.",
                "implications": "Unsupported encoder/pixel-format combinations can make FFmpeg fail.",
                "related": "Interacts with codec, source bit depth, matrix conversion, and player compatibility.",
            },
        },
        "bitrate": {
            "beginner": {
                "title": "Exact bitrate",
                "what": "This sets a target video data rate such as 8M.",
                "how": "If filled in, FFmpeg receives that value as the video bitrate.",
                "implications": "Use this when you need predictable file size. Leave it empty when you want the Quality number to drive the result.",
                "related": "Codec, quality number, maxrate, and bufsize all affect size and quality.",
            },
            "advanced": {
                "title": "Bitrate",
                "what": "Maps to FFmpeg -b:v when non-empty.",
                "how": "Passed as a raw FFmpeg bitrate string, for example 6M or 12000k.",
                "implications": "Combining bitrate with CQ, maxrate, and bufsize creates constrained VBR behavior under NVENC.",
                "related": "Coordinate with nvenc_rc, cq, maxrate, bufsize, and codec.",
            },
        },
        "audio_codec": {
            "beginner": {
                "title": "Audio",
                "what": "This controls what happens to the original video's audio track.",
                "how": "Keep original audio copies it without re-encoding. AAC or Opus re-encodes it. No audio removes it.",
                "implications": "Copy is fastest and avoids quality loss, but re-encoding can improve compatibility with some output containers or players.",
                "related": "Audio comes from the original video input, not from the VapourSynth video pipe.",
            },
            "advanced": {
                "title": "Audio codec",
                "what": "Maps to FFmpeg -c:a, or -an when set to none.",
                "how": "If the source has audio and audio is not none, FFmpeg maps 1:a:0? from the original file.",
                "implications": "Only the first audio stream is considered by the current command. Subtitles are not carried through this UI path.",
                "related": "Independent of source_filter because VapourSynth only supplies video frames.",
            },
        },
        "gif_max_fps": {
            "beginner": {
                "title": "GIF frame rate limit",
                "what": "This caps the frame rate used when VidSmoother writes animated GIF output.",
                "how": "GIF inputs still go through RIFE first, then FFmpeg drops evenly spaced frames before palette generation when the interpolated frame rate is above this limit.",
                "implications": "Lower values make smaller GIFs and more reliable browser playback, but reduce smoothness. Use 0 to keep every interpolated frame.",
                "related": "Works with the smoothness boost and GIF width limit.",
            },
            "advanced": {
                "title": "GIF max FPS",
                "what": "Maps to --gif-max-fps.",
                "how": "The GIF encoder filter uses min(probed_fps * factor_num / factor_den, gif_max_fps).",
                "implications": "GIF timing is quantized by GIF players, so very high frame rates bloat files without reliably improving playback.",
                "related": "Interacts with factor_num, factor_den, and palette generation.",
            },
        },
        "gif_max_width": {
            "beginner": {
                "title": "GIF width limit",
                "what": "This caps the width used when VidSmoother writes animated GIF output.",
                "how": "FFmpeg scales the finished animation before palette generation and keeps the aspect ratio.",
                "implications": "Smaller dimensions reduce GIF size quickly. Use 0 to keep the full processed size.",
                "related": "Works with the GIF frame rate limit.",
            },
            "advanced": {
                "title": "GIF max width",
                "what": "Maps to --gif-max-width.",
                "how": "The GIF encoder filter applies a Lanczos scale with min(input_width, gif_max_width) before generating the palette.",
                "implications": "The limit only applies to GIF output; normal video outputs still use the configured video pixel format and encoder settings.",
                "related": "Interacts with palette generation and final GIF file size.",
            },
        },
        "gif_timeline_smoothing": {
            "beginner": {
                "title": "GIF timeline smoothing",
                "what": "Preserves original GIF pacing with an attempt to smooth it out.",
                "how": "Magic!",
                "implications": "Disable it to force a constant-rate GIF.",
                "related": "GIF max FPS, frame deduplication.",
            },
            "advanced": {
                "title": "GIF timeline smoothing     CURRENTLY BROKEN",
                "what": "Uses a GIF-specific path that preserves source frame delays instead of flattening the animation to constant frame rate.",
                "how": "VidSmoother extracts GIF frames and delays, smooths only non-hold transitions, then reassembles a variable-delay GIF.",
                "implications": "This keeps long pauses compact and better preserves original pacing. Disable it to use the older constant-rate GIF path.",
                "related": "GIF hard hold percentile, GIF max FPS, and frame deduplication.",
            },
        },
        "gif_hard_hold_percentile": {
            "advanced": {
                "title": "GIF hard hold percentile",
                "what": "Controls which long GIF frame delays are treated as intentional holds.",
                "how": "Delays at or above this percentile, and above the median delay, are emitted as a single delayed frame instead of being morphed toward the next frame.",
                "implications": "Lower values preserve more holds. Higher values smooth through more delayed frames.",
                "related": "Only applies when GIF timeline smoothing is enabled.",
            },
        },
        "dedup_preset": {
            "beginner": {
                "title": "Frame deduplication",
                "what": "This removes repeated or nearly repeated frames after smoothing.",
                "how": "Medium and Strong map to numeric duplicate-detection strengths. Removed frames keep their timestamp gaps, so GIFs can store the skipped time as longer frame delays.",
                "implications": "Medium is conservative. Strong can make held animation much smaller, but may remove subtle motion if the source barely changes.",
                "related": "Works with smoothness boost, GIF frame rate limit, and the advanced dedup strength.",
            },
            "advanced": {
                "title": "Dedup preset",
                "what": "Beginner-facing preset for dedup strength.",
                "how": "None sets strength to 0, Medium to 50, and Strong to 80.",
                "implications": "Advanced mode can use the raw strength slider for finer control.",
                "related": "Dedup strength and algorithm.",
            },
        },
        "dedup_strength": {
            "advanced": {
                "title": "Dedup strength",
                "what": "Controls how aggressively near-duplicate frames are removed.",
                "how": "The value maps to FFmpeg duplicate thresholds. 0 disables deduplication; higher values tolerate more pixel difference before a frame is kept.",
                "implications": "High values are useful for GIF holds and very static video, but can remove subtle movement or low-contrast animation.",
                "related": "Dedup algorithm and GIF frame delay behavior.",
            },
        },
        "dedup_algorithm": {
            "advanced": {
                "title": "Dedup algorithm",
                "what": "Chooses the filter chain used for duplicate detection.",
                "how": "mpdecimate uses FFmpeg's pixel-threshold duplicate detector. cuda-mpdecimate runs CUDA upload/scale/download filtering before the same duplicate scoring stage.",
                "implications": "The CUDA option requires an FFmpeg build with CUDA filters and NVIDIA support; unsupported builds will fail in ffmpeg.log.",
                "related": "Dedup strength and FFmpeg path.",
            },
        },
        "ffmpeg": {
            "beginner": {
                "title": "FFmpeg app",
                "what": "FFmpeg writes the final video file.",
                "how": "VidSmoother sends smoothed video frames from VapourSynth into FFmpeg, then FFmpeg encodes video and handles audio.",
                "implications": "Leave this alone if auto-detection works. Set it only if FFmpeg is installed somewhere custom.",
                "related": "FFprobe and VapourSynth runner are the other external tools.",
            },
            "advanced": {
                "title": "FFmpeg path",
                "what": "Maps to --ffmpeg and ToolPaths.ffmpeg.",
                "how": "Used for GIF preparation, final encoding, and encoder capability checks.",
                "implications": "Different FFmpeg builds expose different NVENC encoders and pixel-format support.",
                "related": "Must match your desired nvenc_codec and audio_codec support.",
            },
        },
        "ffprobe": {
            "beginner": {
                "title": "FFprobe app",
                "what": "FFprobe reads information about each source video before processing.",
                "how": "VidSmoother uses it to detect size, frame rate, codec, pixel format, audio, and GIF status.",
                "implications": "If FFprobe is missing or wrong, Scan may work but processing can fail when the video is inspected.",
                "related": "FFmpeg usually ships with FFprobe.",
            },
            "advanced": {
                "title": "FFprobe path",
                "what": "Maps to --ffprobe and ToolPaths.ffprobe.",
                "how": "probe_video() uses this before deciding output encoder, pixel format, audio mapping, and GIF preparation.",
                "implications": "Bad probe data can cascade into wrong encoder selection or output format.",
                "related": "Feeds resolve_video_encoder(), build_ffmpeg_command(), and write_vapoursynth_script().",
            },
        },
        "vspipe": {
            "beginner": {
                "title": "VapourSynth runner",
                "what": "vspipe runs the generated smoothing script.",
                "how": "VidSmoother writes a .vpy script, then vspipe streams the processed video frames into FFmpeg.",
                "implications": "Leave this alone if auto-detection works. A wrong path stops interpolation before FFmpeg can encode anything.",
                "related": "Uses source filter, model version, scale, scene detection, and TensorRT settings.",
            },
            "advanced": {
                "title": "vspipe path",
                "what": "Maps to --vspipe and ToolPaths.vspipe.",
                "how": "Invoked as vspipe --container y4m script.vpy - and piped to FFmpeg stdin.",
                "implications": "The vspipe environment must be able to import vapoursynth and vsrife.",
                "related": "Directly exercises the generated .vpy script and all VapourSynth/RIFE options.",
            },
        },
        "trt_opt_shape": {
            "advanced": {
                "title": "TensorRT opt shape",
                "what": "Preferred width x height used when building dynamic TensorRT engines.",
                "how": "Parsed as WIDTHxHEIGHT and passed to vs-rife as trt_opt_shape.",
                "implications": "Best performance usually occurs near this shape. Keep it close to the resolution you process most often.",
                "related": "Relevant with dynamic shape engines and should sit between min and max shape.",
            },
        },
        "trt_max_shape": {
            "advanced": {
                "title": "TensorRT max shape",
                "what": "Largest width x height allowed for a dynamic TensorRT engine.",
                "how": "Parsed as WIDTHxHEIGHT and passed to vs-rife as trt_max_shape.",
                "implications": "Too low can fail on larger videos. Too high can increase engine build time and memory needs.",
                "related": "Coordinate with trt_min_shape, trt_opt_shape, scale, and source resolution.",
            },
        },
    }

    def help_for(setting_key: str, mode: str) -> dict[str, str]:
        topic = HELP_CONTENT.get(setting_key, {})
        return (
            topic.get(mode)
            or topic.get("advanced")
            or topic.get("beginner")
            or {
                "title": setting_key.replace("_", " ").title(),
                "what": "This setting is passed through to the processing configuration.",
                "how": "It is applied when you start a processing job.",
                "implications": "Invalid values can cause processing to fail.",
                "related": "Check nearby settings that affect the same stage of the pipeline.",
            }
        )

    @component
    def HelpButton(setting_key: str, label: str, mode: str, open_help):
        def handle_click(event):
            open_help(setting_key, label)

        title = help_for(setting_key, mode)["title"]
        return html.button(
            {
                "type": "button",
                "style": styles["help_button"],
                "on_click": handle_click,
                "aria-label": f"Open help for {title}",
                "title": f"Open help for {title}",
            },
            "?",
        )

    @component
    def FieldLabel(label: str, setting_key: str, mode: str, open_help):
        return html.div(
            {"style": styles["label_row"]},
            html.span({"style": styles["label"]}, label),
            HelpButton(setting_key, label, mode, open_help),
        )

    @component
    def TextField(
        label: str,
        setting_key: str,
        settings: dict[str, Any],
        set_settings,
        mode: str,
        open_help,
        help_key: str | None = None,
    ):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["value"]
            set_settings(next_settings)

        return html.div(
            {"style": styles["field"]},
            FieldLabel(label, help_key or setting_key, mode, open_help),
            html.input(
                {
                    "style": styles["input"],
                    "value": settings[setting_key],
                    "on_change": handle_change,
                }
            ),
        )

    @component
    def RangeField(
        label: str,
        setting_key: str,
        settings: dict[str, Any],
        set_settings,
        mode: str,
        open_help,
        *,
        minimum: int = 0,
        maximum: int = 100,
    ):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["value"]
            set_settings(next_settings)

        value = str(settings[setting_key])
        return html.div(
            {"style": styles["field"]},
            FieldLabel(label, setting_key, mode, open_help),
            html.div(
                {
                    "style": {
                        "display": "grid",
                        "gridTemplateColumns": "1fr 48px",
                        "gap": "10px",
                        "alignItems": "center",
                    }
                },
                html.input(
                    {
                        "type": "range",
                        "min": str(minimum),
                        "max": str(maximum),
                        "step": "1",
                        "style": {"width": "100%"},
                        "value": value,
                        "on_change": handle_change,
                    }
                ),
                html.input(
                    {
                        "style": styles["input"],
                        "value": value,
                        "on_change": handle_change,
                    }
                ),
            ),
        )

    @component
    def SelectField(
        label: str,
        setting_key: str,
        options: list[str] | list[tuple[str, str]],
        settings: dict[str, Any],
        set_settings,
        mode: str,
        open_help,
        help_key: str | None = None,
    ):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["value"]
            set_settings(next_settings)

        option_nodes = []
        for option in options:
            if isinstance(option, tuple):
                value, text = option
            else:
                value = text = option
            option_nodes.append(html.option({"value": value}, text))

        return html.div(
            {"style": styles["field"]},
            FieldLabel(label, help_key or setting_key, mode, open_help),
            html.select(
                {
                    "style": styles["input"],
                    "value": settings[setting_key],
                    "on_change": handle_change,
                },
                option_nodes,
            ),
        )

    @component
    def Toggle(
        label: str,
        setting_key: str,
        settings: dict[str, Any],
        set_settings,
        mode: str,
        open_help,
    ):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["checked"]
            set_settings(next_settings)

        return html.div(
            {"style": styles["toggle_row"]},
            html.input(
                {
                    "type": "checkbox",
                    "checked": bool(settings[setting_key]),
                    "on_change": handle_change,
                }
            ),
            html.label(label),
            HelpButton(setting_key, label, mode, open_help),
        )

    @component
    def SmoothnessSelect(settings: dict[str, Any], set_settings, mode: str, open_help):
        current = f"{settings['factor_num']}:{settings['factor_den']}"

        def handle_change(event):
            numerator, denominator = event["target"]["value"].split(":", maxsplit=1)
            next_settings = dict(settings)
            next_settings["factor_num"] = numerator
            next_settings["factor_den"] = denominator
            set_settings(next_settings)

        return html.div(
            {"style": styles["field"]},
            FieldLabel("Smoothness boost", "factor", mode, open_help),
            html.select(
                {
                    "style": styles["input"],
                    "value": current,
                    "on_change": handle_change,
                },
                [
                    html.option({"value": "2:1"}, "2x smoother"),
                    html.option({"value": "3:1"}, "3x smoother"),
                    html.option({"value": "4:1"}, "4x smoother"),
                    html.option({"value": "1:1"}, "Keep original frame rate"),
                ],
            ),
        )

    @component
    def DedupPresetSelect(settings: dict[str, Any], set_settings, mode: str, open_help):
        preset_strengths = {"none": "0", "medium": "50", "strong": "80"}

        def handle_change(event):
            preset = event["target"]["value"]
            next_settings = dict(settings)
            next_settings["dedup_preset"] = preset
            next_settings["dedup_strength"] = preset_strengths[preset]
            set_settings(next_settings)

        return html.div(
            {"style": styles["field"]},
            FieldLabel("Frame deduplication", "dedup_preset", mode, open_help),
            html.select(
                {
                    "style": styles["input"],
                    "value": settings["dedup_preset"],
                    "on_change": handle_change,
                },
                [
                    html.option({"value": "none"}, "None"),
                    html.option({"value": "medium"}, "Medium"),
                    html.option({"value": "strong"}, "Strong"),
                ],
            ),
        )

    def advanced_settings(
        settings: dict[str, Any], set_settings, mode: str, open_help
    ) -> list[Any]:
        return [
            TextField(
                "Input directory", "input_dir", settings, set_settings, mode, open_help
            ),
            TextField(
                "Output directory",
                "output_dir",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            TextField(
                "Work directory", "work_dir", settings, set_settings, mode, open_help
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "Factor numerator",
                    "factor_num",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "Factor denominator",
                    "factor_den",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "RIFE model", "rife_model", settings, set_settings, mode, open_help
                ),
                TextField(
                    "RIFE scale", "scale", settings, set_settings, mode, open_help
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "CUDA device index",
                    "device_index",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "Parallel workers",
                    "workers",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            SelectField(
                "VapourSynth source filter",
                "source_filter",
                ["lsmas", "ffms2", "bestsource"],
                settings,
                set_settings,
                mode,
                open_help,
            ),
            SelectField(
                "NVENC video encoder",
                "nvenc_codec",
                ["auto", "hevc_nvenc", "h264_nvenc", "av1_nvenc"],
                settings,
                set_settings,
                mode,
                open_help,
            ),
            html.div(
                {"style": styles["row"]},
                TextField("NVENC CQ", "cq", settings, set_settings, mode, open_help),
                TextField(
                    "Output pixel format",
                    "pix_fmt",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "GIF max FPS",
                    "gif_max_fps",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "GIF max width",
                    "gif_max_width",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                RangeField(
                    "GIF hard hold percentile",
                    "gif_hard_hold_percentile",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                # Toggle(
                #    "GIF timeline smoothing",
                #    "gif_timeline_smoothing",
                #    settings,
                #    set_settings,
                #    mode,
                #    open_help,
                # ),
            ),
            html.div(
                {"style": styles["row"]},
                RangeField(
                    "Dedup strength",
                    "dedup_strength",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                SelectField(
                    "Dedup algorithm",
                    "dedup_algorithm",
                    [
                        ("mpdecimate", "FFmpeg mpdecimate"),
                        ("cuda-mpdecimate", "CUDA-assisted mpdecimate"),
                    ],
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "TRT opt shape",
                    "trt_opt_shape",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "TRT max shape",
                    "trt_max_shape",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            Toggle(
                "Recursive scan", "recursive", settings, set_settings, mode, open_help
            ),
            Toggle(
                "Overwrite outputs",
                "overwrite",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            Toggle("Dry run", "dry_run", settings, set_settings, mode, open_help),
            Toggle(
                "RIFE ensemble mode",
                "ensemble",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            Toggle(
                "Disable scene change detection",
                "no_scene_change",
                settings,
                set_settings,
                mode,
                open_help,
            ),
        ]

    def beginner_settings(
        settings: dict[str, Any], set_settings, mode: str, open_help
    ) -> list[Any]:
        return [
            html.div({"style": styles["group_title"]}, "Folders"),
            TextField(
                "Videos to scan", "input_dir", settings, set_settings, mode, open_help
            ),
            TextField(
                "Save finished videos in",
                "output_dir",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            TextField(
                "Store temporary files in",
                "work_dir",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            Toggle(
                "Include videos inside subfolders",
                "recursive",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            Toggle(
                "Replace finished videos with the same name",
                "overwrite",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            html.div({"style": styles["group_title"]}, "Smoothness and speed"),
            SmoothnessSelect(settings, set_settings, mode, open_help),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "Motion model version",
                    "rife_model",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "NVIDIA GPU number",
                    "device_index",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "Motion detail scale",
                    "scale",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "Videos to run at once",
                    "workers",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            # UNSUPPORTED ON RIFE 4.6
            # Toggle(
            #    "Use extra smoothing passes",
            #    "ensemble",
            #    settings,
            #    set_settings,
            #    mode,
            #    open_help,
            # ),
            Toggle(
                "Ignore hard scene cuts",
                "no_scene_change",
                settings,
                set_settings,
                mode,
                open_help,
            ),
            html.div({"style": styles["group_title"]}, "Output video"),
            DedupPresetSelect(settings, set_settings, mode, open_help),
            SelectField(
                "Finished video format",
                "nvenc_codec",
                [
                    ("auto", "Auto match source"),
                    ("hevc_nvenc", "H.265 / HEVC"),
                    ("h264_nvenc", "H.264 / AVC"),
                    ("av1_nvenc", "AV1"),
                ],
                settings,
                set_settings,
                mode,
                open_help,
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "Picture quality number",
                    "cq",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "Color storage format",
                    "pix_fmt",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "Target bitrate", "bitrate", settings, set_settings, mode, open_help
                ),
                SelectField(
                    "Audio handling",
                    "audio_codec",
                    [
                        ("copy", "Keep original audio"),
                        ("aac", "AAC"),
                        ("libopus", "Opus"),
                        ("none", "No audio"),
                    ],
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div(
                {"style": styles["row"]},
                TextField(
                    "GIF frame rate limit",
                    "gif_max_fps",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
                TextField(
                    "GIF width limit",
                    "gif_max_width",
                    settings,
                    set_settings,
                    mode,
                    open_help,
                ),
            ),
            html.div({"style": styles["group_title"]}, "Tools"),
            TextField(
                "FFmpeg program", "ffmpeg", settings, set_settings, mode, open_help
            ),
            TextField(
                "FFprobe program", "ffprobe", settings, set_settings, mode, open_help
            ),
            TextField(
                "VapourSynth runner", "vspipe", settings, set_settings, mode, open_help
            ),
            Toggle(
                "Preview commands without processing",
                "dry_run",
                settings,
                set_settings,
                mode,
                open_help,
            ),
        ]

    @component
    def HelpModal(help_state: dict[str, str] | None, mode: str, close_help):
        if not help_state:
            return None

        topic = help_for(help_state["key"], mode)

        def handle_close(event):
            close_help()

        return html.div(
            {"style": styles["modal_backdrop"]},
            html.div(
                {"style": styles["modal"], "role": "dialog", "aria-modal": "true"},
                html.div(
                    {"style": styles["modal_header"]},
                    html.div(
                        html.h2(
                            {"style": {"margin": "0 0 4px", "fontSize": "20px"}},
                            topic["title"],
                        ),
                        html.div(
                            {"style": styles["muted"]},
                            f"{mode.title()} mode explanation for {help_state['label']}.",
                        ),
                    ),
                    html.button(
                        {
                            "style": styles["secondary"],
                            "type": "button",
                            "on_click": handle_close,
                        },
                        "Close",
                    ),
                ),
                html.div(
                    {"style": styles["help_section"]},
                    html.div({"style": styles["help_heading"]}, "What it does"),
                    html.p({"style": styles["help_text"]}, topic["what"]),
                ),
                html.div(
                    {"style": styles["help_section"]},
                    html.div(
                        {"style": styles["help_heading"]}, "How VidSmoother uses it"
                    ),
                    html.p({"style": styles["help_text"]}, topic["how"]),
                ),
                html.div(
                    {"style": styles["help_section"]},
                    html.div(
                        {"style": styles["help_heading"]}, "Why your choice matters"
                    ),
                    html.p({"style": styles["help_text"]}, topic["implications"]),
                ),
                html.div(
                    {"style": styles["help_section"]},
                    html.div(
                        {"style": styles["help_heading"]}, "Settings that affect it"
                    ),
                    html.p({"style": styles["help_text"]}, topic["related"]),
                ),
            ),
        )

    @component
    def App():
        settings, set_settings = hooks.use_state(default_settings())
        mode, set_mode = hooks.use_state("beginner")
        help_state, set_help_state = hooks.use_state(None)
        shutting_down, set_shutting_down = hooks.use_state(False)
        media, set_media = hooks.use_state([])
        selected, set_selected = hooks.use_state([])
        notice, set_notice = hooks.use_state("")
        job, set_job = hooks.use_state(_job_snapshot())

        def auto_refresh_job():
            stop = asyncio.Event()

            async def refresh_loop():
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if not stop.is_set():
                            set_job(_job_snapshot())

            task = asyncio.create_task(refresh_loop())

            def cleanup():
                stop.set()
                task.cancel()

            return cleanup

        hooks.use_effect(auto_refresh_job, [])

        def refresh_media(event):
            videos, error = scan_media(settings)
            set_media([str(path) for path in videos])
            set_selected([str(path) for path in videos])
            set_notice(error or f"Found {len(videos)} media file(s).")

        def refresh_job(event):
            set_job(_job_snapshot())

        def run_selected(event):
            started = start_job(settings, selected)
            set_job(_job_snapshot())
            if not started:
                set_notice("A processing job is already running.")

        def shut_down(event):
            set_shutting_down(True)
            requested = request_app_shutdown()
            set_job(_job_snapshot())
            set_notice(
                "Shutting down VidSmoother and child processes."
                if requested
                else "Shutdown is already in progress."
            )

        def toggle_media(path: str):
            def handle_change(event):
                if event["target"]["checked"]:
                    set_selected(sorted(set(selected) | {path}))
                else:
                    set_selected([item for item in selected if item != path])

            return handle_change

        def choose_mode(next_mode: str):
            def handle_click(event):
                set_mode(next_mode)

            return handle_click

        def open_help(setting_key: str, label: str):
            set_help_state({"key": setting_key, "label": label})

        def close_help():
            set_help_state(None)

        media_rows = [
            html.div(
                {"style": styles["media"]},
                html.input(
                    {
                        "type": "checkbox",
                        "checked": path in selected,
                        "on_change": toggle_media(path),
                    }
                ),
                html.div(
                    html.div(Path(path).name),
                    html.div({"style": styles["muted"]}, str(Path(path).parent)),
                ),
                html.div(
                    {"style": styles["muted"]},
                    Path(path).suffix.lower().lstrip(".").upper(),
                ),
            )
            for path in media
        ]

        return html.div(
            {"style": styles["page"]},
            HelpModal(help_state, mode, close_help),
            html.div(
                {
                    "style": {
                        "maxWidth": "1180px",
                        "margin": "0 auto 18px",
                        "display": "flex",
                        "justifyContent": "space-between",
                        "gap": "16px",
                        "alignItems": "flex-start",
                    }
                },
                html.div(
                    html.h1(
                        {"style": {"margin": "0 0 4px", "fontSize": "28px"}}, APP_TITLE
                    ),
                    html.div(
                        {"style": styles["muted"]},
                        "ReactPy control surface for interpolation settings and batch media selection.",
                    ),
                ),
                html.button(
                    {
                        "style": styles["danger"],
                        "on_click": shut_down,
                        "disabled": shutting_down,
                    },
                    "Shutting down..." if shutting_down else "Shut down",
                ),
            ),
            html.div(
                {"style": styles["shell"]},
                html.section(
                    {"style": styles["panel"]},
                    html.h2(
                        {"style": {"marginTop": 0, "fontSize": "18px"}}, "Settings"
                    ),
                    html.div(
                        {"style": styles["segmented"]},
                        html.button(
                            {
                                "style": (
                                    styles["segment_active"]
                                    if mode == "beginner"
                                    else styles["segment"]
                                ),
                                "on_click": choose_mode("beginner"),
                            },
                            "Beginner",
                        ),
                        html.button(
                            {
                                "style": (
                                    styles["segment_active"]
                                    if mode == "advanced"
                                    else styles["segment"]
                                ),
                                "on_click": choose_mode("advanced"),
                            },
                            "Advanced",
                        ),
                    ),
                    (
                        beginner_settings(settings, set_settings, mode, open_help)
                        if mode == "beginner"
                        else advanced_settings(settings, set_settings, mode, open_help)
                    ),
                ),
                html.section(
                    {"style": styles["panel"]},
                    html.div(
                        {
                            "style": {
                                "display": "flex",
                                "justifyContent": "space-between",
                                "gap": "10px",
                                "alignItems": "center",
                            }
                        },
                        html.h2({"style": {"margin": 0, "fontSize": "18px"}}, "Media"),
                        html.div(
                            {"style": {"display": "flex", "gap": "8px"}},
                            html.button(
                                {
                                    "style": styles["secondary"],
                                    "on_click": refresh_media,
                                },
                                "Scan",
                            ),
                            html.button(
                                {
                                    "style": styles["button"],
                                    "on_click": run_selected,
                                    "disabled": job.running,
                                },
                                "Process",
                            ),
                        ),
                    ),
                    html.p(
                        {"style": styles["muted"]},
                        notice or "Scan the input folder to populate this list.",
                    ),
                    html.div(
                        media_rows
                        or [html.p({"style": styles["muted"]}, "No media loaded.")]
                    ),
                    html.h2(
                        {"style": {"fontSize": "18px", "margin": "22px 0 8px"}}, "Job"
                    ),
                    html.div(
                        {
                            "style": {
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "8px",
                            }
                        },
                        html.span({"style": styles["muted"]}, f"Status: {job.status}"),
                        html.button(
                            {"style": styles["secondary"], "on_click": refresh_job},
                            "Refresh",
                        ),
                    ),
                    html.pre(
                        {"style": styles["log"]},
                        "\n".join(job.log) if job.log else "No job output yet.",
                    ),
                ),
            ),
        )

    return App


def _open_browser(host: str, port: int) -> None:
    webbrowser.open(f"http://{host}:{port}")


def run_ui(
    host: str = "127.0.0.1", port: int = 8764, *, open_browser: bool = True
) -> None:
    from reactpy import run

    if open_browser:
        threading.Timer(1.0, _open_browser, args=(host, port)).start()
    run(make_app_component(), host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the VidSmoother ReactPy interface."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8764)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser.",
    )
    args = parser.parse_args(argv)
    run_ui(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
