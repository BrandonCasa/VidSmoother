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
        subtitle_mode=settings["subtitle_mode"],
    )


def _job_snapshot() -> UiJob:
    with _JOB_LOCK:
        return UiJob(status=_JOB.status, log=list(_JOB.log), running=_JOB.running)


def _set_job(*, status: str | None = None, running: bool | None = None, message: str | None = None) -> None:
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
        videos = [Path(path) for path in selected_media] or iter_videos(config.input_dir, recursive=config.recursive)
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

    thread = threading.Thread(target=_run_job, args=(dict(settings), list(selected_media)), daemon=True)
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
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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

    _set_job(status="shutting down", running=False, message="Shutting down VidSmoother and child processes.")
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
        "label": {"fontSize": "12px", "fontWeight": "700", "color": "#52606d", "textTransform": "uppercase"},
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
        "group_title": {"fontSize": "14px", "fontWeight": "800", "margin": "18px 0 10px", "color": "#334e68"},
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
    }

    @component
    def TextField(label: str, setting_key: str, settings: dict[str, Any], set_settings):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["value"]
            set_settings(next_settings)

        return html.label(
            {"style": styles["field"]},
            html.span({"style": styles["label"]}, label),
            html.input({"style": styles["input"], "value": settings[setting_key], "on_change": handle_change}),
        )

    @component
    def SelectField(
        label: str,
        setting_key: str,
        options: list[str] | list[tuple[str, str]],
        settings: dict[str, Any],
        set_settings,
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

        return html.label(
            {"style": styles["field"]},
            html.span({"style": styles["label"]}, label),
            html.select(
                {"style": styles["input"], "value": settings[setting_key], "on_change": handle_change},
                option_nodes,
            ),
        )

    @component
    def Toggle(label: str, setting_key: str, settings: dict[str, Any], set_settings):
        def handle_change(event):
            next_settings = dict(settings)
            next_settings[setting_key] = event["target"]["checked"]
            set_settings(next_settings)

        return html.label(
            {"style": {"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "10px"}},
            html.input({"type": "checkbox", "checked": bool(settings[setting_key]), "on_change": handle_change}),
            html.span(label),
        )

    @component
    def SmoothnessSelect(settings: dict[str, Any], set_settings):
        current = f"{settings['factor_num']}:{settings['factor_den']}"

        def handle_change(event):
            numerator, denominator = event["target"]["value"].split(":", maxsplit=1)
            next_settings = dict(settings)
            next_settings["factor_num"] = numerator
            next_settings["factor_den"] = denominator
            set_settings(next_settings)

        return html.label(
            {"style": styles["field"]},
            html.span({"style": styles["label"]}, "Smoothness boost"),
            html.select(
                {"style": styles["input"], "value": current, "on_change": handle_change},
                [
                    html.option({"value": "2:1"}, "2x smoother"),
                    html.option({"value": "3:1"}, "3x smoother"),
                    html.option({"value": "4:1"}, "4x smoother"),
                    html.option({"value": "1:1"}, "Keep original frame rate"),
                ],
            ),
        )

    def advanced_settings(settings: dict[str, Any], set_settings) -> list[Any]:
        return [
            TextField("Input folder", "input_dir", settings, set_settings),
            TextField("Output folder", "output_dir", settings, set_settings),
            TextField("Work folder", "work_dir", settings, set_settings),
            html.div(
                {"style": styles["row"]},
                TextField("Factor numerator", "factor_num", settings, set_settings),
                TextField("Factor denominator", "factor_den", settings, set_settings),
            ),
            html.div(
                {"style": styles["row"]},
                TextField("RIFE model", "rife_model", settings, set_settings),
                TextField("Scale", "scale", settings, set_settings),
            ),
            html.div(
                {"style": styles["row"]},
                TextField("CUDA device", "device_index", settings, set_settings),
                TextField("Workers", "workers", settings, set_settings),
            ),
            SelectField("Source filter", "source_filter", ["lsmas", "ffms2", "bestsource"], settings, set_settings),
            SelectField("NVENC codec", "nvenc_codec", ["auto", "hevc_nvenc", "h264_nvenc", "av1_nvenc"], settings, set_settings),
            html.div(
                {"style": styles["row"]},
                TextField("CQ", "cq", settings, set_settings),
                TextField("Pixel format", "pix_fmt", settings, set_settings),
            ),
            html.div(
                {"style": styles["row"]},
                TextField("TRT opt shape", "trt_opt_shape", settings, set_settings),
                TextField("TRT max shape", "trt_max_shape", settings, set_settings),
            ),
            Toggle("Recursive scan", "recursive", settings, set_settings),
            Toggle("Overwrite outputs", "overwrite", settings, set_settings),
            Toggle("Dry run", "dry_run", settings, set_settings),
            Toggle("Ensemble mode", "ensemble", settings, set_settings),
            Toggle("Disable scene change detection", "no_scene_change", settings, set_settings),
        ]

    def beginner_settings(settings: dict[str, Any], set_settings) -> list[Any]:
        return [
            html.div({"style": styles["group_title"]}, "Folders"),
            TextField("Videos folder", "input_dir", settings, set_settings),
            TextField("Save finished videos to", "output_dir", settings, set_settings),
            TextField("Temporary files folder", "work_dir", settings, set_settings),
            Toggle("Include videos inside subfolders", "recursive", settings, set_settings),
            Toggle("Replace finished videos with the same name", "overwrite", settings, set_settings),
            html.div({"style": styles["group_title"]}, "Smoothness and speed"),
            SmoothnessSelect(settings, set_settings),
            html.div(
                {"style": styles["row"]},
                TextField("Model version", "rife_model", settings, set_settings),
                TextField("GPU number", "device_index", settings, set_settings),
            ),
            html.div(
                {"style": styles["row"]},
                TextField("Processing scale", "scale", settings, set_settings),
                TextField("Videos at once", "workers", settings, set_settings),
            ),
            Toggle("Use extra smoothing passes", "ensemble", settings, set_settings),
            Toggle("Ignore hard scene cuts", "no_scene_change", settings, set_settings),
            html.div({"style": styles["group_title"]}, "Output video"),
            SelectField(
                "Video codec",
                "nvenc_codec",
                [
                    ("auto", "Auto match source"),
                    ("hevc_nvenc", "H.265 / HEVC"),
                    ("h264_nvenc", "H.264 / AVC"),
                    ("av1_nvenc", "AV1"),
                ],
                settings,
                set_settings,
            ),
            html.div(
                {"style": styles["row"]},
                TextField("Quality, lower is better", "cq", settings, set_settings),
                TextField("Pixel format", "pix_fmt", settings, set_settings),
            ),
            html.div(
                {"style": styles["row"]},
                TextField("Exact bitrate", "bitrate", settings, set_settings),
                SelectField(
                    "Audio",
                    "audio_codec",
                    [("copy", "Keep original audio"), ("aac", "AAC"), ("libopus", "Opus"), ("none", "No audio")],
                    settings,
                    set_settings,
                ),
            ),
            html.div({"style": styles["group_title"]}, "Tools"),
            TextField("FFmpeg app", "ffmpeg", settings, set_settings),
            TextField("FFprobe app", "ffprobe", settings, set_settings),
            TextField("VapourSynth runner", "vspipe", settings, set_settings),
            Toggle("Preview commands only", "dry_run", settings, set_settings),
        ]

    @component
    def App():
        settings, set_settings = hooks.use_state(default_settings())
        mode, set_mode = hooks.use_state("beginner")
        shutting_down, set_shutting_down = hooks.use_state(False)
        media, set_media = hooks.use_state([])
        selected, set_selected = hooks.use_state([])
        notice, set_notice = hooks.use_state("")
        job, set_job = hooks.use_state(_job_snapshot())

        async def auto_refresh_job():
            while True:
                await asyncio.sleep(1.0)
                set_job(_job_snapshot())

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
            set_notice("Shutting down VidSmoother and child processes." if requested else "Shutdown is already in progress.")

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

        media_rows = [
            html.div(
                {"style": styles["media"]},
                html.input({"type": "checkbox", "checked": path in selected, "on_change": toggle_media(path)}),
                html.div(
                    html.div(Path(path).name),
                    html.div({"style": styles["muted"]}, str(Path(path).parent)),
                ),
                html.div({"style": styles["muted"]}, Path(path).suffix.lower().lstrip(".").upper()),
            )
            for path in media
        ]

        return html.div(
            {"style": styles["page"]},
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
                    html.h1({"style": {"margin": "0 0 4px", "fontSize": "28px"}}, APP_TITLE),
                    html.div({"style": styles["muted"]}, "ReactPy control surface for interpolation settings and batch media selection."),
                ),
                html.button(
                    {"style": styles["danger"], "on_click": shut_down, "disabled": shutting_down},
                    "Shutting down..." if shutting_down else "Shut down",
                ),
            ),
            html.div(
                {"style": styles["shell"]},
                html.section(
                    {"style": styles["panel"]},
                    html.h2({"style": {"marginTop": 0, "fontSize": "18px"}}, "Settings"),
                    html.div(
                        {"style": styles["segmented"]},
                        html.button(
                            {"style": styles["segment_active"] if mode == "beginner" else styles["segment"], "on_click": choose_mode("beginner")},
                            "Beginner",
                        ),
                        html.button(
                            {"style": styles["segment_active"] if mode == "advanced" else styles["segment"], "on_click": choose_mode("advanced")},
                            "Advanced",
                        ),
                    ),
                    beginner_settings(settings, set_settings) if mode == "beginner" else advanced_settings(settings, set_settings),
                ),
                html.section(
                    {"style": styles["panel"]},
                    html.div(
                        {"style": {"display": "flex", "justifyContent": "space-between", "gap": "10px", "alignItems": "center"}},
                        html.h2({"style": {"margin": 0, "fontSize": "18px"}}, "Media"),
                        html.div(
                            {"style": {"display": "flex", "gap": "8px"}},
                            html.button({"style": styles["secondary"], "on_click": refresh_media}, "Scan"),
                            html.button({"style": styles["button"], "on_click": run_selected, "disabled": job.running}, "Process"),
                        ),
                    ),
                    html.p({"style": styles["muted"]}, notice or "Scan the input folder to populate this list."),
                    html.div(media_rows or [html.p({"style": styles["muted"]}, "No media loaded.")]),
                    html.h2({"style": {"fontSize": "18px", "margin": "22px 0 8px"}}, "Job"),
                    html.div(
                        {"style": {"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}},
                        html.span({"style": styles["muted"]}, f"Status: {job.status}"),
                        html.button({"style": styles["secondary"], "on_click": refresh_job}, "Refresh"),
                    ),
                    html.pre({"style": styles["log"]}, "\n".join(job.log) if job.log else "No job output yet."),
                ),
            ),
        )

    return App


def _open_browser(host: str, port: int) -> None:
    webbrowser.open(f"http://{host}:{port}")


def run_ui(host: str = "127.0.0.1", port: int = 8000, *, open_browser: bool = True) -> None:
    from reactpy import run

    if open_browser:
        threading.Timer(1.0, _open_browser, args=(host, port)).start()
    run(make_app_component(), host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the VidSmoother ReactPy interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser.")
    args = parser.parse_args(argv)
    run_ui(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
