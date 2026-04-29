from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .config import VIDEO_EXTENSIONS
from .errors import VidSmootherError


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    pix_fmt: str
    has_audio: bool


def iter_videos(input_dir: Path, *, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    videos = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.name.lower())


def parse_rate(rate: str) -> float:
    if not rate or rate == "0/0":
        return 0.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return float(rate)


def probe_video(ffprobe: Path, video: Path) -> VideoInfo:
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video,
    ]
    result = subprocess.run(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise VidSmootherError(f"ffprobe failed for {video}: {result.stderr.strip()}")

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise VidSmootherError(f"No video stream found in {video}")

    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps = parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/0")
    if fps <= 0:
        raise VidSmootherError(f"Could not determine frame rate for {video}")

    duration = float(video_stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise VidSmootherError(f"Could not determine duration for {video}")

    return VideoInfo(
        path=video,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        duration=duration,
        codec=str(video_stream.get("codec_name", "unknown")),
        pix_fmt=str(video_stream.get("pix_fmt", "yuv420p")),
        has_audio=audio_stream is not None,
    )
