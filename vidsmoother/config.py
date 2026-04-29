from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class ToolPaths:
    ffmpeg: Path
    ffprobe: Path
    rife: Path
    rife_model: Path | None


@dataclass(frozen=True)
class RifeOptions:
    gpu: str | None
    threads: str | None
    tta_spatial: bool
    tta_temporal: bool
    uhd: bool
    output_pattern: str = "%08d.png"


@dataclass(frozen=True)
class EncodeOptions:
    video_codec: str
    audio_codec: str
    pix_fmt: str
    crf: int | None
    preset: str | None
    video_bitrate: str | None


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    work_dir: Path
    tools: ToolPaths
    rife: RifeOptions
    encode: EncodeOptions
    scene_mode: str
    scene_threshold: float
    subtitle_mode: str
    overwrite: bool
    keep_work: bool
    dry_run: bool
    recursive: bool
    workers: int
