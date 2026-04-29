from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class ToolPaths:
    ffmpeg: Path
    ffprobe: Path
    vspipe: Path


@dataclass(frozen=True)
class VapourSynthOptions:
    source_filter: str
    matrix: str
    fp16: bool
    max_cache_size_mb: int | None


@dataclass(frozen=True)
class RifeTensorRtOptions:
    model: str
    device_index: int
    factor_num: int
    factor_den: int
    scale: float
    ensemble: bool
    scene_change: bool
    scene_threshold: float | None
    auto_download: bool
    trt_static_shape: bool
    trt_cache_dir: Path
    trt_workspace_size: int
    trt_optimization_level: int | None
    trt_max_aux_streams: int | None
    trt_min_shape: tuple[int, int]
    trt_opt_shape: tuple[int, int]
    trt_max_shape: tuple[int, int]


@dataclass(frozen=True)
class NvencOptions:
    codec: str
    preset: str
    rate_control: str
    cq: int | None
    qp: int | None
    bitrate: str | None
    maxrate: str | None
    bufsize: str | None
    pix_fmt: str
    audio_codec: str


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    work_dir: Path
    tools: ToolPaths
    vapoursynth: VapourSynthOptions
    rife: RifeTensorRtOptions
    nvenc: NvencOptions
    subtitle_mode: str
    overwrite: bool
    dry_run: bool
    recursive: bool
    workers: int
