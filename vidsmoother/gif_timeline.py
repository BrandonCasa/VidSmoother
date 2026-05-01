from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .errors import VidSmootherError
from .filters import build_dedup_filter
from .media import VideoInfo
from .runner import run_command, run_vspipe_to_ffmpeg
from .tools import bundled_runtime_env
from .vpy import write_vapoursynth_script


@dataclass(frozen=True)
class GifTimelineSegment:
    frame_index: int
    next_index: int
    durations: tuple[float, ...]
    hard_hold: bool


@dataclass(frozen=True)
class GifTimelineFrame:
    path: Path
    duration: float


def process_gif_timeline(
    video: Path,
    info: VideoInfo,
    output: Path,
    video_work: Path,
    logs: Path,
    config: PipelineConfig,
) -> Path:
    frames_dir = video_work / "gif_frames"
    timeline_dir = video_work / "gif_timeline"
    transition_dir = video_work / "gif_transitions"
    manifest = video_work / "gif_timeline.ffconcat"
    frames_dir.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    transition_dir.mkdir(parents=True, exist_ok=True)

    if config.dry_run:
        print(f"[dry-run] timeline-aware GIF smoothing for {video} -> {output}")
        print(f"[dry-run] extract GIF frames and frame delays under {video_work}")
        return output

    clear_generated_frames(frames_dir, "frame_*.png")
    fallback_ms = 1000.0 / info.fps
    delay_payload = probe_gif_frame_payload(config.tools.ffprobe, video)
    delays_ms, used_fallback = parse_gif_frame_delays(delay_payload, fallback_ms)
    if used_fallback:
        (logs / "gif-timeline.log").write_text(
            f"Frame delay metadata was incomplete. Falling back to {fallback_ms:.3f} ms per frame.\n",
            encoding="utf-8",
        )

    extract_gif_frames(config.tools.ffmpeg, video, frames_dir, logs / "gif-extract.log")
    source_frames = sorted(frames_dir.glob("frame_*.png"))
    if not source_frames:
        raise VidSmootherError(f"No GIF frames were extracted from {video}")

    delays_ms = normalize_delay_count(delays_ms, len(source_frames), fallback_ms)
    quantum_ms = 1000.0 / config.gif.max_fps if config.gif.max_fps else fallback_ms
    segments = build_timeline_segments(
        delays_ms,
        quantum_ms=quantum_ms,
        hard_hold_percentile=config.gif.hard_hold_percentile,
    )

    rendered: list[GifTimelineFrame] = []
    for segment in segments:
        frame = source_frames[segment.frame_index]
        next_frame = source_frames[segment.next_index]
        if should_hold_segment(frame, next_frame, segment, config):
            rendered.append(GifTimelineFrame(frame, sum(segment.durations)))
            continue

        transition_frames = render_transition_frames(
            frame,
            next_frame,
            len(segment.durations),
            info,
            transition_dir,
            logs,
            config,
        )
        rendered.extend(
            GifTimelineFrame(path, duration)
            for path, duration in zip(transition_frames, segment.durations, strict=True)
        )

    write_concat_manifest(manifest, rendered)
    encode_timeline_gif(manifest, output, logs / "gif-encode.log", config)
    return output


def probe_gif_frame_payload(ffprobe: Path, video: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-print_format",
            "json",
            str(video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise VidSmootherError(
            f"ffprobe failed for GIF frame timing: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def parse_gif_frame_delays(
    payload: dict[str, Any], fallback_ms: float
) -> tuple[list[float], bool]:
    frames = [
        frame
        for frame in payload.get("frames", [])
        if frame.get("media_type") == "video"
    ]
    delays = [_frame_duration_ms(frame) for frame in frames]
    if delays and all(delay is not None and delay > 0 for delay in delays):
        return [float(delay) for delay in delays if delay is not None], False

    timestamps = [_frame_timestamp_ms(frame) for frame in frames]
    if len(timestamps) >= 2 and all(timestamp is not None for timestamp in timestamps):
        inferred: list[float] = []
        for index, timestamp in enumerate(timestamps):
            if index + 1 < len(timestamps):
                next_timestamp = timestamps[index + 1]
                assert timestamp is not None and next_timestamp is not None
                inferred.append(max(1.0, next_timestamp - timestamp))
            else:
                inferred.append(fallback_ms)
        return inferred, True

    return [fallback_ms for _ in frames], True


def normalize_delay_count(
    delays_ms: list[float], frame_count: int, fallback_ms: float
) -> list[float]:
    if len(delays_ms) >= frame_count:
        return delays_ms[:frame_count]
    return delays_ms + [fallback_ms for _ in range(frame_count - len(delays_ms))]


def build_timeline_segments(
    delays_ms: list[float],
    *,
    quantum_ms: float,
    hard_hold_percentile: float,
) -> list[GifTimelineSegment]:
    if not delays_ms:
        return []

    threshold = percentile(delays_ms, hard_hold_percentile)
    median = percentile(delays_ms, 50.0)
    segments: list[GifTimelineSegment] = []
    for index, delay_ms in enumerate(delays_ms):
        hard_hold = delay_ms >= threshold and delay_ms > median
        slot_count = 1 if hard_hold else slots_for_delay(delay_ms, quantum_ms)
        durations = centisecond_durations(delay_ms, slot_count)
        segments.append(
            GifTimelineSegment(
                frame_index=index,
                next_index=(index + 1) % len(delays_ms),
                durations=tuple(durations),
                hard_hold=hard_hold,
            )
        )
    return segments


def slots_for_delay(delay_ms: float, quantum_ms: float) -> int:
    if quantum_ms <= 0:
        return 1
    total_cs = max(1, round(delay_ms / 10.0))
    return max(1, min(total_cs, round(delay_ms / quantum_ms)))


def centisecond_durations(delay_ms: float, slot_count: int) -> list[float]:
    total_cs = max(1, round(delay_ms / 10.0))
    slots = max(1, min(slot_count, total_cs))
    base = total_cs // slots
    remainder = total_cs % slots
    return [
        float(base + (1 if index < remainder else 0)) / 100.0 for index in range(slots)
    ]


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    bounded = max(0.0, min(100.0, percent))
    position = (len(ordered) - 1) * bounded / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def should_hold_segment(
    frame: Path,
    next_frame: Path,
    segment: GifTimelineSegment,
    config: PipelineConfig,
) -> bool:
    if segment.hard_hold or len(segment.durations) <= 1:
        return True
    if file_sha256(frame) == file_sha256(next_frame):
        return True
    if config.dedup.strength > 0 and frames_are_near_duplicate(
        frame, next_frame, config
    ):
        return True
    return False


def render_transition_frames(
    frame: Path,
    next_frame: Path,
    slot_count: int,
    source_info: VideoInfo,
    transition_dir: Path,
    logs: Path,
    config: PipelineConfig,
) -> list[Path]:
    cache_key = transition_cache_key(frame, next_frame, slot_count, config)
    output_dir = transition_dir / cache_key
    output_pattern = output_dir / "transition_%06d.png"
    cached = sorted(output_dir.glob("transition_*.png"))
    if len(cached) >= slot_count:
        return cached[:slot_count]

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest = output_dir / "pair.ffconcat"
    pair_video = output_dir / "pair.mkv"
    script = output_dir / "pair.vpy"
    write_pair_manifest(pair_manifest, frame, next_frame)
    run_command(
        [
            config.tools.ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            pair_manifest,
            "-map",
            "0:v:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv444p",
            pair_video,
        ],
        log_file=logs / f"gif-transition-{cache_key}-prepare.log",
        dry_run=config.dry_run,
    )

    transition_info = replace(
        source_info,
        path=pair_video,
        codec="ffv1",
        pix_fmt="yuv444p",
        has_audio=False,
    )
    transition_config = replace(
        config,
        rife=replace(
            config.rife, factor_num=slot_count, factor_den=1, scene_change=False
        ),
    )
    write_vapoursynth_script(pair_video, transition_info, script, transition_config)
    run_vspipe_to_ffmpeg(
        [config.tools.vspipe, "--container", "y4m", script, "-"],
        [
            config.tools.ffmpeg,
            "-y",
            "-i",
            "pipe:0",
            "-frames:v",
            str(slot_count),
            output_pattern,
        ],
        vspipe_log=logs / f"gif-transition-{cache_key}-vspipe.log",
        ffmpeg_log=logs / f"gif-transition-{cache_key}-ffmpeg.log",
        dry_run=config.dry_run,
    )

    rendered = sorted(output_dir.glob("transition_*.png"))
    if len(rendered) < slot_count:
        raise VidSmootherError(
            f"Expected {slot_count} transition frames, got {len(rendered)}"
        )
    return rendered[:slot_count]


def extract_gif_frames(
    ffmpeg: Path, video: Path, frames_dir: Path, log_file: Path
) -> None:
    run_command(
        [
            ffmpeg,
            "-y",
            "-ignore_loop",
            "1",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            frames_dir / "frame_%06d.png",
        ],
        log_file=log_file,
    )


def clear_generated_frames(directory: Path, pattern: str) -> None:
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def encode_timeline_gif(
    manifest: Path, output: Path, log_file: Path, config: PipelineConfig
) -> None:
    run_command(
        [
            config.tools.ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            manifest,
            "-filter_complex",
            build_timeline_gif_filter_chain(config),
            "-gifflags",
            "+transdiff+offsetting",
            "-loop",
            "0",
            "-fps_mode:v",
            "vfr",
            output,
        ],
        log_file=log_file,
        dry_run=config.dry_run,
    )


def build_timeline_gif_filter_chain(config: PipelineConfig) -> str:
    filters: list[str] = []
    if config.gif.max_width is not None:
        filters.append(f"scale=w='min(iw\\,{config.gif.max_width})':h=-2:flags=lanczos")

    dedup_filter = build_dedup_filter(config)
    if dedup_filter:
        filters.append(dedup_filter)

    pre_palette = ",".join(filters)
    source = f"[0:v]{pre_palette},split" if pre_palette else "[0:v]split"
    return (
        f"{source}=2[gif_frames][gif_palette_src];"
        "[gif_palette_src]palettegen=stats_mode=full[gif_palette];"
        "[gif_frames][gif_palette]paletteuse=dither=sierra2_4a"
    )


def write_concat_manifest(manifest: Path, frames: list[GifTimelineFrame]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines.append(f"file '{escape_ffconcat_path(frame.path)}'")
        lines.append(f"duration {frame.duration:.6f}")
    if frames:
        lines.append(f"file '{escape_ffconcat_path(frames[-1].path)}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pair_manifest(manifest: Path, frame: Path, next_frame: Path) -> None:
    lines = [
        "ffconcat version 1.0",
        f"file '{escape_ffconcat_path(frame)}'",
        "duration 1",
        f"file '{escape_ffconcat_path(next_frame)}'",
        "duration 1",
        f"file '{escape_ffconcat_path(next_frame)}'",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transition_cache_key(
    frame: Path, next_frame: Path, slot_count: int, config: PipelineConfig
) -> str:
    digest = hashlib.sha256()
    digest.update(file_sha256(frame).encode("ascii"))
    digest.update(file_sha256(next_frame).encode("ascii"))
    digest.update(str(slot_count).encode("ascii"))
    digest.update(config.rife.model.encode("utf-8"))
    digest.update(str(config.rife.scale).encode("ascii"))
    digest.update(str(config.rife.ensemble).encode("ascii"))
    digest.update(str(config.vapoursynth.fp16).encode("ascii"))
    return digest.hexdigest()[:24]


def frames_are_near_duplicate(
    frame: Path, next_frame: Path, config: PipelineConfig
) -> bool:
    result = subprocess.run(
        [
            str(config.tools.ffmpeg),
            "-hide_banner",
            "-i",
            str(frame),
            "-i",
            str(next_frame),
            "-filter_complex",
            "[0:v][1:v]blend=all_mode=difference,format=gray,signalstats,metadata=print:file=-",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=bundled_runtime_env(),
    )
    if result.returncode != 0:
        return False

    match = re.search(
        r"lavfi\.signalstats\.YAVG=([0-9.]+)", result.stdout + result.stderr
    )
    if not match:
        return False
    average_difference = float(match.group(1))
    threshold = 0.25 + max(0.0, min(100.0, config.dedup.strength)) * 0.08
    return average_difference <= threshold


def _frame_duration_ms(frame: dict[str, Any]) -> float | None:
    for key in ("pkt_duration_time", "duration_time"):
        value = frame.get(key)
        if value not in (None, "N/A"):
            return float(value) * 1000.0
    return None


def _frame_timestamp_ms(frame: dict[str, Any]) -> float | None:
    for key in ("best_effort_timestamp_time", "pkt_pts_time", "pts_time"):
        value = frame.get(key)
        if value not in (None, "N/A"):
            return float(value) * 1000.0
    return None
