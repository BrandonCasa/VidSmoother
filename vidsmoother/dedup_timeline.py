from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
import re
from threading import Lock
from pathlib import Path

from .config import PipelineConfig
from .errors import VidSmootherError
from .filters import build_dedup_filter
from .media import VideoInfo
from .runner import run_command, run_vspipe_to_ffmpeg
from .vpy import write_vapoursynth_script

_TRT_CACHE_BUILD_LOCK = Lock()


@dataclass(frozen=True)
class DedupedSourceFrame:
    path: Path
    pts: float


@dataclass(frozen=True)
class TimelineFrame:
    path: Path
    duration: float


@dataclass(frozen=True)
class TransitionRenderJob:
    index: int
    frame: Path
    next_frame: Path
    slot_count: int
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class RenderedTransition:
    index: int
    frame_indices: tuple[int, ...]
    frames: list[Path]


def process_dedup_timeline(
    video: Path,
    info: VideoInfo,
    output: Path,
    video_work: Path,
    logs: Path,
    config: PipelineConfig,
) -> Path:
    frames_dir = video_work / "dedup_source_frames"
    transition_dir = video_work / "dedup_transitions"
    manifest = video_work / "dedup_timeline.ffconcat"
    frames_dir.mkdir(parents=True, exist_ok=True)
    transition_dir.mkdir(parents=True, exist_ok=True)

    target_fps = info.fps * config.rife.factor_num / config.rife.factor_den
    if target_fps <= 0:
        raise VidSmootherError("Target interpolation frame rate must be greater than zero")

    if config.dry_run:
        print(f"[dry-run] pre-interpolation dedup timeline for {video} -> {output}")
        print(f"[dry-run] extract kept source frames under {frames_dir}")
        return output

    source_frames = extract_deduped_source_frames(
        video,
        frames_dir,
        logs / "dedup-extract.log",
        config,
    )
    if len(source_frames) < 2:
        raise VidSmootherError(
            f"Deduplication left fewer than two unique frames for {video}; cannot interpolate"
        )

    rendered = render_timeline_frames(source_frames, info, transition_dir, logs, config)
    write_timeline_manifest(manifest, rendered)
    run_command(
        build_timeline_encode_command(manifest, info, output, config),
        log_file=logs / "dedup-encode.log",
        dry_run=config.dry_run,
    )
    return output


def extract_deduped_source_frames(
    video: Path,
    frames_dir: Path,
    log_file: Path,
    config: PipelineConfig,
) -> list[DedupedSourceFrame]:
    dedup_filter = build_dedup_filter(config)
    if dedup_filter is None:
        raise VidSmootherError("Pre-interpolation dedup requires deduplication to be enabled")

    clear_generated_frames(frames_dir, "frame_*.png")
    run_command(
        [
            config.tools.ffmpeg,
            "-y",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-vf",
            f"{dedup_filter},showinfo",
            "-fps_mode",
            "passthrough",
            frames_dir / "frame_%06d.png",
        ],
        log_file=log_file,
        dry_run=config.dry_run,
    )

    paths = sorted(frames_dir.glob("frame_*.png"))
    pts_values = parse_showinfo_pts(log_file.read_text(encoding="utf-8", errors="replace"))
    if len(pts_values) != len(paths):
        raise VidSmootherError(
            "Could not match deduplicated frames to their source timestamps. "
            f"Found {len(paths)} frame file(s) but {len(pts_values)} showinfo timestamp(s)."
        )

    return [
        DedupedSourceFrame(path=path, pts=pts)
        for path, pts in zip(paths, pts_values, strict=True)
    ]


def parse_showinfo_pts(log_text: str) -> list[float]:
    pts_values: list[float] = []
    for line in log_text.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = re.search(r"pts_time:(?P<pts>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)", line)
        if match:
            pts_values.append(float(match.group("pts")))
    return pts_values


def render_timeline_frames(
    source_frames: list[DedupedSourceFrame],
    info: VideoInfo,
    transition_dir: Path,
    logs: Path,
    config: PipelineConfig,
) -> list[TimelineFrame]:
    target_fps = info.fps * config.rife.factor_num / config.rife.factor_den
    frame_period = 1.0 / info.fps
    target_period = 1.0 / target_fps
    timeline_origin = source_frames[0].pts
    jobs: list[TransitionRenderJob] = []
    tail_frame: TimelineFrame | None = None

    for index, source_frame in enumerate(source_frames):
        if index + 1 >= len(source_frames):
            tail_duration = max(frame_period, info.duration - source_frame.pts)
            tail_frame = TimelineFrame(source_frame.path, tail_duration)
            continue

        next_frame = source_frames[index + 1]
        duration = max(frame_period, next_frame.pts - source_frame.pts)
        render_factor = render_factor_for_duration(duration, info.fps, config)
        frame_indices = transition_frame_indices(
            source_frame.pts,
            next_frame.pts,
            timeline_origin,
            target_fps,
            render_factor,
        )
        jobs.append(
            TransitionRenderJob(
                index=index,
                frame=source_frame.path,
                next_frame=next_frame.path,
                slot_count=render_factor,
                frame_indices=tuple(frame_indices),
            )
        )

    transitions = render_transition_jobs(jobs, info, transition_dir, logs, config)
    rendered: list[TimelineFrame] = []
    for job in jobs:
        transition_frames = transitions[job.index].frames
        rendered.extend(
            TimelineFrame(transition_frames[frame_index], target_period)
            for frame_index in job.frame_indices
        )
    if tail_frame is not None:
        rendered.append(tail_frame)

    return rendered


def render_transition_jobs(
    jobs: list[TransitionRenderJob],
    source_info: VideoInfo,
    transition_dir: Path,
    logs: Path,
    config: PipelineConfig,
) -> dict[int, RenderedTransition]:
    if not jobs:
        return {}

    max_workers = min(max(1, config.workers), len(jobs))
    rendered: dict[int, RenderedTransition] = {}
    remaining_jobs = list(jobs)

    if max_workers > 1:
        warmup_job = next(
            (job for job in jobs if not transition_frames_cached(job, transition_dir, config)),
            None,
        )
        if warmup_job is not None:
            with _TRT_CACHE_BUILD_LOCK:
                transition = render_transition_job(warmup_job, source_info, transition_dir, logs, config)
            rendered[transition.index] = transition
            remaining_jobs = [job for job in jobs if job.index != warmup_job.index]

    if max_workers <= 1:
        return {
            job.index: render_transition_job(job, source_info, transition_dir, logs, config)
            for job in jobs
        }
    if not remaining_jobs:
        return rendered

    cache_locks: dict[str, Lock] = {}
    cache_locks_guard = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                render_transition_job,
                job,
                source_info,
                transition_dir,
                logs,
                config,
                cache_locks,
                cache_locks_guard,
            )
            for job in remaining_jobs
        ]
        for future in as_completed(futures):
            transition = future.result()
            rendered[transition.index] = transition
    return rendered


def render_transition_job(
    job: TransitionRenderJob,
    source_info: VideoInfo,
    transition_dir: Path,
    logs: Path,
    config: PipelineConfig,
    cache_locks: dict[str, Lock] | None = None,
    cache_locks_guard: Lock | None = None,
) -> RenderedTransition:
    cache_key = transition_cache_key(job.frame, job.next_frame, job.slot_count, config)
    if cache_locks is not None and cache_locks_guard is not None:
        with cache_locks_guard:
            cache_lock = cache_locks.setdefault(cache_key, Lock())
        with cache_lock:
            frames = render_transition_frames(
                job.frame,
                job.next_frame,
                job.slot_count,
                source_info,
                transition_dir,
                logs,
                config,
                cache_key=cache_key,
            )
    else:
        frames = render_transition_frames(
            job.frame,
            job.next_frame,
            job.slot_count,
            source_info,
            transition_dir,
            logs,
            config,
            cache_key=cache_key,
        )
    return RenderedTransition(job.index, job.frame_indices, frames)


def transition_frames_cached(
    job: TransitionRenderJob,
    transition_dir: Path,
    config: PipelineConfig,
) -> bool:
    if job.slot_count <= 1:
        return True
    cache_key = transition_cache_key(job.frame, job.next_frame, job.slot_count, config)
    return len(list((transition_dir / cache_key).glob("transition_*.png"))) >= job.slot_count


def slot_count_for_duration(duration: float, target_fps: float) -> int:
    return max(1, round(max(0.0, duration) * target_fps))


def render_factor_for_duration(duration: float, source_fps: float, config: PipelineConfig) -> int:
    source_gap_frames = max(1, round(max(0.0, duration) * source_fps))
    return max(1, source_gap_frames * config.rife.factor_num)


def transition_frame_indices(
    start_pts: float,
    end_pts: float,
    timeline_origin: float,
    target_fps: float,
    render_factor: int,
) -> list[int]:
    start_time = max(0.0, start_pts - timeline_origin)
    end_time = max(start_time, end_pts - timeline_origin)
    start_slot = round(start_time * target_fps)
    end_slot = max(start_slot + 1, round(end_time * target_fps))
    duration = max(0.0, end_time - start_time)
    if duration == 0:
        return [0]

    indices: list[int] = []
    last_index = -1
    for slot in range(start_slot, end_slot):
        slot_time = slot / target_fps
        t = max(0.0, min(1.0, (slot_time - start_time) / duration))
        frame_index = max(0, min(render_factor - 1, round(t * render_factor)))
        if frame_index <= last_index:
            frame_index = last_index + 1
        if frame_index >= render_factor:
            break
        indices.append(frame_index)
        last_index = frame_index

    return indices or [0]


def render_transition_frames(
    frame: Path,
    next_frame: Path,
    slot_count: int,
    source_info: VideoInfo,
    transition_dir: Path,
    logs: Path,
    config: PipelineConfig,
    *,
    cache_key: str | None = None,
) -> list[Path]:
    if slot_count <= 1:
        return [frame]

    cache_key = cache_key or transition_cache_key(frame, next_frame, slot_count, config)
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
        log_file=logs / f"dedup-transition-{cache_key}-prepare.log",
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
        rife=replace(config.rife, factor_num=slot_count, factor_den=1),
    )
    write_vapoursynth_script(
        pair_video,
        transition_info,
        script,
        transition_config,
        frame_limit=slot_count,
    )
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
        vspipe_log=logs / f"dedup-transition-{cache_key}-vspipe.log",
        ffmpeg_log=logs / f"dedup-transition-{cache_key}-ffmpeg.log",
        dry_run=config.dry_run,
    )

    rendered = sorted(output_dir.glob("transition_*.png"))
    if len(rendered) < slot_count:
        raise VidSmootherError(f"Expected {slot_count} transition frames, got {len(rendered)}")
    return rendered[:slot_count]


def build_timeline_encode_command(
    manifest: Path,
    info: VideoInfo,
    output: Path,
    config: PipelineConfig,
) -> list[object]:
    from .pipeline import append_video_encode_options, resolve_video_encoder

    output.parent.mkdir(parents=True, exist_ok=True)
    command: list[object] = [
        config.tools.ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        manifest,
        "-i",
        info.path,
        "-map",
        "0:v:0",
    ]

    if info.has_audio and config.nvenc.audio_codec != "none":
        command.extend(["-map", "1:a:0?"])

    append_video_encode_options(command, resolve_video_encoder(info, config), info, config)
    command.extend(["-fps_mode:v", "vfr"])

    if info.has_audio and config.nvenc.audio_codec == "none":
        command.append("-an")
    elif info.has_audio:
        command.extend(["-c:a", config.nvenc.audio_codec])

    command.extend(["-movflags", "+faststart", "-shortest", output])
    return command


def write_timeline_manifest(manifest: Path, frames: list[TimelineFrame]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines.append(f"file '{escape_ffconcat_path(frame.path)}'")
        lines.append(f"duration {frame.duration:.9f}")
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
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_generated_frames(directory: Path, pattern: str) -> None:
    for path in directory.glob(pattern):
        if path.is_file():
            path.unlink()


def escape_ffconcat_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "'\\''")


def transition_cache_key(frame: Path, next_frame: Path, slot_count: int, config: PipelineConfig) -> str:
    digest = hashlib.sha256()
    digest.update(file_sha256(frame).encode("ascii"))
    digest.update(file_sha256(next_frame).encode("ascii"))
    digest.update(str(slot_count).encode("ascii"))
    digest.update(config.rife.model.encode("utf-8"))
    digest.update(str(config.rife.scale).encode("ascii"))
    digest.update(str(config.rife.ensemble).encode("ascii"))
    digest.update(str(config.rife.scene_change).encode("ascii"))
    digest.update(str(config.vapoursynth.fp16).encode("ascii"))
    return digest.hexdigest()[:24]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
