from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import PipelineConfig
from .errors import VidSmootherError
from .media import VideoInfo, iter_videos, probe_video
from .runner import run_command
from .scenes import Scene, detect_scenes
from .subtitles import escape_subtitle_path, matching_subtitle


def process_all(config: PipelineConfig) -> None:
    videos = iter_videos(config.input_dir, recursive=config.recursive)
    if not videos:
        print(f"No videos found in {config.input_dir}")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    if config.workers <= 1 or len(videos) == 1:
        for video in videos:
            process_video(video, config)
        return

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(process_video, video, config) for video in videos]
        for future in as_completed(futures):
            future.result()


def process_video(video: Path, config: PipelineConfig) -> Path:
    info = probe_video(config.tools.ffprobe, video)
    output = output_path_for(video, config)
    if output.exists() and not config.overwrite:
        print(f"Skipping existing output: {output}")
        return output

    video_work = config.work_dir / video.stem
    logs = video_work / "logs"
    frames = video_work / "frames"
    interpolated = video_work / "interpolated"
    combined = video_work / "combined"

    if video_work.exists() and not config.keep_work:
        shutil.rmtree(video_work)
    for directory in [logs, frames, interpolated, combined]:
        directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing {video.name}: {info.width}x{info.height}, "
        f"{info.fps:.3f}fps -> {info.fps * 2:.3f}fps"
    )

    scenes = detect_scenes(video, info, mode=config.scene_mode, threshold=config.scene_threshold)
    print(f"  Scenes: {len(scenes)}")

    for scene in scenes:
        extract_scene_frames(video, scene, frames / scene_name(scene), config, logs)
        interpolate_scene(scene, frames / scene_name(scene), interpolated / scene_name(scene), config, logs)

    collect_frames(scenes, interpolated, combined)
    encode_video(info, combined, output, config, logs)

    if not config.keep_work and not config.dry_run:
        shutil.rmtree(video_work, ignore_errors=True)

    print(f"  Output: {output}")
    return output


def output_path_for(video: Path, config: PipelineConfig) -> Path:
    return config.output_dir / f"{video.stem}_rife_2x.mp4"


def scene_name(scene: Scene) -> str:
    return f"scene_{scene.index:04d}"


def extract_scene_frames(
    video: Path,
    scene: Scene,
    output_dir: Path,
    config: PipelineConfig,
    logs: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[object] = [
        config.tools.ffmpeg,
        "-y",
        "-ss",
        f"{scene.start:.6f}",
        "-i",
        video,
        "-t",
        f"{scene.end - scene.start:.6f}",
        "-vsync",
        "0",
        "-q:v",
        "2",
        output_dir / "frame_%08d.png",
    ]
    run_command(command, log_file=logs / f"extract_{scene_name(scene)}.log", dry_run=config.dry_run)


def interpolate_scene(
    scene: Scene,
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    logs: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command: list[object] = [
        config.tools.rife,
        "-i",
        input_dir,
        "-o",
        output_dir,
    ]
    if config.tools.rife_model is not None:
        command.extend(["-m", config.tools.rife_model])
    if config.rife.gpu:
        command.extend(["-g", config.rife.gpu])
    if config.rife.threads:
        command.extend(["-j", config.rife.threads])
    if config.rife.tta_spatial:
        command.append("-x")
    if config.rife.tta_temporal:
        command.append("-z")
    if config.rife.uhd:
        command.append("-u")
    command.extend(["-f", config.rife.output_pattern])
    run_command(command, log_file=logs / f"rife_{scene_name(scene)}.log", dry_run=config.dry_run)


def collect_frames(scenes: list[Scene], interpolated: Path, combined: Path) -> None:
    combined.mkdir(parents=True, exist_ok=True)
    frame_number = 1
    for scene in scenes:
        scene_dir = interpolated / scene_name(scene)
        frames = sorted(scene_dir.glob("*.png")) + sorted(scene_dir.glob("*.jpg")) + sorted(scene_dir.glob("*.webp"))
        for frame in sorted(frames):
            shutil.copy2(frame, combined / f"frame_{frame_number:08d}{frame.suffix.lower()}")
            frame_number += 1
    if frame_number == 1:
        raise VidSmootherError(f"No interpolated frames found under {interpolated}")


def encode_video(info: VideoInfo, frames_dir: Path, output: Path, config: PipelineConfig, logs: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    first_frame = next(iter(sorted(frames_dir.glob("frame_*"))), None)
    if first_frame is None:
        raise VidSmootherError(f"No frames found in {frames_dir}")

    input_pattern = frames_dir / f"frame_%08d{first_frame.suffix.lower()}"
    command: list[object] = [
        config.tools.ffmpeg,
        "-y",
        "-framerate",
        f"{info.fps * 2:.6f}",
        "-i",
        input_pattern,
        "-i",
        info.path,
        "-map",
        "0:v:0",
    ]
    if info.has_audio:
        command.extend(["-map", "1:a:0?"])

    command.extend(["-c:v", config.encode.video_codec])
    if config.encode.preset:
        command.extend(["-preset", config.encode.preset])
    if config.encode.crf is not None:
        command.extend(["-crf", str(config.encode.crf)])
    if config.encode.video_bitrate:
        command.extend(["-b:v", config.encode.video_bitrate])
    command.extend(["-pix_fmt", config.encode.pix_fmt])

    subtitle = matching_subtitle(info.path)
    if subtitle and config.subtitle_mode == "burn":
        command.extend(["-vf", escape_subtitle_path(subtitle)])

    if info.has_audio:
        command.extend(["-c:a", config.encode.audio_codec])
    command.extend(["-movflags", "+faststart", output])

    run_command(command, log_file=logs / "encode.log", dry_run=config.dry_run)
