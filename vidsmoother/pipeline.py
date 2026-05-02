from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .dedup_timeline import process_dedup_timeline
from .gif_timeline import process_gif_timeline
from .media import VideoInfo, iter_videos, probe_video
from .runner import run_command, run_vspipe_to_ffmpeg
from .tools import bundled_runtime_env
from .vpy import write_vapoursynth_script


NVENC_ENCODERS_BY_CODEC = {
    "h264": "h264_nvenc",
    "hevc": "hevc_nvenc",
    "av1": "av1_nvenc",
}


def process_all(config: PipelineConfig) -> None:
    videos = iter_videos(config.input_dir, recursive=config.recursive)
    if not videos:
        print(f"No videos found in {config.input_dir}")
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    config.rife.trt_cache_dir.mkdir(parents=True, exist_ok=True)

    if config.workers <= 1 or len(videos) == 1:
        for video in videos:
            process_video(video, config)
        return

    per_video_config = replace(config, workers=1)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(process_video, video, per_video_config) for video in videos]
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
    script = video_work / f"{video.stem}.vpy"
    logs.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing {video.name}: {info.width}x{info.height}, "
        f"{info.fps:.3f}fps x {config.rife.factor_num}/{config.rife.factor_den} via TensorRT"
    )

    if info.is_gif and (config.gif.timeline_smoothing or config.dedup.strength > 0):
        output = process_gif_timeline(video, info, output, video_work, logs, config)
        print(f"  Output: {output}")
        return output

    if config.dedup.strength > 0:
        output = process_dedup_timeline(video, info, output, video_work, logs, config)
        print(f"  Output: {output}")
        return output

    source_video, source_info = prepare_source_video(video, info, video_work, logs, config)
    write_vapoursynth_script(source_video, source_info, script, config)
    run_vspipe_to_ffmpeg(
        build_vspipe_command(script, config),
        build_ffmpeg_command(info, output, config),
        vspipe_log=logs / "vspipe.log",
        ffmpeg_log=logs / "ffmpeg.log",
        dry_run=config.dry_run,
    )

    print(f"  Output: {output}")
    return output


def output_path_for(video: Path, config: PipelineConfig) -> Path:
    suffix = "trt_rife"
    extension = ".gif" if video.suffix.lower() == ".gif" else ".mp4"
    return config.output_dir / f"{video.stem}_{suffix}_{config.rife.factor_num}x{extension}"


def prepare_source_video(
    video: Path,
    info: VideoInfo,
    video_work: Path,
    logs: Path,
    config: PipelineConfig,
) -> tuple[Path, VideoInfo]:
    if not info.is_gif:
        return video, info

    prepared = video_work / f"{video.stem}_source.mkv"
    run_command(
        [
            config.tools.ffmpeg,
            "-y",
            "-ignore_loop",
            "1",
            "-i",
            video,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv444p",
            prepared,
        ],
        log_file=logs / "gif-prepare.log",
        dry_run=config.dry_run,
    )
    return prepared, replace(info, path=prepared, codec="ffv1", pix_fmt="yuv444p", has_audio=False)


def build_vspipe_command(script: Path, config: PipelineConfig) -> list[object]:
    return [config.tools.vspipe, "--container", "y4m", script, "-"]


def build_ffmpeg_command(info: VideoInfo, output: Path, config: PipelineConfig) -> list[object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if info.is_gif:
        return build_gif_ffmpeg_command(info, output, config)

    command: list[object] = [
        config.tools.ffmpeg,
        "-y",
        "-i",
        "pipe:0",
        "-i",
        info.path,
        "-map",
        "0:v:0",
    ]

    if info.has_audio and config.nvenc.audio_codec != "none":
        command.extend(["-map", "1:a:0?"])

    video_encoder = resolve_video_encoder(info, config)
    append_video_encode_options(command, video_encoder, info, config)

    if info.has_audio and config.nvenc.audio_codec == "none":
        command.append("-an")
    elif info.has_audio:
        command.extend(["-c:a", config.nvenc.audio_codec])

    command.extend(["-movflags", "+faststart", output])
    return command


def build_gif_ffmpeg_command(info: VideoInfo, output: Path, config: PipelineConfig) -> list[object]:
    filters = build_gif_filter_chain(info, config)
    command: list[object] = [
        config.tools.ffmpeg,
        "-y",
        "-i",
        "pipe:0",
        "-filter_complex",
        filters,
        "-gifflags",
        "+transdiff+offsetting",
        "-loop",
        "0",
    ]
    command.append(output)
    return command


def build_gif_filter_chain(info: VideoInfo, config: PipelineConfig) -> str:
    frame_rate = info.fps * config.rife.factor_num / config.rife.factor_den
    if config.gif.max_fps is not None:
        frame_rate = min(frame_rate, config.gif.max_fps)

    pre_palette_filters = [f"fps=fps={frame_rate:.6f}:round=near"]
    if config.gif.max_width is not None:
        pre_palette_filters.append(
            f"scale=w='min(iw\\,{config.gif.max_width})':h=-2:flags=lanczos"
        )

    pre_palette = ",".join(pre_palette_filters)
    return (
        f"[0:v]{pre_palette},split=2[gif_frames][gif_palette_src];"
        "[gif_palette_src]palettegen=stats_mode=full[gif_palette];"
        "[gif_frames][gif_palette]paletteuse=dither=sierra2_4a"
    )


def resolve_video_encoder(info: VideoInfo, config: PipelineConfig) -> str:
    if config.nvenc.codec is not None:
        return config.nvenc.codec

    nvenc_encoder = NVENC_ENCODERS_BY_CODEC.get(info.codec)
    if nvenc_encoder and encoder_available(config.tools.ffmpeg, nvenc_encoder):
        return nvenc_encoder

    return info.codec


def append_video_encode_options(
    command: list[object],
    video_encoder: str,
    info: VideoInfo,
    config: PipelineConfig,
) -> None:
    command.extend(["-c:v", video_encoder])

    using_nvenc = video_encoder.endswith("_nvenc")
    if using_nvenc:
        command.extend(["-preset", config.nvenc.preset, "-rc", config.nvenc.rate_control])

    if using_nvenc and config.nvenc.cq is not None:
        command.extend(["-cq", str(config.nvenc.cq)])
    if using_nvenc and config.nvenc.qp is not None:
        command.extend(["-qp", str(config.nvenc.qp)])
    if config.nvenc.bitrate:
        command.extend(["-b:v", config.nvenc.bitrate])
    if config.nvenc.maxrate:
        command.extend(["-maxrate", config.nvenc.maxrate])
    if config.nvenc.bufsize:
        command.extend(["-bufsize", config.nvenc.bufsize])

    pix_fmt = info.pix_fmt if config.nvenc.pix_fmt == "auto" else config.nvenc.pix_fmt
    command.extend(["-pix_fmt", pix_fmt])


def encoder_available(ffmpeg: Path, encoder: str) -> bool:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=bundled_runtime_env(),
    )
    if result.returncode != 0:
        return False
    return encoder in result.stdout
