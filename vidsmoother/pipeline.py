from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .media import VideoInfo, iter_videos, probe_video
from .runner import run_vspipe_to_ffmpeg
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
    script = video_work / f"{video.stem}.vpy"
    logs.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing {video.name}: {info.width}x{info.height}, "
        f"{info.fps:.3f}fps x {config.rife.factor_num}/{config.rife.factor_den} via TensorRT"
    )

    write_vapoursynth_script(video, info, script, config)
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
    return config.output_dir / f"{video.stem}_{suffix}_{config.rife.factor_num}x.mp4"


def build_vspipe_command(script: Path, config: PipelineConfig) -> list[object]:
    return [config.tools.vspipe, "--container", "y4m", script, "-"]


def build_ffmpeg_command(info: VideoInfo, output: Path, config: PipelineConfig) -> list[object]:
    output.parent.mkdir(parents=True, exist_ok=True)
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

    if info.has_audio:
        command.extend(["-map", "1:a:0?"])

    video_encoder = resolve_video_encoder(info, config)
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

    if info.has_audio:
        command.extend(["-c:a", config.nvenc.audio_codec])

    command.extend(["-movflags", "+faststart", output])
    return command


def resolve_video_encoder(info: VideoInfo, config: PipelineConfig) -> str:
    if config.nvenc.codec is not None:
        return config.nvenc.codec

    nvenc_encoder = NVENC_ENCODERS_BY_CODEC.get(info.codec)
    if nvenc_encoder and encoder_available(config.tools.ffmpeg, nvenc_encoder):
        return nvenc_encoder

    return info.codec


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
