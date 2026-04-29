from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import EncodeOptions, PipelineConfig, RifeOptions
from .errors import VidSmootherError
from .pipeline import process_all
from .tools import default_tool_paths, repo_root


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        prog="vidsmoother",
        description="Interpolate videos to 2x frame rate with FFmpeg and RIFE ncnn Vulkan.",
    )
    parser.add_argument("--input-dir", type=Path, default=root / "input")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--work-dir", type=Path, default=root / "output" / "_work")
    parser.add_argument("--recursive", action="store_true", help="Process videos in nested input folders.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--keep-work", action="store_true", help="Keep extracted and interpolated frames.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--workers", type=int, default=1, help="Number of videos to process in parallel.")

    parser.add_argument("--ffmpeg", help="Path or command name for ffmpeg.")
    parser.add_argument("--ffprobe", help="Path or command name for ffprobe.")
    parser.add_argument("--rife", help="Path or command name for rife-ncnn-vulkan.")
    parser.add_argument("--rife-model", help="Path to a RIFE model directory.")

    parser.add_argument(
        "--scene-mode",
        choices=["auto", "none", "pyscenedetect"],
        default="auto",
        help="Scene handling. auto uses PySceneDetect when installed, otherwise whole-video processing.",
    )
    parser.add_argument("--scene-threshold", type=float, default=30.0)

    parser.add_argument("--gpu", default="0", help="RIFE GPU id. Use -1 for CPU or omit with --gpu auto.")
    parser.add_argument("--rife-threads", default="2:4:2", help="RIFE load:proc:save thread counts.")
    parser.add_argument("--tta-spatial", action="store_true", help="Enable RIFE spatial TTA.")
    parser.add_argument("--tta-temporal", action="store_true", help="Enable RIFE temporal TTA.")
    parser.add_argument("--uhd", action="store_true", help="Enable RIFE UHD mode.")

    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--audio-codec", default="copy")
    parser.add_argument("--pix-fmt", default="yuv420p")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--video-bitrate", help="Optional video bitrate, e.g. 10M. Usually leave unset with CRF.")
    parser.add_argument(
        "--subtitle-mode",
        choices=["none", "burn"],
        default="none",
        help="Burn same-stem subtitle files into the output when set to burn.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    gpu = None if args.gpu == "auto" else args.gpu
    tools = default_tool_paths(
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        rife=args.rife,
        rife_model=args.rife_model,
    )
    return PipelineConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        tools=tools,
        rife=RifeOptions(
            gpu=gpu,
            threads=args.rife_threads,
            tta_spatial=args.tta_spatial,
            tta_temporal=args.tta_temporal,
            uhd=args.uhd,
        ),
        encode=EncodeOptions(
            video_codec=args.video_codec,
            audio_codec=args.audio_codec,
            pix_fmt=args.pix_fmt,
            crf=args.crf,
            preset=args.preset,
            video_bitrate=args.video_bitrate,
        ),
        scene_mode=args.scene_mode,
        scene_threshold=args.scene_threshold,
        subtitle_mode=args.subtitle_mode,
        overwrite=args.overwrite,
        keep_work=args.keep_work,
        dry_run=args.dry_run,
        recursive=args.recursive,
        workers=max(1, args.workers),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        if config.dry_run:
            os.environ["PYTHONUNBUFFERED"] = "1"
        process_all(config)
    except VidSmootherError as error:
        parser.exit(2, f"error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "interrupted\n")
    return 0
