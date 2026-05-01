from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import (
    DedupOptions,
    GifOptions,
    NvencOptions,
    PipelineConfig,
    RifeTensorRtOptions,
    VapourSynthOptions,
)
from .errors import VidSmootherError
from .pipeline import process_all
from .tools import default_tool_paths, repo_root


def build_parser() -> argparse.ArgumentParser:
    root = repo_root()
    parser = argparse.ArgumentParser(
        prog="vidsmoother",
        description="Interpolate videos with VapourSynth, RIFE TensorRT, vspipe, and FFmpeg NVENC.",
    )
    parser.add_argument("--input-dir", type=Path, default=root / "input")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--work-dir", type=Path, default=root / "output" / "_work")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process videos in nested input folders.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate scripts and print commands without running.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of videos to process in parallel.",
    )

    parser.add_argument("--ffmpeg", help="Path or command name for ffmpeg.")
    parser.add_argument("--ffprobe", help="Path or command name for ffprobe.")
    parser.add_argument("--vspipe", help="Path or command name for vspipe.")

    parser.add_argument(
        "--source-filter",
        choices=["lsmas", "ffms2", "bestsource"],
        default="lsmas",
        help="VapourSynth source plugin used in generated .vpy scripts.",
    )
    parser.add_argument(
        "--matrix",
        default="709",
        help="VapourSynth resize matrix, usually 709 for HD video.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use RGBS/FP32 instead of RGBH/FP16 for RIFE input.",
    )
    parser.add_argument(
        "--vs-cache-mb", type=int, help="VapourSynth core.max_cache_size in MB."
    )

    parser.add_argument("--rife-model", default="4.26", help="vs-rife model version.")
    parser.add_argument(
        "--device-index", type=int, default=0, help="CUDA device index for vs-rife."
    )
    parser.add_argument(
        "--factor-num", type=int, default=2, help="Frame-rate multiplier numerator."
    )
    parser.add_argument(
        "--factor-den", type=int, default=1, help="Frame-rate multiplier denominator."
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="RIFE process scale: 0.25, 0.5, 1.0, 2.0, or 4.0.",
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Enable RIFE ensemble mode."
    )
    parser.add_argument(
        "--no-scene-change",
        action="store_true",
        help="Disable scene-change guarded interpolation.",
    )
    parser.add_argument("--scene-threshold", type=float, default=0.15)
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable vs-rife model auto-download.",
    )

    parser.add_argument(
        "--trt-dynamic-shape",
        action="store_true",
        help="Build dynamic-shape TensorRT engines.",
    )
    parser.add_argument(
        "--trt-cache-dir", type=Path, default=root / "output" / "_trt_cache"
    )
    parser.add_argument("--trt-workspace-size", type=int, default=0)
    parser.add_argument("--trt-optimization-level", type=int)
    parser.add_argument("--trt-max-aux-streams", type=int)
    parser.add_argument("--trt-min-shape", default="128x128")
    parser.add_argument("--trt-opt-shape", default="1920x1080")
    parser.add_argument("--trt-max-shape", default="1920x1080")

    parser.add_argument(
        "--nvenc-codec",
        choices=["auto", "hevc_nvenc", "h264_nvenc", "av1_nvenc"],
        default="auto",
        help="Output video encoder. auto follows the input codec and uses NVENC when available.",
    )
    parser.add_argument("--nvenc-preset", default="p7")
    parser.add_argument("--nvenc-rc", default="vbr")
    parser.add_argument("--cq", type=int, default=18)
    parser.add_argument("--qp", type=int)
    parser.add_argument("--bitrate")
    parser.add_argument("--maxrate")
    parser.add_argument("--bufsize")
    parser.add_argument(
        "--pix-fmt",
        default="auto",
        help="Output pixel format. auto follows the input pixel format.",
    )
    parser.add_argument("--audio-codec", default="copy")
    parser.add_argument(
        "--gif-max-fps",
        type=float,
        default=50.0,
        help="Maximum frame rate for GIF output. Use 0 to keep the full interpolated frame rate.",
    )
    parser.add_argument(
        "--gif-max-width",
        type=int,
        default=720,
        help="Maximum GIF output width. Use 0 to keep the interpolated frame size.",
    )
    parser.add_argument(
        "--dedup-strength",
        type=float,
        default=0.0,
        help="Timestamp-preserving duplicate frame removal strength from 0 to 100. 0 disables deduplication.",
    )
    parser.add_argument(
        "--dedup-algorithm",
        choices=["mpdecimate", "cuda-mpdecimate"],
        default="cuda-mpdecimate",
        help="Duplicate detection algorithm. cuda-mpdecimate uses CUDA filtering before FFmpeg duplicate scoring.",
    )
    parser.add_argument(
        "--subtitle-mode",
        choices=["none"],
        default="none",
        help="Subtitles are not burned in the vspipe/NVENC path yet.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    tools = default_tool_paths(
        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe, vspipe=args.vspipe
    )
    return PipelineConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        tools=tools,
        vapoursynth=VapourSynthOptions(
            source_filter=args.source_filter,
            matrix=args.matrix,
            fp16=not args.fp32,
            max_cache_size_mb=args.vs_cache_mb,
        ),
        rife=RifeTensorRtOptions(
            model=args.rife_model,
            device_index=args.device_index,
            factor_num=args.factor_num,
            factor_den=args.factor_den,
            scale=args.scale,
            ensemble=args.ensemble,
            scene_change=not args.no_scene_change,
            scene_threshold=args.scene_threshold,
            auto_download=not args.no_auto_download,
            trt_static_shape=not args.trt_dynamic_shape,
            trt_cache_dir=args.trt_cache_dir.resolve(),
            trt_workspace_size=args.trt_workspace_size,
            trt_optimization_level=args.trt_optimization_level,
            trt_max_aux_streams=args.trt_max_aux_streams,
            trt_min_shape=parse_shape(args.trt_min_shape),
            trt_opt_shape=parse_shape(args.trt_opt_shape),
            trt_max_shape=parse_shape(args.trt_max_shape),
        ),
        nvenc=NvencOptions(
            codec=None if args.nvenc_codec == "auto" else args.nvenc_codec,
            preset=args.nvenc_preset,
            rate_control=args.nvenc_rc,
            cq=args.cq,
            qp=args.qp,
            bitrate=args.bitrate,
            maxrate=args.maxrate,
            bufsize=args.bufsize,
            pix_fmt=args.pix_fmt,
            audio_codec=args.audio_codec,
        ),
        gif=GifOptions(
            max_fps=(
                args.gif_max_fps if args.gif_max_fps and args.gif_max_fps > 0 else None
            ),
            max_width=(
                args.gif_max_width
                if args.gif_max_width and args.gif_max_width > 0
                else None
            ),
        ),
        dedup=DedupOptions(
            strength=max(0.0, min(100.0, args.dedup_strength)),
            algorithm=args.dedup_algorithm,
        ),
        subtitle_mode=args.subtitle_mode,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        recursive=args.recursive,
        workers=max(1, args.workers),
    )


def parse_shape(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", maxsplit=1)
        return int(width), int(height)
    except ValueError as exc:
        raise VidSmootherError(
            f"shape must look like WIDTHxHEIGHT, got {value!r}"
        ) from exc


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
