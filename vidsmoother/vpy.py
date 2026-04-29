from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .media import VideoInfo


def write_vapoursynth_script(video: Path, info: VideoInfo, script_path: Path, config: PipelineConfig) -> None:
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(render_script(video, info, config), encoding="utf-8")


def render_script(video: Path, info: VideoInfo, config: PipelineConfig) -> str:
    source = _source_expression(video, config.vapoursynth.source_filter)
    output_pix_fmt = info.pix_fmt if config.nvenc.pix_fmt == "auto" else config.nvenc.pix_fmt
    output_format = _output_format(output_pix_fmt)
    precision_format = "vs.RGBH" if config.vapoursynth.fp16 else "vs.RGBS"
    max_cache = (
        f"core.max_cache_size = {config.vapoursynth.max_cache_size_mb}\n"
        if config.vapoursynth.max_cache_size_mb
        else ""
    )
    scene_threshold = "None" if config.rife.scene_threshold is None else repr(config.rife.scene_threshold)
    optimization_level = (
        "None" if config.rife.trt_optimization_level is None else str(config.rife.trt_optimization_level)
    )
    max_aux_streams = "None" if config.rife.trt_max_aux_streams is None else str(config.rife.trt_max_aux_streams)

    return f'''from pathlib import Path

import vapoursynth as vs
from vsrife import rife

core = vs.core
{max_cache}
clip = {source}
clip = core.resize.Bicubic(clip, format={precision_format}, matrix_in_s={config.vapoursynth.matrix!r})
clip = rife(
    clip,
    device_index={config.rife.device_index},
    model={config.rife.model!r},
    auto_download={config.rife.auto_download},
    factor_num={config.rife.factor_num},
    factor_den={config.rife.factor_den},
    scale={config.rife.scale},
    ensemble={config.rife.ensemble},
    sc={config.rife.scene_change},
    sc_threshold={scene_threshold},
    trt=True,
    trt_static_shape={config.rife.trt_static_shape},
    trt_min_shape={list(config.rife.trt_min_shape)!r},
    trt_opt_shape={list(config.rife.trt_opt_shape)!r},
    trt_max_shape={list(config.rife.trt_max_shape)!r},
    trt_workspace_size={config.rife.trt_workspace_size},
    trt_max_aux_streams={max_aux_streams},
    trt_optimization_level={optimization_level},
    trt_cache_dir={str(config.rife.trt_cache_dir)!r},
)
clip = core.resize.Bicubic(clip, format={output_format}, matrix_s={config.vapoursynth.matrix!r})
clip.set_output()
'''


def _source_expression(video: Path, source_filter: str) -> str:
    path = str(video)
    match source_filter:
        case "lsmas":
            return f"core.lsmas.LWLibavSource(source={path!r}, cache=0)"
        case "ffms2":
            return f"core.ffms2.Source(source={path!r})"
        case "bestsource":
            return f"core.bs.VideoSource(source={path!r})"
        case _:
            raise ValueError(f"Unsupported source filter: {source_filter}")


def _output_format(pix_fmt: str) -> str:
    match pix_fmt:
        case "yuv420p" | "yuvj420p":
            return "vs.YUV420P8"
        case "yuv420p10le":
            return "vs.YUV420P10"
        case "yuv422p" | "yuvj422p":
            return "vs.YUV422P8"
        case "yuv422p10le":
            return "vs.YUV422P10"
        case "yuv444p" | "yuvj444p":
            return "vs.YUV444P8"
        case "yuv444p10le":
            return "vs.YUV444P10"
        case _:
            return "vs.YUV420P10"
