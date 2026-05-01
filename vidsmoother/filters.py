from __future__ import annotations

from .config import PipelineConfig


def build_dedup_filter(config: PipelineConfig) -> str | None:
    if config.dedup.strength <= 0:
        return None

    mpdecimate = build_mpdecimate_filter(config.dedup.strength)
    if config.dedup.algorithm == "cuda-mpdecimate":
        return ",".join(
            [
                "format=yuv420p",
                "hwupload_cuda",
                "scale_cuda=w=iw:h=ih:format=yuv420p",
                "hwdownload",
                "format=yuv420p",
                mpdecimate,
            ]
        )

    return mpdecimate


def build_mpdecimate_filter(strength: float) -> str:
    normalized = max(0.0, min(100.0, strength)) / 100.0
    hi = round(64 * (6 + 22 * normalized))
    lo = round(64 * (3 + 12 * normalized))
    frac = 0.08 + 0.42 * normalized
    return f"mpdecimate=max=0:hi={hi}:lo={lo}:frac={frac:.3f}"
