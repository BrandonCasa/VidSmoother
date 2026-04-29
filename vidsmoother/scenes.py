from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .media import VideoInfo


@dataclass(frozen=True)
class Scene:
    index: int
    start: float
    end: float


def detect_scenes(video: Path, info: VideoInfo, *, mode: str, threshold: float) -> list[Scene]:
    if mode == "none":
        return [Scene(index=1, start=0.0, end=info.duration)]

    if mode in {"auto", "pyscenedetect"}:
        try:
            return _detect_with_pyscenedetect(video, info, threshold)
        except ImportError:
            if mode == "pyscenedetect":
                raise

    return [Scene(index=1, start=0.0, end=info.duration)]


def _detect_with_pyscenedetect(video: Path, info: VideoInfo, threshold: float) -> list[Scene]:
    from scenedetect import ContentDetector, detect

    detected = detect(str(video), ContentDetector(threshold=threshold))
    if not detected:
        return [Scene(index=1, start=0.0, end=info.duration)]

    scenes: list[Scene] = []
    for index, (start_time, end_time) in enumerate(detected, start=1):
        start = max(0.0, start_time.get_seconds())
        end = min(info.duration, end_time.get_seconds())
        if end > start:
            scenes.append(Scene(index=index, start=start, end=end))
    return scenes or [Scene(index=1, start=0.0, end=info.duration)]
