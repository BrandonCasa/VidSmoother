from __future__ import annotations

from pathlib import Path


SUBTITLE_EXTENSIONS = [".srt", ".ass", ".ssa", ".vtt"]


def matching_subtitle(video: Path) -> Path | None:
    for extension in SUBTITLE_EXTENSIONS:
        candidate = video.with_suffix(extension)
        if candidate.exists():
            return candidate
    return None


def escape_subtitle_path(path: Path) -> str:
    text = path.as_posix()
    text = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"subtitles='{text}'"
