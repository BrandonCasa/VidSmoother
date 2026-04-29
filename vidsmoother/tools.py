from __future__ import annotations

import shutil
from pathlib import Path

from .config import ToolPaths
from .errors import ToolMissingError


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_executable(value: str | None, bundled: list[Path], names: list[str]) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
        raise ToolMissingError(f"Executable not found: {value}")

    bundled_match = _first_existing(bundled)
    if bundled_match:
        return bundled_match.resolve()

    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    raise ToolMissingError(f"Could not locate executable. Tried: {', '.join(names)}")


def resolve_optional_path(value: str | None, bundled: list[Path]) -> Path | None:
    if value:
        path = Path(value).expanduser()
        if not path.exists():
            raise ToolMissingError(f"Path not found: {value}")
        return path.resolve()
    return _first_existing(bundled)


def default_tool_paths(
    *,
    ffmpeg: str | None,
    ffprobe: str | None,
    vspipe: str | None,
) -> ToolPaths:
    root = repo_root()
    ffmpeg_dir = root / "libs" / "ffmpeg"

    return ToolPaths(
        ffmpeg=resolve_executable(
            ffmpeg,
            [ffmpeg_dir / "ffmpeg.exe", ffmpeg_dir / "ffmpeg"],
            ["ffmpeg.exe", "ffmpeg"],
        ),
        ffprobe=resolve_executable(
            ffprobe,
            [ffmpeg_dir / "ffprobe.exe", ffmpeg_dir / "ffprobe"],
            ["ffprobe.exe", "ffprobe"],
        ),
        vspipe=resolve_executable(
            vspipe,
            [root / "venv" / "Scripts" / "vspipe.exe", root / "venv" / "bin" / "vspipe"],
            ["vspipe.exe", "vspipe"],
        ),
    )
