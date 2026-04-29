from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import ToolPaths
from .errors import ToolMissingError


def bundled_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundled_runtime_env() -> dict[str, str]:
    root = bundled_root()
    env = os.environ.copy()
    python_runtime = root / "python-runtime"
    path_entries = [
        python_runtime,
        python_runtime / "DLLs",
        root / "libs" / "ffmpeg",
        root / "libs" / "vapoursynth",
        root / "libs" / "vapoursynth" / "core",
    ]
    python_entries = [
        python_runtime / "Lib",
        root / "python" / "site-packages",
    ]

    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([str(path) for path in path_entries if path.exists()] + [existing_path])

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in python_entries if path.exists()] + ([existing_pythonpath] if existing_pythonpath else [])
    )
    if python_runtime.exists():
        env.setdefault("PYTHONHOME", str(python_runtime))

    return env


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
    root = bundled_root()
    ffmpeg_dir = root / "libs" / "ffmpeg"
    vapoursynth_dir = root / "libs" / "vapoursynth"

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
            [
                vapoursynth_dir / "vspipe.exe",
                root / "venv" / "Scripts" / "vspipe.exe",
                root / "venv" / "bin" / "vspipe",
            ],
            ["vspipe.exe", "vspipe"],
        ),
    )
