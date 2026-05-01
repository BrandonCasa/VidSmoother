from __future__ import annotations

import os
import shutil
import sys
import threading
import tomllib
from pathlib import Path

from .config import ToolPaths
from .errors import ToolMissingError


_vapoursynth_config_lock = threading.Lock()
_vapoursynth_configured = False


def bundled_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _python_symbol_path(python_runtime: Path) -> Path | None:
    versioned = python_runtime / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    if versioned.exists():
        return versioned

    for candidate in sorted(python_runtime.glob("python*.dll")):
        if candidate.name.lower() != "python3.dll":
            return candidate

    stable_abi = python_runtime / "python3.dll"
    if stable_abi.exists():
        return stable_abi

    return None


def _ensure_vapoursynth_config(root: Path) -> None:
    global _vapoursynth_configured

    if sys.platform != "win32":
        return

    if _vapoursynth_configured:
        return

    with _vapoursynth_config_lock:
        if _vapoursynth_configured:
            return

        vapoursynth_dir = root / "libs" / "vapoursynth"
        vsscript = vapoursynth_dir / "vsscript.dll"
        python_runtime = root / "python-runtime"
        python_exe = python_runtime / "python.exe"
        python_symbol = _python_symbol_path(python_runtime)

        if not (vsscript.exists() and python_exe.exists() and python_symbol and python_symbol.exists()):
            return

        appdata = os.environ.get("APPDATA")
        if appdata:
            config_dir = Path(appdata) / "vapoursynth"
        else:
            config_dir = Path.home() / "AppData" / "Roaming" / "vapoursynth"

        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "vapoursynth.toml"
        contents: dict[str, object] = {}

        if config_path.exists():
            try:
                with config_path.open("rb") as config_file:
                    contents = tomllib.load(config_file)
            except tomllib.TOMLDecodeError:
                contents = {}

        key = str(vsscript.resolve()).lower()
        value = [str(python_exe.resolve()), str(python_symbol.resolve())]

        if contents.get(key) == value:
            _vapoursynth_configured = True
            return

        contents[key] = value
        lines = []
        for entry_key, entry_value in contents.items():
            if isinstance(entry_value, list) and len(entry_value) == 2:
                lines.append(
                    f"{_toml_string(str(entry_key))} = "
                    f"[{_toml_string(str(entry_value[0]))},{_toml_string(str(entry_value[1]))}]"
                )

        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _vapoursynth_configured = True


def _python_package_dll_dirs(root: Path) -> list[Path]:
    site_packages = root / "python" / "site-packages"
    candidates = [site_packages / "torch" / "lib"]

    nvidia_root = site_packages / "nvidia"
    if nvidia_root.exists():
        for package_dir in sorted(nvidia_root.iterdir()):
            if package_dir.is_dir():
                candidates.extend(
                    [
                        package_dir / "bin",
                        package_dir / "lib",
                        package_dir / "lib" / "x64",
                    ]
                )

    return [path for path in candidates if path.exists()]


def bundled_runtime_env() -> dict[str, str]:
    root = bundled_root()
    _ensure_vapoursynth_config(root)

    env = os.environ.copy()
    python_runtime = root / "python-runtime"
    vapoursynth_plugins = root / "libs" / "vapoursynth" / "vs-plugins"
    path_entries = [
        python_runtime,
        python_runtime / "DLLs",
        root / "libs" / "ffmpeg",
        root / "libs" / "vapoursynth",
        root / "libs" / "vapoursynth" / "core",
        *_python_package_dll_dirs(root),
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

    if vapoursynth_plugins.exists():
        env["VAPOURSYNTH_EXTRA_PLUGIN_PATH"] = str(vapoursynth_plugins)

    return env


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def locate_executable(value: str | None, bundled: list[Path], names: list[str]) -> Path | None:
    if value:
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
        found = shutil.which(value)
        if found:
            return Path(found).resolve()
        return None

    bundled_match = _first_existing(bundled)
    if bundled_match:
        return bundled_match.resolve()

    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()

    return None


def resolve_executable(value: str | None, bundled: list[Path], names: list[str]) -> Path:
    located = locate_executable(value, bundled, names)
    if located:
        return located
    if value:
        raise ToolMissingError(f"Executable not found: {value}")
    raise ToolMissingError(f"Could not locate executable. Tried: {', '.join(names)}")


def ffmpeg_search_dirs(root: Path) -> list[Path]:
    return [
        root / "libs" / "ffmpeg",
        root / "ffmpeg" / "bin",
        root / "ffmpeg",
        root / "bin",
    ]


def _ffmpeg_candidates(directories: list[Path], executable: str) -> list[Path]:
    return [directory / f"{executable}.exe" for directory in directories] + [
        directory / executable for directory in directories
    ]


def default_ffmpeg_paths(*, ffmpeg: str | None = None, ffprobe: str | None = None) -> tuple[Path | None, Path | None]:
    root = bundled_root()
    ffmpeg_dirs = ffmpeg_search_dirs(root)
    return (
        locate_executable(
            ffmpeg,
            _ffmpeg_candidates(ffmpeg_dirs, "ffmpeg"),
            ["ffmpeg.exe", "ffmpeg"],
        ),
        locate_executable(
            ffprobe,
            _ffmpeg_candidates(ffmpeg_dirs, "ffprobe"),
            ["ffprobe.exe", "ffprobe"],
        ),
    )


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
    ffmpeg_dirs = ffmpeg_search_dirs(root)
    vapoursynth_dir = root / "libs" / "vapoursynth"

    return ToolPaths(
        ffmpeg=resolve_executable(
            ffmpeg,
            _ffmpeg_candidates(ffmpeg_dirs, "ffmpeg"),
            ["ffmpeg.exe", "ffmpeg"],
        ),
        ffprobe=resolve_executable(
            ffprobe,
            _ffmpeg_candidates(ffmpeg_dirs, "ffprobe"),
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
