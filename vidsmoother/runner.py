from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, TextIO

from .errors import CommandError
from .tools import bundled_runtime_env


def format_command(command: Iterable[object]) -> str:
    return " ".join(str(part) for part in command)


def run_command(
    command: list[object],
    *,
    log_file: Path | None = None,
    dry_run: bool = False,
    stdout: TextIO | int | None = None,
    stderr: TextIO | int | None = None,
) -> subprocess.CompletedProcess[str] | None:
    printable = format_command(command)
    if dry_run:
        print(f"[dry-run] {printable}")
        return None

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(part) for part in command],
                stdout=log,
                stderr=log,
                text=True,
                env=bundled_runtime_env(),
            )
    else:
        result = subprocess.run(
            [str(part) for part in command],
            stdout=stdout,
            stderr=stderr,
            text=True,
            env=bundled_runtime_env(),
        )

    if result.returncode != 0:
        detail = f" See log: {log_file}" if log_file else ""
        raise CommandError(f"Command failed with exit code {result.returncode}: {printable}.{detail}")
    return result


def run_vspipe_to_ffmpeg(
    vspipe_command: list[object],
    ffmpeg_command: list[object],
    *,
    vspipe_log: Path,
    ffmpeg_log: Path,
    dry_run: bool = False,
) -> None:
    if dry_run:
        print(f"[dry-run] {format_command(vspipe_command)} | {format_command(ffmpeg_command)}")
        return

    vspipe_log.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_log.parent.mkdir(parents=True, exist_ok=True)

    with vspipe_log.open("w", encoding="utf-8") as vp_log, ffmpeg_log.open("w", encoding="utf-8") as ff_log:
        vspipe = subprocess.Popen(
            [str(part) for part in vspipe_command],
            stdout=subprocess.PIPE,
            stderr=vp_log,
            env=bundled_runtime_env(),
        )
        ffmpeg = subprocess.Popen(
            [str(part) for part in ffmpeg_command],
            stdin=vspipe.stdout,
            stdout=ff_log,
            stderr=ff_log,
            env=bundled_runtime_env(),
        )
        if vspipe.stdout is not None:
            vspipe.stdout.close()

        ffmpeg_return = ffmpeg.wait()
        vspipe_return = vspipe.wait()

    if vspipe_return != 0:
        raise CommandError(f"vspipe failed with exit code {vspipe_return}. See log: {vspipe_log}")
    if ffmpeg_return != 0:
        raise CommandError(f"ffmpeg failed with exit code {ffmpeg_return}. See log: {ffmpeg_log}")
