from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, TextIO

from .errors import CommandError


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
            )
    else:
        result = subprocess.run(
            [str(part) for part in command],
            stdout=stdout,
            stderr=stderr,
            text=True,
        )

    if result.returncode != 0:
        detail = f" See log: {log_file}" if log_file else ""
        raise CommandError(f"Command failed with exit code {result.returncode}: {printable}.{detail}")
    return result
