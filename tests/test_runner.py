from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from vidsmoother.errors import CommandError
from vidsmoother.runner import run_vspipe_to_ffmpeg


class _FakeStdout:
    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, returncode: int, stdout: _FakeStdout | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout

    def wait(self) -> int:
        return self.returncode


class RunnerTests(unittest.TestCase):
    def test_vspipe_failure_is_reported_when_both_pipe_processes_fail(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            vspipe = _FakeProcess(1, _FakeStdout())
            ffmpeg = _FakeProcess(1)

            def fake_popen(command, **kwargs):
                if kwargs.get("stdout") == subprocess.PIPE:
                    return vspipe
                return ffmpeg

            with patch("vidsmoother.runner.subprocess.Popen", side_effect=fake_popen):
                with self.assertRaisesRegex(CommandError, "vspipe failed"):
                    run_vspipe_to_ffmpeg(
                        ["vspipe", "script.vpy", "-"],
                        ["ffmpeg", "-i", "pipe:0", "out.mp4"],
                        vspipe_log=root / "vspipe.log",
                        ffmpeg_log=root / "ffmpeg.log",
                    )


if __name__ == "__main__":
    unittest.main()
