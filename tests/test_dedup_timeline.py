from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from vidsmoother.config import (
    DedupOptions,
    GifOptions,
    NvencOptions,
    PipelineConfig,
    RifeTensorRtOptions,
    ToolPaths,
    VapourSynthOptions,
)
from vidsmoother.dedup_timeline import (
    TimelineFrame,
    TransitionRenderJob,
    build_timeline_encode_command,
    parse_showinfo_pts,
    render_transition_jobs,
    render_factor_for_duration,
    render_transition_frames,
    slot_count_for_duration,
    transition_frame_indices,
    write_timeline_manifest,
)
from vidsmoother.media import VideoInfo
from vidsmoother.pipeline import build_ffmpeg_command
from vidsmoother.vpy import render_script


def config(*, dedup_strength: float = 50.0) -> PipelineConfig:
    return PipelineConfig(
        input_dir=Path("input"),
        output_dir=Path("output"),
        work_dir=Path("work"),
        tools=ToolPaths(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"), vspipe=Path("vspipe")),
        vapoursynth=VapourSynthOptions(
            source_filter="lsmas",
            matrix="709",
            fp16=True,
            max_cache_size_mb=None,
        ),
        rife=RifeTensorRtOptions(
            model="4.26",
            device_index=0,
            factor_num=2,
            factor_den=1,
            scale=1.0,
            ensemble=False,
            scene_change=True,
            scene_threshold=0.15,
            auto_download=True,
            trt_static_shape=True,
            trt_cache_dir=Path("cache"),
            trt_workspace_size=0,
            trt_optimization_level=None,
            trt_max_aux_streams=None,
            trt_min_shape=(128, 128),
            trt_opt_shape=(1920, 1080),
            trt_max_shape=(1920, 1080),
        ),
        nvenc=NvencOptions(
            codec="h264_nvenc",
            preset="p7",
            rate_control="vbr",
            cq=18,
            qp=None,
            bitrate=None,
            maxrate=None,
            bufsize=None,
            pix_fmt="auto",
            audio_codec="copy",
        ),
        gif=GifOptions(
            max_fps=50.0,
            max_width=720,
            timeline_smoothing=True,
            hard_hold_percentile=85.0,
        ),
        dedup=DedupOptions(strength=dedup_strength, algorithm="mpdecimate"),
        subtitle_mode="none",
        overwrite=False,
        dry_run=False,
        recursive=False,
        workers=1,
    )


def video_info(*, has_audio: bool = True) -> VideoInfo:
    return VideoInfo(
        path=Path("input.mp4"),
        width=1920,
        height=1080,
        fps=24.0,
        duration=10.0,
        codec="h264",
        pix_fmt="yuv420p",
        has_audio=has_audio,
    )


class DedupTimelineTests(unittest.TestCase):
    def test_parse_showinfo_pts_uses_kept_source_timestamps(self) -> None:
        log_text = """
[Parsed_showinfo_1 @ 000001] n:   0 pts:      0 pts_time:0 pos:0
[Parsed_showinfo_1 @ 000001] n:   1 pts:   3003 pts_time:0.125125 pos:42
[Parsed_showinfo_1 @ 000001] n:   2 pts:   4004 pts_time:0.166833 pos:84
"""

        self.assertEqual(parse_showinfo_pts(log_text), [0.0, 0.125125, 0.166833])

    def test_slot_count_scales_with_removed_source_gap(self) -> None:
        self.assertEqual(slot_count_for_duration(3 / 24.0, 48.0), 6)

    def test_transition_indices_follow_fractional_factor_timing_grid(self) -> None:
        cfg = config()
        cfg = PipelineConfig(
            **{
                **cfg.__dict__,
                "rife": RifeTensorRtOptions(
                    **{
                        **cfg.rife.__dict__,
                        "factor_num": 3,
                        "factor_den": 2,
                    }
                ),
            }
        )
        render_factor = render_factor_for_duration(1 / 24.0, 24.0, cfg)

        first_interval = transition_frame_indices(0.0, 1 / 24.0, 0.0, 36.0, render_factor)
        second_interval = transition_frame_indices(1 / 24.0, 2 / 24.0, 0.0, 36.0, render_factor)

        self.assertEqual(render_factor, 3)
        self.assertEqual(first_interval, [0, 2])
        self.assertEqual(second_interval, [1])

    def test_timeline_manifest_preserves_per_frame_durations(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "a.png"
            second = root / "b.png"
            first.touch()
            second.touch()
            manifest = root / "timeline.ffconcat"

            write_timeline_manifest(
                manifest,
                [
                    TimelineFrame(first, 0.020833333),
                    TimelineFrame(second, 0.0625),
                ],
            )

            text = manifest.read_text(encoding="utf-8")
            self.assertIn("duration 0.020833333", text)
            self.assertIn("duration 0.062500000", text)
            self.assertTrue(text.rstrip().endswith("b.png'"))

    def test_timeline_encode_command_has_no_post_dedup_filter(self) -> None:
        command = build_timeline_encode_command(
            Path("timeline.ffconcat"),
            video_info(),
            Path("output.mp4"),
            config(),
        )
        command_text = " ".join(str(part) for part in command)

        self.assertNotIn("mpdecimate", command_text)
        self.assertIn("-fps_mode:v vfr", command_text)
        self.assertIn("-map 1:a:0?", command_text)

    def test_transition_render_limits_vspipe_clip_to_requested_slots(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frame = root / "a.png"
            next_frame = root / "b.png"
            frame.write_bytes(b"a")
            next_frame.write_bytes(b"b")

            def fake_pipe(vspipe_command, ffmpeg_command, **kwargs) -> None:
                pattern = Path(ffmpeg_command[-1])
                for index in range(1, 5):
                    (pattern.parent / f"transition_{index:06d}.png").write_bytes(b"png")

            with (
                patch("vidsmoother.dedup_timeline.run_command"),
                patch("vidsmoother.dedup_timeline.run_vspipe_to_ffmpeg", side_effect=fake_pipe),
                patch("vidsmoother.dedup_timeline.write_vapoursynth_script") as write_script,
            ):
                rendered = render_transition_frames(
                    frame,
                    next_frame,
                    4,
                    video_info(has_audio=False),
                    root / "transitions",
                    root / "logs",
                    config(),
                )

            self.assertEqual(len(rendered), 4)
            self.assertEqual(write_script.call_args.kwargs["frame_limit"], 4)

    def test_parallel_transition_render_prewarms_uncached_transition(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            frames = [root / f"{name}.png" for name in ("a", "b", "c")]
            for index, frame in enumerate(frames):
                frame.write_bytes(f"frame-{index}".encode("ascii"))

            cfg = config()
            cfg = PipelineConfig(**{**cfg.__dict__, "workers": 2})
            calls: list[str] = []

            def fake_pipe(vspipe_command, ffmpeg_command, **kwargs) -> None:
                calls.append(threading.current_thread().name)
                pattern = Path(ffmpeg_command[-1])
                slot_count = int(ffmpeg_command[ffmpeg_command.index("-frames:v") + 1])
                for index in range(1, slot_count + 1):
                    (pattern.parent / f"transition_{index:06d}.png").write_bytes(b"png")

            jobs = [
                TransitionRenderJob(0, frames[0], frames[1], 4, (0, 1, 2, 3)),
                TransitionRenderJob(1, frames[1], frames[2], 4, (0, 1, 2, 3)),
            ]

            with (
                patch("vidsmoother.dedup_timeline.run_command"),
                patch("vidsmoother.dedup_timeline.run_vspipe_to_ffmpeg", side_effect=fake_pipe),
                patch("vidsmoother.dedup_timeline.write_vapoursynth_script"),
            ):
                rendered = render_transition_jobs(
                    jobs,
                    video_info(has_audio=False),
                    root / "transitions",
                    root / "logs",
                    cfg,
                )

            self.assertEqual(set(rendered), {0, 1})
            self.assertEqual(calls[0], "MainThread")

    def test_vapoursynth_script_applies_frame_limit_before_output_format(self) -> None:
        script = render_script(Path("pair.mkv"), video_info(has_audio=False), config(), frame_limit=4)

        self.assertIn(
            "clip = clip[:4]\nclip = core.resize.Bicubic(clip, format=vs.YUV420P8",
            script,
        )

    def test_regular_ffmpeg_command_has_no_post_dedup_filter(self) -> None:
        command = build_ffmpeg_command(video_info(), Path("output.mp4"), config())
        command_text = " ".join(str(part) for part in command)

        self.assertNotIn("mpdecimate", command_text)
        self.assertNotIn("-vf", command_text)

    def test_legacy_gif_command_has_no_post_dedup_filter(self) -> None:
        gif_info = VideoInfo(
            path=Path("input.gif"),
            width=320,
            height=240,
            fps=10.0,
            duration=1.0,
            codec="gif",
            pix_fmt="bgra",
            has_audio=False,
        )

        command = build_ffmpeg_command(gif_info, Path("output.gif"), config())
        command_text = " ".join(str(part) for part in command)

        self.assertIn("palettegen", command_text)
        self.assertNotIn("mpdecimate", command_text)


if __name__ == "__main__":
    unittest.main()
