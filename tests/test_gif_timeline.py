from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
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
from vidsmoother.gif_timeline import (
    GifTimelineFrame,
    build_timeline_gif_filter_chain,
    build_timeline_segments,
    centisecond_durations,
    coalesce_duplicate_gif_frames,
    parse_gif_frame_delays,
    render_transition_frames,
    write_concat_manifest,
)
from vidsmoother.media import VideoInfo
from vidsmoother.pipeline import build_ffmpeg_command


def config(*, dedup_strength: float = 0.0) -> PipelineConfig:
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
            codec=None,
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


class GifTimelineTests(unittest.TestCase):
    def test_parse_frame_delays_uses_explicit_duration(self) -> None:
        payload = {
            "frames": [
                {"media_type": "video", "pkt_duration_time": "0.12"},
                {"media_type": "video", "pkt_duration_time": "0.04"},
            ]
        }

        delays, used_fallback = parse_gif_frame_delays(payload, fallback_ms=20.0)

        self.assertFalse(used_fallback)
        self.assertEqual(delays, [120.0, 40.0])

    def test_parse_frame_delays_falls_back_from_timestamps(self) -> None:
        payload = {
            "frames": [
                {"media_type": "video", "best_effort_timestamp_time": "0.00"},
                {"media_type": "video", "best_effort_timestamp_time": "0.07"},
                {"media_type": "video", "best_effort_timestamp_time": "0.10"},
            ]
        }

        delays, used_fallback = parse_gif_frame_delays(payload, fallback_ms=20.0)

        self.assertTrue(used_fallback)
        self.assertEqual(delays, [70.0, 30.0, 20.0])

    def test_timeline_segments_keep_long_delays_as_hard_holds(self) -> None:
        segments = build_timeline_segments(
            [40.0, 40.0, 300.0, 40.0],
            quantum_ms=20.0,
            hard_hold_percentile=85.0,
        )

        self.assertFalse(segments[0].hard_hold)
        self.assertEqual(len(segments[0].durations), 2)
        self.assertTrue(segments[2].hard_hold)
        self.assertEqual(segments[2].durations, (0.3,))

    def test_equal_delays_are_not_treated_as_outlier_holds(self) -> None:
        segments = build_timeline_segments(
            [40.0, 40.0, 40.0, 40.0],
            quantum_ms=20.0,
            hard_hold_percentile=85.0,
        )

        self.assertFalse(any(segment.hard_hold for segment in segments))

    def test_centisecond_durations_sum_to_quantized_source_delay(self) -> None:
        durations = centisecond_durations(95.0, 4)

        self.assertEqual(len(durations), 4)
        self.assertAlmostEqual(sum(durations), 0.10)

    def test_coalesce_duplicate_gif_frames_merges_delays_before_interpolation(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "a.png"
            duplicate = root / "b.png"
            first.write_bytes(b"same")
            duplicate.write_bytes(b"same")

            frames, delays = coalesce_duplicate_gif_frames(
                [first, duplicate],
                [40.0, 40.0],
                config(dedup_strength=50.0),
            )

        self.assertEqual(frames, [first])
        self.assertEqual(delays, [80.0])

    def test_concat_manifest_uses_explicit_durations_and_last_file_repeat(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "a.png"
            second = root / "b.png"
            first.touch()
            second.touch()
            manifest = root / "frames.ffconcat"

            write_concat_manifest(
                manifest,
                [
                    GifTimelineFrame(first, 0.02),
                    GifTimelineFrame(second, 0.30),
                ],
            )

            text = manifest.read_text(encoding="utf-8")
            self.assertIn("ffconcat version 1.0", text)
            self.assertIn("duration 0.020000", text)
            self.assertIn("duration 0.300000", text)
            self.assertTrue(text.rstrip().endswith("b.png'"))

    def test_timeline_filter_applies_width_without_post_dedup(self) -> None:
        self.assertIn("scale=w='min(iw\\,720)'", build_timeline_gif_filter_chain(config()))
        self.assertNotIn("mpdecimate=", build_timeline_gif_filter_chain(config(dedup_strength=50.0)))

    def test_timeline_filter_is_valid_without_width_or_dedup(self) -> None:
        base = config()
        no_filters = PipelineConfig(
            **{
                **base.__dict__,
                "gif": GifOptions(
                    max_fps=50.0,
                    max_width=None,
                    timeline_smoothing=True,
                    hard_hold_percentile=85.0,
                ),
            }
        )

        self.assertTrue(build_timeline_gif_filter_chain(no_filters).startswith("[0:v]split=2"))

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
                patch("vidsmoother.gif_timeline.run_command"),
                patch("vidsmoother.gif_timeline.run_vspipe_to_ffmpeg", side_effect=fake_pipe),
                patch("vidsmoother.gif_timeline.write_vapoursynth_script") as write_script,
            ):
                rendered = render_transition_frames(
                    frame,
                    next_frame,
                    4,
                    VideoInfo(
                        path=Path("input.gif"),
                        width=320,
                        height=240,
                        fps=10.0,
                        duration=1.0,
                        codec="gif",
                        pix_fmt="bgra",
                        has_audio=False,
                    ),
                    root / "transitions",
                    root / "logs",
                    config(dedup_strength=50.0),
                )

            self.assertEqual(len(rendered), 4)
            self.assertEqual(write_script.call_args.kwargs["frame_limit"], 4)

    def test_legacy_gif_command_still_generates_palette_pipeline(self) -> None:
        info = VideoInfo(
            path=Path("input.gif"),
            width=320,
            height=240,
            fps=10.0,
            duration=1.0,
            codec="gif",
            pix_fmt="bgra",
            has_audio=False,
        )

        command = build_ffmpeg_command(info, Path("output.gif"), config())

        self.assertIn("-filter_complex", [str(part) for part in command])
        self.assertIn("palettegen", " ".join(str(part) for part in command))


if __name__ == "__main__":
    unittest.main()
