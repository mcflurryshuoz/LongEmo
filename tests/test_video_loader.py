"""Temporary synthetic media only; no dataset reads or network calls."""

import base64
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from evaluation.inference import video_loader


def jpeg_dimensions(raw):
    """Read JPEG SOF dimensions with the standard library, without Pillow."""
    position = 2
    while position < len(raw):
        if raw[position] != 0xFF:
            raise AssertionError("invalid JPEG marker")
        marker = raw[position + 1]
        length = int.from_bytes(raw[position + 2 : position + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", raw[position + 5 : position + 9])
            return width, height
        position += length + 2
    raise AssertionError("JPEG has no dimensions")


class FrameValidationTest(unittest.TestCase):
    def test_invalid_sampling_options_fail_before_reading_video(self):
        for name, extra_invalid_values in (
            ("max_frames", ()),
            ("max_pixels", (video_loader.VIDEO_MIN_PIXELS - 1,)),
        ):
            for value in (
                0,
                -1,
                True,
                False,
                1.5,
                "2",
                None,
                math.nan,
                math.inf,
                *extra_invalid_values,
            ):
                with (
                    self.subTest(name=name, value=value),
                    patch.object(video_loader, "file_hash") as hashed,
                ):
                    with self.assertRaisesRegex(ValueError, name):
                        video_loader.sample_video_frames({}, **{name: value})
                    hashed.assert_not_called()

    def test_invalid_fps_fails_before_reading_video(self):
        for fps in (0, -1, True, False, "2", None, math.nan, math.inf, -math.inf):
            with (
                self.subTest(fps=fps),
                patch.object(video_loader, "file_hash") as hashed,
            ):
                with self.assertRaisesRegex(ValueError, "fps"):
                    video_loader.sample_video_frames({}, fps=fps)
                hashed.assert_not_called()

    def test_changed_file_is_rejected_before_tools_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.write_bytes(b"original")
            video = {"path": str(path), "sha256": video_loader.file_hash(path)}
            path.write_bytes(b"changed")
            with patch.object(video_loader, "_run_tool") as runner:
                with self.assertRaisesRegex(ValueError, "video changed"):
                    video_loader.sample_video_frames(video)
                runner.assert_not_called()

    def test_missing_tools_have_actionable_errors(self):
        for missing in ("ffmpeg", "ffprobe"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as temporary,
            ):
                path = Path(temporary) / "input.mp4"
                path.write_bytes(b"synthetic")
                video = {"path": str(path), "sha256": video_loader.file_hash(path)}
                with patch.object(
                    video_loader.shutil,
                    "which",
                    side_effect=lambda name: None
                    if name == missing
                    else "/tool/" + name,
                ):
                    with self.assertRaisesRegex(RuntimeError, missing + ".*PATH"):
                        video_loader.sample_video_frames(video)

    def test_invalid_probe_metadata_is_rejected(self):
        valid = {"duration": "2", "width": 160, "height": 90, "avg_frame_rate": "4/1"}
        cases = [({"streams": []}, "no video stream")]
        for duration in ("0", "-1", "NaN", "inf", "N/A", None):
            cases.append(({"streams": [{**valid, "duration": duration}]}, "duration"))
        for dimension in ("width", "height"):
            cases.append(({"streams": [{**valid, dimension: 0}]}, dimension))
        cases.append(({"streams": [{**valid, "avg_frame_rate": "0/0"}]}, "frame rate"))
        for data, message in cases:
            with (
                self.subTest(data=data),
                patch.object(video_loader, "_run_tool", return_value=json.dumps(data)),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    video_loader._probe_video(
                        Path("synthetic.mp4"), "ffprobe", time.monotonic() + 10
                    )

    def test_invalid_dimension_probe_metadata_is_rejected(self):
        for name in ("width", "height"):
            for value in (0, -1, True, 28.0, "28", None):
                data = {"streams": [{"width": 160, "height": 90, name: value}]}
                with (
                    self.subTest(name=name, value=value),
                    patch.object(
                        video_loader, "_run_tool", return_value=json.dumps(data)
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, name):
                        video_loader._probe_video_dimensions(
                            Path("synthetic.mp4"), "ffprobe", time.monotonic() + 10
                        )

    def test_subprocess_timeouts_are_reported_and_bounded(self):
        with patch.object(
            video_loader.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("ffmpeg", 1),
        ) as runner:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                video_loader._run_tool(["ffmpeg"], time.monotonic() + 10)
            self.assertGreater(runner.call_args.kwargs["timeout"], 0)
            self.assertLessEqual(runner.call_args.kwargs["timeout"], 10)


class FrameSizingTest(unittest.TestCase):
    def test_qwen_omni_numeric_defaults(self):
        self.assertEqual(video_loader.FPS, 2.0)
        self.assertEqual(video_loader.MAX_FRAMES, 768)
        self.assertEqual(video_loader.IMAGE_FACTOR, 28)
        self.assertEqual(video_loader.VIDEO_MIN_PIXELS, 100352)
        self.assertEqual(video_loader.VIDEO_MAX_PIXELS, 602112)
        self.assertEqual(video_loader.VIDEO_TOTAL_PIXELS, 90316800)

    def test_total_video_budget_reduces_resolution_for_long_clips(self):
        for count, expected in (
            (1, 602112),
            (300, 602112),
            (384, 470400),
            (768, 235200),
        ):
            with self.subTest(count=count):
                budget = video_loader._frame_pixel_budget(count)
                self.assertEqual(budget, expected)
                width, height = video_loader._resize_dimensions(1920, 1080, budget)
                self.assertLessEqual(width * height, budget)
                self.assertGreaterEqual(width * height, video_loader.VIDEO_MIN_PIXELS)
                self.assertEqual(width % 28, 0)
                self.assertEqual(height % 28, 0)
        self.assertEqual(video_loader._frame_pixel_budget(768, 200704), 200704)
        self.assertEqual(video_loader._frame_pixel_budget(768, 10**400), 235200)
        self.assertEqual(video_loader._frame_pixel_budget(1, 10**400), 602112)
        self.assertEqual(video_loader._frame_pixel_budget(10000), 105369)

    def test_resizing_rounds_upscales_and_downscales_on_qwen_grid(self):
        for dimensions, expected in (
            ((1920, 1080), (1008, 560)),
            ((1080, 1920), (560, 1008)),
            ((640, 360), (644, 364)),
            ((160, 90), (448, 252)),
            ((90, 160), (252, 448)),
            ((2, 2), (336, 336)),
        ):
            with self.subTest(dimensions=dimensions):
                resized = video_loader._resize_dimensions(
                    *dimensions, video_loader.VIDEO_MAX_PIXELS
                )
                self.assertEqual(resized, expected)
                self.assertLessEqual(math.prod(resized), video_loader.VIDEO_MAX_PIXELS)
                self.assertGreaterEqual(
                    math.prod(resized), video_loader.VIDEO_MIN_PIXELS
                )

    def test_extreme_ratio_guard_and_minimum_grid_dimension(self):
        for dimensions in ((402, 2), (2, 402)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "aspect ratio"):
                    video_loader._resize_dimensions(
                        *dimensions, video_loader.VIDEO_MAX_PIXELS
                    )
        for dimensions, expected in (
            ((40000, 200), (4480, 28)),
            ((200, 40000), (28, 4480)),
        ):
            with self.subTest(dimensions=dimensions):
                self.assertEqual(
                    video_loader._resize_dimensions(
                        *dimensions, video_loader.VIDEO_MIN_PIXELS
                    ),
                    expected,
                )
        # Qwen's ceil-to-grid minimum-size branch can exceed a tight max target.
        self.assertEqual(
            video_loader._resize_dimensions(160, 90, video_loader.VIDEO_MIN_PIXELS),
            (448, 252),
        )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg and ffprobe are required",
)
class SyntheticFrameTest(unittest.TestCase):
    def make_video(self, directory, *, duration=2, size="160x90", extra=()):
        path = Path(directory) / "synthetic.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-hide_banner",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size={size}:rate=4:duration={duration}",
                *extra,
                "-c:v",
                "mpeg4",
                "-bf",
                "2",
                "-y",
                str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return {"path": str(path), "sha256": video_loader.file_hash(path)}

    def test_frames_cover_whole_clip_with_actual_timestamps_and_bounded_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, size="1280x720")
            original = Path(video["path"]).read_bytes()
            with patch.object(
                video_loader, "_run_tool", wraps=video_loader._run_tool
            ) as runner:
                blocks, metadata = video_loader.sample_video_frames(
                    video, fps=2.5, max_pixels=200704
                )
            self.assertEqual(len(blocks), 10)
            self.assertEqual(metadata["timestamps_seconds"], [0, 0.5, 1, 1.5, 1.75])
            self.assertEqual(metadata["num_frames"], 5)
            self.assertEqual(metadata["target_num_frames"], 5)
            self.assertEqual(metadata["requested_fps"], 2.5)
            self.assertEqual(metadata["source_fps"], 4)
            self.assertEqual(metadata["max_frames"], 768)
            self.assertEqual(metadata["effective_fps"], 2.5)
            self.assertEqual(
                metadata["sampling"], "fps_capped_uniform_timestamp_seeks_v2"
            )
            self.assertNotIn("fps", metadata)
            self.assertNotIn("requested_num_frames", metadata)
            self.assertEqual(metadata["duration_seconds"], 2)
            self.assertNotIn("max_edge", metadata)
            self.assertEqual(metadata["requested_max_pixels"], 200704)
            self.assertEqual(metadata["max_pixels"], 200704)
            self.assertEqual(metadata["min_pixels"], 100352)
            self.assertEqual(metadata["total_pixels"], 90316800)
            self.assertEqual(metadata["resized_width"], 588)
            self.assertEqual(metadata["resized_height"], 336)
            images = []
            for index, timestamp in enumerate(metadata["timestamps_seconds"]):
                self.assertEqual(
                    blocks[2 * index],
                    {"type": "text", "text": f"Frame at {timestamp:.6f} seconds:"},
                )
                part = blocks[2 * index + 1]
                self.assertEqual(part["type"], "image_url")
                url = part["image_url"]["url"]
                self.assertTrue(url.startswith("data:image/jpeg;base64,"))
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                self.assertEqual(jpeg_dimensions(raw), (588, 336))
                images.append(raw)
            self.assertGreater(len(set(images)), 1)
            self.assertNotIn("base64", json.dumps(metadata))
            self.assertEqual(Path(video["path"]).read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [Path(video["path"])])
            extraction_calls = [
                call
                for call in runner.call_args_list
                if Path(call.args[0][0]).name == "ffmpeg"
            ]
            self.assertEqual(len(extraction_calls), 2)  # Four seeks share a process.

    def test_default_sampling_and_pixel_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            blocks, metadata = video_loader.sample_video_frames(video)
            self.assertEqual(metadata["requested_fps"], 2)
            self.assertEqual(metadata["max_frames"], 768)
            self.assertEqual(metadata["timestamps_seconds"], [0, 0.75, 1.25, 1.75])
            self.assertEqual(metadata["requested_max_pixels"], 602112)
            self.assertEqual(metadata["max_pixels"], 602112)
            raw = base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1])
            self.assertEqual(jpeg_dimensions(raw), (448, 252))
            self.assertEqual(
                (metadata["resized_width"], metadata["resized_height"]), (448, 252)
            )

    def test_short_clips_and_one_frame_requests(self):
        for duration, fps, max_frames, expected in (
            (0.25, 128, 32, [0.0]),
            (0.5, 128, 32, [0.0, 0.25]),
            (2, 128, 1, [0.0]),
            (0.5, 128, 1024, [0.0, 0.25]),
        ):
            with (
                self.subTest(duration=duration, fps=fps, max_frames=max_frames),
                tempfile.TemporaryDirectory() as temporary,
            ):
                video = self.make_video(temporary, duration=duration)
                blocks, metadata = video_loader.sample_video_frames(
                    video, fps=fps, max_frames=max_frames
                )
                self.assertEqual(metadata["timestamps_seconds"], expected)
                self.assertEqual(len(blocks), len(expected) * 2)
                self.assertEqual(metadata["target_num_frames"], len(expected))

    def test_fps_plans_below_cap_fractional_and_capped_samples_across_whole_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            for fps, max_frames, expected in (
                (1, 32, [0.0, 1.75]),
                (4, 3, [0.0, 1.0, 1.75]),
                (0.1, 32, [0.0]),
                (0.75, 32, [0.0, 1.75]),
                (1.25, 32, [0.0, 1.75]),
                (1.75, 32, [0.0, 0.75, 1.25, 1.75]),
                (1e308, 3, [0.0, 1.0, 1.75]),
            ):
                with self.subTest(fps=fps, max_frames=max_frames):
                    blocks, metadata = video_loader.sample_video_frames(
                        video, fps=fps, max_frames=max_frames
                    )
                    self.assertEqual(metadata["timestamps_seconds"], expected)
                    self.assertEqual(len(blocks), 2 * len(expected))
                    self.assertEqual(metadata["target_num_frames"], len(expected))
                    self.assertEqual(metadata["num_frames"], len(expected))
                    self.assertEqual(metadata["effective_fps"], len(expected) / 2)
                    self.assertEqual(metadata["requested_fps"], fps)
                    self.assertEqual(metadata["max_frames"], max_frames)

    def test_source_frame_count_caps_planned_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            _, metadata = video_loader.sample_video_frames(video, fps=100, max_frames=32)
            self.assertEqual(metadata["target_num_frames"], 8)
            self.assertEqual(metadata["num_frames"], 8)
            self.assertEqual(metadata["timestamps_seconds"], [i / 4 for i in range(8)])
            self.assertEqual(metadata["effective_fps"], 4)

    def test_missing_source_frame_count_uses_source_rate_for_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            original_probe = video_loader._probe_video

            def probe_without_frame_count(*args):
                duration, source_fps, start, _ = original_probe(*args)
                return duration, source_fps, start, 0

            with patch.object(
                video_loader, "_probe_video", side_effect=probe_without_frame_count
            ):
                _, metadata = video_loader.sample_video_frames(video, fps=100, max_frames=32)
            self.assertEqual(metadata["target_num_frames"], 8)
            self.assertEqual(metadata["num_frames"], 8)

    def test_portrait_images_upscale_or_downscale_to_pixel_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            for size, max_pixels, expected in (
                ("90x160", 602112, (252, 448)),
                ("1080x1920", 602112, (560, 1008)),
                ("1080x1920", 200704, (336, 588)),
                ("1080x1920", 602112 * 2, (560, 1008)),
            ):
                with self.subTest(size=size, max_pixels=max_pixels):
                    video = self.make_video(temporary, duration=0.25, size=size)
                    blocks, metadata = video_loader.sample_video_frames(
                        video, fps=1, max_pixels=max_pixels
                    )
                    raw = base64.b64decode(
                        blocks[1]["image_url"]["url"].split(",", 1)[1]
                    )
                    self.assertEqual(jpeg_dimensions(raw), expected)
                    self.assertEqual(metadata["requested_max_pixels"], max_pixels)
                    self.assertEqual(metadata["max_pixels"], min(max_pixels, 602112))
                    self.assertEqual(
                        (metadata["resized_width"], metadata["resized_height"]),
                        expected,
                    )

    def test_nonzero_stream_start_uses_relative_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, extra=("-output_ts_offset", "5"))
            _, metadata = video_loader.sample_video_frames(video, fps=1.5)
            self.assertEqual(metadata["timestamps_seconds"], [0, 1, 1.75])

    def test_variable_rate_covers_last_decodable_frame_and_collapses_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(
                temporary,
                extra=(
                    "-vf",
                    "setpts=if(lt(N\\,3)\\,N*0.25/TB\\,(N-3)*0.75/TB+0.75/TB)",
                    "-fps_mode",
                    "vfr",
                ),
            )
            data = json.loads(
                subprocess.run(
                    [
                        shutil.which("ffprobe"),
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_frames",
                        "-show_entries",
                        "frame=pts_time",
                        "-of",
                        "json",
                        video["path"],
                    ],
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=30,
                ).stdout
            )
            available = [float(frame["pts_time"]) for frame in data["frames"]]
            _, metadata = video_loader.sample_video_frames(video, fps=8, max_frames=8)
            timestamps = metadata["timestamps_seconds"]
            self.assertEqual(timestamps[0], available[0])
            self.assertEqual(timestamps[-1], available[-1])
            self.assertEqual(timestamps, sorted(set(timestamps)))
            self.assertTrue(set(timestamps).issubset(available))
            self.assertEqual(metadata["target_num_frames"], 8)
            self.assertLess(metadata["num_frames"], metadata["target_num_frames"])
            self.assertEqual(
                metadata["effective_fps"],
                metadata["num_frames"] / metadata["duration_seconds"],
            )

    def test_hash_change_during_extraction_is_rejected_and_temporary_files_are_cleaned(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            original_run = video_loader._run_tool
            output_directories = []

            def mutate_after_extract(command, *args, **kwargs):
                output = original_run(command, *args, **kwargs)
                if Path(command[0]).name == "ffmpeg":
                    output_directories.append(Path(command[-1]).parent)
                    with Path(video["path"]).open("ab") as stream:
                        stream.write(b"changed during extraction")
                return output

            with patch.object(
                video_loader, "_run_tool", side_effect=mutate_after_extract
            ):
                with self.assertRaisesRegex(ValueError, "video changed"):
                    video_loader.sample_video_frames(video, max_frames=1)
            self.assertTrue(output_directories)
            self.assertTrue(all(not path.exists() for path in output_directories))

    def test_empty_frame_output_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            original_run = video_loader._run_tool
            output_directories = []

            def remove_frames(command, *args, **kwargs):
                output = original_run(command, *args, **kwargs)
                if Path(command[0]).name == "ffmpeg":
                    directory = Path(command[-1]).parent
                    output_directories.append(directory)
                    for path in directory.glob("*.jpg"):
                        path.write_bytes(b"")
                return output

            with patch.object(video_loader, "_run_tool", side_effect=remove_frames):
                with self.assertRaisesRegex(ValueError, "empty or invalid JPEG"):
                    video_loader.sample_video_frames(video, max_frames=1)
            self.assertTrue(all(not path.exists() for path in output_directories))

    def test_audio_only_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audio.m4a"
            subprocess.run(
                [
                    shutil.which("ffmpeg"),
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=0.25",
                    "-c:a",
                    "aac",
                    str(path),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            with self.assertRaisesRegex(ValueError, "no video stream"):
                video_loader.sample_video_frames(
                    {"path": str(path), "sha256": video_loader.file_hash(path)}
                )


if __name__ == "__main__":
    unittest.main()
