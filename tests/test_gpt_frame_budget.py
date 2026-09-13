"""Explicit image-area budgets use synthetic videos only, without API calls."""

import base64
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evaluation.inference import video_loader


def jpeg_dimensions(block):
    raw = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
    position = 2
    while position < len(raw):
        marker = raw[position + 1]
        length = int.from_bytes(raw[position + 2 : position + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            height, width = struct.unpack(">HH", raw[position + 5 : position + 9])
            return width, height
        position += length + 2
    raise AssertionError("JPEG has no dimensions")


class TotalPixelValidationTest(unittest.TestCase):
    def test_invalid_total_budget_fails_before_media_tools(self):
        for value in (0, -1, True, False, 1.5, "100352", math.nan, math.inf,
                      video_loader.VIDEO_MIN_PIXELS - 1):
            with self.subTest(value=value), patch.object(video_loader, "_run_tool") as run:
                with self.assertRaisesRegex(ValueError, "total_pixels"):
                    video_loader.sample_video_frames({}, total_pixels=value)
                run.assert_not_called()


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg and ffprobe are required",
)
class ExplicitFrameBudgetTest(unittest.TestCase):
    def make_video(self, directory, *, size="160x90"):
        path = Path(directory) / "synthetic.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error",
                "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=4:duration=2",
                "-c:v", "mpeg4", "-bf", "2", "-y", str(path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return {"path": str(path)}

    def assert_actual_budget(self, blocks, metadata):
        dimensions = [jpeg_dimensions(block) for block in blocks
                      if block["type"] == "image_url"]
        self.assertEqual(len(dimensions), metadata["num_frames"])
        actual = 0
        for width, height in dimensions:
            self.assertEqual((width, height),
                             (metadata["resized_width"], metadata["resized_height"]))
            self.assertEqual(width % video_loader.IMAGE_FACTOR, 0)
            self.assertEqual(height % video_loader.IMAGE_FACTOR, 0)
            self.assertGreater(min(width, height), 0)
            self.assertLessEqual(width * height, metadata["max_pixels"])
            actual += width * height
        self.assertEqual(actual, metadata["actual_total_pixels"])
        self.assertLessEqual(actual, metadata["total_pixels"])

    def test_none_keeps_qwen_defaults_and_pixel_ceiling(self):
        self.assertEqual(video_loader._frame_pixel_budget(768), 235200)
        self.assertEqual(video_loader._frame_pixel_budget(10000), 105369)
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            _, default = video_loader.sample_video_frames(video)
            _, explicit_none = video_loader.sample_video_frames(
                video, max_pixels=video_loader.VIDEO_MAX_PIXELS * 2, total_pixels=None
            )
        for metadata in (default, explicit_none):
            self.assertEqual(metadata["requested_fps"], 2)
            self.assertEqual(metadata["max_frames"], 768)
            self.assertEqual(metadata["max_pixels"], 602112)
            self.assertEqual(metadata["total_pixels"], 90316800)
            self.assertEqual(metadata["timestamps_seconds"], [0, 0.75, 1.25, 1.75])
            self.assertEqual((metadata["resized_width"], metadata["resized_height"]),
                             (448, 252))
            self.assertEqual(metadata["actual_total_pixels"], 448 * 252 * 4)

    def test_explicit_frame_and_total_budgets_replace_qwen_ceiling(self):
        larger = video_loader.VIDEO_MAX_PIXELS * 2
        smaller = video_loader.VIDEO_MIN_PIXELS * 2
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, size="1280x720")
            for maximum, total in ((larger, larger), (smaller, larger), (larger, smaller)):
                with self.subTest(max_pixels=maximum, total_pixels=total):
                    blocks, metadata = video_loader.sample_video_frames(
                        video, max_frames=1, max_pixels=maximum, total_pixels=total
                    )
                    self.assertEqual(metadata["requested_max_pixels"], maximum)
                    self.assertEqual(metadata["max_pixels"], min(maximum, total))
                    self.assertEqual(metadata["total_pixels"], total)
                    self.assert_actual_budget(blocks, metadata)
                    if maximum == total == larger:
                        self.assertGreater(metadata["actual_total_pixels"],
                                           video_loader.VIDEO_MAX_PIXELS)

    def test_minimum_size_and_narrow_frames_stay_within_explicit_budget(self):
        total = video_loader.VIDEO_MIN_PIXELS
        with tempfile.TemporaryDirectory() as temporary:
            for size in ("160x90", "90x160", "2x2", "400x2", "2x400"):
                with self.subTest(size=size):
                    video = self.make_video(temporary, size=size)
                    blocks, metadata = video_loader.sample_video_frames(
                        video, max_frames=1, total_pixels=total
                    )
                    self.assert_actual_budget(blocks, metadata)

    def test_total_budget_reduces_count_and_retains_video_endpoints(self):
        total = video_loader.VIDEO_MIN_PIXELS * 2
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            blocks, metadata = video_loader.sample_video_frames(
                video, fps=100, max_frames=8, total_pixels=total
            )
        self.assertEqual(metadata["max_frames"], 8)
        self.assertEqual(metadata["target_num_frames"], 2)
        self.assertEqual(metadata["num_frames"], 2)
        self.assertEqual(metadata["timestamps_seconds"], [0, 1.75])
        self.assertEqual(metadata["max_pixels"], video_loader.VIDEO_MIN_PIXELS)
        self.assert_actual_budget(blocks, metadata)


if __name__ == "__main__":
    unittest.main()
