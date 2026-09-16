"""Real codec regressions: arbitrary end PTS and delayed audio alignment."""
import base64
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import wave

from evaluation.inference.video_loader import extract_audio, sample_video_frames
from methods.longemo.media import window_input


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.video = Path(self.directory.name) / "offset.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i",
            "color=blue:size=160x120:rate=24:duration=3", "-itsoffset", "0.4", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=16000:duration=2.6", "-c:v", "libx264", "-threads", "1",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "3", "-y", str(self.video)], check=True, timeout=30)

    def tearDown(self):
        self.directory.cleanup()

    def test_arbitrary_window_end_has_only_in_range_actual_pts(self):
        _, meta = sample_video_frames({"path": str(self.video)}, start_seconds=.17, end_seconds=1.337,
                                     fps=4, max_frames=8, max_pixels=200704)
        self.assertGreater(meta["num_frames"], 0)
        self.assertTrue(all(.17 <= timestamp <= 1.337 for timestamp in meta["timestamps_seconds"]))
        self.assertEqual(meta["timestamps_seconds"], sorted(set(meta["timestamps_seconds"])))

    def test_window_audio_matches_aligned_full_audio_slice(self):
        full, _ = extract_audio({"path": str(self.video)}, required=True)
        content, _ = window_input(self.video, .75, 1.25, fps=2, max_frames=2,
                                 max_pixels=200704, with_audio=True, subtitle_rows=[])
        part = next(block for block in content if block["type"] == "input_audio")

        def pcm(block):
            with wave.open(io.BytesIO(base64.b64decode(block["input_audio"]["data"]))) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getsampwidth()), (16000, 1, 2))
                return wav.readframes(wav.getnframes())

        self.assertEqual(pcm(part), pcm(full)[12000*2:20000*2])


if __name__ == "__main__":
    unittest.main()
