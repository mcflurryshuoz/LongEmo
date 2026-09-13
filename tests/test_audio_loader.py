"""Audio alignment and silent-video checks use temporary synthetic media only."""

import array
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from evaluation.inference import video_loader


class AudioValidationTest(unittest.TestCase):
    def test_changed_source_is_rejected_before_running_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.write_bytes(b"original")
            video = {"path": str(path), "sha256": video_loader.file_hash(path)}
            path.write_bytes(b"changed")
            for function, kwargs in (
                (video_loader.extract_audio, {}),
                (video_loader.has_audio_track, {}),
                (video_loader.encode_video, {}),
                (video_loader.encode_video, {"include_audio": False}),
            ):
                with self.subTest(function=function.__name__, kwargs=kwargs):
                    with patch.object(video_loader, "_run_tool") as runner:
                        with self.assertRaisesRegex(ValueError, "video changed"):
                            function(video, **kwargs)
                        runner.assert_not_called()

    def test_missing_tools_have_actionable_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.write_bytes(b"synthetic")
            video = {"path": str(path), "sha256": video_loader.file_hash(path)}
            for function, kwargs, missing in (
                (video_loader.extract_audio, {}, "ffprobe"),
                (video_loader.extract_audio, {}, "ffmpeg"),
                (video_loader.has_audio_track, {}, "ffprobe"),
                (video_loader.encode_video, {"include_audio": False}, "ffmpeg"),
            ):
                with (
                    self.subTest(function=function.__name__, missing=missing),
                    patch.object(
                        video_loader, "_probe_audio", return_value={"index": 1}
                    ),
                    patch.object(
                        video_loader.shutil,
                        "which",
                        side_effect=lambda name: None
                        if name == missing
                        else "/tool/" + name,
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, missing + ".*PATH"):
                        function(video, **kwargs)

    def test_original_video_payload_needs_no_media_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mp4"
            path.write_bytes(b"original video bytes")
            video = {"path": str(path), "sha256": video_loader.file_hash(path)}
            with patch.object(video_loader.shutil, "which", side_effect=AssertionError):
                block = video_loader.encode_video(video)
            self.assertEqual(
                base64.b64decode(block["video_url"]["url"].split(",", 1)[1]),
                path.read_bytes(),
            )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg and ffprobe are required",
)
class SyntheticAudioTest(unittest.TestCase):
    def make_video(
        self,
        directory,
        *,
        video_start=0,
        audio_start=0,
        audio_duration=2,
        audio_tracks=1,
    ):
        path = Path(directory) / "synthetic.mp4"
        command = [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-copyts",
            "-itsoffset",
            str(video_start),
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=4:duration=2",
        ]
        if audio_tracks:
            command += [
                "-itsoffset",
                str(audio_start),
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=44100:duration={audio_duration}",
            ]
        command += ["-map", "0:v:0"]
        for _ in range(audio_tracks):
            command += ["-map", "1:a:0"]
        command += [
            "-c:v",
            "mpeg4",
            "-bf",
            "0",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-fps_mode",
            "passthrough",
            str(path),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return {"path": str(path), "sha256": video_loader.file_hash(path)}

    def decode_audio(self, block):
        self.assertEqual(block["type"], "input_audio")
        self.assertEqual(block["input_audio"]["format"], "wav")
        data = block["input_audio"]["data"]
        self.assertFalse(data.startswith("data:"))
        raw = base64.b64decode(data, validate=True)
        with wave.open(io.BytesIO(raw), "rb") as audio:
            self.assertEqual(
                (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()),
                (1, 16000, 2),
            )
            self.assertEqual(audio.getcomptype(), "NONE")
            samples = array.array("h", audio.readframes(audio.getnframes()))
        return raw, samples

    def volume(self, samples, start, end):
        section = samples[round(start * 16000) : round(end * 16000)]
        return sum(abs(sample) for sample in section) / len(section)

    def probe(self, path, *arguments):
        return json.loads(
            subprocess.run(
                [
                    shutil.which("ffprobe"),
                    "-v",
                    "error",
                    *arguments,
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        )

    def test_wav_payload_metadata_source_hash_and_temporary_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            source = Path(video["path"]).read_bytes()
            with patch.object(
                video_loader, "_run_tool", wraps=video_loader._run_tool
            ) as runner:
                block, metadata = video_loader.extract_audio(video)
            raw, samples = self.decode_audio(block)
            self.assertEqual(len(samples), 32000)
            self.assertGreater(self.volume(samples, 0.2, 1.8), 500)
            self.assertEqual(
                metadata,
                {
                    "audio": "separate_audio",
                    "sample_rate": 16000,
                    "channels": 1,
                    "duration_seconds": 2.0,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "video_start_seconds": 0.0,
                },
            )
            self.assertNotIn("base64", json.dumps(metadata))
            self.assertEqual(Path(video["path"]).read_bytes(), source)
            output_paths = [
                Path(call.args[0][-1])
                for call in runner.call_args_list
                if Path(call.args[0][0]).name == "ffmpeg"
            ]
            self.assertEqual(len(output_paths), 1)
            self.assertTrue(all(not path.parent.exists() for path in output_paths))

    def test_delayed_audio_keeps_silence_relative_to_video_start(self):
        for start in (0, 5):
            with (
                self.subTest(video_start=start),
                tempfile.TemporaryDirectory() as temporary,
            ):
                video = self.make_video(
                    temporary,
                    video_start=start,
                    audio_start=start + 0.5,
                    audio_duration=1,
                )
                block, metadata = video_loader.extract_audio(video)
                _, samples = self.decode_audio(block)
                self.assertEqual(metadata["video_start_seconds"], start)
                self.assertEqual(len(samples), 32000)
                self.assertLess(self.volume(samples, 0, 0.4), 5)
                self.assertGreater(self.volume(samples, 0.6, 1.4), 500)
                self.assertLess(self.volume(samples, 1.7, 2), 5)

    def test_audio_before_video_is_trimmed_and_short_track_is_padded(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, video_start=1, audio_duration=2)
            block, metadata = video_loader.extract_audio(video)
            _, samples = self.decode_audio(block)
            self.assertEqual(metadata["video_start_seconds"], 1)
            self.assertGreater(self.volume(samples, 0.1, 0.9), 500)
            self.assertLess(self.volume(samples, 1.2, 2), 5)

    def test_nonoverlapping_audio_track_produces_full_duration_silence(self):
        for video_start, audio_start, audio_duration in ((1, 0, 0.5), (0, 3, 1)):
            with (
                self.subTest(video_start=video_start),
                tempfile.TemporaryDirectory() as temporary,
            ):
                video = self.make_video(
                    temporary,
                    video_start=video_start,
                    audio_start=audio_start,
                    audio_duration=audio_duration,
                )
                block, metadata = video_loader.extract_audio(video)
                _, samples = self.decode_audio(block)
                self.assertEqual(metadata["duration_seconds"], 2)
                self.assertEqual(len(samples), 32000)
                self.assertEqual(set(samples), {0})

    def test_no_audio_is_explicit_and_required_mode_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, audio_tracks=0)
            self.assertFalse(video_loader.has_audio_track(video))
            self.assertEqual(
                video_loader.extract_audio(video), (None, {"audio": "no_audio_track"})
            )
            with self.assertRaisesRegex(ValueError, "no audio track.*required"):
                video_loader.extract_audio(video, required=True)

    def test_track_detection_never_encodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary)
            with patch.object(
                video_loader, "_run_tool", wraps=video_loader._run_tool
            ) as runner:
                self.assertTrue(video_loader.has_audio_track(video))
            self.assertEqual(len(runner.call_args_list), 1)
            self.assertEqual(Path(runner.call_args.args[0][0]).name, "ffprobe")

    def test_silent_mp4_removes_all_audio_and_copies_encoded_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = self.make_video(temporary, audio_tracks=2)
            source = Path(video["path"]).read_bytes()
            with patch.object(
                video_loader, "_run_tool", wraps=video_loader._run_tool
            ) as runner:
                block = video_loader.encode_video(video, include_audio=False)
            raw = base64.b64decode(
                block["video_url"]["url"].split(",", 1)[1], validate=True
            )
            output = Path(temporary) / "returned.mp4"
            output.write_bytes(raw)
            streams = self.probe(output, "-show_entries", "stream=codec_type")[
                "streams"
            ]
            self.assertEqual([stream["codec_type"] for stream in streams], ["video"])
            arguments = (
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "packet=data_hash",
            )
            self.assertEqual(
                self.probe(video["path"], *arguments)["packets"],
                self.probe(output, *arguments)["packets"],
            )
            self.assertEqual(Path(video["path"]).read_bytes(), source)
            self.assertFalse(Path(runner.call_args.args[0][-1]).parent.exists())

    def test_hash_change_during_media_processing_is_rejected_and_cleaned(self):
        for function, kwargs in (
            (video_loader.extract_audio, {}),
            (video_loader.encode_video, {"include_audio": False}),
        ):
            with (
                self.subTest(function=function.__name__),
                tempfile.TemporaryDirectory() as temporary,
            ):
                video = self.make_video(temporary)
                original_run = video_loader._run_tool
                directories = []

                def mutate_source(command, *args, **kwargs):
                    result = original_run(command, *args, **kwargs)
                    if Path(command[0]).name == "ffmpeg":
                        directories.append(Path(command[-1]).parent)
                        with Path(video["path"]).open("ab") as stream:
                            stream.write(b"changed")
                    return result

                with patch.object(video_loader, "_run_tool", side_effect=mutate_source):
                    with self.assertRaisesRegex(ValueError, "video changed"):
                        function(video, **kwargs)
                self.assertTrue(directories)
                self.assertTrue(all(not path.exists() for path in directories))

    def test_ffmpeg_failure_and_invalid_wav_clean_temporary_files(self):
        for failure in ("subprocess", "invalid_wave"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                video = self.make_video(temporary)
                original_run = video_loader._run_tool
                directories = []

                def fail_extract(command, *args, **kwargs):
                    if Path(command[0]).name != "ffmpeg":
                        return original_run(command, *args, **kwargs)
                    path = Path(command[-1])
                    directories.append(path.parent)
                    path.write_bytes(b"not a WAV")
                    if failure == "subprocess":
                        raise RuntimeError("ffmpeg failed: synthetic failure")
                    return ""

                expected = RuntimeError if failure == "subprocess" else ValueError
                with patch.object(video_loader, "_run_tool", side_effect=fail_extract):
                    with self.assertRaisesRegex(expected, "ffmpeg"):
                        video_loader.extract_audio(video)
                self.assertTrue(directories)
                self.assertTrue(all(not path.exists() for path in directories))
                self.assertEqual(video_loader.file_hash(video["path"]), video["sha256"])


if __name__ == "__main__":
    unittest.main()
