"""Audio-only inference contracts using synthetic PCM and a local HTTP server."""

import base64
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from evaluation import eval as eval_module, io_utils
from evaluation.inference import run, video_loader
from helpers import question, service
from test_audio_routing import AUDIO, AUDIO_META


MODEL = "Qwen/Qwen2-Audio-7B-Instruct"


def invoke(module, arguments):
    with contextlib.redirect_stdout(io.StringIO()):
        return module.run(module.parser().parse_args(arguments))


def long_audio_fixture():
    # Non-constant samples expose lost, duplicated or reordered segment boundaries.
    count = 16000 * 65 + 137
    pattern = bytes(range(251))
    samples = (pattern * ((count * 2 + 250) // 251))[: count * 2]
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples)
    raw = output.getvalue()
    return (
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(raw).decode("ascii"),
                "format": "wav",
            },
        },
        {
            "audio": "separate_audio",
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": count / 16000,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "video_start_seconds": 0.0,
        },
        samples,
    )


class QwenAudioTest(unittest.TestCase):
    def arguments(self, root, url, q=None, model=MODEL):
        q = question() if q is None else q
        source = root / "question.json"
        source.write_text(json.dumps(q))
        videos = root / "videos"
        videos.mkdir()
        (videos / "V1.mp4").write_bytes(b"immutable synthetic media")
        return [
            "--data-path",
            str(source),
            "-g",
            q["granularity"],
            "--modality",
            "audio",
            "--model",
            model,
            "--base-url",
            url + "/v1",
            "--output-dir",
            str(root / "out"),
            "--tries",
            "1",
        ]

    def test_audio_request_isolated_from_gold_with_subtitle_and_prompt_modes(self):
        for granularity in ("clip", "episode"):
            for prompt_mode in ("system", "merge"):
                for subtitles in (False, True):
                    with (
                        self.subTest(
                            granularity=granularity, prompt_mode=prompt_mode, subtitles=subtitles
                        ),
                        tempfile.TemporaryDirectory() as td,
                        service(lambda *_: "Initially happy, then sad.") as (
                            url,
                            calls,
                        ),
                        patch.object(
                            run, "extract_audio", return_value=(AUDIO, AUDIO_META)
                        ) as audio,
                        patch.object(run, "sample_video_frames") as frames,
                        patch.object(run, "encode_video") as video,
                        patch.object(
                            io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "subtitles"
                        ),
                    ):
                        root = Path(td)
                        q = question(granularity=granularity, task="emotion trajectory")
                        q["answer_details"]["description"] = "PRIVATE_DETAIL_GUIDANCE"
                        q["answer_details"]["items"][0]["description"] = (
                            "PRIVATE_STAGE_DESCRIPTION"
                        )
                        q["rubric"]["criterion"] = "PRIVATE_SCORING_GUIDANCE"
                        if granularity == "episode":
                            io_utils.EPISODE_SUBTITLES_DIR.mkdir()
                            (io_utils.EPISODE_SUBTITLES_DIR / "V1.json").write_text(
                                json.dumps(q["subtitles"])
                            )
                            q["subtitles"] = None
                        args = self.arguments(root, url, q) + ["--prompt-mode", prompt_mode]
                        if subtitles:
                            args += ["--with-subtitle"]
                        original = (root / "question.json").read_bytes()
                        self.assertEqual(invoke(run, args), 0)
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0][0], "/v1/chat/completions")
                        messages = calls[0][1]["messages"]
                        self.assertEqual(
                            [m["role"] for m in messages],
                            ["system", "user"] if prompt_mode == "system" else ["user"],
                        )
                        content = messages[-1]["content"]
                        self.assertEqual(
                            sum(x["type"] == "input_audio" for x in content), 1
                        )
                        self.assertFalse(
                            any(
                                x["type"] in {"video_url", "image_url"} for x in content
                            )
                        )
                        payload = json.dumps(messages)
                        for secret in (
                            "REFERENCE_NOT_FOR_INFERENCE",
                            "PRIVATE_DETAIL_GUIDANCE",
                            "PRIVATE_STAGE_DESCRIPTION",
                            "PRIVATE_SCORING_GUIDANCE",
                            "Synthetic calibration level",
                            "SYNTHETIC_SOURCE",
                            "PRIVATE_SPEAKER",
                        ):
                            self.assertNotIn(secret, payload)
                        text = "\n".join(
                            x["text"] for x in content if x["type"] == "text"
                        )
                        self.assertEqual("First subtitle." in text, subtitles)
                        if subtitles:
                            audio_position = next(
                                i
                                for i, x in enumerate(content)
                                if x["type"] == "input_audio"
                            )
                            subtitle_position = next(
                                i
                                for i, x in enumerate(content)
                                if "First subtitle." in x.get("text", "")
                            )
                            self.assertLess(audio_position, subtitle_position)
                            self.assertLess(
                                text.index("First subtitle."), text.index(q["question"])
                            )
                        audio.assert_called_once()
                        self.assertEqual(audio.call_args.kwargs, {"required": True})
                        frames.assert_not_called()
                        video.assert_not_called()
                        self.assertEqual(
                            (root / "question.json").read_bytes(), original
                        )
                        result = io_utils.load_records(root / "out/predictions.jsonl")[
                            0
                        ]
                        self.assertEqual(result["granularity"], granularity)
                        self.assertEqual(
                            result["media"]["representation"], "audio_only"
                        )

    def test_long_audio_preserves_every_sample_in_one_request_and_stores_references(
        self,
    ):
        audio_block, metadata, original_samples = long_audio_fixture()
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "happiness") as (url, calls),
            patch.object(run, "extract_audio", return_value=(audio_block, metadata)),
            patch.object(run, "sample_video_frames") as frames,
            patch.object(run, "encode_video") as video,
        ):
            root = Path(td)
            self.assertEqual(invoke(run, self.arguments(root, url)), 0)
            self.assertEqual(len(calls), 1)
            content = calls[0][1]["messages"][-1]["content"]
            audio_parts = [x for x in content if x["type"] == "input_audio"]
            self.assertEqual(len(audio_parts), 3)
            self.assertEqual(
                [
                    content[i - 1]["text"]
                    for i, part in enumerate(content)
                    if part["type"] == "input_audio"
                ],
                [
                    "Audio at 0.000–30.000 seconds:",
                    "Audio at 30.000–60.000 seconds:",
                    "Audio at 60.000–65.009 seconds:",
                ],
            )
            restored, lengths, digests = [], [], []
            for part in audio_parts:
                raw = base64.b64decode(part["input_audio"]["data"], validate=True)
                digests.append(hashlib.sha256(raw).hexdigest())
                with wave.open(io.BytesIO(raw), "rb") as audio:
                    self.assertEqual(
                        (
                            audio.getframerate(),
                            audio.getnchannels(),
                            audio.getsampwidth(),
                        ),
                        (16000, 1, 2),
                    )
                    self.assertLessEqual(audio.getnframes(), 30 * 16000)
                    lengths.append(audio.getnframes())
                    restored.append(audio.readframes(audio.getnframes()))
            self.assertEqual(b"".join(restored), original_samples)
            self.assertEqual(lengths, [480000, 480000, 80137])
            records_path = root / "out/predictions.jsonl"
            record = io_utils.load_records(records_path)[0]
            segments = record["media"]["audio_details"]["segments"]
            self.assertEqual([s["start_seconds"] for s in segments], [0, 30, 60])
            self.assertEqual(
                [s["end_seconds"] for s in segments], [30, 60, 65 + 137 / 16000]
            )
            self.assertEqual([s["sha256"] for s in segments], digests)
            recorded_parts = record["messages"][-1]["content"]
            references = [x for x in recorded_parts if x["type"] == "audio_reference"]
            self.assertEqual(len(references), 3)
            self.assertTrue(
                all(
                    x["path"] == str((root / "videos/V1.mp4").resolve())
                    for x in references
                )
            )
            self.assertEqual(
                [x["audio_details"]["sha256"] for x in references], digests
            )
            saved = records_path.read_text()
            for part in audio_parts:
                self.assertNotIn(part["input_audio"]["data"], saved)
            self.assertNotIn('"input_audio"', saved)
            frames.assert_not_called()
            video.assert_not_called()

    def test_service_rejection_never_retries_with_a_shorter_waveform(self):
        audio_block, metadata, _ = long_audio_fixture()
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: {"error": {"message": "Too many audio inputs"}}) as (
                url,
                calls,
            ),
            patch.object(run, "extract_audio", return_value=(audio_block, metadata)),
        ):
            root = Path(td)
            self.assertEqual(invoke(run, self.arguments(root, url)), 1)
            self.assertEqual(len(calls), 1)
            parts = calls[0][1]["messages"][-1]["content"]
            self.assertEqual(sum(x["type"] == "input_audio" for x in parts), 3)
            record = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertNotEqual(record["status"], "ok")
            self.assertNotIn("prediction", record)

    def test_no_audio_track_prevents_model_request(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "must not be called") as (url, calls),
            patch.object(
                video_loader, "_require_tool", return_value="synthetic-ffprobe"
            ),
            patch.object(video_loader, "_probe_audio", return_value=None),
            patch.object(video_loader, "_run_tool") as extraction,
            patch.object(run, "sample_video_frames") as frames,
            patch.object(run, "encode_video") as video,
        ):
            root = Path(td)
            self.assertEqual(invoke(run, self.arguments(root, url)), 1)
            self.assertEqual(calls, [])
            record = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertNotEqual(record["status"], "ok")
            self.assertIn("no audio track", record["error"])
            extraction.assert_not_called()
            frames.assert_not_called()
            video.assert_not_called()

    def test_unsupported_input_combinations_fail_before_reading_media(self):
        cases = [
            (MODEL, ["--no-audio"], "cannot be combined"),
            (MODEL, ["--force-frames"], "requires --modality video"),
            ("Qwen/Qwen-Audio-Chat", [], "Qwen2-Audio-7B-Instruct"),
            ("Qwen/Qwen3-VL-8B-Instruct", [], "not supported"),
            ("gpt-4.1", [], "not supported"),
            ("claude-sonnet-4-5", [], "not supported"),
            ("custom-audio-model", [], "not supported"),
        ]
        with service(lambda *_: "must not be called") as (url, calls):
            for model, flags, error in cases:
                with (
                    self.subTest(model=model, flags=flags),
                    tempfile.TemporaryDirectory() as td,
                    patch.object(run, "extract_audio") as audio,
                    patch.object(run, "sample_video_frames") as frames,
                    patch.object(run, "encode_video") as video,
                ):
                    root = Path(td)
                    with self.assertRaisesRegex(ValueError, error):
                        invoke(run, self.arguments(root, url, model=model) + flags)
                    self.assertFalse((root / "out").exists())
                    for operation in (audio, frames, video):
                        operation.assert_not_called()
            self.assertEqual(calls, [])

    def test_audio_predictions_use_existing_clip_and_episode_scoring(self):
        for granularity in ("clip", "episode"):
            answer = "happiness" if granularity == "clip" else "Yes"
            with (
                self.subTest(granularity=granularity),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: answer) as (url, calls),
                service(lambda *_: '{"score": 1, "reason": "Correct."}') as (
                    judge_url,
                    judge_calls,
                ),
                patch.object(run, "extract_audio", return_value=(AUDIO, AUDIO_META)),
            ):
                root = Path(td)
                q = question(granularity=granularity)
                self.assertEqual(invoke(run, self.arguments(root, url, q)), 0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(
                    invoke(
                        eval_module,
                        [
                            "--data-path",
                            str(root / "question.json"),
                            "--predictions",
                            str(root / "out/predictions.jsonl"),
                            "-g",
                            granularity,
                            "--model",
                            "local-judge",
                            "--base-url",
                            judge_url + "/v1",
                            "--output-dir",
                            str(root / "scores"),
                            "--tries",
                            "1",
                        ],
                    ),
                    0,
                )
                outputs = list((root / "scores").glob("*/scores.jsonl"))
                self.assertEqual(len(outputs), 1)
                result = io_utils.load_records(outputs[0])[0]
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["granularity"], granularity)
                self.assertEqual(result["question_id"], q["question_id"])
                self.assertEqual(len(judge_calls), int(granularity == "episode"))


if __name__ == "__main__":
    unittest.main()
