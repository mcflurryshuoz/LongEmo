"""Exercise CLI sampling defaults through run(), without media or API requests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import io_utils
from evaluation.clients import Client
from evaluation.inference import run
from helpers import question


class FrameSamplingDefaultsTest(unittest.TestCase):
    def invoke_sampling(self, model, *options, endpoint="/v1/chat/completions"):
        blocks = [
            {"type": "text", "text": "Frame at 0.000000 seconds:"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}},
        ]
        metadata = {"timestamps_seconds": [0.0], "num_frames": 1,
                    "duration_seconds": 1.0}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(run, "sample_video_frames", return_value=(blocks, metadata)) as frames,
            patch.object(Client, "generate", return_value={
                "content": "happiness", "raw_response": {}, "usage": None,
            }) as generate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(temporary)
            source = root / "question.json"
            source.write_text(json.dumps(question()), encoding="utf-8")
            (root / "videos").mkdir()
            video = root / "videos" / "V1.mp4"
            video.write_bytes(b"synthetic video")
            output = root / "output"
            args = run.parser().parse_args([
                "--data-path", str(source), "--model", model,
                "--base-url", "http://localhost:1" + endpoint,
                "--api-key", "synthetic-test-key", "--sample-frames",
                "--output-dir", str(output), "--tries", "1", *options,
            ])
            self.assertEqual(run.run(args), 0)
            frames.assert_called_once()
            self.assertEqual(frames.call_args.args[0]["path"], str(video.resolve()))
            generate.assert_called_once()
            messages = generate.call_args.args[0]
            self.assertTrue(any(
                part.get("type") == "image_url"
                for message in messages if isinstance(message["content"], list)
                for part in message["content"]
            ))
            record = io_utils.load_records(output / "predictions.jsonl")[0]
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["prediction"], ["happiness"])
            self.assertEqual(record["media"]["representation"], "frames")
            return dict(frames.call_args.kwargs)

    def test_image_model_defaults_reach_sampler(self):
        for model, endpoint in (
            ("gpt-4.1", "/v1/responses"),
            ("gpt-4.1", "/v1/chat/completions"),
        ):
            with self.subTest(model=model, endpoint=endpoint):
                self.assertEqual(self.invoke_sampling(model, endpoint=endpoint), {
                    "fps": 1.0, "max_frames": 128, "max_pixels": 262144,
                    "total_pixels": 33554432,
                })

    def test_claude_verified_budget_reaches_sampler(self):
        self.assertEqual(self.invoke_sampling(
            "claude-sonnet-4-6", endpoint="/v1/messages"
        ), {
            "fps": 2.0, "max_frames": 80, "max_pixels": 262144,
            "total_pixels": 8028160,
        })

    def test_deepseek_verified_capacity_reaches_sampler(self):
        self.assertEqual(self.invoke_sampling("deepseek-v4-flash"), {
            "fps": 2.0, "max_frames": 600, "max_pixels": 602112,
            "total_pixels": 361267200,
        })

    def test_explicit_cli_values_override_gpt_defaults(self):
        for fps, frames, pixels, total in (
            (2.0, 768, 602112, 90316800),
            (0.5, 32, 200704, 6422528),
        ):
            with self.subTest(fps=fps, frames=frames, pixels=pixels, total=total):
                actual = self.invoke_sampling(
                    "gpt-4.1", "--fps", str(fps), "--max-frames", str(frames),
                    "--frame-max-pixels", str(pixels), "--total-pixels", str(total),
                )
                self.assertEqual(actual, {
                    "fps": fps, "max_frames": frames, "max_pixels": pixels,
                    "total_pixels": total,
                })

    def test_partial_override_preserves_other_gpt_defaults(self):
        actual = self.invoke_sampling("gpt-4.1", "--max-frames", "1000")
        self.assertEqual(actual, {
            "fps": 1.0, "max_frames": 1000, "max_pixels": 262144,
            "total_pixels": 33554432,
        })

    def test_other_models_keep_original_defaults_and_service_caps(self):
        for model, frame_cap in (
            ("Qwen/Qwen3-VL-8B-Instruct", 768),
            ("glm-4.6v", 768),
            ("doubao-seed-2-0-pro-260215", 768),
            ("test-model", 768),
        ):
            with self.subTest(model=model):
                self.assertEqual(self.invoke_sampling(model), {
                    "fps": 2.0, "max_frames": frame_cap, "max_pixels": 602112,
                    "total_pixels": None,
                })

    def test_claude_and_deepseek_keep_their_service_caps(self):
        for model, endpoint in (
            ("claude-sonnet-4-6", "/v1/messages"),
            ("deepseek-v4-flash", "/v1/chat/completions"),
        ):
            with self.subTest(model=model):
                actual = self.invoke_sampling(
                    model, "--max-frames", "700", endpoint=endpoint
                )
                self.assertEqual(actual["max_frames"], 600)

    def test_explicit_zero_is_validated_instead_of_replaced_by_default(self):
        for option in ("--fps", "--max-frames", "--frame-max-pixels"):
            with (
                self.subTest(option=option),
                patch.object(run, "sample_video_frames") as frames,
                patch.object(Client, "generate") as generate,
            ):
                args = run.parser().parse_args([
                    "--data-path", "unused", "--model", "gpt-4.1",
                    "--base-url", "http://localhost:1/v1", "--api-key", "synthetic-test-key",
                    option, "0",
                ])
                with self.assertRaisesRegex(ValueError, "frame sampling requires"):
                    run.run(args)
                frames.assert_not_called()
                generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
