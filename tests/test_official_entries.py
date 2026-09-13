"""Independent official-model CLIs without model weights or GPU libraries."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from evaluation import eval as eval_module, io_utils
from evaluation.inference.adapters import emollm_runner as common
from helpers import question


ROOT = Path(__file__).resolve().parents[1]


class OfficialEntryTest(unittest.TestCase):
    def test_adapter_imports_do_not_load_model_dependencies(self):
        code = """
import sys
from evaluation.inference.adapters import (
    affectgpt, emotion_llama, r1_omni, emollm_runner,
    openai, anthropic, gemini, model_settings, qwen_audio
)
loaded = {"torch", "transformers", "minigpt4", "my_affectgpt"}.intersection(sys.modules)
assert not loaded, f"Adapters eagerly imported model dependencies: {sorted(loaded)}"
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_independent_help_without_weights(self):
        for backend in common.REPOSITORIES:
            for launch in ("script", "module"):
                with (
                    self.subTest(backend=backend, launch=launch),
                    tempfile.TemporaryDirectory() as td,
                ):
                    if launch == "script":
                        entry = (
                            ROOT / "evaluation/inference/adapters" / (backend + ".py")
                        )
                        prefix = [sys.executable, str(entry)]
                        cwd = td
                    else:
                        prefix = [
                            sys.executable,
                            "-m",
                            "evaluation.inference.adapters." + backend,
                        ]
                        cwd = ROOT
                    help_run = subprocess.run(
                        [*prefix, "--help"],
                        capture_output=True,
                        text=True,
                        cwd=cwd,
                        timeout=15,
                    )
                    self.assertEqual(help_run.returncode, 0, help_run.stderr)
                    self.assertNotIn("--dry-run", help_run.stdout)
                    self.assertNotIn("--lang", help_run.stdout)

    def test_all_independent_runs_with_mock_runtime(self):
        for backend in common.REPOSITORIES:
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                q = question(task="emotion trajectory")
                (root / "q.json").write_text(
                    json.dumps([q, question("Q2", video_id="unselected_video")])
                )
                (root / "runtime.json").write_text("{}")
                (root / "videos").mkdir()
                (root / "videos/V1.mp4").write_bytes(b"synthetic-video")
                runtime = SimpleNamespace(
                    generate=Mock(return_value="The person becomes happier.")
                )
                factory = Mock(return_value=runtime)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = common.main(
                        backend,
                        factory,
                        [
                            "--data-path",
                            str(root / "q.json"),
                            "--runtime-config",
                            str(root / "runtime.json"),
                            "--with-subtitle",
                            "--limit",
                            "1",
                            "--output-dir",
                            str(root / "out"),
                        ],
                    )
                self.assertEqual(result, 0)
                factory.assert_called_once()
                records = io_utils.load_records(root / "out/predictions.jsonl")
                self.assertEqual(len(records), 1)
                record = records[0]
                prompt = record["request"]["prompt"]
                runtime.generate.assert_called_once_with(
                    str((root / "videos/V1.mp4").resolve()), prompt
                )
                self.assertIn("### Question", prompt)
                self.assertIn(q["question"], prompt)
                self.assertLess(
                    prompt.index("First subtitle."), prompt.index(q["question"])
                )
                for hidden in (
                    q["answer"],
                    "Required stages",
                    "SYNTHETIC_SOURCE",
                    "Synthetic calibration",
                    "PRIVATE_SPEAKER",
                ):
                    self.assertNotIn(hidden, prompt)
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["prediction"], "The person becomes happier.")

    def test_reused_runtime_resume_and_shared_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qs = [question("Q1"), question("Q2")]
            (root / "q.json").write_text(json.dumps(qs))
            (root / "runtime.json").write_text("{}")
            (root / "videos").mkdir()
            (root / "videos/V1.mp4").write_bytes(b"synthetic-video")
            runtime = SimpleNamespace(
                generate=Mock(return_value="happiness"),
                metadata={"visual_input": "official sampling"},
            )
            factory = Mock(return_value=runtime)
            args = common.parser("r1_omni").parse_args(
                [
                    "--data-path",
                    str(root / "q.json"),
                    "--videos-dir",
                    str(root / "videos"),
                    "--runtime-config",
                    str(root / "runtime.json"),
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(common.run("r1_omni", factory, args), 0)
                self.assertEqual(common.run("r1_omni", factory, args), 0)
            factory.assert_called_once()
            self.assertEqual(runtime.generate.call_count, 2)
            self.assertNotIn("First subtitle.", runtime.generate.call_args[0][1])
            records = io_utils.load_records(root / "out/predictions.jsonl")
            self.assertEqual(records[0]["prediction"], ["happiness"])
            self.assertEqual(records[0]["media"], runtime.metadata)
            scoring = eval_module.parser().parse_args(
                [
                    "--data-path",
                    str(root / "q.json"),
                    "--predictions",
                    str(root / "out/predictions.jsonl"),
                    "--base-url",
                    "http://localhost:8000/v1",
                    "--output-dir",
                    str(root / "scores"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(eval_module.run(scoring), 0)

    def test_input_and_execution_errors_not_submitted_as_answers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "q.json").write_text(
                json.dumps([question("Q1"), question("Q2", video_id="missing")])
            )
            (root / "runtime.json").write_text("{}")
            (root / "videos").mkdir()
            (root / "videos/V1.mp4").write_bytes(b"synthetic-video")
            runtime = SimpleNamespace(
                generate=Mock(side_effect=RuntimeError("decode failed"))
            )
            args = common.parser("affectgpt").parse_args(
                [
                    "--data-path",
                    str(root / "q.json"),
                    "--videos-dir",
                    str(root / "videos"),
                    "--runtime-config",
                    str(root / "runtime.json"),
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(common.run("affectgpt", lambda _: runtime, args), 1)
            rows = io_utils.load_records(root / "out/predictions.jsonl")
            self.assertEqual([row["status"] for row in rows], ["error", "input_error"])
            self.assertTrue(all(row.get("prediction") is None for row in rows))

    def test_relative_assets_are_relative_to_config(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "runtime.json"
            source.write_text('{"model_path":"weights","checkpoint":"emotion.pth"}')
            config = common.read_config(source, "emotion_llama")
            self.assertEqual(config["model_path"], str((root / "weights").resolve()))
            self.assertEqual(
                config["checkpoint"], str((root / "emotion.pth").resolve())
            )

    def test_existing_output_is_reused_until_force(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "q.json").write_text(json.dumps(question()))
            (root / "videos").mkdir()
            (root / "videos/V1.mp4").write_bytes(b"video")
            (root / "audio").mkdir()
            (root / "audio/V1.wav").write_bytes(b"audio-v1")
            (root / "model.yaml").write_text("sampling: 8")
            (root / "runtime.json").write_text(
                '{"audio_dir":"audio","config_path":"model.yaml"}'
            )
            runtime = SimpleNamespace(generate=Mock(return_value="happiness"))
            args = common.parser("affectgpt").parse_args(
                [
                    "--data-path",
                    str(root / "q.json"),
                    "--videos-dir",
                    str(root / "videos"),
                    "--runtime-config",
                    str(root / "runtime.json"),
                    "--output-dir",
                    str(root / "out"),
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(common.run("affectgpt", lambda _: runtime, args), 0)
                (root / "audio/V1.wav").write_bytes(b"audio-v2")
                self.assertEqual(common.run("affectgpt", lambda _: runtime, args), 0)
                self.assertEqual(runtime.generate.call_count, 1)
                (root / "model.yaml").write_text("sampling: 16")
                args.force = True
                self.assertEqual(common.run("affectgpt", lambda _: runtime, args), 0)
                self.assertEqual(runtime.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
