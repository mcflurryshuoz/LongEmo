"""Weightless contracts for official AffectGPT adaptation; no model execution."""

import ast
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from evaluation.inference.adapters import affectgpt


def upstream_prompt_method():
    # Exercise the actual upstream pure-text formatter without importing its
    # unrelated training dependencies or loading benchmark/dataset annotations.
    path = (
        Path(affectgpt.__file__).resolve().parents[1]
        / "emollm/AffectGPT/AffectGPT/my_affectgpt/datasets/datasets/base_dataset.py"
    )
    tree = ast.parse(path.read_text())
    dataset = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseDataset"
    )
    method = next(
        node
        for node in dataset.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_prompt_for_multimodal"
    )
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


class AffectGPTTest(unittest.TestCase):
    def runtime(self, root):
        runtime = object.__new__(affectgpt.AffectGPTRuntime)
        runtime.generation = {
            "max_new_tokens": 64,
            "max_length": 4096,
            "do_sample": False,
            "temperature": 1.0,
        }
        runtime.audio_dir = root
        runtime.torch = SimpleNamespace(inference_mode=nullcontext)
        dataset_cls = type(
            "Dataset", (), {"get_prompt_for_multimodal": upstream_prompt_method()}
        )
        runtime.dataset = dataset_cls()
        runtime.dataset.read_frame_face_audio_text = Mock(
            return_value={"prepared": True}
        )
        runtime.chat = SimpleNamespace(
            replace_token_for_multimodal=lambda value: value,
            tokenizer=lambda value, **kwargs: {"input_ids": list(range(len(value)))},
            postprocess_audio=Mock(return_value=("audio hidden", "audio embeds")),
            postprocess_frame=Mock(return_value=("frame hidden", "frame embeds")),
            postprocess_multi=Mock(return_value=("merged hidden", "merged embeds")),
            answer_sample=Mock(return_value='["happiness", "surprise"]'),
        )
        return runtime

    def test_real_upstream_prompt_and_multimodal_route_keep_benchmark_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "V1.mp4"
            video.write_bytes(b"video fixture")
            (root / "V1.wav").write_bytes(b"audio fixture")
            runtime = self.runtime(root)
            question = "Subtitles: Why are you here?\nWhy did Alice and Bob react differently? Return a JSON list."
            self.assertEqual(
                runtime.generate(str(video), question), '["happiness", "surprise"]'
            )
            call = runtime.chat.answer_sample.call_args.kwargs
            self.assertIn(question, call["prompt"])
            self.assertIn("<FrameHere>", call["prompt"])
            self.assertIn("<AudioHere>", call["prompt"])
            self.assertIn("<MultiHere>", call["prompt"])
            self.assertIn("<Subtitle></Subtitle>", call["prompt"])
            self.assertEqual(call["max_new_tokens"], 64)
            self.assertFalse(call["do_sample"])
            self.assertEqual(call["img_list"]["multi"], "merged embeds")
            runtime.dataset.read_frame_face_audio_text.assert_called_once_with(
                video_path=str(video.resolve()),
                audio_path=str((root / "V1.wav").resolve()),
                face_npy=None,
                image_path=None,
            )
            runtime.chat.postprocess_multi.assert_called_once_with(
                "frame hidden", "audio hidden"
            )

    def test_long_questions_error_before_preprocessing_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "V1.mp4"
            video.touch()
            runtime = self.runtime(root)
            runtime.generation["max_length"] = 10
            with self.assertRaisesRegex(ValueError, "exceeding context limit"):
                runtime.generate(str(video), "Preserve this question")
            runtime.dataset.read_frame_face_audio_text.assert_not_called()
            runtime.chat.answer_sample.assert_not_called()

    def test_missing_audio_sidecar_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = self.runtime(root)
            with self.assertRaisesRegex(ValueError, "audio sidecar"):
                with runtime._audio_path(root / "V1.mp4"):
                    self.fail("missing audio was accepted")

    def test_ffmpeg_failure_never_substitutes_silent_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.runtime(Path(tmp))
            runtime.audio_dir = None
            runtime.ffmpeg = "/mock/ffmpeg"
            runtime.audio_timeout = 30
            failure = SimpleNamespace(returncode=1, stderr="no audio stream")
            with patch.object(affectgpt.subprocess, "run", return_value=failure) as run:
                with self.assertRaisesRegex(ValueError, "no audio stream"):
                    with runtime._audio_path(Path(tmp) / "video space.mp4"):
                        self.fail("missing audio was accepted")
            command = run.call_args.args[0]
            self.assertIn("0:a:0", command)
            self.assertIn("16000", command)
            self.assertIn(str(Path(tmp) / "video space.mp4"), command)

    def test_generation_options_and_missing_assets_are_validated_without_model_imports(
        self,
    ):
        options = affectgpt._generation_options(
            {"generation": {"max_new_tokens": 4096, "temperature": 0}}
        )
        self.assertFalse(options["do_sample"])
        self.assertEqual(options["temperature"], 1.0)
        for generation in (
            {"max_new_tokens": 0},
            {"max_new_tokens": True},
            {"do_sample": True, "temperature": 0},
            {"unexpected": 1},
        ):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                affectgpt._generation_options({"generation": generation})
        with self.assertRaisesRegex(ValueError, "model_path"):
            affectgpt.create_runtime({})


if __name__ == "__main__":
    unittest.main()
