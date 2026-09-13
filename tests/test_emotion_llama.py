"""Weightless tests of the Emotion-LLaMA benchmark adapter and official runtime."""

import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.inference.adapters import emotion_llama


class EmotionLLaMAAdapterTest(unittest.TestCase):
    def test_construction_defers_official_imports_and_model_loading(self):
        before = set(sys.modules)
        with patch.object(emotion_llama, "_load_runtime_class") as load:
            runtime = emotion_llama.create_runtime({})
        load.assert_not_called()
        self.assertIsNone(runtime._runtime)
        imported = set(sys.modules) - before
        self.assertFalse(
            any(
                name.split(".")[0] in {"torch", "transformers", "minigpt4"}
                for name in imported
            )
        )
        self.assertEqual(runtime.metadata["visual_input"], "first video frame only")
        self.assertIn("mean-pooled", runtime.metadata["audio_input"])
        self.assertIn("zero-filled", runtime.metadata["temporal_feature"])

    def test_repo_dir_alias_is_explicit(self):
        runtime = emotion_llama.create_runtime(
            {"repo_dir": str(emotion_llama.DEFAULT_REPO)}
        )
        self.assertEqual(runtime.repo_path, emotion_llama.DEFAULT_REPO)
        with self.assertRaisesRegex(ValueError, "only one"):
            emotion_llama.create_runtime(
                {
                    "repo_dir": str(emotion_llama.DEFAULT_REPO),
                    "repo_path": str(emotion_llama.DEFAULT_REPO),
                }
            )

    def test_official_runtime_receives_full_prompt_and_fresh_conversations(self):
        # Execute the actual official lightweight runtime without importing its
        # heavy package __init__. Inject Chat components, never model weights.
        name = "emotion_llama_official_runtime_under_test"
        source = emotion_llama.DEFAULT_REPO / "minigpt4/inference/runtime.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        conversations = []
        chat = Mock()
        chat.answer.side_effect = [("first answer", None), ("second answer", None)]

        def conversation_factory():
            conversation = object()
            conversations.append(conversation)
            return conversation

        with patch.dict(sys.modules, {name: module}):
            spec.loader.exec_module(module)

            def factory(**kwargs):
                return module.EmotionLLaMARuntime(
                    **kwargs,
                    component_loader=lambda: (chat, conversation_factory, "cpu"),
                )

            with (
                tempfile.TemporaryDirectory() as td,
                patch.object(
                    emotion_llama, "_load_runtime_class", return_value=factory
                ) as load,
            ):
                video = Path(td) / "clip.mp4"
                video.write_bytes(b"not decoded by fake chat")
                prompt = "SYSTEM: Answer the supplied question.\n\nQuestion: Why did Alex become upset?\nChoices: A. betrayal B. fear\n"
                runtime = emotion_llama.create_runtime(
                    {"generation": {"max_new_tokens": 64, "temperature": 0.3}}
                )
                self.assertEqual(runtime.generate(str(video), prompt), "first answer")
                followup = "Question: How did Alex's emotions change?"
                self.assertEqual(
                    runtime.generate(str(video), followup), "second answer"
                )
                load.assert_called_once()
                self.assertEqual(chat.ask.call_args_list[0].args[0], prompt)
                self.assertEqual(chat.ask.call_args_list[1].args[0], followup)
                self.assertIsNot(conversations[0], conversations[1])
                self.assertEqual(
                    chat.upload_img.call_args_list[0].args[0], str(video.resolve())
                )
                self.assertEqual(chat.answer.call_args.kwargs["max_new_tokens"], 64)
                self.assertEqual(chat.answer.call_args.kwargs["temperature"], 0.3)

    def test_paths_device_and_generation_map_to_official_runtime(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            llama = root / "llama model=chat"
            audio = root / "hubert"
            llama.mkdir()
            audio.mkdir()
            checkpoint = root / "checkpoint.pth"
            checkpoint.touch()
            video = root / "video.mp4"
            video.touch()
            factory = Mock()
            factory.return_value.analyze.return_value = "B"
            runtime = emotion_llama.create_runtime(
                {
                    "model_path": str(llama),
                    "checkpoint": str(checkpoint),
                    "audio_model_path": str(audio),
                    "device": "cuda:1",
                    "options": ["model.low_resource=false"],
                    "generation": {"seed": 3, "max_length": 4096},
                }
            )
            with patch.object(
                emotion_llama, "_load_runtime_class", return_value=factory
            ):
                self.assertEqual(runtime.generate(str(video), "Which emotion?"), "B")
            kwargs = factory.call_args.kwargs
            self.assertEqual(kwargs["device"], "cuda:1")
            self.assertEqual(kwargs["dataset_name"], "feature_face_caption")
            overrides = dict(value.split("=", 1) for value in kwargs["options"])
            self.assertEqual(
                json.loads(overrides["model.llama_model"]), str(llama.resolve())
            )
            self.assertEqual(
                json.loads(overrides["model.ckpt"]), str(checkpoint.resolve())
            )
            self.assertEqual(
                json.loads(overrides["model.audio_model_path"]), str(audio.resolve())
            )
            factory.return_value.analyze.assert_called_once_with(
                str(video.resolve()), "Which emotion?", seed=3, max_length=4096
            )

    def test_bad_inputs_fail_before_importing_model_dependencies(self):
        runtime = emotion_llama.create_runtime({})
        with patch.object(emotion_llama, "_load_runtime_class") as load:
            with self.assertRaisesRegex(ValueError, "prompt"):
                runtime.generate("missing.mp4", " ")
            with self.assertRaises(FileNotFoundError):
                runtime.generate("missing.mp4", "A valid question")
        load.assert_not_called()
        for config in (
            {"question": "replace user question"},
            {"options": "model.low_resource=false"},
            {"options": ["model.low_resource"]},
            {"generation": []},
            {"generation": {"do_sample": False}},
            {"generation": {"max_new_tokens": 4096}},
            {"generation": {"max_new_tokens": 2000, "max_length": 2000}},
            {"generation": {"max_new_tokens": True}},
            {"generation": {"max_length": 0}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                emotion_llama.create_runtime(config)

    def test_context_guard_checks_real_official_preparation_before_truncation(self):
        # Execute the actual pure-Python official answer_prepare function. Model
        # embeddings are mocked; this specifically checks the boundary before
        # upstream slices embeddings, without Torch or downloaded model weights.
        source = emotion_llama.DEFAULT_REPO / "minigpt4/conversation/conversation.py"
        tree = ast.parse(source.read_text())
        chat_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Chat"
        )
        prepare = next(
            node
            for node in chat_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "answer_prepare"
        )
        logger = Mock()
        scope = {"logger": logger}
        exec(
            compile(ast.Module(body=[prepare], type_ignores=[]), str(source), "exec"),
            scope,
        )
        official_chat = type(
            "OfficialChatPreparation", (), {"answer_prepare": scope["answer_prepare"]}
        )

        class Embeddings:
            def __init__(self, tokens):
                self.shape = (1, tokens, 32)
                self.slices = []

            def __getitem__(self, selection):
                self.slices.append(selection[1])
                return Embeddings(max(0, self.shape[1] - selection[1].start))

        for tokens, overflows in ((1500, False), (1501, True)):
            with self.subTest(tokens=tokens):
                embeddings = Embeddings(tokens)
                model = types.SimpleNamespace(
                    get_context_emb=Mock(return_value=embeddings)
                )
                chat = official_chat()
                chat.model = model
                chat.stopping_criteria = []
                loader = Mock(return_value=types.SimpleNamespace(chat=chat))
                runtime = types.SimpleNamespace(_component_loader=loader)
                emotion_llama._install_context_guard(runtime)
                loader.assert_not_called()
                runtime._component_loader()
                conv = Mock(roles=["user", "assistant"])
                conv.get_prompt.return_value = "Complete benchmark question"
                arguments = {
                    "conv": conv,
                    "img_list": ["visual and audio embeddings"],
                    "max_new_tokens": 500,
                    "max_length": 2000,
                }
                if overflows:
                    with self.assertRaisesRegex(ValueError, "needs 2001 tokens"):
                        chat.answer_prepare(**arguments)
                    self.assertEqual(embeddings.slices, [])
                else:
                    result = chat.answer_prepare(**arguments)
                    self.assertEqual(result["inputs_embeds"].shape[1], 1500)
                    self.assertEqual(result["max_new_tokens"], 500)
                self.assertIs(chat.model, model)
                model.get_context_emb.assert_called_once_with(
                    "Complete benchmark question", ["visual and audio embeddings"]
                )
                logger.warning.assert_not_called()

    def test_missing_assets_and_non_string_answers_are_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            for key in (
                "model_path",
                "checkpoint",
                "audio_model_path",
                "config_path",
                "repo_path",
            ):
                with self.subTest(key=key), self.assertRaises(FileNotFoundError):
                    emotion_llama.create_runtime({key: str(Path(td) / "missing")})
            video = Path(td) / "video.mp4"
            video.touch()
            runtime = emotion_llama.create_runtime({})
            runtime._runtime = Mock()
            runtime._runtime.analyze.return_value = {"answer": "bad contract"}
            with self.assertRaisesRegex(TypeError, "non-string"):
                runtime.generate(str(video), "Question")

    def test_another_minigpt4_checkout_requires_process_isolation(self):
        package = types.ModuleType("minigpt4")
        package.__file__ = "/another/model/minigpt4/__init__.py"
        with patch.dict(sys.modules, {"minigpt4": package}):
            with self.assertRaisesRegex(RuntimeError, "own Python process"):
                emotion_llama._load_runtime_class(emotion_llama.DEFAULT_REPO)


if __name__ == "__main__":
    unittest.main()
