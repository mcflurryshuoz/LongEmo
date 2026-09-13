"""Weightless checks using the vendored inference functions with fake tensors."""

import ast
import copy
import tempfile
import types
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.inference.adapters import r1_omni


def upstream_function(relative_path, name, namespace):
    """Execute the actual function body without importing GPU dependencies."""
    path = r1_omni.OFFICIAL_REPO / relative_path
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


class FakeTensor:
    def half(self):
        return self

    def cuda(self):
        return self

    def long(self):
        return self

    def unsqueeze(self, _dimension):
        return self

    def ne(self, _value):
        return self

    def to(self, _device):
        return self


class R1OmniTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.config = {
            "device": "cuda:2",
            "seed": 7,
            "generation": {"max_new_tokens": 137},
        }
        for key, dirname in (
            ("model_path", "R1-Omni-0.5B"),
            ("bert_model_path", "bert-base-uncased"),
            ("vision_model_path", "siglip-base-patch16-224"),
            ("audio_model_path", "whisper-large-v3"),
        ):
            directory = self.root / dirname
            directory.mkdir()
            self.config[key] = str(directory)
        self.checkpoint = Path(self.config["model_path"]) / "config.json"
        self.checkpoint.write_text(
            '{"mm_vision_tower":"author-path","mm_audio_tower":"author-path"}'
        )
        self.video = self.root / "episode.mp4"
        self.video.write_bytes(b"fake video; processor is mocked")

    def dependencies(self):
        self.video_tensor, self.audio_tensor = FakeTensor(), FakeTensor()
        self.tokenizer = types.SimpleNamespace(
            pad_token_id=0,
            eos_token_id=2,
            eos_token="<eos>",
            apply_chat_template=Mock(return_value="chat prompt"),
            batch_decode=Mock(
                return_value=["  <think>Some evidence.</think>\n<answer>B</answer>  "]
            ),
        )
        self.bert_tokenizer = Mock(return_value={"input_ids": FakeTensor()})
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(
                model_type="HumanOmni_qwen2", mm_use_x_start_end=False, num_frames=8
            ),
            eval=Mock(),
            generate=Mock(return_value=[[101, 102]]),
        )
        torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=Mock(return_value=True),
                device=Mock(side_effect=lambda _: nullcontext()),
            ),
            inference_mode=nullcontext,
            manual_seed=Mock(),
            float32="float32",
            zeros=Mock(),
        )
        transformer = types.SimpleNamespace(
            image_processing_base=types.SimpleNamespace(
                BatchFeature=type("BatchFeature", (dict,), {})
            ),
            BertTokenizer=types.SimpleNamespace(
                from_pretrained=Mock(return_value=self.bert_tokenizer)
            ),
        )
        mm_namespace = {
            "torch": torch,
            "transformers": transformer,
            "copy": copy,
            "DEFAULT_IMAGE_TOKEN": "<image>",
            "DEFAULT_VIDEO_TOKEN": "<video>",
            "DEFAULT_AUDIO_TOKEN": "<audio>",
            "tokenizer_multimodal_token": Mock(return_value=FakeTensor()),
            "KeywordsStoppingCriteria": Mock(),
        }
        infer = upstream_function("humanomni/__init__.py", "mm_infer", mm_namespace)
        self.video_processor = Mock(return_value=self.video_tensor)
        self.audio_processor = Mock(return_value=(self.audio_tensor, 16000))
        self.audio_processor.keywords = {
            "processor": types.SimpleNamespace(chunk_length=30)
        }
        processor = {"video": self.video_processor, "audio": self.audio_processor}
        architecture = types.SimpleNamespace(
            BertModel=types.SimpleNamespace(
                from_pretrained=Mock(return_value="bert model")
            ),
            BertTokenizer=transformer.BertTokenizer,
        )
        loader = types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(
                from_pretrained=Mock(
                    side_effect=lambda *a, **k: types.SimpleNamespace()
                )
            )
        )

        def model_init(path, **kwargs):
            self.loaded_config = loader.AutoConfig.from_pretrained(path)
            architecture.BertModel.from_pretrained(
                "/author/hard-coded/bert-base-uncased"
            )
            architecture.BertTokenizer.from_pretrained(
                "/author/hard-coded/bert-base-uncased"
            )
            return self.model, processor, self.tokenizer

        return types.SimpleNamespace(
            torch=torch,
            transformers=transformer,
            loader=loader,
            architecture=architecture,
            mm_utils=types.SimpleNamespace(torch=torch),
            upstream=types.SimpleNamespace(
                model_init=Mock(side_effect=model_init), mm_infer=infer
            ),
        )

    def test_official_mm_infer_gets_complete_benchmark_prompt_and_modalities(self):
        deps = self.dependencies()
        original_bert = deps.architecture.BertModel
        original_config = deps.loader.AutoConfig
        original_checkpoint = self.checkpoint.read_bytes()
        prompt = 'How did the emotion change?\nA. calm → angry\nB. angry → calm\nReturn {"answer": "A or B"}.'
        with patch.object(r1_omni, "_load_dependencies", return_value=deps):
            runtime = r1_omni.create_runtime(self.config)
        output = runtime.generate(str(self.video), prompt)
        self.assertEqual(output, "<think>Some evidence.</think>\n<answer>B</answer>")
        self.assertEqual(
            self.tokenizer.apply_chat_template.call_args.args[0],
            [{"role": "user", "content": "<video>\n<audio>\n" + prompt}],
        )
        self.bert_tokenizer.assert_called_once_with(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        kwargs = self.model.generate.call_args.kwargs
        self.assertEqual(kwargs["images"], [(self.video_tensor, "video")])
        self.assertIs(kwargs["audios"], self.audio_tensor)
        self.assertEqual(kwargs["max_new_tokens"], 137)
        self.assertFalse(kwargs["do_sample"])
        self.assertIsNotNone(kwargs["prompts"])
        self.assertEqual(
            self.loaded_config.mm_vision_tower, self.config["vision_model_path"]
        )
        self.assertEqual(
            self.loaded_config.mm_audio_tower, self.config["audio_model_path"]
        )
        original_bert.from_pretrained.assert_called_once_with(
            self.config["bert_model_path"], local_files_only=True
        )
        self.assertIs(deps.architecture.BertModel, original_bert)
        self.assertIs(deps.loader.AutoConfig, original_config)
        self.assertEqual(self.checkpoint.read_bytes(), original_checkpoint)
        self.assertEqual(runtime.metadata["audio_max_seconds"], 30)
        self.assertEqual(runtime.metadata["num_frames"], 8)
        deps.torch.manual_seed.assert_called_once_with(7)
        self.assertEqual(
            deps.upstream.model_init.call_args.kwargs["device_map"], {"": "cuda:2"}
        )

    def test_video_mode_does_not_read_audio(self):
        deps = self.dependencies()
        with patch.object(r1_omni, "_load_dependencies", return_value=deps):
            runtime = r1_omni.create_runtime({**self.config, "modal": "video"})
        runtime.generate(str(self.video), "What happened at the end?")
        self.audio_processor.assert_not_called()
        self.assertIsNone(self.model.generate.call_args.kwargs["audios"])
        self.assertEqual(runtime.metadata["audio_sampling"], "disabled")

    def test_actual_upstream_audio_failure_raises_instead_of_generating_on_silence(
        self,
    ):
        deps = self.dependencies()
        mm_utils = types.ModuleType("fake_mm_utils")
        mm_utils.torch = deps.torch
        mm_utils.AudioReader = Mock(side_effect=ValueError("no audio stream"))
        mm_utils.cpu = Mock()
        process_audio = upstream_function(
            "humanomni/mm_utils.py", "process_audio", mm_utils.__dict__
        )
        deps.mm_utils = mm_utils
        with patch.object(r1_omni, "_load_dependencies", return_value=deps):
            runtime = r1_omni.create_runtime(self.config)
        runtime.processor["audio"] = partial(process_audio, processor=None)
        with self.assertRaisesRegex(RuntimeError, "silent-audio fallback") as caught:
            runtime.generate(str(self.video), "What emotion is expressed?")
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.model.generate.assert_not_called()
        deps.torch.zeros.assert_not_called()
        self.assertIs(mm_utils.torch, deps.torch)

    def test_invalid_device_or_ignored_generation_options_fail_before_loading(self):
        with patch.object(r1_omni, "_load_dependencies") as load:
            with self.assertRaisesRegex(ValueError, "requires device cuda"):
                r1_omni.create_runtime({**self.config, "device": "cpu"})
            with self.assertRaisesRegex(ValueError, "Unsupported.*generation"):
                r1_omni.create_runtime({**self.config, "generation": {"num_beams": 2}})
            with self.assertRaisesRegex(FileNotFoundError, "bert_model_path"):
                r1_omni.create_runtime(
                    {**self.config, "bert_model_path": "/missing/bert"}
                )
            load.assert_not_called()

    def test_local_aliases_restore_when_model_load_fails(self):
        deps = self.dependencies()
        original_bert = deps.architecture.BertModel
        original_config = deps.loader.AutoConfig
        deps.upstream.model_init.side_effect = RuntimeError("missing checkpoint shard")
        with patch.object(r1_omni, "_load_dependencies", return_value=deps):
            with self.assertRaisesRegex(RuntimeError, "missing checkpoint"):
                r1_omni.create_runtime(self.config)
        self.assertIs(deps.architecture.BertModel, original_bert)
        self.assertIs(deps.loader.AutoConfig, original_config)


if __name__ == "__main__":
    unittest.main()
