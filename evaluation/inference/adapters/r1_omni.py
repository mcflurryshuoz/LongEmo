"""Benchmark adapter for the checked-out official R1-Omni/HumanOmni inference.

The public CLI is intentionally not imported: it changes CUDA_VISIBLE_DEVICES.
This adapter follows its model_init -> video/audio processors -> mm_infer calls,
passing the benchmark prompt unchanged as both instruct and the BERT question.

Only local model directories are accepted. The official loader's tower config
and hard-coded BERT location are redirected in memory while loading; no vendor
files or checkpoint config files are edited. Run this backend in its own Python
environment because upstream imports a specific Torch/Transformers stack.

Upstream limitations are retained: CUDA/float16 inference, checkpoint-defined
uniform video sampling, BERT question truncation, and Whisper's configured audio
window (normally the first 30 seconds). The upstream silent-audio fallback is
turned into an error. There is no long-video chunk aggregation in this adapter.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


OFFICIAL_REPO = Path(__file__).resolve().parents[1] / "emollm" / "R1-Omni"
OFFICIAL_REVISION = "17cafcae0b0a1c454896f4eb7a1c006bf13e1f4d"
_GENERATION_OPTIONS = {"do_sample", "temperature", "top_p", "max_new_tokens"}


def _local_directory(value, label):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"R1-Omni requires {label} as a local model directory")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"R1-Omni {label} directory does not exist: {path}")
    return str(path)


def _load_dependencies(repo):
    """Keep heavy dependencies out of CLI help, request assembly and tests."""
    if not (repo / "inference.py").is_file():
        raise FileNotFoundError(f"Official R1-Omni checkout is missing: {repo}")
    existing = sys.modules.get("humanomni")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin is None or repo not in Path(origin).resolve().parents:
            raise RuntimeError(
                "Another humanomni package is already imported; run R1-Omni "
                "in a separate model Python process"
            )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        return SimpleNamespace(
            torch=importlib.import_module("torch"),
            transformers=importlib.import_module("transformers"),
            upstream=importlib.import_module("humanomni"),
            loader=importlib.import_module("humanomni.model"),
            architecture=importlib.import_module("humanomni.model.humanomni_arch"),
            mm_utils=importlib.import_module("humanomni.mm_utils"),
        )
    except ImportError as exc:
        raise RuntimeError(
            "R1-Omni dependencies are unavailable. Use a separate Python "
            "environment prepared according to emollm/R1-Omni/README.md "
            f"and setup.sh. Original import error: {exc}"
        ) from exc


class _StrictAudioTorch:
    """Delegate tensor operations, but reject process_audio's silence fallback."""

    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def zeros(self, *args, **kwargs):
        raise RuntimeError(
            "Official R1-Omni audio decoding failed; refusing its silent-audio fallback"
        ) from sys.exc_info()[1]


class R1OmniRuntime:
    """Load one official model, then answer arbitrary benchmark video prompts."""

    def __init__(self, config):
        if not isinstance(config, dict):
            raise TypeError("R1-Omni configuration must be a JSON object")
        self.model_path = _local_directory(config.get("model_path"), "model_path")
        bert_path = _local_directory(config.get("bert_model_path"), "bert_model_path")
        repo = Path(_local_directory(config.get("repo_dir", OFFICIAL_REPO), "repo_dir"))
        self.modal = config.get("modal", "video_audio")
        if self.modal not in {"video_audio", "video"}:
            raise ValueError("R1-Omni benchmark modal must be video_audio or video")
        self.device = config.get("device", "cuda:0")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"cuda(?::\d+)?", self.device
        ):
            raise ValueError(
                "Official R1-Omni inference requires device cuda or cuda:N"
            )
        self.generation = {"do_sample": False, "max_new_tokens": 2048}
        generation = config.get("generation", {})
        if not isinstance(generation, dict):
            raise ValueError("R1-Omni generation must be a JSON object")
        unknown = set(generation) - _GENERATION_OPTIONS
        if unknown:
            raise ValueError(
                f"Unsupported R1-Omni generation options: {sorted(unknown)}"
            )
        self.generation.update(generation)
        token_limit = self.generation["max_new_tokens"]
        if (
            isinstance(token_limit, bool)
            or not isinstance(token_limit, int)
            or token_limit < 1
        ):
            raise ValueError("R1-Omni max_new_tokens must be a positive integer")
        self.seed = config.get("seed")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("R1-Omni seed must be a nonnegative integer")

        checkpoint_config = json.loads(
            (Path(self.model_path) / "config.json").read_text()
        )
        tower_paths = {}
        for option, field in (
            ("vision_model_path", "mm_vision_tower"),
            ("audio_model_path", "mm_audio_tower"),
        ):
            value = config.get(option, checkpoint_config.get(field))
            # HumanOmni loads the configured audio tower even in video-only mode.
            if field == "mm_audio_tower" and not value and self.modal == "video":
                continue
            tower_paths[field] = _local_directory(value, option)
        if "siglip" not in tower_paths["mm_vision_tower"]:
            raise ValueError(
                "The official R1-Omni SigLIP directory path must contain 'siglip'"
            )
        if (
            "mm_audio_tower" in tower_paths
            and "whisper" not in tower_paths["mm_audio_tower"]
        ):
            raise ValueError(
                "The official R1-Omni Whisper directory path must contain 'whisper'"
            )

        deps = _load_dependencies(repo)
        self._torch = deps.torch
        self._mm_utils = deps.mm_utils
        if not self._torch.cuda.is_available():
            raise RuntimeError(
                "Official R1-Omni inference requires a CUDA-enabled PyTorch runtime"
            )
        self._mm_infer = deps.upstream.mm_infer
        self.bert_tokenizer = deps.transformers.BertTokenizer.from_pretrained(
            bert_path, local_files_only=True
        )

        original_config_loader = deps.loader.AutoConfig.from_pretrained

        def load_config(*args, **kwargs):
            kwargs["local_files_only"] = True
            model_config = original_config_loader(*args, **kwargs)
            for name, value in tower_paths.items():
                setattr(model_config, name, value)
            return model_config

        def bert_loader(cls):
            def from_pretrained(_upstream_path, *args, **kwargs):
                kwargs["local_files_only"] = True
                return cls.from_pretrained(bert_path, *args, **kwargs)

            return SimpleNamespace(from_pretrained=from_pretrained)

        # Patch module aliases only, for this load. The upstream source remains
        # byte-for-byte intact, and Transformers' global classes are untouched.
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    deps.loader,
                    "AutoConfig",
                    SimpleNamespace(from_pretrained=load_config),
                )
            )
            stack.enter_context(
                patch.object(
                    deps.architecture,
                    "BertModel",
                    bert_loader(deps.architecture.BertModel),
                )
            )
            stack.enter_context(
                patch.object(
                    deps.architecture,
                    "BertTokenizer",
                    bert_loader(deps.architecture.BertTokenizer),
                )
            )
            stack.enter_context(self._torch.cuda.device(self.device))
            self.model, self.processor, self.tokenizer = deps.upstream.model_init(
                self.model_path,
                device=self.device,
                device_map={"": self.device},
                use_flash_attn=config.get("use_flash_attn", False),
                local_files_only=True,
            )
        self.model.eval()
        audio_processor = getattr(self.processor.get("audio"), "keywords", {}).get(
            "processor"
        )
        self.metadata = {
            "source_revision": OFFICIAL_REVISION,
            "modal": self.modal,
            "device": self.device,
            "num_frames": getattr(self.model.config, "num_frames", 8),
            "video_sampling": "official uniform sampling across the full video",
            "audio_max_seconds": getattr(audio_processor, "chunk_length", None)
            if self.modal == "video_audio"
            else None,
            "audio_sampling": "start of video, truncated/padded to the Whisper window; no chunk aggregation"
            if self.modal == "video_audio"
            else "disabled",
            "audio_decode_failure": "raise an error instead of substituting silence",
        }

    def generate(self, video_path: str, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("R1-Omni requires a nonempty benchmark prompt")
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"R1-Omni video does not exist: {video}")
        with self._torch.cuda.device(self.device):
            if self.seed is not None:
                self._torch.manual_seed(self.seed)
            video_tensor = self.processor["video"](str(video))
            audio = None
            if self.modal == "video_audio":
                with patch.object(
                    self._mm_utils, "torch", _StrictAudioTorch(self._mm_utils.torch)
                ):
                    audio = self.processor["audio"](str(video))[0]
            output = self._mm_infer(
                video_tensor,
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                modal=self.modal,
                question=prompt,
                bert_tokeni=self.bert_tokenizer,
                audio=audio,
                **self.generation,
            )
        if not isinstance(output, str):
            raise TypeError("Official R1-Omni mm_infer did not return response text")
        return output


def create_runtime(config: dict) -> R1OmniRuntime:
    return R1OmniRuntime(config)


if __name__ == "__main__":
    if __package__ in {None, ""}:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        __package__ = "evaluation.inference.adapters"
    from .emollm_runner import main

    raise SystemExit(main("r1_omni", create_runtime))
