"""Run the vendored official AffectGPT raw-frame model on benchmark questions.

The model path is the *frame* AffectGPT .pth checkpoint, not the Qwen base
model. Separate local directories supply Qwen2.5, CLIP, and Chinese HuBERT.
See ../emollm/AffectGPT/AffectGPT/inference_sample.py for the upstream flow.
Heavy dependencies are imported only by create_runtime(), in the model environment.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace


OFFICIAL_REVISION = "fffca794c6792023234565370e3f69f0488aede9"
DEFAULT_CONFIG = "mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml"
FRAME_MODALITY = "multiframe_audio_frame_text"


def _path(value, label, *, directory=False):
    if not value:
        raise ValueError(f"AffectGPT requires {label}")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"AffectGPT {label} does not exist: {path}")
    return path


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _generation_options(config):
    options = dict(config.get("generation") or {})
    allowed = {
        "num_beams",
        "temperature",
        "do_sample",
        "top_p",
        "max_new_tokens",
        "min_length",
        "max_length",
        "repetition_penalty",
        "length_penalty",
    }
    unsupported = set(options) - allowed
    if unsupported:
        raise ValueError(
            f"unsupported AffectGPT generation options: {sorted(unsupported)}"
        )
    options.setdefault("max_new_tokens", 4096)
    options.setdefault("temperature", 0.0)
    options.setdefault("do_sample", options["temperature"] > 0)
    if (
        isinstance(options["max_new_tokens"], bool)
        or not isinstance(options["max_new_tokens"], int)
        or options["max_new_tokens"] < 1
    ):
        raise ValueError("AffectGPT max_new_tokens must be a positive integer")
    if not isinstance(options["do_sample"], bool):
        raise ValueError("AffectGPT do_sample must be a boolean")
    if options["temperature"] < 0 or (
        options["do_sample"] and options["temperature"] <= 0
    ):
        raise ValueError("AffectGPT sampling requires temperature > 0")
    # Transformers validates the value even when sampling is disabled in some
    # releases. Greedy decoding is still controlled by do_sample=False.
    if not options["do_sample"]:
        options["temperature"] = 1.0
    return options


class AffectGPTRuntime:
    def __init__(self, config):
        self.generation = _generation_options(config)
        default_repo = Path(__file__).resolve().parents[1] / "emollm" / "AffectGPT"
        self.repo_dir = _path(
            config.get("repo_dir", default_repo), "repo_dir", directory=True
        )
        # The official repository includes three projects; its AffectGPT Python
        # application is nested one level below the repository root.
        self.application_dir = self.repo_dir / "AffectGPT"
        _path(
            self.application_dir / "my_affectgpt" / "models" / "affectgpt.py",
            "official model source",
        )
        config_path = _path(
            config.get(
                "config_path", self.application_dir / "train_configs" / DEFAULT_CONFIG
            ),
            "config_path",
        )
        checkpoint = _path(
            config.get("model_path"), "model_path (raw-frame AffectGPT .pth checkpoint)"
        )
        asset_paths = {
            "llm_path": _path(
                config.get(
                    "llm_path", self.application_dir / "models" / "Qwen2.5-7B-Instruct"
                ),
                "llm_path",
                directory=True,
            ),
            "visual_encoder_path": _path(
                config.get(
                    "visual_encoder_path",
                    self.application_dir / "models" / "clip-vit-large-patch14",
                ),
                "visual_encoder_path",
                directory=True,
            ),
            "audio_encoder_path": _path(
                config.get(
                    "audio_encoder_path",
                    self.application_dir / "models" / "chinese-hubert-large",
                ),
                "audio_encoder_path",
                directory=True,
            ),
        }
        self.audio_dir = (
            _path(config["audio_dir"], "audio_dir", directory=True)
            if config.get("audio_dir")
            else None
        )
        self.ffmpeg = str(config.get("ffmpeg_path") or shutil.which("ffmpeg") or "")
        if self.audio_dir is None and not self.ffmpeg:
            raise ValueError(
                "AffectGPT needs ffmpeg for audio extraction, or audio_dir/<video_id>.wav"
            )
        self.audio_timeout = float(config.get("audio_timeout", 300))
        self.device = str(config.get("device", "cuda:0"))
        if not (self.device == "cuda" or self.device.startswith("cuda:")):
            raise ValueError("the official AffectGPT runtime requires a CUDA device")

        # Local model paths are mandatory. Prevent upstream Hugging Face loaders
        # from trying network downloads when an asset is incomplete.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(self.application_dir))
        existing_config = sys.modules.get("config")
        if (
            existing_config is not None
            and Path(getattr(existing_config, "__file__", "")).resolve()
            != self.application_dir / "config.py"
        ):
            raise RuntimeError(
                "AffectGPT must run as its own entry point: another package has imported 'config'"
            )

        with _working_directory(self.application_dir):
            try:
                import torch
                from omegaconf import OmegaConf

                upstream_config = importlib.import_module("config")
                upstream_config.PATH_TO_LLM["Qwen25"] = str(asset_paths["llm_path"])
                upstream_config.PATH_TO_VISUAL["CLIP_VIT_LARGE"] = str(
                    asset_paths["visual_encoder_path"]
                )
                upstream_config.PATH_TO_AUDIO["HUBERT_LARGE"] = str(
                    asset_paths["audio_encoder_path"]
                )

                # Validate the selected architecture before allocating model weights.
                yaml_config = OmegaConf.load(config_path)
                model_cfg = yaml_config.model
                modality = {
                    value.face_or_frame for value in yaml_config.datasets.values()
                }
                if modality != {FRAME_MODALITY}:
                    raise ValueError(
                        "AffectGPT adapter requires the raw-frame YAML and matching raw-frame checkpoint; face checkpoints need OpenFace inputs"
                    )
                expected = {
                    "arch": "affectgpt",
                    "llama_model": "Qwen25",
                    "visual_encoder": "CLIP_VIT_LARGE",
                    "acoustic_encoder": "HUBERT_LARGE",
                }
                if any(model_cfg.get(key) != value for key, value in expected.items()):
                    raise ValueError(
                        f"AffectGPT raw-frame adapter requires model settings {expected}"
                    )
                if any(
                    model_cfg.get(key) != "attention"
                    for key in (
                        "multi_fusion_type",
                        "video_fusion_type",
                        "audio_fusion_type",
                    )
                ):
                    raise ValueError(
                        "AffectGPT adapter supports the released attention-fusion frame configuration"
                    )
                if model_cfg.get("ckpt") or model_cfg.get("ckpt_2"):
                    raise ValueError(
                        "use the standalone frame checkpoint in model_path; ckpt/ckpt_2 must be empty"
                    )
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "AffectGPT needs CUDA in the selected model Python environment"
                    )

                from my_affectgpt.common.config import Config
                from my_affectgpt.common.registry import registry
                from my_affectgpt.conversation.conversation_video import Chat
                from my_affectgpt.datasets.datasets.base_dataset import BaseDataset

                cfg = Config(SimpleNamespace(cfg_path=str(config_path), options=None))
                cfg.model_cfg.ckpt_3 = str(checkpoint)
                model_cls = registry.get_model_class(cfg.model_cfg.arch)
                self.model = model_cls.from_config(cfg.model_cfg).to(self.device).eval()
                self.chat = Chat(self.model, cfg.model_cfg, device=self.device)
                # MER2025OV_Dataset() eagerly reads a dataset CSV, even for a
                # single video. Its actual preprocessing and prompt methods are
                # inherited unchanged from BaseDataset, which needs no labels.
                self.dataset = BaseDataset()
                self.dataset.needed_data = self.dataset.get_needed_data(FRAME_MODALITY)
                processor_cfg = cfg.inference_cfg.vis_processor.train
                self.dataset.vis_processor = registry.get_processor_class(
                    processor_cfg.name
                ).from_config(processor_cfg)
                self.dataset.n_frms = int(cfg.model_cfg.vis_processor.train.n_frms)
                self.torch = torch
            except ImportError as exc:
                raise RuntimeError(
                    "AffectGPT dependencies are missing or incompatible. Run affectgpt.py in a separate environment matching official AffectGPT/environment.yml."
                ) from exc

        context_limit = getattr(
            self.model.llama_model.config, "max_position_embeddings", None
        )
        if not isinstance(context_limit, int) or context_limit < 1:
            raise ValueError(
                "AffectGPT base model must declare max_position_embeddings"
            )
        selected_limit = self.generation.get("max_length", context_limit)
        if (
            not isinstance(selected_limit, int)
            or not 1 <= selected_limit <= context_limit
        ):
            raise ValueError(
                f"AffectGPT generation.max_length must be within 1..{context_limit}"
            )
        self.generation["max_length"] = selected_limit
        self.metadata = {
            "official_revision": OFFICIAL_REVISION,
            "input_modalities": ["video_frames", "audio", "benchmark_prompt"],
            "checkpoint_variant": "raw_frame",
            "num_frames": self.dataset.n_frms,
            "frame_sampling": "official uniform samples across full video; short videos repeat final frame",
            "frame_size": [224, 224],
            "audio_sampling": "official eight 2-second clips distributed across audio; short audio is zero-padded",
            "audio_sample_rate": 16000,
            "audio_source": "audio_dir sidecar"
            if self.audio_dir
            else "video audio extracted by ffmpeg",
            "transcript": "only when included by the common benchmark prompt; no automatic ASR",
            "context_limit": selected_limit,
            "overflow_policy": "error; question and modality tokens are never silently truncated",
        }

    @contextmanager
    def _audio_path(self, video_path):
        if self.audio_dir is not None:
            yield _path(self.audio_dir / (video_path.stem + ".wav"), "audio sidecar")
            return
        with tempfile.TemporaryDirectory(prefix="affectgpt-audio-") as tmp:
            audio_path = Path(tmp) / "audio.wav"
            process = subprocess.run(
                [
                    self.ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.audio_timeout,
                check=False,
            )
            if (
                process.returncode
                or not audio_path.is_file()
                or audio_path.stat().st_size <= 44
            ):
                raise ValueError(
                    f"AffectGPT could not extract audio from {video_path}: {process.stderr.strip()}"
                )
            yield audio_path

    def generate(self, video_path: str, prompt: str) -> str:
        video = _path(video_path, "video_path")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("AffectGPT requires a non-empty benchmark prompt")
        # Keep the common question, answer format, and optional transcript in the
        # actual user-message slot. Never substitute the upstream OV-label task.
        model_prompt = self.dataset.get_prompt_for_multimodal(
            FRAME_MODALITY, "", prompt
        )
        expanded = self.chat.replace_token_for_multimodal(model_prompt)
        input_ids = self.chat.tokenizer(
            expanded, add_special_tokens=False, truncation=False
        )["input_ids"]
        needed = len(input_ids) + self.generation["max_new_tokens"]
        if needed > self.generation["max_length"]:
            raise ValueError(
                f"AffectGPT input plus generation needs {needed} tokens, exceeding context limit {self.generation['max_length']}; reduce max_new_tokens or input text"
            )

        with self._audio_path(video) as audio_path, self.torch.inference_mode():
            data = self.dataset.read_frame_face_audio_text(
                video_path=str(video),
                audio_path=str(audio_path),
                face_npy=None,
                image_path=None,
            )
            audio_hidden, audio_embeds = self.chat.postprocess_audio(data)
            frame_hidden, frame_embeds = self.chat.postprocess_frame(data)
            _, multi_embeds = self.chat.postprocess_multi(frame_hidden, audio_hidden)
            if any(item is None for item in (audio_embeds, frame_embeds, multi_embeds)):
                raise ValueError(
                    "AffectGPT failed to produce required audio, frame, or fused embeddings"
                )
            images = {
                "audio": audio_embeds,
                "frame": frame_embeds,
                "multi": multi_embeds,
                "face": None,
                "image": None,
            }
            # answer_sample decodes generated output directly; unlike a generic
            # causal-LM adapter, it must not slice away an input-token prefix.
            response = self.chat.answer_sample(
                prompt=model_prompt, img_list=images, **self.generation
            )
        if not isinstance(response, str):
            raise TypeError(
                "official AffectGPT Chat.answer_sample returned non-text output"
            )
        return response


def create_runtime(config: dict) -> AffectGPTRuntime:
    """Load official AffectGPT once; reuse it for all benchmark questions."""
    return AffectGPTRuntime(config)


if __name__ == "__main__":
    if __package__ in {None, ""}:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        __package__ = "evaluation.inference.adapters"
    from .emollm_runner import main

    raise SystemExit(main("affectgpt", create_runtime))
