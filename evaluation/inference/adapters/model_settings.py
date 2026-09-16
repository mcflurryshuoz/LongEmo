"""Model-specific generation settings, independent of the HTTP adapter."""

from __future__ import annotations

import re
from urllib.parse import urlparse


def configure_model(args, config):
    """Apply supported model presets after the service format is selected."""
    family = args.model_family
    model = (args.model or "").lower().rsplit("/", 1)[-1]
    if model.startswith("gpt-6"):
        args.temperature = None
        for key in ("temperature", "top_p", "top_logprobs", "logprobs"):
            if key in config:
                raise ValueError(f"GPT-6 does not support {key}")
        if args.thinking == "off":
            raise ValueError("GPT-6 requires reasoning; use --thinking default")
        effort = "high" if args.thinking == "on" else "medium"
        if args.api_format == "responses":
            config.setdefault("reasoning", {"effort": effort})
        elif urlparse(args.base_url).hostname == "openrouter.ai":
            config.setdefault("reasoning", {"effort": effort})
        else:
            config.setdefault("reasoning_effort", effort)
        args.thinking = "default"
    if family == "gemini":
        _gemini(args, config, model)
    elif family in {"qwen_vl", "qwen_omni"}:
        _qwen(args, config, model)
    elif family == "qwen_audio":
        if not model.startswith("qwen2-audio"):
            raise ValueError(
                "the original Qwen-Audio requires its official custom inference code; "
                "use Qwen/Qwen2-Audio-7B-Instruct with this API entry"
            )
        if args.thinking == "on":
            raise ValueError("Qwen2-Audio has no thinking-mode switch")
    elif family == "glm":
        if args.thinking != "default":
            version = re.match(r"glm-(\d+)(?:\.(\d+))?", model)
            if not version or tuple(int(n or 0) for n in version.groups()) < (4, 5):
                raise ValueError(
                    "this GLM model has no preset thinking switch; use --thinking default "
                    "with the model's supported --config settings"
                )
            _thinking_type(args, config)
    elif family == "seed":
        _seed(args, config, model)
    elif family == "deepseek":
        if args.thinking != "default":
            if args.api_format != "chat":
                raise ValueError(
                    "this DeepSeek thinking preset uses Chat Completions; "
                    "use --config with --thinking default for other APIs"
                )
            _thinking_type(args, config)
    else:
        if args.thinking != "default":
            raise ValueError(
                "this model has no preset thinking switch; use --config with --thinking default"
            )
        if family == "gpt" and args.api_format == "chat":
            if "max_tokens" in config:
                if "max_completion_tokens" in config:
                    raise ValueError("provide only one output token limit in --config")
                config["max_completion_tokens"] = config.pop("max_tokens")
            config.setdefault("max_completion_tokens", args.max_tokens)


def _thinking_type(args, config):
    config.setdefault(
        "thinking", {"type": "enabled" if args.thinking == "on" else "disabled"}
    )


def _gemini(args, config, model):
    if args.thinking == "default":
        return
    gemini3 = model.startswith(("gemini-3-", "gemini-3."))
    gemini25 = model.startswith("gemini-2.5-")
    if not (gemini3 or gemini25):
        raise ValueError(
            "no thinking preset for this Gemini model; use --thinking default "
            "with --config for the model's supported settings"
        )
    if args.thinking == "off":
        if gemini3:
            raise ValueError(
                "Gemini 3 cannot fully disable thinking; for Flash, use "
                "--thinking default with thinkingLevel=minimal in native "
                "request options (reasoning_effort=minimal for chat)"
            )
        if model.startswith("gemini-2.5-pro"):
            raise ValueError("Gemini 2.5 Pro cannot disable thinking")

    if args.api_format == "gemini":
        settings = config.setdefault("generationConfig", {})
        if not isinstance(settings, dict):
            raise ValueError("generationConfig must be an object")
        thinking = settings.setdefault("thinkingConfig", {})
        if not isinstance(thinking, dict):
            raise ValueError("thinkingConfig must be an object")
        key = "thinkingLevel" if gemini3 else "thinkingBudget"
        other_key = "thinkingBudget" if gemini3 else "thinkingLevel"
        if other_key in thinking:
            raise ValueError(
                f"{other_key} conflicts with this model's thinking preset; "
                "use --thinking default with explicit request options"
            )
        value = "high" if gemini3 else (-1 if args.thinking == "on" else 0)
        thinking.setdefault(key, value)
    else:
        config.setdefault(
            "reasoning_effort",
            "high" if gemini3 else ("medium" if args.thinking == "on" else "none"),
        )


def _qwen(args, config, model):
    omni = args.model_family == "qwen_omni"
    if omni:
        config.setdefault("modalities", ["text"])
        _omni_output(config, config.get("enable_thinking") is True)
    if args.thinking == "default":
        return

    enabled = args.thinking == "on"
    # Instruct and Thinking checkpoints are separate weights, not API modes.
    if "-instruct" in model or "-thinking" in model:
        checkpoint_thinks = "-thinking" in model
        if enabled != checkpoint_thinks:
            raise ValueError(
                "this Qwen checkpoint has a fixed thinking mode; select the "
                "matching Instruct/Thinking checkpoint or use --thinking default "
                "with server-specific request options"
            )
        if omni:
            _omni_output(config, checkpoint_thinks)
        return

    hybrid = model.startswith(("qwen3-vl-plus", "qwen3-vl-flash", "qwen3-omni-flash"))
    if not hybrid:
        raise ValueError(
            "no thinking switch is defined for this Qwen model; use "
            "--thinking default with server-specific --config"
        )
    config.setdefault("enable_thinking", enabled)
    if omni:
        _omni_output(config, config.get("enable_thinking") is True)


def _omni_output(config, thinking):
    if thinking and (config.get("modalities") != ["text"] or "audio" in config):
        raise ValueError("Qwen-Omni thinking mode supports text output only")


def _seed(args, config, model):
    # Ark ignores requested temperatures for these published model versions.
    if urlparse(args.base_url).hostname == "ark.cn-beijing.volces.com" and model in {
        "doubao-seed-2-0-pro-260215",
        "doubao-seed-2-0-lite-260215",
    }:
        args.temperature = 1.0
        if "temperature" in config:
            config["temperature"] = 1.0
    if "max_tokens" in config and "max_completion_tokens" in config:
        raise ValueError("provide only one output token limit in --config")
    if args.thinking != "default":
        if "thinking" in model and args.thinking == "off":
            raise ValueError("this Seed thinking model cannot disable thinking")
        _thinking_type(args, config)
