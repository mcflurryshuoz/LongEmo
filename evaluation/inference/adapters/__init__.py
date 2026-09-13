"""OpenAI, Anthropic and Gemini API adapters for shared model inference."""

from __future__ import annotations

from urllib.parse import urlparse

from . import anthropic, gemini, openai
from .model_settings import configure_model


API_ADAPTERS = {
    "chat": openai,
    "responses": openai,
    "anthropic": anthropic,
    "gemini": gemini,
}

# Model-specific features have only been adapted for these request formats.
MODEL_API_FORMATS = {
    "gpt": {"chat", "responses"},
    "claude": {"chat", "anthropic"},
    "gemini": {"chat", "gemini"},
    "glm": {"chat"},
    "seed": {"chat"},
    "qwen_vl": {"chat"},
    "qwen_omni": {"chat"},
    "qwen_audio": {"chat"},
}


def prepare_request(args, config):
    """Select the service's request format, then apply model-specific settings."""
    args.api_format = _request_format(args.base_url, args.model_family)
    configure_model(args, config)


def _request_format(base_url, model_family):
    endpoint = urlparse(base_url)
    path = endpoint.path.rstrip("/")
    if path.endswith(":streamGenerateContent"):
        raise ValueError(
            "use a generateContent endpoint for non-streaming Gemini requests"
        )

    allowed = MODEL_API_FORMATS.get(model_family, API_ADAPTERS.keys())
    # A complete endpoint takes precedence over base-path and provider defaults.
    for suffix, api_format in (
        ("/chat/completions", "chat"),
        ("/responses", "responses"),
        ("/messages", "anthropic"),
        (":generateContent", "gemini"),
    ):
        if path.endswith(suffix):
            if api_format not in allowed:
                raise ValueError(
                    f"{model_family} is not configured for the {suffix} endpoint"
                )
            return api_format

    segments = path.lower().split("/")
    if {"openai", "compatible-mode"}.intersection(segments):
        return "chat"
    if "anthropic" in segments and "anthropic" in allowed:
        return "anthropic"
    if path.endswith("/v1beta") and "gemini" in allowed:
        return "gemini"
    return {
        ("gpt", "api.openai.com"): "responses",
        ("claude", "api.anthropic.com"): "anthropic",
        ("gemini", "generativelanguage.googleapis.com"): "gemini",
    }.get((model_family, endpoint.hostname), "chat")
