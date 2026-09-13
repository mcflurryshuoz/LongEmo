"""Anthropic Messages wire format."""

from .message_content import content_blocks, inline_image


def _message_content(content, *, system=False):
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content_blocks(content):
        kind = chunk["type"]
        if kind == "text":
            parts.append({"type": "text", "text": chunk["text"]})
        elif kind == "image_url" and not system:
            media_type, data = inline_image(chunk["image_url"]["url"], "Anthropic")
            parts.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                }
            )
        elif system:
            raise ValueError("Anthropic system content must contain text only")
        elif kind == "video_url":
            raise ValueError(
                "anthropic adapter does not support native video_url; use sampled images"
            )
        else:
            raise ValueError(f"unsupported anthropic content block: {kind}")
    return parts


def build_payload(client, messages):
    result = {"model": client.model, "messages": [], "max_tokens": client.max_tokens}
    if client.temperature is not None:
        result["temperature"] = client.temperature
    for message in messages:
        role = message["role"]
        content = _message_content(message["content"], system=role == "system")
        if role == "system":
            result.setdefault("system", []).extend(content_blocks(content))
        elif role in {"user", "assistant"}:
            result["messages"].append({"role": role, "content": content})
        else:
            raise ValueError("Anthropic messages require user or assistant roles")
    result.update(client.options)
    return result


def endpoint(client):
    base = client.base_url.rstrip("/")
    if base.endswith("/messages"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/messages"


def headers(client):
    result = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if client.api_key:
        result["x-api-key"] = client.api_key
    return result


def parse_response(client, raw):
    finish = raw.get("stop_reason")
    if finish == "max_tokens":
        raise RuntimeError("model output exceeded the token limit")
    if finish not in {"end_turn", "stop_sequence", "refusal"}:
        raise RuntimeError(
            f"Anthropic generation stopped before a final answer: {finish}"
        )
    blocks = raw.get("content")
    if not isinstance(blocks, list) or not any(
        block.get("type") == "text" for block in blocks
    ):
        raise RuntimeError("Anthropic service returned no answer text")
    content = "".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    )
    if not isinstance(content, str):
        raise RuntimeError("model answer must be text")
    return {
        "content": content,
        "raw_response": raw,
        "finish_reason": finish,
        "usage": raw.get("usage", raw.get("usageMetadata")),
    }
