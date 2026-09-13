"""OpenAI Chat Completions and Responses wire formats."""

import copy
from urllib import parse

from .message_content import content_blocks, inline_audio


def _responses_content(content):
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content_blocks(content):
        kind = chunk["type"]
        if kind == "text":
            parts.append({"type": "input_text", "text": chunk["text"]})
        elif kind == "image_url":
            image = chunk["image_url"]
            part = {"type": "input_image", "image_url": image["url"]}
            if "detail" in image:
                part["detail"] = image["detail"]
            parts.append(part)
        elif kind == "video_url":
            raise ValueError(
                "responses adapter does not support native video_url; use sampled images"
            )
        else:
            raise ValueError(f"unsupported responses content block: {kind}")
    return parts


def _chat_messages(client, messages):
    dashscope = (parse.urlparse(client.base_url).hostname or "").endswith(
        ".aliyuncs.com"
    )
    if dashscope:
        messages = copy.deepcopy(messages)
        for message in messages:
            for chunk in content_blocks(message["content"]):
                if chunk["type"] == "video_url":
                    url = chunk["video_url"]["url"]
                    if (
                        url.startswith("data:")
                        and len(url.partition(",")[2]) >= 10 * 1024 * 1024
                    ):
                        raise ValueError(
                            "DashScope inline video base64 must be smaller than "
                            "10 MiB; use a smaller prepared video or sampled images"
                        )
                elif chunk["type"] == "input_audio":
                    media_type, data = inline_audio(chunk)
                    chunk["input_audio"]["data"] = f"data:{media_type};base64,{data}"
    else:
        for message in messages:
            for chunk in content_blocks(message["content"]):
                if chunk["type"] == "input_audio":
                    inline_audio(chunk)
    return messages


def build_payload(client, messages):
    if client.api_format == "chat":
        result = {
            "model": client.model,
            "messages": _chat_messages(client, messages),
        }
        if "max_completion_tokens" not in client.options:
            result["max_tokens"] = client.max_tokens
    else:
        result = {
            "model": client.model,
            "input": [
                {**message, "content": _responses_content(message["content"])}
                for message in messages
            ],
            "max_output_tokens": client.max_tokens,
        }
    if client.temperature is not None:
        result["temperature"] = client.temperature
    result.update(client.options)
    return result


def endpoint(client):
    base = client.base_url.rstrip("/")
    suffix = "/chat/completions" if client.api_format == "chat" else "/responses"
    if base.endswith(suffix):
        return base
    if not parse.urlparse(base).path:
        base += "/v1"
    return base + suffix


def headers(client):
    result = {"Content-Type": "application/json"}
    if client.api_key:
        result["Authorization"] = "Bearer " + client.api_key
    return result


def parse_response(client, raw):
    if client.api_format == "chat":
        choices = raw.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError("chat service returned no answer message")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError("model output exceeded the token limit")
        message = choice["message"]
        content = message.get("content") or message.get("refusal") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if part.get("type") == "text"
            )
        finish = choice.get("finish_reason")
    else:
        if raw.get("status") in {
            "incomplete",
            "failed",
            "cancelled",
            "queued",
            "in_progress",
        }:
            raise RuntimeError(f"Responses request was {raw['status']}")
        messages = [
            item for item in raw.get("output", []) if item.get("type") == "message"
        ]
        if not messages:
            raise RuntimeError("Responses service returned no answer message")
        content = "".join(
            part.get("text", part.get("refusal", ""))
            for message in messages
            for part in message.get("content", [])
            if part.get("type") in {"output_text", "refusal"}
        )
        finish = raw.get("status")
    if not isinstance(content, str):
        raise RuntimeError("model answer must be text")
    return {
        "content": content,
        "raw_response": raw,
        "finish_reason": finish,
        "usage": raw.get("usage", raw.get("usageMetadata")),
    }
