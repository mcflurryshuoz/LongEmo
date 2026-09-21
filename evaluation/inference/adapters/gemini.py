"""Google Gemini generateContent wire format."""

from urllib import parse

from .message_content import content_blocks, inline_audio, inline_image


def _message_parts(content):
    parts = []
    for chunk in content_blocks(content):
        if chunk["type"] == "text":
            parts.append({"text": chunk["text"]})
        elif chunk["type"] == "video_url":
            url = chunk["video_url"]["url"]
            if not url.startswith("data:video/mp4;base64,"):
                raise ValueError("native Gemini adapter expects encoded MP4 content")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "video/mp4",
                        "data": url.split(",", 1)[1],
                    },
                }
            )
        elif chunk["type"] == "image_url":
            media_type, data = inline_image(chunk["image_url"]["url"], "Gemini")
            parts.append({"inline_data": {"mime_type": media_type, "data": data}})
        elif chunk["type"] == "input_audio":
            media_type, data = inline_audio(chunk)
            parts.append({"inline_data": {"mime_type": media_type, "data": data}})
        else:
            raise ValueError("unsupported content block")
    return parts


def build_payload(client, messages):
    result = {
        "contents": [],
        "generationConfig": {"maxOutputTokens": client.max_tokens},
    }
    if client.temperature is not None:
        result["generationConfig"]["temperature"] = client.temperature
    for message in messages:
        parts = _message_parts(message["content"])
        if message["role"] == "system":
            result.setdefault("systemInstruction", {"parts": []})["parts"].extend(parts)
        else:
            result["contents"].append(
                {
                    "role": "model" if message["role"] == "assistant" else "user",
                    "parts": parts,
                }
            )
    for key, value in client.options.items():
        if key == "generationConfig":
            if not isinstance(value, dict):
                raise ValueError("generationConfig must be an object")
            result[key].update(value)
        else:
            result[key] = value
    return result


def endpoint(client):
    base = client.base_url.rstrip("/")
    if base.endswith(":generateContent"):
        model_path = parse.urlparse(base).path.rsplit("/models/", 1)[-1]
        endpoint_model = parse.unquote(model_path[:-len(":generateContent")] if model_path.endswith(":generateContent") else model_path)
        if endpoint_model != client.model[len("models/"):] if client.model.startswith("models/") else client.model:
            raise ValueError("Gemini endpoint model must match --model")
        return base
    if not base.endswith(("/v1beta", "/v1")):
        base += "/v1beta"
    return (
        base
        + "/models/"
        + parse.quote(client.model[len("models/"):] if client.model.startswith("models/") else client.model, safe="")
        + ":generateContent"
    )


def headers(client):
    result = {"Content-Type": "application/json"}
    if client.api_key:
        result["x-goog-api-key"] = client.api_key
    return result


def parse_response(client, raw):
    candidates = raw.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no answer candidate")
    finish = candidates[0].get("finishReason")
    if finish not in {None, "STOP"}:
        raise RuntimeError(f"Gemini generation stopped: {finish}")
    content = "".join(
        part.get("text", "")
        for part in candidates[0].get("content", {}).get("parts", [])
        if not part.get("thought")
    )
    if not isinstance(content, str):
        raise RuntimeError("model answer must be text")
    return {
        "content": content,
        "raw_response": raw,
        "finish_reason": finish,
        "usage": raw.get("usage", raw.get("usageMetadata")),
    }
