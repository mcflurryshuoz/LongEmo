"""Canonical message content shared by the API protocol adapters."""


def inline_image(url, adapter):
    """Split the canonical image data URL for native multimodal APIs."""
    if isinstance(url, str):
        prefix, separator, data = url.partition(";base64,")
        media_type = prefix[len("data:"):] if prefix.startswith("data:") else prefix
        if (
            prefix.startswith("data:")
            and separator
            and data
            and media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}
        ):
            return media_type, data
    raise ValueError(f"{adapter} adapter expects a base64 JPEG, PNG, GIF or WebP image")


def inline_audio(chunk):
    """Canonical audio contains raw base64, never a data URL or a file path."""
    audio = chunk.get("input_audio", {})
    data, fmt = audio.get("data"), audio.get("format")
    if fmt not in {"wav", "mp3"} or not isinstance(data, str) or not data:
        raise ValueError("input_audio requires base64 data and wav or mp3 format")
    if data.startswith("data:"):
        raise ValueError("canonical input_audio data must be raw base64")
    return "audio/wav" if fmt == "wav" else "audio/mpeg", data


def content_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise ValueError("message content must be text or a list of content blocks")
    return content
