"""Validated HTTP transport shared by the API protocol adapters."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from urllib import request, error, parse

from .inference.adapters import API_ADAPTERS


# Published service limits; validate the encoded body, not a pixel estimate.
# Ark states 64 MB; use decimal bytes conservatively. DeepSeek states 48 MiB.
REQUEST_BODY_LIMITS = {
    "ark.cn-beijing.volces.com": 64_000_000,
    "api.deepseek.com": 48 * 1024 * 1024,
}


class ServiceError(RuntimeError):
    """Safe service metadata for bounded retries; never retain provider error text."""
    def __init__(self, status_code, code=None):
        self.status_code = status_code
        value = str(code) if code is not None else ""
        self.code = value if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) else None
        self.retryable = status_code in (429, 500, 502, 503, 504) and self.code != 'content_filter'
        super().__init__(f"model service returned HTTP {status_code}" + (f" ({self.code})" if self.code else ""))


@dataclass
class Client:
    model: str
    base_url: str
    api_format: str = "chat"
    api_key: str = field(default="", repr=False)
    timeout: float = 180
    max_tokens: int = 16384
    temperature: float | None = 0.0
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.api_format not in API_ADAPTERS:
            raise ValueError("unsupported API format")
        if self.max_tokens <= 0 or self.timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        if not isinstance(self.options, dict):
            raise ValueError("request options must be a JSON object")
        parsed = parse.urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base URL must be an http(s) service URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("supply credentials separately from the base URL")
        # An explicitly named endpoint must agree with the chosen model adapter.
        path = parsed.path.rstrip("/")
        for suffix, expected in (
            ("/chat/completions", "chat"),
            ("/responses", "responses"),
            ("/messages", "anthropic"),
            (":generateContent", "gemini"),
        ):
            if path.endswith(suffix) and self.api_format != expected:
                raise ValueError(
                    f"the {suffix} endpoint requires the {expected} API format"
                )
        if path.endswith(":streamGenerateContent"):
            raise ValueError(
                "use a generateContent endpoint for non-streaming Gemini requests"
            )

    def configuration(self):
        value = {
            name: getattr(self, name)
            for name in (
                "model",
                "base_url",
                "api_format",
                "max_tokens",
                "temperature",
                "options",
                "timeout",
            )
        }
        from . import azure_transport
        if azure_transport.is_azure(self.base_url):
            value['transport'] = azure_transport.configuration()
        return value

    def payload(self, messages):
        return API_ADAPTERS[self.api_format].build_payload(self, messages)

    def _url(self):
        return API_ADAPTERS[self.api_format].endpoint(self)

    def generate(self, messages):
        payload = self.payload(messages)
        from . import azure_transport
        if azure_transport.is_azure(self.base_url):
            raw = azure_transport.generate(self, payload)
            return API_ADAPTERS[self.api_format].parse_response(self, raw)
        data = json.dumps(payload, allow_nan=False).encode()
        body_limit = REQUEST_BODY_LIMITS.get(parse.urlparse(self.base_url).hostname)
        if body_limit is not None and len(data) > body_limit:
            raise ValueError(
                f"encoded request body is {len(data)} bytes; this service permits "
                f"at most {body_limit} bytes. Reduce --max-frames or "
                "--frame-max-pixels for frame input, or reduce the input size"
            )
        adapter = API_ADAPTERS[self.api_format]
        req = request.Request(
            self._url(),
            data=data,
            headers=adapter.headers(self),
            method="POST",
        )
        print(
            f"Sending request: {len(data) / (1024 * 1024):.2f} MiB; "
            f"timeout: {self.timeout:g}s",
            flush=True,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = json.load(response)
        except error.HTTPError as exc:
            # Error messages can echo prompts or credentials; retain only a safe code.
            try:
                detail = json.load(exc).get('error', {})
                code = detail.get('code', detail.get('status')) if isinstance(detail, dict) else None
            except (ValueError, AttributeError):
                code = None
            raise ServiceError(exc.code, code) from None
        if self.api_format == 'gemini':
            blocked = (raw.get('promptFeedback') or {}).get('blockReason')
            finish = (raw.get('candidates') or [{}])[0].get('finishReason')
            if blocked or finish in {'SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'IMAGE_SAFETY'}:
                raise ServiceError(200, 'content_filter')
        elif self.api_format == 'chat':
            if any(c.get('finish_reason') == 'content_filter' for c in (raw.get('choices') or [])):
                raise ServiceError(200, 'content_filter')
        return adapter.parse_response(self, raw)
