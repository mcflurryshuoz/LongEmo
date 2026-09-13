"""Exercise provider wire formats with synthetic media and a local HTTP server."""

import copy
import json
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from evaluation.clients import Client, REQUEST_BODY_LIMITS


IMAGE_URL = "data:image/jpeg;base64,c3ludGhldGljLWltYWdl"
USER = [{"role": "user", "content": "What emotion is shown?"}]


@contextmanager
def native_service(response):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def anthropic_answer(stop_reason="end_turn", text="happiness"):
    return {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "PRIVATE_REASONING"},
            {"type": "redacted_thinking", "data": "PRIVATE_REDACTED"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


class ClientTest(unittest.TestCase):
    def test_mismatched_endpoint_is_rejected_without_guessing_another_protocol(self):
        for suffix, api in (
            ("/messages", "chat"),
            ("/responses", "anthropic"),
            ("/chat/completions", "responses"),
            ("/models/example:generateContent", "chat"),
        ):
            with self.subTest(suffix=suffix, api=api):
                with self.assertRaisesRegex(ValueError, "endpoint requires"):
                    Client("test", "https://gateway.example" + suffix, api)

    def test_actual_encoded_request_size_is_checked_before_http(self):
        raw = {"choices": [{"message": {"content": "joy"}, "finish_reason": "stop"}]}
        messages = [{"role": "user", "content": '情绪\n"what"'}]
        with native_service(raw) as (url, calls):
            client = Client("synthetic", url + "/v1")
            size = len(json.dumps(client.payload(messages), allow_nan=False).encode())
            with patch.dict(REQUEST_BODY_LIMITS, {"127.0.0.1": size - 1}):
                with self.assertRaisesRegex(
                    ValueError, "encoded request body.*max-frames"
                ):
                    client.generate(messages)
                self.assertEqual(calls, [])
            with patch.dict(REQUEST_BODY_LIMITS, {"127.0.0.1": size}):
                self.assertEqual(client.generate(messages)["content"], "joy")
                self.assertEqual(calls[0]["body"]["messages"], messages)

    def test_chat_reasoning_token_limit_replaces_legacy_field(self):
        raw = {"choices": [{"message": {"content": "joy"}, "finish_reason": "stop"}]}
        with native_service(raw) as (url, calls):
            result = Client(
                "gpt-test",
                url + "/v1",
                max_tokens=4096,
                options={"max_completion_tokens": 2000, "reasoning_effort": "low"},
            ).generate(USER)
            self.assertEqual(result["content"], "joy")
            self.assertEqual(calls[0]["path"], "/v1/chat/completions")
            body = calls[0]["body"]
            self.assertEqual(body["max_completion_tokens"], 2000)
            self.assertNotIn("max_tokens", body)
            self.assertEqual(body["reasoning_effort"], "low")

    def test_dashscope_inline_video_limit_is_specific_to_provider(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "data:video/mp4;base64," + "A" * (10 * 1024 * 1024)
                        },
                    }
                ],
            }
        ]
        client = Client(
            "qwen-test", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        with self.assertRaisesRegex(ValueError, "10 MiB.*sampled images"):
            client.payload(messages)
        self.assertEqual(
            Client("custom", "http://localhost/v1").payload(messages)["messages"],
            messages,
        )
        messages[0]["content"][0]["video_url"]["url"] = (
            "data:video/mp4;base64,c3ludGhldGlj"
        )
        self.assertEqual(client.payload(messages)["messages"], messages)

    def test_anthropic_http_contract_and_final_text(self):
        messages = [
            {"role": "system", "content": "Evaluate emotion."},
            {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
            {"role": "assistant", "content": "Ready."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1 at 0 seconds:"},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": "What emotion is shown?"},
                ],
            },
        ]
        original = copy.deepcopy(messages)
        raw = anthropic_answer()
        raw["content"].append({"type": "text", "text": "."})
        for suffix in ("", "/", "/v1", "/v1/"):
            with self.subTest(suffix=suffix), native_service(raw) as (url, calls):
                client = Client(
                    "claude-test",
                    url + suffix,
                    "anthropic",
                    "SYNTHETIC_KEY",
                    max_tokens=2000,
                    temperature=0.2,
                    options={"thinking": {"type": "adaptive"}},
                )
                result = client.generate(messages)
                self.assertEqual(result["content"], "happiness.")
                self.assertEqual(result["finish_reason"], "end_turn")
                self.assertEqual(result["usage"], raw["usage"])
                self.assertEqual(result["raw_response"], raw)
                self.assertEqual(calls[0]["path"], "/v1/messages")
                headers = {k.lower(): v for k, v in calls[0]["headers"].items()}
                self.assertEqual(headers["anthropic-version"], "2023-06-01")
                self.assertEqual(headers["x-api-key"], "SYNTHETIC_KEY")
                self.assertNotIn("authorization", headers)
                body = calls[0]["body"]
                self.assertEqual(body["model"], "claude-test")
                self.assertEqual(body["max_tokens"], 2000)
                self.assertEqual(body["temperature"], 0.2)
                self.assertEqual(body["thinking"], {"type": "adaptive"})
                self.assertEqual(
                    body["system"],
                    [
                        {"type": "text", "text": "Evaluate emotion."},
                        {"type": "text", "text": "Be concise."},
                    ],
                )
                self.assertEqual(body["messages"][0], messages[2])
                self.assertEqual(
                    body["messages"][1]["content"],
                    [
                        {"type": "text", "text": "Frame 1 at 0 seconds:"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "c3ludGhldGljLWltYWdl",
                            },
                        },
                        {"type": "text", "text": "What emotion is shown?"},
                    ],
                )
                self.assertNotIn("SYNTHETIC_KEY", json.dumps(client.configuration()))
                self.assertNotIn("SYNTHETIC_KEY", repr(client))
        self.assertEqual(messages, original)

    def test_anthropic_string_input_and_refusal_are_preserved(self):
        for finish in ("end_turn", "stop_sequence", "refusal"):
            raw = anthropic_answer(finish, "I cannot answer that request.")
            with self.subTest(finish=finish), native_service(raw) as (url, calls):
                result = Client("test", url, "anthropic").generate(USER)
                self.assertEqual(result["content"], "I cannot answer that request.")
                self.assertEqual(result["finish_reason"], finish)
                self.assertEqual(calls[0]["body"]["messages"], USER)
                self.assertNotIn("system", calls[0]["body"])

    def test_anthropic_rejects_unfinished_answers(self):
        for reason in (
            "max_tokens",
            "pause_turn",
            "tool_use",
            "model_context_window_exceeded",
            "unknown",
            None,
        ):
            with (
                self.subTest(reason=reason),
                native_service(anthropic_answer(reason, "partial")) as (url, _),
            ):
                with self.assertRaises(RuntimeError):
                    Client("test", url, "anthropic").generate(USER)

    def test_anthropic_requires_answer_text(self):
        raw = anthropic_answer()
        raw["content"] = [{"type": "thinking", "thinking": "PRIVATE_REASONING"}]
        with native_service(raw) as (url, _):
            with self.assertRaisesRegex(RuntimeError, "no answer text"):
                Client("test", url, "anthropic").generate(USER)

    def test_anthropic_rejects_nontext_system_and_unsupported_roles(self):
        client = Client("test", "https://api.anthropic.com", "anthropic")
        with self.assertRaisesRegex(ValueError, "system content.*text only"):
            client.payload(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "image_url", "image_url": {"url": IMAGE_URL}}
                        ],
                    },
                    *USER,
                ]
            )
        with self.assertRaisesRegex(ValueError, "user or assistant"):
            client.payload([{"role": "tool", "content": "tool result"}])

    def test_responses_maps_images_and_preserves_text_messages(self):
        messages = [
            {"role": "system", "content": "Evaluate emotion."},
            {"role": "assistant", "content": "Ready."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Frame 1:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_URL, "detail": "high"},
                    },
                    {"type": "text", "text": "Frame 2:"},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": "Question"},
                ],
            },
        ]
        original = copy.deepcopy(messages)
        raw = {
            "status": "completed",
            "output": [
                {"type": "reasoning", "summary": [{"text": "PRIVATE_REASONING"}]},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "joy"}],
                },
            ],
            "usage": {"input_tokens": 12, "output_tokens": 1},
        }
        with native_service(raw) as (url, calls):
            result = Client(
                "gpt-test",
                url + "/v1",
                "responses",
                "SYNTHETIC_KEY",
                max_tokens=500,
                options={"reasoning": {"effort": "low"}},
            ).generate(messages)
            self.assertEqual(result["content"], "joy")
            self.assertEqual(result["raw_response"], raw)
            self.assertEqual(result["usage"], raw["usage"])
            self.assertEqual(calls[0]["path"], "/v1/responses")
            headers = {k.lower(): v for k, v in calls[0]["headers"].items()}
            self.assertEqual(headers["authorization"], "Bearer SYNTHETIC_KEY")
            body = calls[0]["body"]
            self.assertEqual(body["input"][:2], messages[:2])
            self.assertEqual(body["max_output_tokens"], 500)
            self.assertEqual(body["reasoning"], {"effort": "low"})
            self.assertEqual(
                body["input"][2]["content"],
                [
                    {"type": "input_text", "text": "Frame 1:"},
                    {"type": "input_image", "image_url": IMAGE_URL, "detail": "high"},
                    {"type": "input_text", "text": "Frame 2:"},
                    {"type": "input_image", "image_url": IMAGE_URL},
                    {"type": "input_text", "text": "Question"},
                ],
            )
        self.assertEqual(messages, original)

    def test_responses_rejects_partial_outputs(self):
        for status in ("incomplete", "failed", "cancelled", "queued", "in_progress"):
            raw = {
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "partial"}],
                    }
                ],
            }
            with self.subTest(status=status), native_service(raw) as (url, _):
                with self.assertRaisesRegex(RuntimeError, status):
                    Client("test", url, "responses").generate(USER)

    def test_responses_and_anthropic_reject_native_video(self):
        for api in ("responses", "anthropic"):
            with self.subTest(api=api):
                client = Client("test", "http://localhost", api)
                with self.assertRaisesRegex(ValueError, "native video_url"):
                    client.payload(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "video_url",
                                        "video_url": {
                                            "url": "data:video/mp4;base64,c3ludGhldGlj"
                                        },
                                    }
                                ],
                            }
                        ]
                    )

    def test_native_image_adapters_reject_nonimage_data(self):
        for api in ("anthropic", "gemini"):
            for url in (
                "data:video/mp4;base64,c3ludGhldGlj",
                "data:image/jpeg;base64,",
                "data:image/jpeg,c3ludGhldGlj",
                "https://example.invalid/frame.jpg",
            ):
                with self.subTest(api=api, url=url):
                    client = Client("test", "http://localhost", api)
                    with self.assertRaisesRegex(ValueError, "base64.*image"):
                        client.payload(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image_url", "image_url": {"url": url}}
                                    ],
                                }
                            ]
                        )

    def test_gemini_accepts_sampled_images_and_combines_system_messages(self):
        raw = {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "parts": [
                            {"text": "PRIVATE_REASONING", "thought": True},
                            {"text": "joy"},
                        ]
                    },
                }
            ],
        }
        messages = [
            {"role": "system", "content": "Evaluate emotion."},
            {"role": "system", "content": "Be concise."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": "Question"},
                ],
            },
        ]
        with native_service(raw) as (url, calls):
            result = Client(
                "models/gemini-test", url, "gemini", "SYNTHETIC_KEY"
            ).generate(messages)
            self.assertEqual(result["content"], "joy")
            self.assertEqual(
                calls[0]["path"], "/v1beta/models/gemini-test:generateContent"
            )
            headers = {k.lower(): v for k, v in calls[0]["headers"].items()}
            self.assertEqual(headers["x-goog-api-key"], "SYNTHETIC_KEY")
            self.assertNotIn("authorization", headers)
            self.assertEqual(
                calls[0]["body"]["systemInstruction"]["parts"],
                [
                    {"text": "Evaluate emotion."},
                    {"text": "Be concise."},
                ],
            )
            self.assertEqual(
                calls[0]["body"]["contents"][0]["parts"],
                [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": "c3ludGhldGljLWltYWdl",
                        }
                    },
                    {"text": "Question"},
                ],
            )

    def test_requests_are_non_streaming_for_every_protocol(self):
        for api in ("chat", "responses", "gemini", "anthropic"):
            with self.subTest(api=api):
                client = Client("test", "http://localhost", api)
                self.assertNotIn("stream", client.payload(USER))
                self.assertNotIn("stream", client.configuration())


if __name__ == "__main__":
    unittest.main()
