"""Audio capability routing and HTTP contracts, without datasets or paid APIs."""

import base64
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from evaluation import io_utils
from evaluation.clients import Client
from evaluation.inference import run
from evaluation.inference.runner import init_client
from helpers import question, service


LOCAL = "http://localhost:8000/v1"
DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GOOGLE = "https://generativelanguage.googleapis.com/v1beta"
QWEN_LOCAL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
IMAGE_DATA = base64.b64encode(b"synthetic-frame").decode("ascii")
VIDEO_DATA = base64.b64encode(b"synthetic-video").decode("ascii")


def audio_fixture():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 32)
    raw = buffer.getvalue()
    return (
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(raw).decode("ascii"),
                "format": "wav",
            },
        },
        {
            "audio": "separate_audio",
            "sample_rate": 16000,
            "channels": 1,
            "duration_seconds": 0.002,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "video_start_seconds": 0.0,
        },
    )


AUDIO, AUDIO_META = audio_fixture()


def frame_fixture(*_args, **_kwargs):
    blocks = []
    for timestamp in (0.0, 1.0):
        blocks.extend(
            [
                {"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + IMAGE_DATA},
                },
            ]
        )
    return blocks, {
        "timestamps_seconds": [0.0, 1.0],
        "num_frames": 2,
        "duration_seconds": 2.0,
    }


def video_fixture(*_args, **_kwargs):
    return {
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64," + VIDEO_DATA},
    }


def invoke(arguments):
    with contextlib.redirect_stdout(io.StringIO()):
        return run.run(run.parser().parse_args(arguments))


def endpoint(base_url, api):
    """Exercise each wire format through an explicit service address."""
    base_url = base_url.rstrip("/")
    if api == "gemini":
        return (
            base_url
            if base_url.endswith("/v1beta")
            else base_url.removesuffix("/v1") + "/v1beta"
        )
    if api == "chat" and base_url == GOOGLE:
        return base_url + "/openai"
    if api == "chat":
        return base_url + "/chat/completions"
    if api in {"responses", "anthropic"}:
        return base_url + ("/responses" if api == "responses" else "/messages")
    return base_url


class AudioRoutingTest(unittest.TestCase):
    def resolved(self, model, api, base_url=LOCAL, no_audio=False):
        args = run.parser().parse_args(
            [
                "--data-path",
                "unused",
                "--model",
                model,
                "--base-url",
                endpoint(base_url, api),
            ]
            + (["--no-audio"] if no_audio else [])
        )
        init_client(args)
        return args

    def prepared_arguments(self, root, model, api, base_url, **options):
        (root / "q.json").write_text(json.dumps(question()))
        (root / "videos").mkdir(exist_ok=True)
        (root / "videos/V1.mp4").write_bytes(b"immutable synthetic source")
        arguments = [
            "--data-path",
            str(root / "q.json"),
            "--videos-dir",
            str(root / "videos"),
            "--model",
            model,
            "--base-url",
            endpoint(base_url, api),
            "--output-dir",
            str(root / "out"),
            "--tries",
            "1",
            "--max-frames",
            "2",
            "--prompt-mode",
            "system",
        ]
        for name, value in options.items():
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    arguments.append(flag)
            else:
                arguments += [flag, value]
        return arguments

    def test_auto_policy_is_specific_to_model_api_and_visual_input(self):
        cases = [
            ("gemini-3-flash-preview", "gemini", GOOGLE, "frames", (False, True)),
            ("gemini-3-flash-preview", "gemini", GOOGLE, "native", (True, False)),
            ("gemini-3-flash-preview", "chat", GOOGLE, "frames", (False, False)),
            (QWEN_LOCAL, "chat", LOCAL, "frames", (False, True)),
            (QWEN_LOCAL, "chat", LOCAL, "native", (False, True)),
            ("qwen3.5-omni-plus", "chat", DASHSCOPE, "frames", (False, True)),
            ("qwen3.5-omni-plus", "chat", DASHSCOPE, "native", (True, False)),
            ("qwen3-omni-flash", "chat", DASHSCOPE, "frames", (False, False)),
            ("qwen3-omni-flash", "chat", DASHSCOPE, "native", (True, False)),
            ("qwen3-omni-flash", "chat", LOCAL, "frames", (False, False)),
            ("Qwen/Qwen2.5-Omni-7B", "chat", LOCAL, "frames", (False, False)),
            ("Qwen/Qwen3-VL-8B-Instruct", "chat", LOCAL, "frames", (False, False)),
            ("Qwen/Qwen3-VL-8B-Instruct", "chat", LOCAL, "native", (False, False)),
            ("gpt-4.1", "chat", LOCAL, "frames", (False, False)),
            ("gpt-4.1", "responses", LOCAL, "frames", (False, False)),
            ("claude-sonnet-4-5", "chat", LOCAL, "frames", (False, False)),
            ("claude-sonnet-4-5", "anthropic", LOCAL, "frames", (False, False)),
            ("custom-model", "chat", LOCAL, "frames", (False, False)),
            ("custom-model", "chat", LOCAL, "native", (True, False)),
        ]
        for model, api, url, visual, expected in cases:
            with self.subTest(model=model, api=api, url=url, visual=visual):
                args = self.resolved(model, api, url)
                self.assertEqual(run.select_audio_input(args, visual), expected)
                args.no_audio = True
                self.assertEqual(run.select_audio_input(args, visual), (False, False))

    def test_auto_unsupported_audio_is_omitted_before_generation(self):
        cases = [
            ("gpt-4.1", "responses", LOCAL, "frames"),
            ("gpt-4.1", "chat", LOCAL, "frames"),
            ("claude-sonnet-4-5", "anthropic", LOCAL, "frames"),
            ("claude-sonnet-4-5", "chat", LOCAL, "frames"),
            ("Qwen/Qwen3-VL-8B-Instruct", "chat", LOCAL, "frames"),
            ("Qwen/Qwen3-VL-8B-Instruct", "chat", LOCAL, "native"),
            ("qwen3-omni-flash", "chat", DASHSCOPE, "frames"),
            ("qwen3-omni-flash", "chat", LOCAL, "frames"),
            ("custom-model", "responses", LOCAL, "frames"),
            ("custom-model", "chat", LOCAL, "frames"),
            ("gemini-3-flash-preview", "chat", GOOGLE, "frames"),
        ]
        for model, api, base_url, visual in cases:
            with (
                self.subTest(model=model, api=api, visual=visual),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: "happiness") as (url, calls),
                patch.object(run, "sample_video_frames", side_effect=frame_fixture) as frames,
                patch.object(run, "encode_video", side_effect=video_fixture) as video,
                patch.object(run, "extract_audio") as audio,
            ):
                root = Path(td)
                args = self.prepared_arguments(
                    root,
                    model,
                    api,
                    url if base_url == LOCAL else base_url,
                    force_frames=visual == "frames",
                )
                path = {"responses": "/responses", "anthropic": "/messages"}.get(
                    api, "/v1/chat/completions"
                )
                with patch.object(Client, "_url", return_value=url + path):
                    self.assertEqual(invoke(args), 0)
                audio.assert_not_called()
                if visual == "native":
                    video.assert_called_once()
                    self.assertEqual(video.call_args.kwargs, {"include_audio": False})
                    frames.assert_not_called()
                else:
                    frames.assert_called_once()
                    video.assert_not_called()
                self.assertEqual(len(calls), 1)
                self.assertNotIn("input_audio", json.dumps(calls[0][1]))
                result = io_utils.load_records(root / "out/predictions.jsonl")[0]
                self.assertEqual(
                    result["media"]["audio"],
                    "removed" if visual == "native" else "not_sent",
                )
                self.assertFalse(result["media"]["no_audio"])

    def test_text_mode_never_extracts_audio(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "happiness") as (url, calls),
            patch.object(run, "extract_audio") as audio,
            patch.object(run, "encode_video") as video,
            patch.object(run, "sample_video_frames") as frames,
        ):
            root = Path(td)
            args = self.prepared_arguments(
                root, QWEN_LOCAL, "chat", url, modality="text"
            )
            self.assertEqual(invoke(args), 0)
            payload = json.dumps(calls[0][1])
            self.assertNotIn("input_audio", payload)
            self.assertIn("First subtitle.", payload)
            self.assertEqual(invoke(args + ["--no-audio", "--force"]), 0)
            self.assertEqual(len(calls), 2)
            for operation in (audio, video, frames):
                operation.assert_not_called()

    def test_frames_and_audio_reach_provider_before_subtitles_and_question(self):
        cases = [
            ("gemini-3-flash-preview", "gemini", "local"),
            (QWEN_LOCAL, "chat", "local"),
            ("qwen3.5-omni-plus", "chat", DASHSCOPE),
        ]
        for model, api, endpoint in cases:
            with (
                self.subTest(model=model, api=api),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: "happiness") as (url, calls),
                patch.object(run, "sample_video_frames", side_effect=frame_fixture) as frames,
                patch.object(
                    run, "extract_audio", return_value=(AUDIO, AUDIO_META)
                ) as audio,
                patch.object(run, "encode_video") as video,
            ):
                root = Path(td)
                args = self.prepared_arguments(
                    root,
                    model,
                    api,
                    url if endpoint == "local" else endpoint,
                    force_frames=True,
                ) + ["--with-subtitle"]
                originals = {
                    p: p.read_bytes() for p in (root / "q.json", root / "videos/V1.mp4")
                }
                # Keep provider capability/serialization detection, redirect only transport.
                local_path = (
                    "/v1beta/models/test:generateContent"
                    if api == "gemini"
                    else "/v1/chat/completions"
                )
                with patch.object(Client, "_url", return_value=url + local_path):
                    self.assertEqual(invoke(args), 0)
                self.assertEqual(len(calls), 1)
                frames.assert_called_once()
                video.assert_not_called()
                self.assertEqual(audio.call_args.kwargs, {"required": False})
                body = calls[0][1]
                self.assertNotIn("stream", body)
                if api == "gemini":
                    parts = body["contents"][0]["parts"]
                    image_indexes = [
                        i
                        for i, p in enumerate(parts)
                        if p.get("inline_data", {}).get("mime_type") == "image/jpeg"
                    ]
                    audio_indexes = [
                        i
                        for i, p in enumerate(parts)
                        if p.get("inline_data", {}).get("mime_type") == "audio/wav"
                    ]
                    sent_audio = parts[audio_indexes[0]]["inline_data"]["data"]
                else:
                    parts = body["messages"][1]["content"]
                    image_indexes = [
                        i for i, p in enumerate(parts) if p["type"] == "image_url"
                    ]
                    audio_indexes = [
                        i for i, p in enumerate(parts) if p["type"] == "input_audio"
                    ]
                    sent_audio = parts[audio_indexes[0]]["input_audio"]["data"]
                    if model.startswith("Qwen/") or model.startswith("qwen"):
                        self.assertEqual(body["modalities"], ["text"])
                expected_audio = AUDIO["input_audio"]["data"]
                if endpoint == DASHSCOPE:
                    expected_audio = "data:audio/wav;base64," + expected_audio
                self.assertEqual(sent_audio, expected_audio)
                self.assertEqual(len(image_indexes), 2)
                self.assertEqual(len(audio_indexes), 1)
                self.assertLess(max(image_indexes), audio_indexes[0])
                prompt_index = next(
                    i
                    for i, p in enumerate(parts)
                    if "First subtitle." in p.get("text", "")
                )
                self.assertLess(audio_indexes[0], prompt_index)
                prompt = parts[prompt_index]["text"]
                self.assertLess(
                    prompt.index("First subtitle."),
                    prompt.index(question()["question"]),
                )
                self.assertTrue(any("time zero" in p.get("text", "") for p in parts))
                self.assertNotIn("SYNTHETIC_SOURCE", json.dumps(body))
                prediction = io_utils.load_records(root / "out/predictions.jsonl")[0]
                self.assertEqual(prediction["prediction"], ["happiness"])
                self.assertEqual(prediction["media"]["audio"], "separate_audio")
                saved = prediction["messages"][1]["content"]
                refs = [p for p in saved if p["type"] == "audio_reference"]
                self.assertEqual(len(refs), 1)
                self.assertEqual(
                    refs[0]["path"], str((root / "videos/V1.mp4").resolve())
                )
                self.assertEqual(
                    refs[0]["sha256"],
                    hashlib.sha256(originals[root / "videos/V1.mp4"]).hexdigest(),
                )
                self.assertEqual(refs[0]["audio_details"], AUDIO_META)
                self.assertEqual(
                    [
                        p["timestamp_seconds"]
                        for p in saved
                        if p["type"] == "frame_reference"
                    ],
                    [0.0, 1.0],
                )
                for secret in ("base64,", AUDIO["input_audio"]["data"], IMAGE_DATA):
                    self.assertNotIn(secret, json.dumps(prediction))
                self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_native_video_uses_one_audio_path_and_no_audio_removes_audio(self):
        cases = [
            (QWEN_LOCAL, "chat", "local", False, False, True),
            ("gemini-3-flash-preview", "gemini", "local", False, True, False),
            ("qwen3-omni-flash", "chat", DASHSCOPE, False, True, False),
            ("custom-model", "chat", "local", False, True, False),
            ("gemini-3-flash-preview", "gemini", "local", True, False, False),
            (QWEN_LOCAL, "chat", "local", True, False, False),
            ("custom-model", "chat", "local", True, False, False),
            ("Qwen/Qwen3-VL-8B-Instruct", "chat", "local", False, False, False),
        ]
        for model, api, endpoint, no_audio, include_audio, separate in cases:
            with (
                self.subTest(model=model, no_audio=no_audio),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: "happiness") as (url, calls),
                patch.object(run, "encode_video", side_effect=video_fixture) as video,
                patch.object(
                    run, "extract_audio", return_value=(AUDIO, AUDIO_META)
                ) as audio,
                patch.object(run, "sample_video_frames") as frames,
            ):
                root = Path(td)
                args = self.prepared_arguments(
                    root,
                    model,
                    api,
                    url if endpoint == "local" else endpoint,
                    no_audio=no_audio,
                )
                originals = {
                    p: p.read_bytes() for p in (root / "q.json", root / "videos/V1.mp4")
                }
                path = (
                    "/v1beta/models/test:generateContent"
                    if api == "gemini"
                    else "/v1/chat/completions"
                )
                with patch.object(Client, "_url", return_value=url + path):
                    self.assertEqual(invoke(args), 0)
                video.assert_called_once()
                self.assertEqual(
                    video.call_args.kwargs, {"include_audio": include_audio}
                )
                frames.assert_not_called()
                if separate:
                    audio.assert_called_once()
                    self.assertEqual(audio.call_args.kwargs, {"required": False})
                else:
                    audio.assert_not_called()
                self.assertEqual(len(calls), 1)
                body = calls[0][1]
                if api == "chat":
                    kinds = [p["type"] for p in body["messages"][1]["content"]]
                    self.assertEqual(kinds.count("video_url"), 1)
                    self.assertEqual(kinds.count("input_audio"), int(separate))
                else:
                    types = [
                        p["inline_data"]["mime_type"]
                        for p in body["contents"][0]["parts"]
                        if "inline_data" in p
                    ]
                    self.assertEqual(types, ["video/mp4"])
                result = io_utils.load_records(root / "out/predictions.jsonl")[0]
                self.assertEqual(result["media"]["no_audio"], no_audio)
                if no_audio:
                    self.assertEqual(result["media"]["audio"], "removed")
                self.assertEqual({p: p.read_bytes() for p in originals}, originals)

    def test_missing_audio_records_fallback_and_still_calls_model(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "happiness") as (url, calls),
            patch.object(run, "sample_video_frames", side_effect=frame_fixture),
            patch.object(
                run, "extract_audio", return_value=(None, {"audio": "no_audio_track"})
            ) as audio,
        ):
            root = Path(td)
            args = self.prepared_arguments(
                root, QWEN_LOCAL, "chat", url, force_frames=True
            )
            self.assertEqual(invoke(args), 0)
            result = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertEqual(result["media"]["audio"], "no_audio_track")
            audio.assert_called_once()
            self.assertEqual(audio.call_args.kwargs, {"required": False})
            self.assertNotIn("input_audio", json.dumps(calls[0][1]))
            self.assertNotIn("audio_reference", json.dumps(result))
            self.assertEqual(len(calls), 1)

    def test_existing_output_is_reused_until_force(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "happiness") as (url, calls),
            patch.object(run, "sample_video_frames", side_effect=frame_fixture),
            patch.object(run, "extract_audio", return_value=(AUDIO, AUDIO_META)),
        ):
            root = Path(td)
            args = self.prepared_arguments(
                root, QWEN_LOCAL, "chat", url, force_frames=True
            )
            self.assertEqual(invoke(args), 0)
            first = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertEqual(invoke(args), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(invoke(args + ["--no-audio"]), 0)
            second = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertEqual(first["media"], second["media"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(invoke(args + ["--no-audio", "--force"]), 0)
            second = io_utils.load_records(root / "out/predictions.jsonl")[0]
            self.assertTrue(second["media"]["no_audio"])
            self.assertEqual(second["media"]["audio"], "not_sent")
            self.assertNotIn("input_audio", json.dumps(calls[1][1]))
            self.assertEqual(len(calls), 2)


class AudioSerializationTest(unittest.TestCase):
    def test_payload_translation_never_mutates_canonical_audio(self):
        for fmt, mime in [("wav", "audio/wav"), ("mp3", "audio/mpeg")]:
            with self.subTest(format=fmt):
                messages = [{"role": "user", "content": [copy.deepcopy(AUDIO)]}]
                messages[0]["content"][0]["input_audio"]["format"] = fmt
                original = copy.deepcopy(messages)
                for api, endpoint in [
                    ("chat", LOCAL),
                    ("chat", DASHSCOPE),
                    ("gemini", GOOGLE),
                ]:
                    client = Client("test", endpoint, api)
                    for _ in range(2):
                        payload = client.payload(messages)
                        if api == "gemini":
                            self.assertEqual(
                                payload["contents"][0]["parts"],
                                [
                                    {
                                        "inline_data": {
                                            "mime_type": mime,
                                            "data": AUDIO["input_audio"]["data"],
                                        }
                                    }
                                ],
                            )
                        else:
                            data = payload["messages"][0]["content"][0]["input_audio"][
                                "data"
                            ]
                            expected = AUDIO["input_audio"]["data"]
                            if endpoint == DASHSCOPE:
                                expected = f"data:{mime};base64," + expected
                            self.assertEqual(data, expected)
                        self.assertEqual(messages, original)

    def test_unsupported_protocols_reject_audio_before_http(self):
        messages = [{"role": "user", "content": [AUDIO]}]
        for api in ("responses", "anthropic"):
            with self.subTest(api=api), service(lambda *_: "happiness") as (url, calls):
                with self.assertRaisesRegex(ValueError, "unsupported.*input_audio"):
                    Client("test", url, api).generate(messages)
                self.assertEqual(calls, [])

    def test_malformed_canonical_audio_rejects_before_http(self):
        for audio in (
            {"data": "data:audio/wav;base64,AAAA", "format": "wav"},
            {"data": "AAAA", "format": "flac"},
            {"data": "", "format": "wav"},
        ):
            for api, endpoint in [
                ("chat", LOCAL),
                ("chat", DASHSCOPE),
                ("gemini", GOOGLE),
            ]:
                with self.subTest(audio=audio, api=api, endpoint=endpoint):
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "input_audio", "input_audio": audio}],
                        }
                    ]
                    with self.assertRaisesRegex(ValueError, "input_audio"):
                        Client("test", endpoint, api).payload(messages)


if __name__ == "__main__":
    unittest.main()
