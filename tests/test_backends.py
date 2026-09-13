"""Model-family defaults and complete public inference/judge routes."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import eval as eval_module, io_utils
from evaluation.inference import run
from evaluation.inference.runner import detect_model_family, init_client
from helpers import question, prediction, service


def invoke(module, arguments):
    with contextlib.redirect_stdout(io.StringIO()):
        return module.run(module.parser().parse_args(arguments))


class BackendTest(unittest.TestCase):
    def test_model_adapters_configure_protocol_and_token_options(self):
        cases = [
            (
                "gemini-3-flash-preview",
                "https://generativelanguage.googleapis.com/v1beta/openai",
                "chat",
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            ),
            (
                "gemini-3-flash-preview",
                "https://gateway.example/v1beta",
                "gemini",
                "https://gateway.example/v1beta/models/gemini-3-flash-preview:generateContent",
            ),
            (
                "gemini-3-flash-preview",
                "https://gateway.example/v1beta/openai",
                "chat",
                "https://gateway.example/v1beta/openai/chat/completions",
            ),
            (
                "gemini-3-flash-preview",
                "https://gateway.example/compatible-mode/v1beta",
                "chat",
                "https://gateway.example/compatible-mode/v1beta/chat/completions",
            ),
            (
                "claude-sonnet-4-5",
                "https://gateway.example/anthropic",
                "anthropic",
                "https://gateway.example/anthropic/v1/messages",
            ),
            (
                "claude-sonnet-4-5",
                "https://api.openai.com/custom/messages",
                "anthropic",
                "https://api.openai.com/custom/messages",
            ),
            (
                "custom-model",
                "https://api.anthropic.com/custom/chat/completions",
                "chat",
                "https://api.anthropic.com/custom/chat/completions",
            ),
            (
                "gpt-4.1",
                "https://generativelanguage.googleapis.com/custom/responses",
                "responses",
                "https://generativelanguage.googleapis.com/custom/responses",
            ),
            (
                "gemini-3-flash-preview",
                "https://api.openai.com/custom/models/gemini-3-flash-preview:generateContent",
                "gemini",
                "https://api.openai.com/custom/models/gemini-3-flash-preview:generateContent",
            ),
            (
                "custom-model",
                "https://api.anthropic.com/custom/openai",
                "chat",
                "https://api.anthropic.com/custom/openai/chat/completions",
            ),
            (
                "custom-model",
                "https://generativelanguage.googleapis.com/v1beta",
                "gemini",
                "https://generativelanguage.googleapis.com/v1beta/models/custom-model:generateContent",
            ),
            (
                "gpt-4.1",
                "https://api.openai.com/openai",
                "chat",
                "https://api.openai.com/openai/chat/completions",
            ),
            (
                "gpt-4.1",
                "https://api.openai.com/compatible-mode/v1",
                "chat",
                "https://api.openai.com/compatible-mode/v1/chat/completions",
            ),
            (
                "claude-sonnet-4-5",
                "https://api.anthropic.com/v1/chat/completions",
                "chat",
                "https://api.anthropic.com/v1/chat/completions",
            ),
            (
                "claude-sonnet-4-5",
                "https://api.anthropic.com/openai",
                "chat",
                "https://api.anthropic.com/openai/chat/completions",
            ),
            (
                "claude-sonnet-4-5",
                "https://gateway.example/service/openai",
                "chat",
                "https://gateway.example/service/openai/chat/completions",
            ),
        ]
        for model in (
            "gpt-4.1",
            "claude-sonnet-4-5",
            "gemini-3-flash-preview",
            "MiniMax-M2.7",
            "kimi-k2.5",
            "glm-4.6v",
            "doubao-seed-2-0-pro-260215",
            "InternVL3-8B",
            "deepseek-v4-pro",
        ):
            cases.append(
                (
                    model,
                    "https://gateway.example/v1",
                    "chat",
                    "https://gateway.example/v1/chat/completions",
                )
            )
        for model, host in (
            ("MiniMax-M2.7", "api.minimax.io"),
            ("MiniMax-M2.7", "api.minimaxi.com"),
            ("kimi-k2.5", "api.moonshot.ai"),
            ("moonshot-v1-8k", "api.moonshot.cn"),
        ):
            for path, protocol, suffix in (
                ("", "chat", "/v1/chat/completions"),
                ("/v1", "chat", "/v1/chat/completions"),
                ("/anthropic", "anthropic", "/anthropic/v1/messages"),
            ):
                cases.append(
                    (
                        model,
                        f"https://{host}{path}",
                        protocol,
                        f"https://{host}{suffix}",
                    )
                )
        for host in (
            "api.openai.com",
            "api.anthropic.com",
            "generativelanguage.googleapis.com",
        ):
            cases.append(
                (
                    "custom-model",
                    f"https://{host}",
                    "chat",
                    f"https://{host}/v1/chat/completions",
                )
            )
        for model, base_url, protocol, final_url in cases:
            with (
                self.subTest(model=model, base_url=base_url),
                patch.dict(os.environ, {}, clear=True),
            ):
                args = run.parser().parse_args(
                    ["--data-path", "unused", "--model", model, "--base-url", base_url]
                )
                client = init_client(args)
                self.assertEqual(args.base_url, base_url)
                self.assertEqual(client.api_format, protocol)
                self.assertEqual(client._url(), final_url)
                if model == "custom-model":
                    self.assertEqual(args.model_family, "other")

        with patch.dict(os.environ, {}, clear=True):
            with tempfile.TemporaryDirectory() as td:
                options = Path(td) / "options.json"
                options.write_text('{"max_tokens":1024}')
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        "gpt-4.1",
                        "--base-url",
                        "https://gateway.example/v1",
                        "--config",
                        str(options),
                    ]
                )
                payload = init_client(args).payload(
                    [{"role": "user", "content": "Question"}]
                )
                self.assertEqual(payload["max_completion_tokens"], 1024)
                self.assertNotIn("max_tokens", payload)

    def test_gateway_routes_depend_on_endpoint_path_not_host(self):
        for model, model_family, path, protocol in (
            (
                "claude-sonnet-4-5",
                "claude",
                "/service/alpha/v1/messages",
                "anthropic",
            ),
            (
                "gemini-3-flash-preview",
                "gemini",
                "/service/beta/v1beta/models/gemini-3-flash-preview:generateContent",
                "gemini",
            ),
            (
                "gpt-4.1",
                "gpt",
                "/service/gamma/responses",
                "responses",
            ),
        ):
            for host in ("gateway.example", "renamed-gateway.example"):
                with (
                    self.subTest(model=model, host=host),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    url = "https://" + host + path
                    args = run.parser().parse_args(
                        ["--data-path", "unused", "--model", model, "--base-url", url]
                    )
                    client = init_client(args)
                    self.assertEqual(args.model_family, model_family)
                    self.assertEqual(args.base_url, url)
                    self.assertEqual(client.api_format, protocol)
                    self.assertEqual(client._url(), url)

        for model in (
            "gpt-4.1",
            "claude-sonnet-4-5",
            "gemini-3-flash-preview",
            "glm-4.6v",
            "doubao-seed-2-0-pro-260215",
            "custom-model",
        ):
            for host in ("gateway.example", "renamed-gateway.example"):
                with (
                    self.subTest(opaque_path=model, host=host),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    url = f"https://{host}/service/vendor"
                    args = run.parser().parse_args(
                        ["--data-path", "unused", "--model", model, "--base-url", url]
                    )
                    client = init_client(args)
                    self.assertEqual(args.model_family, detect_model_family(model))
                    self.assertEqual(client.api_format, "chat")
                    self.assertEqual(client.base_url, url)
                    self.assertEqual(client._url(), url + "/chat/completions")

    def test_complete_endpoints_preserve_exact_request_urls(self):
        for model, path, protocol in (
            ("gpt-4.1", "/v1/chat/completions", "chat"),
            ("gpt-4.1", "/v1/responses", "responses"),
            ("gpt-4.1", "/openai/responses", "responses"),
            ("custom-model", "/v1/responses", "responses"),
            ("custom-model", "/v1/messages", "anthropic"),
            (
                "custom-model",
                "/v1beta/models/custom-model:generateContent",
                "gemini",
            ),
            ("claude-sonnet-4-5", "/v1/messages", "anthropic"),
            ("claude-sonnet-4-5", "/custom/messages", "anthropic"),
            ("claude-sonnet-4-5", "/messages", "anthropic"),
            (
                "gemini-3-flash-preview",
                "/custom/models/gemini-3-flash-preview:generateContent",
                "gemini",
            ),
            (
                "gemini-3-flash-preview",
                "/openai/models/gemini-3-flash-preview:generateContent",
                "gemini",
            ),
        ):
            with (
                self.subTest(path=path),
                patch.dict(os.environ, {}, clear=True),
                service(lambda *_: "answer") as (url, calls),
            ):
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        model,
                        "--base-url",
                        url + path + "/",
                    ]
                )
                client = init_client(args)
                self.assertEqual(client.api_format, protocol)
                self.assertEqual(client._url(), url + path)
                repeated = init_client(args)
                self.assertEqual(repeated.configuration(), client.configuration())
                self.assertEqual(repeated._url(), url + path)
                self.assertEqual(
                    client.generate([{"role": "user", "content": "Question"}])[
                        "content"
                    ],
                    "answer",
                )
                self.assertEqual(calls[0][0], path)

        for endpoint, error in (
            ("gemini-2.5-flash:generateContent", "match --model"),
            ("gemini-3-flash-preview:streamGenerateContent", "non-streaming"),
        ):
            with (
                self.subTest(endpoint=endpoint),
                patch.dict(os.environ, {}, clear=True),
            ):
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        "gemini-3-flash-preview",
                        "--base-url",
                        "https://gateway.example/models/" + endpoint,
                    ]
                )
                with self.assertRaisesRegex(ValueError, error):
                    init_client(args)._url()

        for model, suffix in (
            ("gpt-4.1", "/messages"),
            ("gpt-4.1", "/models/gpt-4.1:generateContent"),
            ("claude-sonnet-4-5", "/responses"),
            ("claude-sonnet-4-5", "/models/claude-sonnet-4-5:generateContent"),
            ("gemini-3-flash-preview", "/messages"),
            ("gemini-3-flash-preview", "/responses"),
        ):
            with (
                self.subTest(unsupported_model=model, endpoint=suffix),
                patch.dict(os.environ, {}, clear=True),
            ):
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        model,
                        "--base-url",
                        "https://gateway.example/custom" + suffix,
                    ]
                )
                with self.assertRaises(ValueError):
                    init_client(args)._url()

    def test_removed_cli_options_are_rejected(self):
        for module in (run, eval_module):
            for option, value in (
                ("--backend", "auto"),
                ("--api-format", "chat"),
                ("--api-key-env", "OPENAI_API_KEY"),
                ("--lang", "en"),
                ("--dry-run", None),
                ("--stream", None),
                ("--video-input", "auto"),
                ("--audio-input", "auto"),
                ("--request-options", "options.json"),
                ("--force", None),
            ):
                if option == "--force" and module is run:
                    continue
                with (
                    self.subTest(module=module.__name__, option=option),
                    contextlib.redirect_stderr(io.StringIO()) as stderr,
                ):
                    arguments = ["--data-path", "unused", option]
                    if value is not None:
                        arguments.append(value)
                    if module is eval_module:
                        arguments += ["--predictions", "unused"]
                    with self.assertRaises(SystemExit) as raised:
                        module.parser().parse_args(arguments)
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("unrecognized arguments", stderr.getvalue())
                    self.assertIn(option, stderr.getvalue())

    def test_direct_credentials_and_environment_fallbacks(self):
        for model, base_url, key_env in (
            ("gpt-4.1", "https://api.openai.com/v1", "OPENAI_API_KEY"),
            (
                "gemini-3-flash-preview",
                "https://generativelanguage.googleapis.com/v1beta",
                "GEMINI_API_KEY",
            ),
            ("claude-sonnet-4-5", "https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
            (
                "qwen3-vl-plus",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "DASHSCOPE_API_KEY",
            ),
            (
                "Qwen/Qwen3-VL-8B-Instruct",
                "http://localhost:8000/v1",
                "DASHSCOPE_API_KEY",
            ),
            (
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "http://localhost:8000/v1",
                "DASHSCOPE_API_KEY",
            ),
            (
                "Qwen/Qwen2-Audio-7B-Instruct",
                "http://localhost:8000/v1",
                "DASHSCOPE_API_KEY",
            ),
            ("MiniMax-M2.7", "https://api.minimax.io/v1", "MINIMAX_API_KEY"),
            ("MiniMax-M2.7", "https://api.minimaxi.com/anthropic", "MINIMAX_API_KEY"),
            ("kimi-k2.5", "https://api.moonshot.ai/v1", "MOONSHOT_API_KEY"),
            ("kimi-k2.5", "https://api.moonshot.cn/anthropic", "MOONSHOT_API_KEY"),
            ("MiniMax-M2.7", "https://gateway.example/v1", "MINIMAX_API_KEY"),
            ("kimi-k2.5", "https://gateway.example/v1", "MOONSHOT_API_KEY"),
            ("glm-4.6v", "https://open.bigmodel.cn/api/paas/v4", "ZHIPUAI_API_KEY"),
            (
                "doubao-seed-2-0-pro-260215",
                "https://ark.cn-beijing.volces.com/api/v3",
                "ARK_API_KEY",
            ),
            ("deepseek-v4-pro", "https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
            (
                "glm-4.6v",
                "https://api.openai.com/v1/chat/completions",
                "ZHIPUAI_API_KEY",
            ),
            (
                "custom-model",
                "https://api.moonshot.ai/v1",
                "OPENAI_API_KEY",
            ),
            (
                "custom-model",
                "https://gateway.example/service/messages",
                "ANTHROPIC_API_KEY",
            ),
            (
                "custom-model",
                "https://gateway.example/models/custom-model:generateContent",
                "GEMINI_API_KEY",
            ),
        ):
            for direct, unified, expected in (
                (True, True, "DIRECT_SECRET"),
                (False, True, "UNIFIED_SECRET"),
                (False, False, "PROVIDER_SECRET"),
            ):
                env = {
                    "OPENAI_API_KEY": "OTHER_SECRET",
                    "ANTHROPIC_API_KEY": "OTHER_SECRET",
                    "GEMINI_API_KEY": "OTHER_SECRET",
                    "MOONSHOT_API_KEY": "OTHER_SECRET",
                    key_env: "PROVIDER_SECRET",
                }
                if unified:
                    env["MODEL_API_KEY"] = "UNIFIED_SECRET"
                with (
                    self.subTest(model=model, direct=direct, unified=unified),
                    patch.dict(os.environ, env, clear=True),
                ):
                    arguments = [
                        "--data-path",
                        "unused",
                        "--model",
                        model,
                        "--base-url",
                        base_url,
                    ]
                    if direct:
                        arguments += ["--api-key", "DIRECT_SECRET"]
                    client = init_client(run.parser().parse_args(arguments))
                    self.assertEqual(client.api_key, expected)
                    self.assertNotIn("SECRET", json.dumps(client.configuration()))
                    self.assertNotIn("SECRET", repr(client))

        for model, expected in (
            ("glm-4.6v", ""),
            ("doubao-seed-2-0-pro-260215", ""),
            ("Qwen/Qwen3-VL-8B-Instruct", ""),
            ("InternVL3-8B", "FALLBACK_SECRET"),
        ):
            with (
                self.subTest(protocol_fallback=model),
                patch.dict(
                    os.environ, {"OPENAI_API_KEY": "FALLBACK_SECRET"}, clear=True
                ),
            ):
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        model,
                        "--base-url",
                        "https://gateway.example/service/vendor",
                    ]
                )
                self.assertEqual(init_client(args).api_key, expected)

    def test_direct_key_is_not_written_to_run_artifacts(self):
        for module in (run, eval_module):
            with (
                self.subTest(module=module.__name__),
                tempfile.TemporaryDirectory() as td,
                service(
                    lambda *_: json.dumps({"score": 1, "reason": "Correct."})
                    if module is eval_module
                    else "happiness"
                ) as (url, calls),
            ):
                root = Path(td)
                q = question(granularity="episode" if module is eval_module else "clip")
                (root / "q.json").write_text(json.dumps(q))
                arguments = [
                    "--data-path",
                    str(root / "q.json"),
                    "--model",
                    "gpt-4.1",
                    "--api-key",
                    "DIRECT_SECRET",
                    "--base-url",
                    url + "/v1",
                    "--output-dir",
                    str(root / "out"),
                ]
                if module is eval_module:
                    (root / "p.json").write_text(json.dumps(prediction(q, "Yes")))
                    arguments += [
                        "--predictions",
                        str(root / "p.json"),
                        "--granularity",
                        "episode",
                    ]
                else:
                    arguments += ["--modality", "text"]
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(
                        module.run(module.parser().parse_args(arguments)), 0
                    )
                self.assertEqual(len(calls), 1)
                self.assertNotIn("DIRECT_SECRET", stdout.getvalue())
                for artifact in (root / "out").rglob("*"):
                    if artifact.is_file():
                        self.assertNotIn(
                            "DIRECT_SECRET", artifact.read_text(), str(artifact)
                        )

    def test_family_defaults_and_credentials(self):
        cases = [
            (
                "gemini-3-flash-preview",
                "gemini",
                "gemini",
                "generativelanguage.googleapis.com",
                "GEMINI_API_KEY",
            ),
            ("gpt-4.1", "gpt", "responses", "api.openai.com", "OPENAI_API_KEY"),
            (
                "claude-sonnet-4-5",
                "claude",
                "anthropic",
                "api.anthropic.com",
                "ANTHROPIC_API_KEY",
            ),
            ("MiniMax-M2.7", "minimax", "chat", "api.minimax.io", "MINIMAX_API_KEY"),
            ("kimi-k2.5", "kimi", "chat", "api.moonshot.ai", "MOONSHOT_API_KEY"),
            ("moonshot-v1-8k", "kimi", "chat", "api.moonshot.ai", "MOONSHOT_API_KEY"),
            ("glm-4.6v", "glm", "chat", "open.bigmodel.cn", "ZHIPUAI_API_KEY"),
            (
                "doubao-seed-2-0-pro-260215",
                "seed",
                "chat",
                "ark.cn-beijing.volces.com",
                "ARK_API_KEY",
            ),
            (
                "deepseek-v4-pro",
                "deepseek",
                "chat",
                "api.deepseek.com",
                "DEEPSEEK_API_KEY",
            ),
        ]
        for model, model_family, protocol, host, key_env in cases:
            with (
                self.subTest(model=model),
                patch.dict(os.environ, {key_env: "SECRET"}, clear=True),
            ):
                args = run.parser().parse_args(
                    ["--data-path", "unused", "--model", model]
                )
                client = init_client(args)
                self.assertEqual(args.model_family, model_family)
                self.assertEqual(client.api_format, protocol)
                self.assertIn(host, client.base_url)
                self.assertEqual(client.api_key, "SECRET")
                self.assertNotIn("SECRET", json.dumps(client.configuration()))

        for name in ("Qwen/Qwen2-Audio-7B-Instruct", "Qwen/Qwen-Audio-Chat"):
            self.assertEqual(detect_model_family(name), "qwen_audio")

    def test_local_services_require_an_explicit_url(self):
        for model, family in (
            ("Qwen/Qwen3-VL-8B-Instruct", "qwen_vl"),
            ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen_omni"),
            ("Qwen/Qwen2-Audio-7B-Instruct", "qwen_audio"),
            ("InternVL3-8B", "internvl"),
            ("custom-model", "other"),
        ):
            arguments = ["--data-path", "unused", "--model", model]
            with (
                self.subTest(missing_address=model),
                patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "--base-url|MODEL_BASE_URL"):
                    init_client(run.parser().parse_args(arguments))

            for explicit, environment, expected in (
                ("http://localhost:8000/v1", None, "http://localhost:8000/v1"),
                (None, "http://localhost:9000/v1", "http://localhost:9000/v1"),
                (
                    "http://localhost:8000/v1",
                    "http://localhost:9000/v1",
                    "http://localhost:8000/v1",
                ),
            ):
                env = {
                    "OPENAI_API_KEY": "LOCAL_SECRET",
                    "DASHSCOPE_API_KEY": "LOCAL_SECRET",
                }
                if environment:
                    env["MODEL_BASE_URL"] = environment
                with (
                    self.subTest(
                        model=model, explicit=explicit, environment=environment
                    ),
                    patch.dict(os.environ, env, clear=True),
                ):
                    configured = arguments + (
                        ["--base-url", explicit] if explicit else []
                    )
                    args = run.parser().parse_args(configured)
                    client = init_client(args)
                    self.assertEqual(args.model_family, family)
                    self.assertEqual(client.api_format, "chat")
                    self.assertEqual(client.base_url, expected)
                    self.assertEqual(client._url(), expected + "/chat/completions")
                    self.assertEqual(client.api_key, "LOCAL_SECRET")
                    self.assertNotIn(
                        "stream",
                        client.payload([{"role": "user", "content": "Question"}]),
                    )
                    if family == "qwen_omni":
                        self.assertEqual(client.options["modalities"], ["text"])
                    if family == "qwen_vl":
                        self.assertNotIn("modalities", client.options)

    def test_glm_seed_settings_preserve_multimodal_input(self):
        messages = [
            {"role": "system", "content": "Answer the question."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,YWJj"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,ZGVm"},
                    },
                    {"type": "text", "text": "Describe the change."},
                ],
            },
        ]
        for model in ("glm-4.6v", "doubao-seed-2-0-pro-260215"):
            for thinking_mode in ("default", "on", "off"):
                with (
                    self.subTest(model=model, thinking_mode=thinking_mode),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    args = run.parser().parse_args(
                        ["--data-path", "unused", "--model", model, "--thinking", thinking_mode]
                    )
                    client = init_client(args)
                    payload = client.payload(messages)
                    self.assertEqual(payload["messages"], messages)
                    self.assertEqual(payload["max_tokens"], 16384)
                    if thinking_mode == "default":
                        self.assertNotIn("thinking", payload)
                    else:
                        self.assertEqual(
                            payload["thinking"],
                            {"type": "enabled" if thinking_mode == "on" else "disabled"},
                        )
                    self.assertEqual(
                        payload["temperature"],
                        1.0 if model.startswith("doubao") else 0.0,
                    )

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {}, clear=True),
        ):
            config = Path(td) / "options.json"
            config.write_text(
                '{"thinking":{"type":"disabled"},"max_completion_tokens":20000}'
            )
            args = run.parser().parse_args(
                [
                    "--data-path",
                    "unused",
                    "--model",
                    "doubao-seed-2-0-pro-260215",
                    "--config",
                    str(config),
                ]
            )
            payload = init_client(args).payload(messages)
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["max_completion_tokens"], 20000)
            self.assertNotIn("max_tokens", payload)
            config.write_text('{"max_tokens":1000,"max_completion_tokens":20000}')
            with self.assertRaisesRegex(ValueError, "only one output token"):
                init_client(args)

    def test_frame_inference_and_native_judging(self):
        for model, path, requested_frames, sampled_frame_cap in [
            ("gpt-4.1", "/v1/responses", 768, 768),
            ("gpt-4.1", "/v1/chat/completions", 768, 768),
            ("claude-sonnet-4-5", "/v1/messages", 768, 600),
            ("claude-sonnet-4-5", "/v1/chat/completions", 768, 600),
            ("glm-4.6v", "/v1/chat/completions", 768, 768),
            ("InternVL3-8B", "/v1/chat/completions", 768, 768),
            ("deepseek-v4-pro", "/v1/chat/completions", 768, 600),
            ("deepseek-v4-pro", "/v1/chat/completions", 128, 128),
        ]:
            with (
                self.subTest(model=model, path=path, max_frames=requested_frames),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: "happiness") as (url, calls),
            ):
                root = Path(td)
                q = question()
                (root / "q.json").write_text(json.dumps(q))
                (root / "videos").mkdir()
                (root / "videos/V1.mp4").write_bytes(b"synthetic")
                blocks = [
                    {"type": "text", "text": "Frame at 0.000 seconds:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,YWJj"},
                    },
                ]
                meta = {
                    "timestamps_seconds": [0.0],
                    "num_frames": 1,
                    "duration_seconds": 1.0,
                }
                args = [
                    "--data-path",
                    str(root / "q.json"),
                    "--videos-dir",
                    str(root / "videos"),
                    "--model",
                    model,
                    "--base-url",
                    url + path,
                    "--with-subtitle",
                    "--prompt-mode",
                    "system",
                    "--output-dir",
                    str(root / "out"),
                    "--tries",
                    "1",
                    "--max-frames",
                    str(requested_frames),
                ]
                with patch.object(
                    run, "sample_video_frames", return_value=(blocks, meta)
                ) as extract:
                    self.assertEqual(invoke(run, args), 0)
                    extract.assert_called_once()
                    self.assertEqual(
                        extract.call_args.kwargs["max_frames"], sampled_frame_cap
                    )
                    self.assertEqual(
                        extract.call_args.args[0]["path"],
                        str((root / "videos/V1.mp4").resolve()),
                    )
                self.assertEqual((root / "videos/V1.mp4").read_bytes(), b"synthetic")
                self.assertEqual(calls[0][0], path)
                payload = json.dumps(calls[0][1])
                self.assertNotIn("video_url", payload)
                self.assertLess(
                    payload.index("Frame at"), payload.index("First subtitle.")
                )
                self.assertLess(
                    payload.index("First subtitle."), payload.index(q["question"])
                )
                pred = io_utils.load_records(root / "out/predictions.jsonl")[0]
                self.assertEqual(pred["prediction"], ["happiness"])
                self.assertEqual(pred["media"]["audio"], "not_sent")
                self.assertNotIn("base64,", json.dumps(pred))
                if model.startswith("gpt") and path.endswith("/chat/completions"):
                    self.assertIn("max_completion_tokens", payload)
                    self.assertNotIn('"max_tokens"', payload)

    def test_seed_defaults_to_native_video_and_versioned_audio(self):
        for model, expected_audio in (
            ("doubao-seed-2-0-pro-260215", (False, False)),
            ("doubao-seed-2-0-lite-260428", (True, False)),
        ):
            with self.subTest(model=model), patch.dict(os.environ, {}, clear=True):
                args = run.parser().parse_args(
                    [
                        "--data-path",
                        "unused",
                        "--model",
                        model,
                        "--base-url",
                        "https://ark.cn-beijing.volces.com/api/v3",
                    ]
                )
                init_client(args)
                self.assertEqual(run.select_audio_input(args, "native"), expected_audio)

        self.assertEqual(run.MODEL_FRAME_CAPS["deepseek"], 600)
        self.assertEqual(run.MODEL_FRAME_CAPS["claude"], 600)
        self.assertEqual(run.MODEL_FRAME_CAPS["seed"], 1280)

        for model, path in (
            ("gpt-4.1", "/v1/responses"),
            ("claude-sonnet-4-5", "/v1/messages"),
            ("gemini-3-flash-preview", "/v1beta"),
            ("Qwen/Qwen3-VL-8B-Instruct", "/v1"),
            ("glm-4.6v", "/v1"),
            ("doubao-seed-2-0-pro-260215", "/v1"),
            ("InternVL3-8B", "/v1"),
            ("deepseek-v4-pro", "/v1"),
        ):
            with (
                self.subTest(judge=model),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: '{"score":1,"reason":"Correct."}') as (url, calls),
            ):
                root = Path(td)
                q = question(granularity="episode")
                (root / "q.json").write_text(json.dumps(q))
                (root / "p.json").write_text(json.dumps(prediction(q, "Yes")))
                args = [
                    "--data-path",
                    str(root / "q.json"),
                    "--predictions",
                    str(root / "p.json"),
                    "--granularity",
                    "episode",
                    "--model",
                    model,
                    "--base-url",
                    url + path,
                    "--output-dir",
                    str(root / "out"),
                    "--tries",
                    "1",
                ]
                self.assertEqual(invoke(eval_module, args), 0)
                self.assertEqual(len(calls), 1)
                result = next((root / "out").glob("*/scores.jsonl"))
                self.assertEqual(io_utils.load_records(result)[0]["score"], 1)

    def test_version_specific_thinking(self):
        for base_url, protocol in (
            (None, "gemini"),
            ("https://gateway.example/v1/chat/completions", "chat"),
        ):
            for thinking in ("default", "on"):
                with (
                    self.subTest(gemini_url=base_url, thinking=thinking),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    arguments = [
                        "--data-path",
                        "unused",
                        "--model",
                        "gemini-3-flash-preview",
                        "--thinking",
                        thinking,
                    ]
                    if base_url:
                        arguments += ["--base-url", base_url]
                    client = init_client(run.parser().parse_args(arguments))
                    self.assertEqual(client.api_format, protocol)
                    payload = client.payload([{"role": "user", "content": "Question"}])
                    if protocol == "gemini":
                        settings = payload["generationConfig"]
                        self.assertNotIn("reasoning_effort", payload)
                        if thinking == "on":
                            self.assertEqual(
                                settings["thinkingConfig"], {"thinkingLevel": "high"}
                            )
                        else:
                            self.assertNotIn("thinkingConfig", settings)
                    else:
                        self.assertNotIn("generationConfig", payload)
                        if thinking == "on":
                            self.assertEqual(payload["reasoning_effort"], "high")
                        else:
                            self.assertNotIn("reasoning_effort", payload)
        for model in ("gemini-3-flash-preview", "Qwen/Qwen3-VL-8B-Thinking"):
            args = run.parser().parse_args(
                [
                    "--data-path",
                    "unused",
                    "--model",
                    model,
                    "--base-url",
                    "http://localhost:8000/v1"
                    if model.startswith("Qwen/")
                    else "https://generativelanguage.googleapis.com/v1beta",
                    "--thinking",
                    "off",
                ]
            )
            with self.subTest(model=model), self.assertRaises(ValueError):
                init_client(args)

        for base_url in (
            "https://api.deepseek.com/v1",
            "https://gateway.example/service/vendor",
        ):
            for thinking in ("default", "on", "off"):
                with (
                    self.subTest(deepseek_url=base_url, thinking=thinking),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    args = run.parser().parse_args(
                        [
                            "--data-path",
                            "unused",
                            "--model",
                            "deepseek-v4-pro",
                            "--base-url",
                            base_url,
                            "--thinking",
                            thinking,
                        ]
                    )
                    payload = init_client(args).payload(
                        [{"role": "user", "content": "Question"}]
                    )
                    if thinking == "default":
                        self.assertNotIn("thinking", payload)
                    else:
                        self.assertEqual(
                            payload["thinking"],
                            {"type": "enabled" if thinking == "on" else "disabled"},
                        )

        for endpoint in ("responses", "messages"):
            for thinking in ("on", "off"):
                with (
                    self.subTest(deepseek_endpoint=endpoint, thinking=thinking),
                    patch.dict(os.environ, {}, clear=True),
                ):
                    args = run.parser().parse_args(
                        [
                            "--data-path",
                            "unused",
                            "--model",
                            "deepseek-v4-pro",
                            "--base-url",
                            f"https://gateway.example/v1/{endpoint}",
                            "--thinking",
                            thinking,
                        ]
                    )
                    with self.assertRaisesRegex(ValueError, "--config"):
                        init_client(args)


if __name__ == "__main__":
    unittest.main()
