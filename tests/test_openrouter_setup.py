"""OpenRouter routing uses service credentials without changing model families."""

import argparse
import os
import unittest
from unittest.mock import patch

from evaluation.inference.runner import model_args, setup_api


class OpenRouterSetupTest(unittest.TestCase):
    def configure(self, model, base_url=None, api_key=None):
        parser = argparse.ArgumentParser()
        model_args(parser)
        values = ["--model", model]
        if base_url is not None:
            values += ["--base-url", base_url]
        if api_key is not None:
            values += ["--api-key", api_key]
        args = parser.parse_args(values)
        setup_api(args, {})
        return args

    def test_openrouter_keeps_model_id_and_uses_service_key(self):
        env = {
            "OPENROUTER_API_KEY": "router-key",
            "ZHIPUAI_API_KEY": "vendor-key",
            "MINIMAX_API_KEY": "vendor-key",
            "MOONSHOT_API_KEY": "vendor-key",
            "MODEL_API_KEY": "generic-key",
        }
        with patch.dict(os.environ, env, clear=True):
            for model, family in (
                ("z-ai/glm-5.3-flash", "glm"),
                ("minimax/minimax-m3", "minimax"),
                ("moonshotai/kimi-k3", "kimi"),
            ):
                for url in (
                    "https://openrouter.ai/api/v1",
                    "https://openrouter.ai/api/v1/chat/completions",
                ):
                    with self.subTest(model=model, url=url):
                        args = self.configure(model, url)
                        self.assertEqual(args.api_key, "router-key")
                        self.assertEqual(args.model, model)
                        self.assertEqual(args.model_family, family)
                        self.assertEqual(args.api_format, "chat")
                        self.assertEqual(args.base_url, url)

    def test_explicit_key_takes_precedence(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "router-key"}, clear=True):
            args = self.configure(
                "moonshotai/kimi-k3", "https://openrouter.ai/api/v1", "explicit-key"
            )
        self.assertEqual(args.api_key, "explicit-key")

    def test_openrouter_does_not_reuse_vendor_credentials(self):
        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "vendor-key"}, clear=True):
            args = self.configure("moonshotai/kimi-k3", "https://openrouter.ai/api/v1")
        self.assertEqual(args.api_key, "")

    def test_generic_service_key_remains_available(self):
        with patch.dict(os.environ, {"MODEL_API_KEY": "generic-key"}, clear=True):
            args = self.configure("minimax/minimax-m3", "https://openrouter.ai/api/v1")
        self.assertEqual(args.api_key, "generic-key")

    def test_official_services_are_unchanged(self):
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "router-key",
                "DEEPSEEK_API_KEY": "deepseek-key",
                "ANTHROPIC_API_KEY": "claude-key",
            },
            clear=True,
        ):
            for model, url, key in (
                ("deepseek-v4-flash", "https://api.deepseek.com/v1", "deepseek-key"),
                ("claude-sonnet-4-6", "https://api.anthropic.com/v1", "claude-key"),
            ):
                with self.subTest(model=model):
                    args = self.configure(model)
                    self.assertEqual(args.base_url, url)
                    self.assertEqual(args.api_key, key)

    def test_model_namespace_does_not_choose_openrouter(self):
        with patch.dict(os.environ, {}, clear=True):
            args = self.configure("moonshotai/kimi-k3")
        self.assertEqual(args.base_url, "https://api.moonshot.ai/v1")


if __name__ == "__main__":
    unittest.main()
