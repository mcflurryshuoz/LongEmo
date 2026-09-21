"""Provider configuration for Claude visual graphs and Gemini 2.5 audio cues."""
from pathlib import Path
import os

from evaluation.clients import Client, ServiceError
from evaluation.inference.adapters import anthropic, gemini
from experiments.zyf.aicodemirror_probe import BASE, mirror_headers
from experiments.zyf.alternative_model_probe import CLAUDE_BASE, claude_headers
from experiments.zyf import aicodemirror_worker as audio_worker
from experiments.zyf.aicodemirror_worker import verified_inherited_audio
from methods.longemo import audio as audio_module
from experiments.zyf.blackai_continuation import read
from experiments.zyf.native_worker import main
from methods.longemo import runner

VISUAL_MODEL = 'claude-opus-5'
AUDIO_MODEL = 'gemini-2.5-pro'


def visual_client(args):
    assert args.model == VISUAL_MODEL and args.base_url == CLAUDE_BASE
    assert args.max_tokens == 8192 and args.thinking == 'default'
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(VISUAL_MODEL, CLAUDE_BASE, 'anthropic', key, args.timeout, 8192, None, {})


def audio_client(args):
    assert args.audio_model == AUDIO_MODEL and args.audio_base_url == BASE
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(AUDIO_MODEL, BASE, 'gemini', key, 180, 4096, None,
                  {'generationConfig': {'thinkingConfig': {'thinkingBudget': 1024}}})


def refusal_aware_parser(original):
    def parse(client, raw):
        if raw.get('stop_reason') == 'refusal':
            raise ServiceError(200, 'content_filter')
        return original(client, raw)
    return parse


if __name__ == '__main__':
    gemini.headers = mirror_headers
    anthropic.headers = claude_headers
    anthropic.parse_response = refusal_aware_parser(anthropic.parse_response)
    runner.client_for = visual_client
    runner.audio_client_for = audio_client
    runner.bridge_audio = verified_inherited_audio
    raise SystemExit(main())
