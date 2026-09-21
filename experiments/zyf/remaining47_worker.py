"""Fixed Sonnet visual / Gemini audio profile using the unchanged graph validator."""
from pathlib import Path
import sys
from evaluation.clients import Client
from evaluation.inference.adapters import gemini
from experiments.zyf.aicodemirror_probe import BASE as AUDIO_BASE, mirror_headers
from experiments.zyf.alternative_model_probe import CLAUDE_BASE as VISUAL_BASE
from experiments.zyf.blackai_continuation import read
from experiments.zyf import blackai31_worker as gate
from experiments.zyf.mirror_stream_client import StreamClient
from experiments.zyf.native_worker import main
from methods.longemo import runner

MODEL = 'claude-sonnet-5'
AUDIO_MODEL = 'gemini-2.5-pro'


def visual_client(args):
    assert args.model == MODEL and args.base_url == VISUAL_BASE
    assert args.max_tokens == 8192 and args.thinking == 'default'
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    client = StreamClient(MODEL, VISUAL_BASE, 'anthropic', key, 240, 8192, None, {'stream':True})
    client.ledger = Path(args.output_dir)/'stream_transport.jsonl'
    return client


def audio_client(args):
    assert args.with_audio and args.audio_model == AUDIO_MODEL and args.audio_base_url == AUDIO_BASE
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(AUDIO_MODEL, AUDIO_BASE, 'gemini', key, 180, 4096, None,
                  {'generationConfig':{'thinkingConfig':{'thinkingBudget':1024}}})


if __name__ == '__main__':
    gemini.headers = mirror_headers
    gate.audio_client = audio_client
    if '--probe-only' in sys.argv:
        sys.argv.remove('--probe-only'); runner._build_video = gate.probe_one
    runner.client_for = visual_client
    runner.audio_client_for = audio_client
    runner.window_input = gate.bounded_media
    raise SystemExit(main())
