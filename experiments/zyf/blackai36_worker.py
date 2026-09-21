"""Frozen BlackAI 3.6 profile matching the successful audio/visual probe."""
from pathlib import Path
from evaluation.clients import Client
from experiments.zyf.blackai_continuation import read, BLACKAI
from experiments.zyf.native_worker import main
from methods.longemo import runner
MODEL = 'gemini-3.6-flash'

def visual_client(args):
    assert args.model == MODEL and args.base_url == BLACKAI
    assert args.max_tokens == 8192 and args.thinking == 'default'
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(MODEL, BLACKAI, 'gemini', key, 180, 8192, None, {})

def audio_client(args):
    if not args.with_audio: return None
    assert args.audio_model == MODEL and args.audio_base_url == BLACKAI
    key = read(Path(args.credential_file))['GEMINI_API_KEY']
    return Client(MODEL, BLACKAI, 'gemini', key, 180, 4096, None, {})

if __name__ == '__main__':
    runner.client_for = visual_client
    runner.audio_client_for = audio_client
    raise SystemExit(main())
