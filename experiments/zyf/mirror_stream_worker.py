"""Original Claude/Gemini perception profile, with declared Claude SSE transport."""
from pathlib import Path
from evaluation.inference.adapters import gemini
from experiments.zyf import claude_gemini_worker as base
from experiments.zyf.mirror_stream_client import StreamClient
from methods.longemo import runner


def visual_client(args):
    original = base.visual_client(args)
    client = StreamClient(original.model, original.base_url, original.api_format, original.api_key,
                          original.timeout, original.max_tokens, original.temperature, {'stream': True})
    client.ledger = Path(args.output_dir)/'stream_transport.jsonl'
    return client


if __name__ == '__main__':
    gemini.headers = base.mirror_headers
    runner.client_for = visual_client
    runner.audio_client_for = base.audio_client
    runner.bridge_audio = base.verified_inherited_audio
    raise SystemExit(base.main())
