"""BlackAI Gemini 3.6 profile for the remaining graph windows."""
import sys
from pathlib import Path
from experiments.zyf import blackai31_worker as gate
from experiments.zyf.blackai_continuation import BLACKAI, read
from experiments.zyf.native_worker import main
from methods.longemo import runner

MODEL = 'gemini-3.6-flash'
AUDIO_MODEL = MODEL
VISUAL_BASE = BLACKAI
AUDIO_BASE = BLACKAI
# Reuse the gate implementation while binding its model/profile constants.
gate.MODEL = MODEL
gate.BLACKAI = BLACKAI

if __name__ == '__main__':
    probe_only = '--probe-only' in sys.argv
    if probe_only:
        sys.argv.remove('--probe-only')
        runner._build_video = gate.probe_one
    runner.client_for = gate.visual_client
    runner.audio_client_for = gate.audio_client
    runner.window_input = gate.bounded_media
    raise SystemExit(main())
