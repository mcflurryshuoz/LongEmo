"""Provider-scoped Gemini auth and exact-input reuse of inherited audio cues."""
import json
from pathlib import Path

from evaluation.clients import Client
from evaluation.inference.adapters import gemini
from experiments.zyf.aicodemirror_probe import mirror_headers
from experiments.zyf.native_worker import main
from methods.longemo import runner
from methods.longemo.audio import AUDIO_PROMPT, bridge_audio
from methods.longemo.common import fingerprint


def verified_inherited_audio(content, interval, client, path, tries=3):
    path = Path(path)
    if path.exists():
        cached = json.loads(path.read_text())
        audio = [part for part in content if part["type"] == "input_audio"]
        messages = [{"role":"system","content":AUDIO_PROMPT}, {"role":"user","content":[
            {"type":"text","text":json.dumps({"clip_start":interval[0],"clip_end":interval[1]})}] + audio}]
        assert fingerprint({"messages":messages,"model":cached["model"]}) == cached["input_fingerprint"], "inherited audio input changed"
        # Exact signature matches: bridge_audio reads and validates this cache,
        # and never calls the old provider. Its original configuration is kept.
        original = Client(**cached["model"])
        result, trace = bridge_audio(content, interval, original, path, tries)
        trace["base_url"] = original.base_url
        trace["inherited"] = True
        return result, trace
    return bridge_audio(content, interval, client, path, tries)


if __name__ == "__main__":
    gemini.headers = mirror_headers
    runner.bridge_audio = verified_inherited_audio
    raise SystemExit(main())
