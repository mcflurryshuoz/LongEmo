"""Observe native response metadata before parsing without changing model behavior."""
from pathlib import Path
import time

from evaluation.inference.adapters import gemini
from methods.longemo import runner
from methods.longemo.common import append_json


def observed_parser(original, ledger):
    def parse(client, raw):
        candidates=raw.get('candidates') or []
        usage=raw.get('usageMetadata') or {}
        append_json(ledger, {'time_unix':time.time(),'model':client.model,
            'response_model':raw.get('modelVersion'),'max_output_tokens':client.max_tokens,
            'candidate_count':len(candidates),'finish_reasons':[c.get('finishReason') for c in candidates],
            'text_characters':[sum(len(p.get('text','')) for p in c.get('content',{}).get('parts',[]) if not p.get('thought')) for c in candidates],
            'usage':{k:usage[k] for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount','cachedContentTokenCount') if isinstance(usage.get(k),int)}})
        return original(client,raw)
    return parse


def main():
    args=runner.parser().parse_args()
    original=gemini.parse_response
    gemini.parse_response=observed_parser(original,Path(args.output_dir)/'native_responses.jsonl')
    try:
        return runner.main()
    finally:
        gemini.parse_response=original


if __name__=='__main__':raise SystemExit(main())
