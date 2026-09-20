"""Observe native response metadata before parsing without changing model behavior."""
from pathlib import Path
import hashlib
import json
import re
import time

from evaluation.inference.adapters import gemini
from methods.longemo import runner
from methods.longemo.common import append_json


def observed_parser(original, ledger):
    def parse(client, raw):
        candidates=raw.get('candidates') or []
        usage=raw.get('usageMetadata') or {}
        safe_code = lambda v: v if isinstance(v, (int, bool)) or isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', v) else None
        feedback = raw.get('promptFeedback') or {}
        error = raw.get('error') or {}
        record = {'time_unix':time.time(),'model':client.model,
            'base_url':client.base_url,'response_keys':sorted(raw),
            'response_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),
            'prompt_block_reason':safe_code(feedback.get('blockReason')),
            'error_code':safe_code(error.get('code')) if isinstance(error,dict) else None,
            'error_status':safe_code(error.get('status')) if isinstance(error,dict) else None,
            'response_model':raw.get('modelVersion'),'max_output_tokens':client.max_tokens,
            'candidate_count':len(candidates),'finish_reasons':[c.get('finishReason') for c in candidates],
            'text_characters':[sum(len(p.get('text','')) for p in c.get('content',{}).get('parts',[]) if not p.get('thought')) for c in candidates],
            'usage':{k:usage[k] for k in ('promptTokenCount','candidatesTokenCount','thoughtsTokenCount','totalTokenCount','cachedContentTokenCount') if isinstance(usage.get(k),int)}}
        try:
            result = original(client,raw)
        except Exception as exc:
            record['parser_error_type'] = type(exc).__name__
            # The adapter's error messages are fixed strings, not response text.
            record['empty_candidates'] = not candidates
            append_json(ledger,record)
            raise
        append_json(ledger,record)
        return result
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
