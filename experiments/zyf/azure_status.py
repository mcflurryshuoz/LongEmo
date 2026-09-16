"""Read-only Azure experiment progress, including usage without invented billing."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def records(paths):
    for p in paths:
        lines=p.read_text().splitlines()
        for i,line in enumerate(lines):
            if not line.strip():continue
            try:yield json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:raise


def snapshot(run):
    status=json.loads((run/'status.json').read_text())
    memories=[json.loads(p.read_text()) for p in (run/'memory').glob('*/memory.json')]
    rows=list(records((run/'azure_transport').glob('*.jsonl')))
    recent=[r for r in rows if r.get('time_unix',0)>time.time()-300]
    success=[r for r in rows if r.get('http_status')==200]
    result={'run':run.name,'snapshot_unix':time.time(),'status':status,
        'completed_windows':sum(len(m['completed_windows']) for m in memories),
        'complete_video_memories':sum(bool(m.get('complete')) for m in memories),
        'memory_progress':{m['video_id']:{'windows':len(m['completed_windows']),'complete':bool(m.get('complete'))} for m in memories},
        'azure':{'attempts':len(rows),'http_statuses':dict(Counter(str(r.get('http_status',r.get('error_type'))) for r in rows)),
            'actual_models':dict(Counter(r.get('response_model') for r in success)),
            'returned_prompt_tokens':sum((r.get('usage') or {}).get('prompt_tokens',0) for r in success),
            'returned_completion_tokens':sum((r.get('usage') or {}).get('completion_tokens',0) for r in success),
            'returned_total_tokens':sum((r.get('usage') or {}).get('total_tokens',0) for r in success),
            'billing_usd':None,'billing_note':'Azure usage is recorded; no Azure billing export is available.',
            'recent_5min_attempts':len(recent),'recent_5min_429':sum(r.get('http_status')==429 for r in recent)}}
    for kind,pattern in [('audio','*/audio/calls.jsonl'),('perception','*/calls.jsonl')]:
        calls=list(records((run/'memory').glob(pattern)))
        result[kind]={'attempts':len(calls),'success':sum(r.get('status')=='ok' for r in calls),
            'error_types':dict(Counter(r.get('validation_error',r.get('error_type')) for r in calls if r.get('status')=='error'))}
        if kind=='audio':
            result[kind]['provider_reported_cost_usd']=sum(((r.get('usage') or {}).get('raw') or {}).get('cost') or 0 for r in calls)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=snapshot(args.run)
    if args.output:
        from evaluation.io_utils import write_json
        write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False))
