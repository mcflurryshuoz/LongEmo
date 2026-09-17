"""Export native Gemini run progress without prompts, credentials or invented costs."""
import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.io_utils import write_json
from experiments.zyf.azure_status import records, snapshot


def usage(paths):
    rows=list(records(paths))
    costs=[(r.get('usage',{}).get('raw') or {}).get('cost') for r in rows]
    known=[c for c in costs if isinstance(c,(int,float))]
    return {'attempts':len(rows), 'success':sum(r.get('status')=='ok' for r in rows),
        'response_models':dict(Counter(r['response_model'] for r in rows if r.get('response_model'))),
        'http_errors':dict(Counter(str(r['http_status']) for r in rows if r.get('status')!='ok' and 'http_status' in r)),
        'error_types':dict(Counter(r.get('error_type','unknown') for r in rows if r.get('status')!='ok')),
        'returned_total_tokens':sum((r.get('usage',{}).get('total_tokens') or 0) for r in rows),
        'attempts_without_usage':sum(not r.get('usage',{}).get('raw') for r in rows),
        'provider_reported_cost_usd':sum(known) if known else None,
        'attempts_without_cost':len(costs)-len(known)}


def export(run, output=None):
    result=snapshot(run)
    inv=json.loads((run/'inventory.json').read_text())
    manifest=json.loads((run/'experiment_manifest.json').read_text())
    launch=json.loads((run/'launch.json').read_text())
    config=manifest['configuration']
    if config.get('embedding_backend')!='gemini-native':
        raise ValueError('this report is for native Gemini experiments only')
    result['configuration']=config
    result['launch']={k:launch[k] for k in ('pid','cwd','started_unix','git_revision','defer_answers')}
    result['expected_windows']=inv['windows']
    result['expected_videos']=inv['expected_videos']
    result['metrics']=json.loads((run/'metrics.json').read_text())
    result['audio']=usage((run/'memory').glob('*/audio/calls.jsonl'))
    result['perception']=usage((run/'memory').glob('*/calls.jsonl'))
    command=launch['command']
    cache=Path(command[command.index('--embedding-cache-dir')+1])/'api'
    result['embedding']=usage([cache/'calls.jsonl'] if (cache/'calls.jsonl').exists() else [])
    result['video_outcomes']=dict(Counter(v['status'] for v in result['status'].get('videos',{}).values()))
    responses=list(records((run/'videos').glob('*/build_memory/native_responses.jsonl')))
    result['response_diagnostics']={'responses':len(responses),
        'finish_reasons':dict(Counter(str(reason) for r in responses for reason in r['finish_reasons'])),
        'returned_total_tokens':sum(r.get('usage',{}).get('totalTokenCount',0) for r in responses),
        'note':'Observability added during resume; overlaps call-ledger tokens, never add these totals together.'}
    result['full_score_complete']=result['metrics']['overall_unweighted']['coverage']==1
    if output:
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/'progress.json',result)
        metric=result['metrics']['overall_unweighted']
        lines=['# E09 native Gemini progress','',
            f"Status: {result['status']['status']}. Scored: {metric['n_scored']}/{metric['n_total']}. "
            f"Complete memories: {result['complete_video_memories']}/{inv['expected_videos']}. "
            f"Saved windows: {result['completed_windows']}/{inv['windows']}.",'',
            f"Execution commit: {launch['git_revision']}. Supervisor PID: {launch['pid']}.",'',
            'Audio and perception use native Gemini 3.8 Flash; embedding is configured for native Gemini Embedding 2. '
            'Azure GPT-6 performs planning, answering and unchanged official judging. No OpenRouter calls or fallback.', '',
            f"Deferred answering: {launch['defer_answers']}. Complete memories awaiting embedding access are not scored results.",'',
            f"Active videos: {len(result['status'].get('active_videos',[]))}; queued: {result['status'].get('queued_videos')}.",'',
            '| Stage | Attempts | Successful | Returned tokens | Reported model IDs |',
            '|---|---:|---:|---:|---|']
        for stage in ('audio','perception','embedding'):
            u=result[stage]
            lines.append(f"| {stage} | {u['attempts']} | {u['success']} | {u['returned_total_tokens']} | {u['response_models']} |")
        lines+=['','Token usage includes returned usage on failed validation attempts. Missing billing is unknown, not zero. '
            'No full benchmark score or improvement is established until coverage and matched comparisons support it.']
        (output/'progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('snapshot_unix','completed_windows','complete_video_memories','video_outcomes','audio','perception','embedding')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    export(args.run,args.output)
