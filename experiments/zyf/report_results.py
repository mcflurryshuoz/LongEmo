"""Summarize a frozen run without exporting gold annotations, prompts or credentials."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def ledger(paths):
    rows=[r for p in paths for r in records(p)]
    usages=[r.get('usage',{}).get('raw',{}) or {} for r in rows]
    return {'attempts':len(rows),'successful_calls':sum(r['status']=='ok' for r in rows),
            'missing_usage_attempts':sum(not u for u in usages),
            'reported_total_tokens':sum(u.get('total_tokens',u.get('totalTokenCount',0)) or 0 for u in usages),
            'provider_reported_cost_usd':sum(u.get('cost') or 0 for u in usages),
            'missing_cost_attempts':sum(u.get('cost') is None for u in usages),
            'elapsed_api_seconds':sum(r['elapsed_seconds'] for r in rows),
            'errors':dict(Counter(r.get('validation_error',r.get('error_type','unknown')) for r in rows if r['status']!='ok'))}


def summarize(run, embedding_cache):
    result={'run':run.name,'generated_unix':time.time(),'methods':{},'memories':{},'costs':{},'paired_scores':[]}
    for name in ('launch','experiment_manifest','status','resources'):
        p=run/(name+'.json')
        if p.exists():result[name]=json.loads(p.read_text())
    for p in sorted((run/'memory').glob('*/memory.json')):
        m=json.loads(p.read_text())
        result['memories'][m['video_id']]={'complete':bool(m.get('complete')),'duration_seconds':m['duration'],
            'windows':len(m['completed_windows']),'entities':len(m['entities']),'events':len(m['events']),
            'states':sum(len(e['states']) for e in m['events']),'observations':len(m['observations']),
            'relations':len(m['relations']),'bytes':p.stat().st_size}
    groups={'perception':list((run/'memory').glob('*/calls.jsonl')),
            'audio_observation':list((run/'memory').glob('*/audio/calls.jsonl')),
            'shared_planning':list((run/'shared_plans/calls').glob('*.jsonl'))}
    if embedding_cache:
        groups['embedding_shared']=list((embedding_cache/'api').glob('calls.jsonl'))
    score_rows={}
    for method in ('graph','graph_inspect','direct'):
        folder=run/method
        groups[method+'_answer']=list((folder/'calls').glob('*.jsonl'))
        groups[method+'_inspection_audio']=list((folder/'inspection_audio').glob('calls.jsonl'))
        preds=records(folder/'predictions.jsonl')
        info={'predictions':len(preds),'successful_predictions':sum(p.get('status')=='ok' for p in preds)}
        scores=sorted((folder/'scores').glob('run_*/metrics.json'))
        if scores:
            latest=scores[-1];info['metrics']=json.loads(latest.read_text())
            scored=records(latest.parent/'scores.jsonl')
            info['scores_source']=str(latest)
            judge_raw=[raw for row in scored for raw in row.get('judge_responses',[])]
            info['judge_returned_attempts']=len(judge_raw)
            info['judge_provider_reported_cost_usd']=sum((r.get('usage') or {}).get('cost') or 0 for r in judge_raw)
            info['judge_missing_cost_attempts']=sum((r.get('usage') or {}).get('cost') is None for r in judge_raw)
            score_rows[method]={r['question_id']:r for r in scored}
        traces=[]
        for p in (folder/'traces').glob('*.json'):
            d=json.loads(p.read_text())
            traces.append({'question_id':p.stem,'evidence_characters':d.get('evidence_characters'),
                'selected_events':len(d.get('retrieval',{}).get('events',[])),
                'invalid_evidence_ids':d.get('invalid_evidence_ids',[]),
                'inspection_count':len(d.get('inspection_trace',[]))})
        info['trace_summary']=traces
        index=folder/'embedding_indexes.json'
        if index.exists():info['embedding_indexes']=json.loads(index.read_text())
        result['methods'][method]=info
    for group,paths in groups.items():result['costs'][group]=ledger(paths)
    qids=sorted({qid for rows in score_rows.values() for qid in rows})
    for qid in qids:
        item={'question_id':qid}
        for method,rows in score_rows.items():
            row=rows.get(qid,{})
            item[method]={'status':row.get('status'),'score':row.get('score'),'max_score':row.get('max_score'),
                          'normalized_score':row.get('normalized_score')}
            item['type']=row.get('type',item.get('type'));item['video_id']=row.get('video_id',item.get('video_id'))
        result['paired_scores'].append(item)
    return result


def markdown(result):
    lines=['# GPT-6 hybrid graph pilot','','This development pilot contains 3 videos and 9 questions. It is not a full benchmark result.','',
           '| Method | Successful predictions | Scored / total | Overall (%) |','|---|---:|---:|---:|']
    for method,data in result['methods'].items():
        m=data.get('metrics',{}).get('overall_unweighted',{})
        pct=f"{m['percent_score']:.2f}" if 'percent_score' in m else 'pending'
        lines.append(f"| {method} | {data['successful_predictions']}/9 | {m.get('n_scored','–')}/{m.get('n_total',9)} | {pct} |")
    lines+=['','| Video | Complete | Windows | Events | States |','|---|---|---:|---:|---:|']
    for video,m in result['memories'].items():lines.append(f"| {video} | {m['complete']} | {m['windows']} | {m['events']} | {m['states']} |")
    lines+=['','| Stage | Attempts | Returned tokens | Provider-reported USD | Attempts without cost |','|---|---:|---:|---:|---:|']
    for name,c in result['costs'].items():
        lines.append(f"| {name} | {c['attempts']} | {c['reported_total_tokens']} | {c['provider_reported_cost_usd']:.6f} | {c['missing_cost_attempts']} |")
    lines+=['','Shared costs must be amortized equally. Cached query embeddings/plans are not a zero-cost ability of the second condition. Judge costs are recorded separately in the JSON. Provider-reported cost can exclude failed requests without returned usage.','',
        'The three conditions use GPT-6 Astra; graph retrieval uses Gemini Embedding 2. Direct uses 128 sampled frames and the same subtitles/audio-only frontend but no graph states. Graph perception uses 20-second windows, so visual budgets differ.','',
        'Score formulas and judge prompts are unchanged. Both pilot reasoning questions are explanation questions; result questions are absent. GPT-6 judging GPT-6 outputs introduces possible judge bias. No statistical superiority claim is supported by nine development questions.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--embedding-cache',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=summarize(a.run,a.embedding_cache);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    (a.output/'results.md').write_text(markdown(r))
    print(json.dumps({'status':r.get('status',{}).get('status','running'),'methods':{m:d['successful_predictions'] for m,d in r['methods'].items()}}))
