"""Report first valid scores, provenance, and unresolved questions without API calls."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from evaluation.metrics import summarize
from evaluation.io_utils import load_records
from experiments.zyf.blackai36_continuation import NAME,SCORE_HASHES,score_guard
from experiments.zyf.blackai_continuation import read,atomic
from experiments.zyf.finish_scoring import scoring_inventory
from experiments.zyf.series_scores import source_series
from methods.longemo.common import file_hash
from experiments.zyf.method_report import render_method_report

ADDITIONAL='aicodemirror_recharge_parallel_20260921'
RECOVERY='aicodemirror_v106_recovery_20260921'
STREAM='aicodemirror_stream_remaining_20260921'
STREAM129='aicodemirror_stream_v129_20260921'


def make_report(runtime,output):
    run=runtime/'runs'/NAME;target=read(run/'questions.json');score_guard(runtime,target)
    questions=read(runtime/'data/questions.json');assert len(questions)==558
    target_ids={q['question_id'] for q in target};accepted={};runs=[];baseline={}
    route={q['question_id']:NAME for q in target};new_targets={NAME:target};states={}
    assigned=set()
    for extra_name in [ADDITIONAL,RECOVERY,STREAM,STREAM129]:
        additional=runtime/'runs'/extra_name/'questions.json'
        if additional.exists():
            extra=read(additional);ids={q['question_id'] for q in extra}
            assert ids<=target_ids
            if extra_name not in [STREAM,STREAM129]:assert not ids.intersection(assigned)
            else:
                cfg=read(additional.parent/'experiment_manifest.json')['configuration']
                assert len(extra)=={STREAM:45,STREAM129:4}[extra_name] and all(cfg['parent_runs'][q['video_id']]==route[q['question_id']] for q in extra)
            assigned.update(ids);new_targets[extra_name]=extra
            route.update({q['question_id']:extra_name for q in extra})
    for index,name in enumerate([*SCORE_HASHES,NAME,ADDITIONAL,RECOVERY,STREAM,STREAM129]):
        p=runtime/'runs'/name/'scores.jsonl'
        if not p.exists():continue
        raw=p.read_bytes();rows=[json.loads(x) for x in raw.splitlines() if x.strip()]
        if index==0:baseline={r['question_id']:r for r in rows}
        for row in rows:
            if row.get('status')!='ok':continue
            qid=row['question_id'];assert qid not in accepted
            accepted[qid]={**row,'source_run':name}
        runs.append({'run':name,'scores_sha256':hashlib.sha256(raw).hexdigest(),'metrics':summarize(rows)})
    state=read(run/'backend_status.json') if (run/'backend_status.json').exists() else {'status':'not_started','videos':{}}
    pending=0
    for name,subset in new_targets.items():
        folder=runtime/'runs'/name
        states[name]=read(folder/'backend_status.json') if (folder/'backend_status.json').exists() else {'status':'not_started','videos':{}}
        pending+=sum(len(v['pending']) for v in scoring_inventory(folder,subset).values())
    flat=[];series=defaultdict(list);dispositions=[]
    for q in questions:
        qid=q['question_id'];vid=q['video_id'];s=source_series(q.get('source',{}).get('from','')) or 'unclassified'
        row={'question_id':qid,'video_id':vid,'type':q['type'],'series':s}
        if qid in accepted:
            metric=accepted[qid];row.update({k:metric.get(k) for k in ['score','max_score','normalized_score','source_run']});row['status']='scored'
        else:
            metric={**baseline[qid],'status':'pending','normalized_score':None}
            if qid in ['G2_Q000259','G2_Q000260']:
                row.update(status='blocked_question_policy',stage='answer',provider='Matrix GPT-6')
            elif qid=='G2_Q000350':row.update(status='blocked_judge_policy',stage='judge',provider='Matrix GPT-6')
            else:
                assert qid in target_ids,'unclassified missing question'
                selected=route[qid];folder=runtime/'runs'/selected
                video_state=states[selected].get('videos',{}).get(vid,{})
                row.update(status=video_state.get('status','pending_memory_or_score'),source_run=selected)
                receipt=folder/'imports'/(vid+'.json')
                if receipt.exists():
                    item=read(receipt);row['graph_outcome']=item['outcome'];row['experiment_fingerprint']=item['experiment_fingerprint']
                if (folder/'memory'/vid/'memory.json').exists():row['graph_complete']=bool(read(folder/'memory'/vid/'memory.json').get('complete'))
                if selected!=NAME:
                    row['previous_blackai_status']=state.get('videos',{}).get(vid,{}).get('status')
        flat.append(metric);series[s].append(metric);dispositions.append(row)
    data={'as_of':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'aggregation':'First valid scores; equal question weights; missing excluded. Mixed models/providers and inherited windows, not a homogeneous complete benchmark.',
        'runs':runs,'combined':summarize(flat),'series':{s:summarize(rows) for s,rows in sorted(series.items())},
        'question_disposition_counts':dict(Counter(x['status'] for x in dispositions)),
        'questions':dispositions,'backend':state,'backends':states,'successful_answers_awaiting_first_judge':pending,'protected_score_files_unchanged':True,
        'continuation_combined':summarize([r for r in flat if r['question_id'] in target_ids]),
        'execution_split':{'BlackAI Gemini 3.6':len(target_ids)-len(assigned),
                           'AICodeMirror Claude Opus 5 + Gemini 2.5 Pro':len(assigned)}}
    output.mkdir(parents=True,exist_ok=True);atomic(output/'report.json',data)
    combined=data['combined']['overall_unweighted']
    new=data['continuation_combined']['overall_unweighted']
    (output/'report.md').write_text(render_method_report(data))
    return {'path':str(output),'new':new,'combined':combined,'pending_judgments':pending,'backend_status':state['status']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(make_report(a.runtime,a.output),ensure_ascii=False))
