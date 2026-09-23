"""Read-only final noevent audit: verified first scores and every missing QID.

No credentials, model calls, raw predictions, rubric text or response bodies are
exported. An unknown cause or live process is an audit failure, not completion.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

SNAPSHOT_SHA='bc6a7f48c45874f2585d82717b5362e3e96191d62eff1a12cd9df5536642867e'
ROOT=Path('/root/longemo/runtime')
BASE=ROOT/'runs/three_level_full558_gemini38_20260922'
DIAG=ROOT/'diagnostics/noevent_resume_20260923'


def regular(path):
    path=Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('non-regular source')
    return path

def sha(p):return hashlib.sha256(regular(p).read_bytes()).hexdigest()
def read(p):return json.loads(regular(p).read_text())
def rows(p):return [json.loads(l) for l in regular(p).read_text().splitlines() if l.strip()] if Path(p).exists() else []

def alive(identity):
    if not identity or not identity.get('pid'):return False
    if identity.get('start_ticks') is None:raise ValueError('unverifiable process start ticks')
    p=Path(f"/proc/{identity['pid']}/stat")
    return p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(identity['start_ticks'])

def series(value):
    match=re.fullmatch(r'(.+?)\s+[sS]\d+[eE]\d+',value.strip())
    return {'jiayouernv':'Home with Kids'}.get(match[1],match[1]) if match else 'unclassified'

def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for x in value.values():yield from strings(x)
    elif isinstance(value,list):
        for x in value:yield from strings(x)

def exact_dlp(response):
    body=response.get('body_text','')
    if response.get('http_status')!=428 or response.get('body_redacted') is True:return False
    if hashlib.sha256(body.encode()).hexdigest()!=response.get('body_sha256'):return False
    try:return any('Data leak protection rejected' in s for s in strings(json.loads(body)))
    except ValueError:return False

def memory_folder(run,video):
    candidate=run/'noevent/videos'/video/'memory'/video
    # A first-window rejection can have a manifest and ledger but no memory.
    if (candidate/'manifest.json').exists():return candidate
    index=read(run/'inheritance/parent_index.json')
    candidate=Path(index['branches']['noevent']['videos'][video]['memory_dir'])
    if not candidate.is_absolute() or not candidate.is_relative_to(run/'inheritance/parent'):raise ValueError('bad inherited checkpoint')
    regular(candidate/'manifest.json');return candidate

def final_failure(folder):
    memory=read(folder/'memory.json') if (folder/'memory.json').exists() else {'completed_windows':[]}
    if memory.get('complete'):raise ValueError('complete perception has no build failure')
    next_window=f"W{len(memory['completed_windows'])+1:05d}"
    selected=[]
    for rel in ('calls.jsonl','audio/calls.jsonl'):
        for row in rows(folder/rel):
            if row.get('purpose','').endswith(next_window) and row.get('status')=='error':selected.append((row,folder/rel))
    if not selected:raise ValueError('no failure evidence for the next incomplete window')
    row,ledger=max(selected,key=lambda item:item[0]['time_unix'])
    return row,ledger,next_window,len(memory['completed_windows'])

def audit_processes(runs):
    result={};unjudged=[];unaccepted=[]
    for run in runs:
        identities=[read(run/n) for n in ('process.json','launch_process.json') if (run/n).exists()]
        if not identities or any(alive(x) for x in identities):raise ValueError(f'coordinator live or missing identity: {run.name}')
        active=[];counts=Counter()
        for p in (run/'tasks').glob('*/*/task.json'):
            task=read(p);counts[task.get('status','unknown')]+=1
            if alive(task.get('child')):active.append(str(p))
            if task.get('status') in ('running','needs_audit'):raise ValueError(f'unresolved task: {p}')
        if active:raise ValueError('live task children')
        lock=run/'coordinator.lock'
        if lock.exists():
            with lock.open('r') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
        result[run.name]={'coordinator_identity_alive':False,'active_task_children':0,'lock_free':True,'task_counts':dict(counts),
            'process_file_sha256':sha(run/'process.json') if (run/'process.json').exists() else None,
            'finished':read(run/'finished.json') if (run/'finished.json').exists() else None}
    return result

def dlp_evidence(run,video,row):
    matches=[];h=row['request_hash']
    for p in (run/'private_diagnostics/build'/video).glob('*.json'):
        wrapper=read(p)
        if wrapper.get('request_hash')!=h or not wrapper.get('responses'):continue
        started=datetime.fromisoformat(wrapper['started_at']).timestamp()
        if abs(started-row['time_unix'])>=1:continue
        for response in wrapper['responses']:
            if exact_dlp(response):matches.append({'path':str(p),'sha256':sha(p),'body_sha256':response['body_sha256'],'kind':'same_attempt_private_diagnostic'})
    for p in DIAG.rglob('summary.json'):
        summary=read(p);response=p.with_name('response.json')
        if summary.get('request_hash')==h and response.exists():
            r=read(response)
            if r.get('request_hash')==h and exact_dlp(r):
                matches.append({'path':str(response),'sha256':sha(response),'body_sha256':r['body_sha256'],'kind':'independent_exact_request_diagnostic'})
    if not matches:raise ValueError(f'HTTP428 has no exact verified DLP evidence: {video}')
    return matches

def backend_failure(run,q):
    qid=q['question_id'];video=q['video_id'];found=[]
    for p in (run/'scores/noevent'/video).glob('run_*/scores.jsonl'):
        for row in rows(p):
            if row.get('question_id')==qid:
                if row.get('status')=='ok':raise ValueError('successful first judge missing accepted')
                if not any('content_filter' in s for s in strings(row.get('error'))):raise ValueError('unknown judge failure')
                found.append({'category':'judge_content_filter','stage':'score','evidence_path':str(p),'evidence_sha256':sha(p),'already_judged':True})
    p=run/'noevent/plans/calls'/f'{qid}.jsonl'
    for row in rows(p):
        if row.get('status')=='error' and row.get('service_error_code')=='content_filter':
            found.append({'category':'planner_content_filter','stage':'plan','evidence_path':str(p),'evidence_sha256':sha(p),'request_hash':row['request_hash'],'already_judged':False})
    if len(found)!=1:raise ValueError(f'backend missing cause is not unique: {qid}')
    return found[0]

def grouped(questions,scored,field):
    result={}
    for key in sorted({q[field] for q in questions}):
        cohort=[q for q in questions if q[field]==key];values=[scored[q['question_id']]['normalized_score'] for q in cohort if q['question_id'] in scored]
        result[key]={'scored':len(values),'total':len(cohort),'mean_scored':100*sum(values)/len(values) if values else None}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot-script',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    if sha(args.snapshot_script)!=SNAPSHOT_SHA:raise ValueError('frozen first-score snapshot implementation differs')
    out=Path(args.output).absolute()
    if out.exists():raise ValueError('final audit output must be new')
    out.mkdir(parents=True,mode=0o700)
    old=sys.argv;sys.argv=[args.snapshot_script,str(out/'snapshot.json')]
    try:state=runpy.run_path(args.snapshot_script,run_name='__final_snapshot__')
    finally:sys.argv=old
    scored=state['all_rows'];questions=list(state['questions'].values());snap=state['result'];names=state['NAMES'];runs=[BASE,*[ROOT/'runs'/n for n in names if (ROOT/'runs'/n/'configuration.json').exists()]]
    process_audit=audit_processes(runs)
    ownership={q['question_id']:BASE for q in questions}
    for run in runs[1:]:
        cfg=read(run/'configuration.json')
        for qid in cfg.get('selection',{}).get('question_ids',cfg.get('question_ids',[])):ownership[qid]=run
    evidence_cache={};public=[]
    for q in questions:
        qid=q['question_id'];video=q['video_id'];item={k:q[k] for k in ('question_id','video_id','type')};item['series']=series(q.get('source',{}).get('from',''))
        if qid in scored:
            row=scored[qid];item.update(status='scored',**{k:row[k] for k in ('score','max_score','normalized_score')})
            if qid in snap['new_first_scores']:item['source']=snap['new_first_scores'][qid]
            else:
                v=state['protected'][qid];item['source']={'run':str(BASE),'accepted_sha256':v['sha256'],'source':v['envelope']['source'],'source_sha256':v['official_source_sha256'],'prediction_sha256':v['envelope']['prediction_sha256']}
        else:
            run=ownership[qid];item.update(status='blocked',latest_run=str(run))
            folder=memory_folder(run,video);mem=read(folder/'memory.json') if (folder/'memory.json').exists() else {}
            if mem.get('complete'):item.update(backend_failure(run,q))
            else:
                key=(str(run),video)
                if key not in evidence_cache:
                    row,ledger,window,count=final_failure(folder);code=row.get('service_error_code');reason={'stage':'audio' if ledger.parent.name=='audio' else 'visual','window_id':window,'completed_windows':count,'request_hash':row['request_hash'],'attempt':row.get('attempt'),'http_status':row.get('http_status'),'service_error_code':code,'evidence_path':str(ledger),'evidence_sha256':sha(ledger),'memory_dir':str(folder)}
                    if code in ('content_filter','content_policy_violation'):reason['category']='perception_content_filter'
                    elif row.get('http_status')==428:reason.update(category='perception_dlp_rejection',response_evidence=dlp_evidence(run,video,row))
                    else:raise ValueError(f'unclassified remaining build failure: {video}/{row.get("error_type")}')
                    evidence_cache[key]=reason
                item.update(evidence_cache[key])
        public.append(item)
    # A successful answer lacking any judgment is actionable, never a blocker.
    unjudged=[];unaccepted=[]
    for run in runs:
        judged={}
        for p in (run/'scores/noevent').glob('*/run_*/scores.jsonl'):
            for row in rows(p):
                judged[row['question_id']]=row
                if row['status']=='ok' and row['question_id'] not in scored:unaccepted.append(row['question_id'])
        for p in (run/'answers/noevent').glob('*/predictions.jsonl'):
            for row in rows(p):
                if row['status']=='ok' and row['question_id'] not in judged and row['question_id'] not in scored:unjudged.append(row['question_id'])
    if unjudged or unaccepted:raise ValueError('actionable unjudged answers or unarchived scores remain')
    seeds={}
    for run in runs:
        for p in (run/'probe_seeds').glob('*/receipt.json'):
            receipt=read(p);video=receipt['video_id'];folder=run/'noevent/videos'/video/'memory'/video
            ledger=folder/('audio/calls.jsonl' if receipt['stage']=='audio' else 'calls.jsonl')
            count=sum(r.get('purpose','').endswith(receipt['window_id']) for r in rows(ledger))
            if count:raise ValueError('successful seed was re-requested')
            seeds[str(p)]={'sha256':sha(p),'stage':receipt['stage'],'window_id':receipt['window_id'],'reissued_calls':0}
    missing=Counter(x['category'] for x in public if x['status']=='blocked')
    assert len(public)==558 and len(scored)+sum(missing.values())==558
    result={'schema_version':1,'as_of':datetime.now(timezone.utc).isoformat(),'condition':'noevent','status':'executable_queues_finished_with_external_blocks','coverage_complete':len(scored)==558,'combined':snap['combined'],'by_type':grouped(public,scored,'type'),'by_series':grouped(public,scored,'series'),'missing_counts':dict(missing),'successful_answers_without_judgment':unjudged,'valid_judgments_without_accepted':unaccepted,'process_audit':process_audit,'seed_reuse_audit':seeds,'questions':public,'snapshot_sha256':sha(out/'snapshot.json'),'audit_implementation_sha256':sha(__file__)}
    (out/'final_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('as_of','combined','by_type','by_series','missing_counts')},ensure_ascii=False))

if __name__=='__main__':main()
