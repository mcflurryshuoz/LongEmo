"""BlackAI 3.6 continuation of 69 unscored questions, preserving old scores."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from evaluation.io_utils import load_records
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import BLACKAI, atomic, backend, export_video, read, recorded_video_block
from experiments.zyf.recover_blackai import transient_failure
from methods.longemo.common import code_hash, file_hash, manifest
from experiments.zyf.blackai36_worker import MODEL

NAME = 'blackai_gemini36_matrix_gpt6_remaining_20260921'
PARENT = 'aicodemirror_claude5_gemini25_remaining_20260921'
V71_PARENT = 'aicodemirror_claude48_gemini25_retry_20260921'
CANARIES = ['G2_V000104', 'G2_V000071']
SCORE_HASHES = {
 'matrix_gemini38_gpt6_full_v1':'de3b5a7b21180cd403536e89506678c3bcd4e52287a92c9bca89a7d267482f09',
 'blackai_gemini38_matrix_gpt6_remaining_v1':'c6981f0882008259bde2e35c67689b4e64b69d1cb157f7aa3f8575563628653c',
 'blackai_gemini37_matrix_gpt6_compact_20260920':'020c62e05af0f74b887c2eff6cf1093f458632064ff780b5da76ffc749674488',
 'aicodemirror_gemini37_recovery_20260921':'b8fd9c04a20cc92e794c58f6ebfc129a9ef8df93d47bf33336f2b92c25a54ad4',
 PARENT:'1b730ca9450627a67be3d5f962dc8cfa4bcba3547f3c13cefdc13bbd919d1bb3',
}

def clone_checkpoint(source, target, parent_name):
    if target.exists():
        receipt=read(target/'checkpoint_source.json')
        assert receipt['parent_run']==parent_name
        return receipt
    assert source.is_dir() and not source.is_symlink()
    assert not any(p.is_symlink() for p in source.rglob('*'))
    old=read(source/'memory.json');assert not old.get('complete')
    files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage=target.with_name(target.name+'.staging');assert not stage.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,stage)
    assert files=={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
    archive=stage/'ancestry'/parent_name;archive.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
        p=stage/name
        if p.exists():
            dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
    # Pending audio from another model must not be relabelled or reused.
    for p in (stage/'audio').glob('W*.json'):
        if p.stem not in old['completed_windows']:
            dest=archive/'pending_audio'/p.name;dest.parent.mkdir(exist_ok=True);p.rename(dest)
    receipt={'parent_run':parent_name,'files':files,'inherited_windows':old['completed_windows'],
             'new_window_model':MODEL,'new_media':{'max_frames':8,'max_pixels':150528},
             'cache_policy':'Completed windows unchanged; old pending audio archived; diagnostic probes are not benchmark checkpoints.'}
    atomic(stage/'checkpoint_source.json',receipt);stage.rename(target);return receipt

def setup(args):
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    if not (out/'experiment_manifest.json').exists():
        assert args.stage=='frontend','transfer startup files before backend launch'
        parent=args.runtime/'runs'/PARENT
        states=read(parent/'frontend_status.json')['videos']
        vids=sorted(v for v,s in states.items() if s['status']!='memory_complete')
        questions=[q for q in read(parent/'questions.json') if q['video_id'] in vids]
        assert len(vids)==13 and len(questions)==69
        sources={v:V71_PARENT if v=='G2_V000071' else PARENT for v in vids}
        config={'protocol':NAME,'method_hash':code_hash(),
            'worker_hashes':{n:file_hash(Path(__file__).parent/n) for n in ['blackai36_worker.py','blackai36_continuation.py','native_worker.py']},
            'question_ids':[q['question_id'] for q in questions],'parent_runs':sources,'protected_score_hashes':SCORE_HASHES,
            'question_sha256':file_hash(args.runtime/'data/questions.json'),
            'visual':{'model':MODEL,'base_url':BLACKAI,'max_tokens':8192,'temperature':None,'options':{},'timeout':180},
            'audio':{'model':MODEL,'base_url':BLACKAI,'max_tokens':4096,'temperature':None,'options':{},'timeout':180},
            'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':8,'max_pixels':150528},
            'answer_judge':'Matrix gpt-6-astra; official scorer unchanged',
            'embedding':'OpenRouter google/gemini-embedding-2',
            'retry':'3 schema attempts per window; at most 3 extra checkpoint attempts per video, only for explicit transient HTTP errors',
            'scope':'69 upstream-unscored questions / 13 videos; mixed inherited and new models; preserve 486 prior first scores. Three earlier Matrix policy-blocked questions excluded.',
            'gate':'Two near-complete canaries, at least one complete before the other 11 videos'}
        manifest(out/'experiment_manifest.json',config);atomic(out/'questions.json',questions)
    cfg=read(out/'experiment_manifest.json')['configuration']
    assert cfg['method_hash']==code_hash()
    for n,sha in cfg['worker_hashes'].items():assert file_hash(Path(__file__).parent/n)==sha
    questions=read(out/'questions.json');assert [q['question_id'] for q in questions]==cfg['question_ids']
    prepare_question_files(out,questions)
    if args.stage=='frontend':
        for vid,parent in cfg['parent_runs'].items():
            clone_checkpoint(args.runtime/'runs'/parent/'memory'/vid,out/'memory'/vid,parent)
    (out/'outbox').mkdir(exist_ok=True)
    return out,questions

def score_guard(runtime,questions):
    seen=set()
    for n,sha in SCORE_HASHES.items():
        path=runtime/'runs'/n/'scores.jsonl';assert file_hash(path)==sha, f'protected score file changed: {n}'
        ids={r['question_id'] for r in load_records(path) if r.get('status')=='ok'}
        assert not seen.intersection(ids);seen.update(ids)
    assert len(seen)==486 and not seen.intersection(q['question_id'] for q in questions)
    return len(seen)

def run_video(args,out,vid):
    folder=out/'videos'/vid;memory=out/'memory'/vid;root=folder/'build_memory';root.mkdir(exist_ok=True)
    if not (root/vid).exists():(root/vid).symlink_to(memory,target_is_directory=True)
    assert (root/vid).resolve()==memory.resolve()
    cmd=list(map(str,[sys.executable,'-u','-m','experiments.zyf.blackai36_worker','build',
        '--data-path',folder/'questions.json','--videos-dir',args.runtime/'data/episode/videos',
        '--subtitles-dir',args.runtime/'data/prepared_subtitles','--output-dir',root,
        '--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',8,'--max-pixels',150528,
        '--with-audio','--workers',1,'--model',MODEL,'--base-url',BLACKAI,
        '--audio-model',MODEL,'--audio-base-url',BLACKAI,'--credential-file',args.credential_file,
        '--thinking','default','--timeout',180,'--tries',3,'--max-tokens',8192]))
    cp=folder/'build_command.json'
    if cp.exists():assert read(cp)['command']==cmd
    else:atomic(cp,{'command':cmd})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1')
    env.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE',None);env['PATH']=str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
    ap=folder/'attempts.json';attempts=read(ap) if ap.exists() else {'attempts':0}
    complete=lambda:read(memory/'memory.json').get('complete',False)
    if attempts.get('in_flight') and not complete():
        return {'video_id':vid,'status':'interrupted_request_needs_audit','attempts':attempts['attempts']}
    while not complete() and attempts['attempts']<4:
        if attempts['attempts'] and not transient_failure(out,vid):break
        attempts.update(attempts=attempts['attempts']+1,in_flight=True,started_unix=time.time());atomic(ap,attempts)
        with (folder/'build.log').open('a') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(folder/'worker_process.json',{'pid':proc.pid,'attempt':attempts['attempts'],'started_unix':time.time()})
            rc=proc.wait()
        attempts.update(in_flight=False,returncode=rc,finished_unix=time.time());atomic(ap,attempts)
        if complete() or not transient_failure(out,vid):break
        if attempts['attempts']<4:time.sleep(15)
    block=recorded_video_block(folder,memory)
    result={'video_id':vid,'status':'memory_complete' if complete() else block['status'] if block else 'frontend_failed',
            'attempts':attempts['attempts'],'transient_remaining':False if complete() else transient_failure(out,vid)}
    if block:result['rejection']=block['rejection']
    last=[]
    for p in [memory/'calls.jsonl',memory/'audio/calls.jsonl']:
        if p.exists() and load_records(p):last.append(load_records(p)[-1])
    if not complete() and any(r.get('http_status') in (401,402,403) for r in last):
        result.update(status='blocked_provider',http_status=next(r['http_status'] for r in last if r.get('http_status') in (401,402,403)))
    export_video(out,vid,result);return result

def frontend(args,out,questions):
    path=out/'frontend_status.json';state=read(path) if path.exists() else {'videos':{}}
    all_vids=sorted({q['video_id'] for q in questions})
    stop=False
    def group(vids,stage):
        nonlocal stop
        todo=[v for v in vids if v not in state['videos']]
        workers=min(args.workers,2 if stage=='canary' else args.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active={}
            while todo or active:
                while todo and len(active)<workers and not stop:
                    v=todo.pop(0);active[pool.submit(run_video,args,out,v)]=v
                state.update(status='paused' if stop else stage,active_videos=sorted(active.values()),queued_videos=todo.copy(),question_count=len(questions),updated_unix=time.time());atomic(path,state)
                if not active:break
                done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    vid=active.pop(f)
                    try:result=f.result()
                    except Exception as exc:result={'video_id':vid,'status':'orchestration_error','error_type':type(exc).__name__}
                    state['videos'][vid]=result
                    if result['status'] in ('blocked_provider','orchestration_error','interrupted_request_needs_audit'):stop=True
                    atomic(path,state);print(json.dumps(result),flush=True)
    group(CANARIES,'canary')
    if not stop and any(state['videos'].get(v,{}).get('status')=='memory_complete' for v in CANARIES):
        remaining=[v for v in all_vids if v not in CANARIES]
        remaining.sort(key=lambda v:math.ceil(read(out/'memory'/v/'memory.json')['duration']/20)-len(read(out/'memory'/v/'memory.json')['completed_windows']))
        group(remaining,'running')
    pending=[v for v in all_vids if v not in state['videos']]
    state.update(status='paused' if pending else 'finished',active_videos=[],queued_videos=pending,updated_unix=time.time());atomic(path,state)
    if not pending:atomic(out/'outbox/producer_done.json',{'video_ids':all_vids,'time_unix':time.time()})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();assert 1<=args.workers<=4
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(args)
        if args.prepare_only:
            print(json.dumps({'run':str(out),'questions':len(questions),'videos':len({q['video_id'] for q in questions})}));return
        atomic(out/(args.stage+'_process.json'),{'pid':os.getpid(),'started_unix':time.time(),'workers':args.workers})
        if args.stage=='frontend':frontend(args,out,questions)
        else:
            score_guard(args.runtime,questions);backend(args,out,questions);score_guard(args.runtime,questions)
if __name__=='__main__':main()
