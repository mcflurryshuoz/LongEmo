"""User-authorized Sonnet 5 / Gemini 2.5 switch for 44 unscored questions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluation.io_utils import load_records
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import BLACKAI, atomic, backend, export_video, read, recorded_video_block
from experiments.zyf.blackai31_continuation import SCORE_HASHES as OLD_HASHES, NAME as OLD_NAME
from experiments.zyf.blackai36_after_sonnet_worker import MODEL, AUDIO_MODEL, VISUAL_BASE, AUDIO_BASE
from methods.longemo.common import code_hash, file_hash, manifest

NAME = 'blackai36_after_sonnet_20260921'
PARENTS = {f'G2_V{x:06d}': 'mirror_sonnet5_remaining_v2_20260921' for x in [104,105,106,112,113,116,117,129]}
SCORE_HASHES = {**OLD_HASHES, OLD_NAME:'7a36580b1b4729265cd6cdca5314a13e74cb8a5265daba7722d2003f5b4ea59b'}
WORKERS = ['blackai36_after_sonnet_continuation.py','blackai36_after_sonnet_worker.py','blackai31_worker.py','native_worker.py','blackai_continuation.py']



def guard(runtime, questions):
    seen = set()
    for name, expected in SCORE_HASHES.items():
        p = runtime/'runs'/name/'scores.jsonl'
        assert file_hash(p) == expected, 'protected first-score file changed: '+name
        ids = {q['question_id'] for q in load_records(p) if q.get('status')=='ok'}
        assert not ids.intersection(seen); seen.update(ids)
    assert len(seen)==511 and not seen.intersection(q['question_id'] for q in questions)
    # Every other prior successor must still have zero valid scores for our scope.
    for parent in [*set(PARENTS.values()),'aicodemirror_recharge_parallel_20260921','aicodemirror_v106_recovery_20260921']:
        p=runtime/'runs'/parent/'scores.jsonl'
        ids={q['question_id'] for q in load_records(p) if q.get('status')=='ok'} if p.exists() else set()
        assert not ids.intersection(q['question_id'] for q in questions)


def clone_checkpoint(source, target, parent):
    if target.exists():
        assert read(target/'checkpoint_source.json')['parent_run']==parent
        return
    assert source.is_dir() and not source.is_symlink() and not any(p.is_symlink() for p in source.rglob('*'))
    graph=read(source/'memory.json'); assert not graph.get('complete')
    hashes={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage=target.with_name(target.name+'.staging'); assert not stage.exists()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,stage)
    assert hashes=={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
    history=stage/'ancestry'/parent;history.mkdir(parents=True,exist_ok=False)
    for name in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl','probe_attempt.json','probe_result.json']:
        p=stage/name
        if p.exists():
            d=history/name;d.parent.mkdir(parents=True,exist_ok=True);p.rename(d)
    for p in (stage/'audio').glob('W*.json'):
        if p.stem not in graph['completed_windows']:
            d=history/'pending_audio'/p.name;d.parent.mkdir(exist_ok=True);p.rename(d)
    atomic(stage/'checkpoint_source.json',{'parent_run':parent,'files':hashes,'inherited_windows':graph['completed_windows'],
        'new_visual_model':MODEL,'new_audio_model':AUDIO_MODEL,'new_provider':'BlackAI',
        'audio_reuse':'Only completed parent windows retained. New unfinished-window audio uses the BlackAI Gemini 3.6 profile.',
        'media':{'max_frames':16,'max_pixels':150528,'window_seconds':20,'padding':2}})
    assert hashes=={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage.rename(target)


def setup(args):
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    if not (out/'experiment_manifest.json').exists():
        assert args.stage=='frontend'
        questions=[q for q in read(args.runtime/'data/questions.json') if q['video_id'] in PARENTS]
        assert len(questions)==44
        guard(args.runtime,questions)
        for vid,parent in PARENTS.items():
            source=args.runtime/'runs'/parent
            state=read(source/'frontend_status.json');assert state['status']=='finished'
            assert state['videos'][vid]['status'] in {'frontend_failed','blocked_input_policy'} and vid not in state.get('active_videos',[])
            worker=read(source/'videos'/vid/'worker_process.json')
            assert not Path('/proc',str(worker['pid'])).exists(), 'source worker still exists; audit PID before clone'
        config={'protocol':NAME,'method_hash':code_hash(),'parent_runs':PARENTS,
            'worker_hashes':{n:file_hash(Path(__file__).parent/n) for n in WORKERS},
            'question_ids':[q['question_id'] for q in questions],'question_sha256':file_hash(args.runtime/'data/questions.json'),
            'protected_score_hashes':SCORE_HASHES,'protected_first_scores':511,
            'visual':{'model':MODEL,'base_url':VISUAL_BASE,'max_tokens':8192,'temperature':None,'timeout':240,'options':{}},
            'audio':{'model':AUDIO_MODEL,'base_url':AUDIO_BASE,'max_tokens':4096,'temperature':None,'timeout':180,'options':{}},
            'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':16,'max_pixels':150528},
            'scheduling':{'max_video_workers':6,'max_media_decoders':2},
            'gate':'Each video: one first unfinished-window attempt per stage. Only a fully validated new window allows continuation; reuse its saved graph/audio without another request.',
            'retry':'Gate: one attempt. Remaining windows: existing three schema attempts. No outer restart in this protocol.',
            'answer_judge':'Matrix gpt-6-astra; official scorer unchanged','embedding':'OpenRouter google/gemini-embedding-2',
            'scope':'44 unscored questions/8 videos after user authorized BlackAI model switch; preserve all 511 first valid scores and inherited perception sources.'}
        manifest(out/'experiment_manifest.json',config);atomic(out/'questions.json',questions)
    cfg=read(out/'experiment_manifest.json')['configuration'];assert cfg['method_hash']==code_hash()
    for n,h in cfg['worker_hashes'].items():assert file_hash(Path(__file__).parent/n)==h
    questions=read(out/'questions.json');assert len(questions)==44 and [q['question_id'] for q in questions]==cfg['question_ids']
    guard(args.runtime,questions);prepare_question_files(out,questions);(out/'outbox').mkdir(exist_ok=True)
    if args.stage=='frontend':
        for vid,parent in PARENTS.items():clone_checkpoint(args.runtime/'runs'/parent/'memory'/vid,out/'memory'/vid,parent)
    return out,questions


def command(args,out,vid):
    folder=out/'videos'/vid;root=folder/'build_memory';root.mkdir(exist_ok=True)
    memory=out/'memory'/vid
    if not (root/vid).exists():(root/vid).symlink_to(memory,target_is_directory=True)
    assert (root/vid).resolve()==memory.resolve()
    return list(map(str,[sys.executable,'-u','-m','experiments.zyf.blackai36_after_sonnet_worker','build',
        '--data-path',folder/'questions.json','--videos-dir',args.runtime/'data/episode/videos',
        '--subtitles-dir',args.runtime/'data/prepared_subtitles','--output-dir',root,
        '--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',16,'--max-pixels',150528,
        '--with-audio','--workers',1,'--model',MODEL,'--base-url',VISUAL_BASE,'--audio-model',AUDIO_MODEL,
        '--audio-base-url',AUDIO_BASE,'--credential-file',args.credential_file,'--thinking','default',
        '--timeout',240,'--tries',3,'--max-tokens',8192]))


def run_video(args,out,vid):
    folder=out/'videos'/vid;memory=out/'memory'/vid;path=folder/'attempts.json'
    assert not path.exists(), 'already attempted; no automatic outer restart'
    cmd=command(args,out,vid);atomic(folder/'build_command.json',{'probe_command':cmd+['--probe-only'],'continuation_command':cmd})
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1',
        LONGEMO_MEDIA_LOCK_DIR=str(args.runtime/'tmp/blackai36-after-sonnet-media-locks'))
    env.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE',None);env['PATH']=str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
    def execute(phase,argv):
        row={'attempts':1,'in_flight':True,'phase':phase,'started_unix':time.time()};atomic(path,row)
        with (folder/'build.log').open('a') as log:
            proc=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
            start_ticks=int(Path('/proc',str(proc.pid),'stat').read_text().split()[21])
            atomic(folder/'worker_process.json',{'pid':proc.pid,'start_ticks':start_ticks,'phase':phase,'started_unix':time.time()})
            rc=proc.wait()
        row.update(in_flight=False,returncode=rc,finished_unix=time.time());atomic(path,row)
        return rc
    rc=execute('first_window_gate',cmd+['--probe-only'])
    if rc==0:
        result=read(memory/'probe_result.json');assert result['status']=='ok'
        assert file_hash(memory/'windows'/(result['window_id']+'.json'))==result['window_sha256']
        if not read(memory/'memory.json').get('complete'):rc=execute('remaining_windows',cmd)
    complete=bool(read(memory/'memory.json').get('complete'))
    block=recorded_video_block(folder,memory)
    result={'video_id':vid,'status':'memory_complete' if complete else block['status'] if block else 'frontend_failed',
            'returncode':rc,'gate_passed':(memory/'probe_result.json').exists(),'outer_recovery_allowed':False}
    if block:result['rejection']=block['rejection']
    last=[load_records(p)[-1] for p in [memory/'calls.jsonl',memory/'audio/calls.jsonl'] if p.exists() and load_records(p)]
    if not complete and any(x.get('http_status') in [401,402,403,429] for x in last):
        result.update(status='blocked_provider',http_status=next(x['http_status'] for x in last if x.get('http_status') in [401,402,403,429]))
    export_video(out,vid,result);return result


def frontend(args,out,questions):
    path=out/'frontend_status.json';state=read(path) if path.exists() else {'videos':{}}
    queue=[v for v in PARENTS if v not in state['videos']]
    queue.sort(key=lambda v:(read(out/'memory'/v/'memory.json')['duration']/20-len(read(out/'memory'/v/'memory.json')['completed_windows'])))
    active={};stop=False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or active:
            while queue and len(active)<args.workers and not stop:
                vid=queue.pop(0);active[pool.submit(run_video,args,out,vid)]=vid
            state.update(status='paused' if stop else 'running',active_videos=sorted(active.values()),queued_videos=queue.copy(),question_count=44,updated_unix=time.time());atomic(path,state)
            if not active:break
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                vid=active.pop(f)
                try:result=f.result()
                except Exception as exc:result={'video_id':vid,'status':'orchestration_error','error_type':type(exc).__name__}
                state['videos'][vid]=result;atomic(path,state);print(json.dumps(result),flush=True)
                if result['status'] in ['orchestration_error','blocked_provider']:stop=True
    state.update(status='paused' if queue else 'finished',active_videos=[],queued_videos=queue,updated_unix=time.time());atomic(path,state)
    if not queue and all((out/'outbox'/(v+'.ready.json')).exists() for v in PARENTS):
        atomic(out/'outbox/producer_done.json',{'video_ids':sorted(PARENTS),'time_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    assert 1<=args.workers<=6
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(args)
        if args.prepare_only:return
        atomic(out/(args.stage+'_process.json'),{'pid':os.getpid(),'started_unix':time.time(),'workers':args.workers})
        if args.stage=='frontend':frontend(args,out,questions)
        else:backend(args,out,questions);guard(args.runtime,questions)


if __name__=='__main__':main()
