"""Disjoint AICodeMirror continuation after the user's renewed quota authorization."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from evaluation.io_utils import load_records
from experiments.zyf import blackai36_continuation as blackai
from experiments.zyf import claude_gemini_continuation as claude
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import atomic, read, backend
from methods.longemo.common import code_hash, file_hash, manifest

NAME = 'aicodemirror_recharge_parallel_20260921'
VIDEOS = ['G2_V000104', 'G2_V000105', 'G2_V000112', 'G2_V000117', 'G2_V000129']
WORKERS = ['aicodemirror_recharge.py', 'claude_gemini_continuation.py', 'claude_gemini_worker.py',
           'aicodemirror_worker.py', 'native_worker.py', 'aicodemirror_probe.py', 'alternative_model_probe.py']


def clone(source, target):
    if target.exists():
        assert read(target/'checkpoint_source.json')['parent_run'] == blackai.NAME
        return
    assert source.is_dir() and not any(p.is_symlink() for p in source.rglob('*'))
    memory = read(source/'memory.json');assert not memory.get('complete')
    files = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage = target.with_name(target.name+'.staging');assert not stage.exists()
    target.parent.mkdir(parents=True, exist_ok=True);shutil.copytree(source,stage)
    assert files == {str(p.relative_to(stage)): file_hash(p) for p in stage.rglob('*') if p.is_file()}
    archive = stage/'ancestry'/blackai.NAME;archive.mkdir(parents=True,exist_ok=False)
    for n in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
        p=stage/n
        if p.exists():
            dest=archive/n;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
    for p in (stage/'audio').glob('W*.json'):
        if p.stem not in memory['completed_windows']:
            dest=archive/'pending_audio'/p.name;dest.parent.mkdir(exist_ok=True);p.rename(dest)
    atomic(stage/'checkpoint_source.json',{'parent_run':blackai.NAME,'files':files,
           'inherited_windows':memory['completed_windows'],'new_visual_model':'claude-opus-5',
           'new_audio_model':'gemini-2.5-pro','scope':'Preserved inherited observations; pending audio regenerated under the declared profile.'})
    stage.rename(target)


def setup(args):
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent;parent=args.runtime/'runs'/blackai.NAME
    if not (out/'experiment_manifest.json').exists():
        assert args.stage=='frontend'
        state=read(parent/'frontend_status.json')['videos']
        assert all(state[v]['status'] in ['blocked_input_policy','frontend_failed','delegated_provider'] for v in VIDEOS)
        for v in ['G2_V000105','G2_V000112','G2_V000129']:
            assert state[v]['status']=='delegated_provider' and state[v]['destination_run']==NAME
        questions=[q for q in read(parent/'questions.json') if q['video_id'] in VIDEOS]
        assert len(questions)==27
        cfg={'protocol':NAME,'method_hash':code_hash(),'parent_run':blackai.NAME,
             'worker_hashes':{n:file_hash(root/n) for n in WORKERS},
             'question_ids':[q['question_id'] for q in questions],
             'visual':{'model':'claude-opus-5','base_url':claude.CLAUDE_BASE,'max_tokens':8192,'temperature':None,'options':{}},
             'audio':{'model':'gemini-2.5-pro','base_url':claude.BASE,'max_tokens':4096,'thinkingBudget':1024},
             'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':16,'max_pixels':150528},
             'retry':'3 schema attempts; at most 3 extra attempts for explicit transient HTTP errors only',
             'answer_judge':'Unchanged Matrix GPT-6 official pipeline','embedding':'OpenRouter Gemini Embedding 2',
             'scope':'27 unscored questions. Three untouched videos reserved from BlackAI; two BlackAI failed videos. Different provider run with preserved first scores and inherited window provenance.'}
        manifest(out/'experiment_manifest.json',cfg);atomic(out/'questions.json',questions)
    cfg=read(out/'experiment_manifest.json')['configuration']
    assert cfg['method_hash']==code_hash()
    for n,sha in cfg['worker_hashes'].items():assert file_hash(root/n)==sha
    questions=read(out/'questions.json');assert [q['question_id'] for q in questions]==cfg['question_ids']
    prepare_question_files(out,questions);(out/'outbox').mkdir(exist_ok=True)
    if args.stage=='frontend':
        for vid in VIDEOS:clone(parent/'memory'/vid,out/'memory'/vid)
    return out,questions


def guard(runtime,questions):
    blackai.score_guard(runtime,questions)
    seen={r['question_id'] for r in load_records(runtime/'runs'/blackai.NAME/'scores.jsonl') if r.get('status')=='ok'}
    assert not seen.intersection(q['question_id'] for q in questions), 'overlap with first BlackAI scores'


def frontend(args,out,questions):
    path=out/'frontend_status.json';state=read(path) if path.exists() else {'videos':{}}
    todo=[v for v in VIDEOS if v not in state['videos']]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active={pool.submit(claude.run_video,args,out,v):v for v in todo}
        while active:
            state.update(status='running',active_videos=sorted(active.values()),queued_videos=[],updated_unix=time.time(),question_count=len(questions));atomic(path,state)
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                v=active.pop(f)
                try:result=f.result()
                except Exception as e:result={'video_id':v,'status':'orchestration_error','error_type':type(e).__name__}
                state['videos'][v]=result;atomic(path,state);print(json.dumps(result),flush=True)
    state.update(status='finished',active_videos=[],updated_unix=time.time());atomic(path,state)
    atomic(out/'outbox/producer_done.json',{'video_ids':VIDEOS,'time_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();assert 1<=a.workers<=8
    out=a.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(a.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(a)
        if a.prepare_only:return
        atomic(out/(a.stage+'_process.json'),{'pid':os.getpid(),'workers':a.workers,'started_unix':time.time()})
        if a.stage=='frontend':frontend(a,out,questions)
        else:guard(a.runtime,questions);backend(a,out,questions);guard(a.runtime,questions)


if __name__=='__main__':main()
