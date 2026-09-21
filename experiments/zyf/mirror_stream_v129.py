"""Four-question stream continuation after the former V129 worker exhausted HTTP recovery."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from evaluation.io_utils import load_records
from experiments.zyf import mirror_stream_continuation as stream
from experiments.zyf import aicodemirror_recharge as mirror
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import atomic,read,backend
from methods.longemo.common import code_hash,file_hash,manifest

NAME='aicodemirror_stream_v129_20260921'
VIDEO='G2_V000129'


def guard(runtime,questions):
    stream.guard(runtime,questions)
    assert VIDEO not in {q['video_id'] for q in read(runtime/'runs'/stream.NAME/'questions.json')}
    p=runtime/'runs'/stream.NAME/'scores.jsonl'
    seen={r['question_id'] for r in load_records(p) if r.get('status')=='ok'} if p.exists() else set()
    assert not seen.intersection(q['question_id'] for q in questions)


def setup(args):
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).parent
    if not (out/'experiment_manifest.json').exists():
        assert args.stage=='frontend'
        parent=args.runtime/'runs'/mirror.NAME
        state=read(parent/'frontend_status.json')
        assert state['status']=='finished' and state['videos'][VIDEO]['status']=='frontend_failed'
        assert state['videos'][VIDEO]['attempts']==4
        graph=read(parent/'memory'/VIDEO/'memory.json');assert len(graph['completed_windows'])==29
        questions=[q for q in read(parent/'questions.json') if q['video_id']==VIDEO];assert len(questions)==4
        guard(args.runtime,questions)
        cfg=read(args.runtime/'runs'/stream.NAME/'experiment_manifest.json')['configuration'].copy()
        cfg.update(protocol=NAME,parent_runs={VIDEO:mirror.NAME},question_ids=[q['question_id'] for q in questions],
            worker_hashes={**cfg['worker_hashes'],'mirror_stream_v129.py':file_hash(Path(__file__))},
            scope='Four disjoint V129 questions after original nonstream worker finished at 29/64 with HTTP524. Preserve frozen 45-question stream run and all first scores. One checkpoint attempt only.')
        manifest(out/'experiment_manifest.json',cfg);atomic(out/'questions.json',questions)
    cfg=read(out/'experiment_manifest.json')['configuration'];assert cfg['method_hash']==code_hash()
    for name,h in cfg['worker_hashes'].items():assert file_hash(root/name)==h
    questions=read(out/'questions.json');assert len(questions)==4 and [q['question_id'] for q in questions]==cfg['question_ids']
    guard(args.runtime,questions);prepare_question_files(out,questions);(out/'outbox').mkdir(exist_ok=True)
    if args.stage=='frontend':
        original=stream.PARENTS
        try:
            stream.PARENTS={VIDEO:mirror.NAME};stream.clone(args.runtime,out,VIDEO)
        finally:stream.PARENTS=original
    return out,questions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert a.workers==1
    out=a.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(a.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(a)
        if a.prepare_only:return
        atomic(out/(a.stage+'_process.json'),{'pid':os.getpid(),'workers':1,'started_unix':time.time()})
        if a.stage=='frontend':
            path=out/'frontend_status.json';state=read(path) if path.exists() else {'videos':{}}
            if VIDEO not in state['videos']:
                state.update(status='running',active_videos=[VIDEO],queued_videos=[],question_count=4,updated_unix=time.time());atomic(path,state)
                state['videos'][VIDEO]=stream.run_video(a,out,VIDEO)
            state.update(status='finished',active_videos=[],updated_unix=time.time());atomic(path,state)
            atomic(out/'outbox/producer_done.json',{'video_ids':[VIDEO],'time_unix':time.time()})
        else:backend(a,out,questions);guard(a.runtime,questions)


if __name__=='__main__':main()
