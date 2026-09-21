"""One fixed six-question alternative-provider recovery; prior runs are immutable."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from evaluation.io_utils import load_records
from experiments.zyf import aicodemirror_recharge as mirror
from experiments.zyf import blackai36_continuation as blackai
from experiments.zyf import claude_gemini_continuation as claude
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import atomic, read, backend
from methods.longemo.common import code_hash, file_hash, manifest

NAME='aicodemirror_v106_recovery_20260921'
VIDEO='G2_V000106'


def setup(args):
    out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    parent=args.runtime/'runs'/blackai.NAME;root=Path(__file__).parent
    if not (out/'experiment_manifest.json').exists():
        assert args.stage=='frontend'
        state=read(parent/'frontend_status.json')
        assert state['videos'][VIDEO]['status']=='frontend_failed'
        assert VIDEO not in state.get('active_videos',[]) and VIDEO not in mirror.VIDEOS
        worker=read(parent/'videos'/VIDEO/'worker_process.json')
        assert not Path('/proc',str(worker['pid'])).exists(), 'source worker still exists'
        source=read(parent/'memory'/VIDEO/'memory.json');assert len(source['completed_windows'])==36 and not source.get('complete')
        questions=[q for q in read(parent/'questions.json') if q['video_id']==VIDEO];assert len(questions)==6
        cfg=read(args.runtime/'runs'/mirror.NAME/'experiment_manifest.json')['configuration'].copy()
        cfg.update(protocol=NAME,parent_run=blackai.NAME,question_ids=[q['question_id'] for q in questions],
                   scope='Fixed six unscored V106 questions after BlackAI audio W37 schema/JSON failures; no content-filter inference. Preserve 36 inherited windows and all first scores.',
                   worker_hashes={n:file_hash(root/n) for n in [*mirror.WORKERS,'aicodemirror_v106.py','blackai_continuation.py']})
        assert cfg['method_hash']==code_hash()
        manifest(out/'experiment_manifest.json',cfg);atomic(out/'questions.json',questions)
    cfg=read(out/'experiment_manifest.json')['configuration'];assert cfg['method_hash']==code_hash()
    for name,sha in cfg['worker_hashes'].items():assert file_hash(root/name)==sha
    questions=read(out/'questions.json');assert len(questions)==6 and {q['video_id'] for q in questions}=={VIDEO}
    assert [q['question_id'] for q in questions]==cfg['question_ids']
    prepare_question_files(out,questions);(out/'outbox').mkdir(exist_ok=True)
    if args.stage=='frontend':mirror.clone(parent/'memory'/VIDEO,out/'memory'/VIDEO)
    return out,questions


def guard(runtime,questions):
    mirror.guard(runtime,questions)
    p=runtime/'runs'/mirror.NAME/'scores.jsonl'
    successful={r['question_id'] for r in load_records(p) if r.get('status')=='ok'} if p.exists() else set()
    assert not successful.intersection(q['question_id'] for q in questions)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();assert 1<=a.workers<=2
    out=a.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(a.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(a)
        if a.prepare_only:return
        atomic(out/(a.stage+'_process.json'),{'pid':os.getpid(),'workers':a.workers,'started_unix':time.time()})
        if a.stage=='frontend':
            p=out/'frontend_status.json';state=read(p) if p.exists() else {'videos':{}}
            if VIDEO not in state['videos']:
                state.update(status='running',active_videos=[VIDEO],queued_videos=[],question_count=6,updated_unix=time.time());atomic(p,state)
                result=claude.run_video(a,out,VIDEO);state['videos'][VIDEO]=result
            state.update(status='finished',active_videos=[],updated_unix=time.time());atomic(p,state)
            atomic(out/'outbox/producer_done.json',{'video_ids':[VIDEO],'time_unix':time.time()})
        else:
            guard(a.runtime,questions);backend(a,out,questions);guard(a.runtime,questions)


if __name__=='__main__':main()
