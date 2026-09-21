"""Continue 79 unscored questions with Claude vision and Gemini 2.5 audio."""
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
from experiments.zyf.aicodemirror_probe import BASE
from experiments.zyf.alternative_model_probe import CLAUDE_BASE
from experiments.zyf.blackai_continuation import atomic, backend, export_video, read, recorded_video_block
from experiments.zyf.claude_gemini_worker import AUDIO_MODEL, VISUAL_MODEL
from experiments.zyf.recover_blackai import transient_failure
from methods.longemo.common import code_hash, file_hash, manifest

NAME = 'aicodemirror_claude5_gemini25_remaining_20260921'
PARENT = 'blackai_gemini37_matrix_gpt6_compact_20260920'
VIDEO_IDS = [f'G2_V{x:06d}' for x in [48,71,104,105,106,108,110,112,113,115,116,117,129,131,137]]
CANARIES = ['G2_V000137', 'G2_V000048', 'G2_V000104']
SCORE_RUNS = ['matrix_gemini38_gpt6_full_v1', 'blackai_gemini38_matrix_gpt6_remaining_v1',
              PARENT, 'aicodemirror_gemini37_recovery_20260921']


def clone_memory(source, target):
    if target.exists():
        receipt = read(target/'checkpoint_source.json')
        assert receipt['parent_run'] == PARENT
        return receipt
    assert source.is_dir() and not source.is_symlink()
    assert not any(p.is_symlink() for p in source.rglob('*'))
    memory = read(source/'memory.json') if (source/'memory.json').exists() else {'completed_windows': []}
    assert not memory.get('complete'), 'completed memory must not be rebuilt'
    files = {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage = target.with_name(target.name+'.staging');assert not stage.exists()
    shutil.copytree(source, stage)
    assert files == {str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
    archive = stage/'ancestry'/PARENT;archive.mkdir(parents=True)
    for name in ['manifest.json', 'checkpoint_source.json', 'calls.jsonl', 'audio/calls.jsonl']:
        p = stage/name
        if p.exists():
            dest = archive/name;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
    receipt = {'parent_run':PARENT, 'files':files, 'inherited_windows':memory['completed_windows'],
               'audio_reuse':'Only exact media/prompt/config fingerprints; retain original source model'}
    atomic(stage/'checkpoint_source.json', receipt)
    stage.rename(target)
    return receipt


def setup(args):
    out = args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    if args.stage == 'frontend':
        parent = args.runtime/'runs'/PARENT
        assert read(parent/'experiment_manifest.json')['configuration']['method_hash'] == code_hash()
        questions = [q for q in read(args.runtime/'data/questions.json') if q['video_id'] in VIDEO_IDS]
        assert len(questions) == 79
        root = Path(__file__).parent
        config = {'protocol':NAME,'method_hash':code_hash(),'parent_run':PARENT,
            'worker_hashes':{name:file_hash(root/name) for name in ['claude_gemini_worker.py','aicodemirror_worker.py','native_worker.py','aicodemirror_probe.py','alternative_model_probe.py']},
            'question_ids':[q['question_id'] for q in questions],'questions_sha256':file_hash(args.runtime/'data/questions.json'),
            'data_revision':read(args.runtime/'data/manifest.json')['revision'],
            'selection':'All 15 upstream-incomplete videos in the 476/558 frozen audit; excludes prior successful scores and 3 GPT-6 rejections',
            'visual':{'model':VISUAL_MODEL,'base_url':CLAUDE_BASE,'max_tokens':8192,'temperature':None,'options':{}},
            'audio':{'model':AUDIO_MODEL,'base_url':BASE,'max_tokens':4096,'thinkingBudget':1024},
            'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':16,'max_pixels':150528},
            'answer_judge':'Matrix gpt-6-astra; unchanged official scorer',
            'embedding':'OpenRouter google/gemini-embedding-2',
            'scope':'New failure-subset experiment with inherited BlackAI Gemini windows and new AICodeMirror Claude/Gemini windows; not a homogeneous full benchmark',
            'canaries':CANARIES,'gate':'At least one complete canary memory before the other 12 videos',
            'retry':'Original 3 schema attempts per window; at most 3 extra checkpoint attempts per video for explicit transient HTTP only'}
        manifest(out/'experiment_manifest.json', config)
        if (out/'questions.json').exists():assert read(out/'questions.json') == questions
        else:atomic(out/'questions.json', questions)
        for vid in VIDEO_IDS:
            clone_memory(parent/'memory'/vid, out/'memory'/vid)
    questions = read(out/'questions.json')
    prepare_question_files(out, questions)
    config = read(out/'experiment_manifest.json')['configuration']
    assert config['method_hash'] == code_hash()
    for name, sha in config['worker_hashes'].items():assert file_hash(Path(__file__).parent/name) == sha
    return out, questions


def build_command(args, out, vid):
    folder = out/'videos'/vid;root = folder/'build_memory';root.mkdir(exist_ok=True)
    memory = out/'memory'/vid
    if not (root/vid).exists():(root/vid).symlink_to(memory,target_is_directory=True)
    assert (root/vid).resolve() == memory.resolve()
    return list(map(str, [sys.executable,'-u','-m','experiments.zyf.claude_gemini_worker','build',
        '--data-path',folder/'questions.json','--videos-dir',args.runtime/'data/episode/videos',
        '--subtitles-dir',args.runtime/'data/prepared_subtitles','--output-dir',root,
        '--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',16,'--max-pixels',150528,
        '--with-audio','--workers',1,'--model',VISUAL_MODEL,'--base-url',CLAUDE_BASE,
        '--audio-model',AUDIO_MODEL,'--audio-base-url',BASE,'--credential-file',args.credential_file,
        '--thinking','default','--timeout',240,'--tries',3,'--max-tokens',8192]))


def run_video(args, out, vid):
    folder = out/'videos'/vid;memory = out/'memory'/vid
    attempts_path = folder/'attempts.json'
    attempts = read(attempts_path) if attempts_path.exists() else {'attempts':0}
    command = build_command(args,out,vid)
    command_path = folder/'build_command.json'
    if command_path.exists():assert read(command_path)['command'] == command
    else:atomic(command_path, {'command':command})
    env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1')
    env.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE',None)
    env['PATH'] = str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
    complete = lambda: (memory/'memory.json').exists() and read(memory/'memory.json').get('complete')
    # A recorded attempt interrupted in-flight is audited before any resubmission.
    if attempts.get('in_flight') and not complete():
        result = {'video_id':vid,'status':'interrupted_request_needs_audit','attempts':attempts['attempts']}
        return result
    while not complete() and attempts['attempts'] < 4:
        if attempts['attempts'] and not transient_failure(out,vid):break
        attempts.update(attempts=attempts['attempts']+1,in_flight=True,started_unix=time.time())
        atomic(attempts_path,attempts)
        with (folder/'build.log').open('a') as log:
            proc = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(folder/'worker_process.json',{'pid':proc.pid,'attempt':attempts['attempts'],'started_unix':time.time()})
            rc = proc.wait()
        attempts.update(in_flight=False,returncode=rc,finished_unix=time.time());atomic(attempts_path,attempts)
        if complete() or not transient_failure(out,vid):break
        if attempts['attempts'] < 4:time.sleep(20)
    block = recorded_video_block(folder,memory)
    result = {'video_id':vid,'status':'memory_complete' if complete() else block['status'] if block else 'frontend_failed',
              'attempts':attempts['attempts'],'transient_remaining':False if complete() else transient_failure(out,vid)}
    if block:result['rejection'] = block['rejection']
    export_video(out,vid,result)
    return result


def canary_passed(state):
    return any(state['videos'].get(v,{}).get('status') == 'memory_complete' for v in CANARIES)


def frontend(args, out, questions):
    path = out/'frontend_status.json'
    state = read(path) if path.exists() else {'videos':{}}
    def run_group(vids, stage):
        todo = [v for v in vids if v not in state['videos']]
        workers = min(args.workers,3 if stage=='canary' else args.workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            while todo or active:
                while todo and len(active) < workers:
                    vid = todo.pop(0);active[pool.submit(run_video,args,out,vid)] = vid
                state.update(status=stage,active_videos=sorted(active.values()),queued_videos=todo.copy(),updated_unix=time.time());atomic(path,state)
                if not active:break
                done,_ = wait(active,timeout=10,return_when=FIRST_COMPLETED)
                for f in done:
                    vid = active.pop(f)
                    try:result = f.result()
                    except Exception as exc:result = {'video_id':vid,'status':'orchestration_error','error_type':type(exc).__name__}
                    state['videos'][vid] = result
                    atomic(path,state)
                    print(json.dumps(result),flush=True)
    run_group(CANARIES,'canary')
    if not canary_passed(state):
        state.update(status='paused_canary_failed',active_videos=[],queued_videos=[v for v in VIDEO_IDS if v not in state['videos']],updated_unix=time.time())
        atomic(path,state)
        return
    run_group([v for v in VIDEO_IDS if v not in CANARIES],'running')
    state.update(status='finished',active_videos=[],queued_videos=[],updated_unix=time.time());atomic(path,state)
    atomic(out/'outbox/producer_done.json',{'video_ids':sorted(state['videos']),'time_unix':time.time()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['frontend','backend']);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--credential-file',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args = p.parse_args();assert 1 <= args.workers <= 8
    out = args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out,questions = setup(args)
        atomic(out/(args.stage+'_process.json'),{'pid':os.getpid(),'started_unix':time.time(),'workers':args.workers})
        if args.stage == 'frontend':frontend(args,out,questions)
        else:
            seen = set()
            for name in SCORE_RUNS:
                rows = load_records(args.runtime/'runs'/name/'scores.jsonl')
                successful = {row['question_id'] for row in rows if row.get('status')=='ok'}
                assert not seen.intersection(successful);seen.update(successful)
            assert len(seen) == 476 and not seen.intersection(q['question_id'] for q in questions)
            backend(args,out,questions)


if __name__ == '__main__':main()
