"""Low-concurrency recovery canary for a failed hybrid-perception video."""
import argparse
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
from experiments.zyf.blackai_continuation import atomic, export_video, read, recorded_video_block
from experiments.zyf.claude_gemini_continuation import (
    NAME as SOURCE, PARENT as GRANDPARENT, CLAUDE_BASE, BASE, VISUAL_MODEL, AUDIO_MODEL,
)
from methods.longemo.common import code_hash, file_hash, manifest

NAME = os.environ.get('CLAUDE_GEMINI_RETRY_NAME', 'aicodemirror_claude5_gemini25_retry_20260921')
DEFAULT_VIDEO = 'G2_V000071'
VISUAL_MODEL = os.environ.get('CLAUDE_GEMINI_VISUAL_MODEL', 'claude-opus-5')
AUDIO_MODEL = os.environ.get('CLAUDE_GEMINI_AUDIO_MODEL', 'gemini-2.5-pro')
MAX_FRAMES = int(os.environ.get('CLAUDE_GEMINI_MAX_FRAMES', '8'))
AUDIO_PROMPT = """Observe only the supplied audio. Return JSON {observations:[{span:[start,end],voice:string,cue:string,uncertainty:string}]}.
Use absolute video seconds. Cover the whole clip and transcribe audible words and tone.
The voice field is REQUIRED and MUST be nonempty text; when identity is unclear use exactly \"unknown speaker\".
The cue field is REQUIRED and MUST be nonempty text; when no speech is intelligible describe the audible sound.
Do not infer visuals, motives, or psychological conclusions."""


def clone_memory(source, target):
    if target.exists():
        receipt = read(target/'checkpoint_source.json')
        assert receipt['parent_run'] == SOURCE
        return receipt
    memory = read(source/'memory.json')
    assert not memory.get('complete')
    assert not any(p.is_symlink() for p in source.rglob('*'))
    files = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage = target.with_name(target.name+'.staging'); assert not stage.exists()
    shutil.copytree(source, stage)
    assert files == {str(p.relative_to(stage)): file_hash(p) for p in stage.rglob('*') if p.is_file()}
    archive = stage/'ancestry'/SOURCE;archive.mkdir(parents=True)
    for name in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
        p=stage/name
        if p.exists():
            d=archive/name;d.parent.mkdir(parents=True,exist_ok=True);p.rename(d)
    receipt={'parent_run':SOURCE,'grandparent_run':GRANDPARENT,'files':files,
             'inherited_windows':memory['completed_windows'],'new_max_frames':MAX_FRAMES,
             'audio_prompt_sha256':__import__('hashlib').sha256(AUDIO_PROMPT.encode()).hexdigest()}
    atomic(stage/'checkpoint_source.json',receipt);stage.rename(target);return receipt


def setup(args):
    source=args.runtime/'runs'/SOURCE;out=args.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    vid=args.video_id
    questions=[q for q in read(source/'questions.json') if q['video_id']==vid]
    assert questions
    config={'protocol':NAME,'method_hash':code_hash(),'source_run':SOURCE,'grandparent_run':GRANDPARENT,
        'question_ids':[q['question_id'] for q in questions],'visual':{'model':VISUAL_MODEL,'base_url':CLAUDE_BASE,'max_tokens':8192,'max_frames':MAX_FRAMES},
        'audio':{'model':AUDIO_MODEL,'base_url':BASE,'max_tokens':4096,'thinkingBudget':1024,'prompt_sha256':__import__('hashlib').sha256(AUDIO_PROMPT.encode()).hexdigest()},
        'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':MAX_FRAMES,'max_pixels':150528},
        'scope':'One failed video only; lower visual frame budget and stricter audio schema; no old score replay'}
    manifest(out/'experiment_manifest.json',config)
    if (out/'questions.json').exists():assert read(out/'questions.json')==questions
    else:atomic(out/'questions.json',questions)
    clone_memory(source/'memory'/vid,out/'memory'/vid)
    prepare_question_files(out,questions)
    return out,questions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--credential-file',type=Path,required=True);p.add_argument('--video-id',default=DEFAULT_VIDEO)
    p.add_argument('--run-name',default=None);p.add_argument('--visual-model',default=None);p.add_argument('--audio-model',default=None);p.add_argument('--max-frames',type=int,default=None)
    args=p.parse_args()
    global NAME, VISUAL_MODEL, AUDIO_MODEL, MAX_FRAMES
    if args.run_name: NAME=args.run_name
    if args.visual_model: VISUAL_MODEL=args.visual_model
    if args.audio_model: AUDIO_MODEL=args.audio_model
    if args.max_frames: MAX_FRAMES=args.max_frames
    out,questions=setup(args);vid=args.video_id
    with (out/'frontend.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=out/'videos'/vid;root=folder/'build_memory';root.mkdir(exist_ok=True);memory=out/'memory'/vid
        link=root/vid
        if not link.exists():link.symlink_to(memory,target_is_directory=True)
        command=list(map(str,[sys.executable,'-u','-m','experiments.zyf.claude_gemini_worker','build',
            '--data-path',folder/'questions.json','--videos-dir',args.runtime/'data/episode/videos','--subtitles-dir',args.runtime/'data/prepared_subtitles',
            '--output-dir',root,'--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',MAX_FRAMES,'--max-pixels',150528,
            '--with-audio','--workers',1,'--model',VISUAL_MODEL,'--base-url',CLAUDE_BASE,'--audio-model',AUDIO_MODEL,'--audio-base-url',BASE,
            '--credential-file',args.credential_file,'--thinking','default','--timeout',300,'--tries',3,'--max-tokens',8192]))
        atomic(folder/'build_command.json',{'command':command,'retry_reason':'lower visual frame budget and single worker'})
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1',CLAUDE_GEMINI_AUDIO_PROMPT=AUDIO_PROMPT)
        env['PATH']=str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
        with (folder/'build.log').open('a') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic(out/'frontend_process.json',{'pid':proc.pid,'started_unix':time.time(),'video_id':vid,'max_frames':MAX_FRAMES})
            rc=proc.wait()
        complete=(memory/'memory.json').exists() and read(memory/'memory.json').get('complete')
        block=recorded_video_block(folder,memory)
        state={'video_id':vid,'status':'memory_complete' if rc==0 and complete else block['status'] if block else 'frontend_failed','returncode':rc,'max_frames':MAX_FRAMES}
        if block:state['rejection']=block['rejection']
        atomic(out/'frontend_status.json',{'videos':{vid:state},'status':'finished','active_videos':[],'updated_unix':time.time()})
        export_video(out,vid,state)
        print(json.dumps(state),flush=True)
        raise SystemExit(0 if complete else 1)


if __name__=='__main__':main()
