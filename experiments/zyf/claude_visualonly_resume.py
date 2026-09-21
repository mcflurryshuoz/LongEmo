"""Finish a hybrid graph with visual-only windows after the audio provider is exhausted.

This is an explicitly separate fallback: inherited windows retain their original
Claude 4.8 + Gemini 2.5 observations; only uncompleted windows omit audio.
"""
import argparse, fcntl, json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path
from evaluation.io_utils import load_questions, write_json
from experiments.zyf.blackai_continuation import atomic, export_video, read, recorded_video_block
from experiments.zyf.claude_gemini_continuation import CLAUDE_BASE
from methods.longemo.common import code_hash, file_hash, manifest

DEFAULT_SOURCE='aicodemirror_claude48_gemini25_retry_20260921'
DEFAULT_NAME='aicodemirror_claude48_visualonly_v71_20260921'
DEFAULT_VIDEO='G2_V000071'
MODEL='claude-opus-4-8'
MAX_FRAMES=8

def setup(runtime, source_name, name, vid):
    source=runtime/'runs'/source_name; out=runtime/'runs'/name
    assert (source/'memory'/vid/'memory.json').exists()
    assert not out.exists(), f'target exists: {out}'
    questions=read(source/'questions.json')
    questions=[q for q in questions if q['video_id']==vid]; assert questions
    out.mkdir(parents=True)
    cfg={'protocol':name,'source_run':source_name,'source_memory_sha256':file_hash(source/'memory'/vid/'memory.json'),
         'method_hash':code_hash(),'question_ids':[q['question_id'] for q in questions],
         'visual':{'model':MODEL,'base_url':CLAUDE_BASE,'max_tokens':8192,'max_frames':MAX_FRAMES},
         'audio':{'status':'disabled_after_provider_balance_insufficient','inherited_model':'gemini-2.5-pro'},
         'media':{'window_seconds':20,'padding':2,'fps':1,'max_frames':MAX_FRAMES,'max_pixels':150528,'with_audio':False},
         'scope':'Fallback continuation: inherited 49 windows keep audio; remaining windows use visual-only Claude 4.8 after Gemini audio HTTP 402.'}
    manifest(out/'experiment_manifest.json',cfg); atomic(out/'questions.json',questions)
    target=out/'memory'/vid; stage=target.with_name(target.name+'.staging'); stage.parent.mkdir(parents=True)
    shutil.copytree(source/'memory'/vid,stage)
    assert not any(p.is_symlink() for p in stage.rglob('*'))
    archive=stage/'ancestry'/source_name; archive.mkdir(parents=True)
    for n in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
        q=stage/n
        if q.exists():
            d=archive/n; d.parent.mkdir(parents=True,exist_ok=True); q.rename(d)
    receipt={'parent_run':source_name,'source_memory_sha256':file_hash(source/'memory'/vid/'memory.json'),
             'inherited_windows':read(source/'memory'/vid/'memory.json')['completed_windows'],
             'fallback':'visual_only','model':MODEL}
    atomic(stage/'checkpoint_source.json',receipt); stage.rename(target)
    folder=out/'videos'/vid; folder.mkdir(parents=True)
    atomic(folder/'questions.json',questions)
    root=folder/'build_memory'; root.mkdir(); (root/vid).symlink_to(target,target_is_directory=True)
    return out,folder,target

def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--source-run',default=DEFAULT_SOURCE); p.add_argument('--run-name',default=DEFAULT_NAME); p.add_argument('--video-id',default=DEFAULT_VIDEO)
    a=p.parse_args(); out,folder,memory=setup(a.runtime,a.source_run,a.run_name,a.video_id)
    lock=out/'frontend.lock'
    with lock.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cmd=[sys.executable,'-u','-m','experiments.zyf.claude_gemini_worker','build','--data-path',folder/'questions.json','--videos-dir',a.runtime/'data/episode/videos','--subtitles-dir',a.runtime/'data/prepared_subtitles','--output-dir',folder/'build_memory','--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',MAX_FRAMES,'--max-pixels',150528,'--workers',1,'--model',MODEL,'--base-url',CLAUDE_BASE,'--credential-file',a.credential_file,'--thinking','default','--timeout',300,'--tries',3,'--max-tokens',8192]
        atomic(folder/'build_command.json',{'command':list(map(str,cmd)),'fallback':'visual_only'})
        env=dict(os.environ,CLAUDE_GEMINI_VISUAL_MODEL=MODEL,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1')
        env['PATH']=str(a.runtime/'tmp')+os.pathsep+env.get('PATH','')
        with (folder/'build.log').open('a') as log:
            atomic(out/'frontend_process.json',{'pid':None,'started_unix':time.time(),'video_id':a.video_id,'visual_only':True})
            rc=subprocess.run(list(map(str,cmd)),cwd='/mnt/data1/zyf/LongEmo-zyf',env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        complete=(memory/'memory.json').exists() and read(memory/'memory.json').get('complete')
        block=recorded_video_block(folder,memory)
        state={'video_id':a.video_id,'status':'memory_complete' if rc==0 and complete else block['status'] if block else 'frontend_failed','returncode':rc,'visual_only':True}
        if block: state['rejection']=block['rejection']
        atomic(out/'frontend_status.json',{'videos':{a.video_id:state},'status':'finished','active_videos':[],'updated_unix':time.time()})
        export_video(out,a.video_id,state)
        print(json.dumps(state,ensure_ascii=False)); raise SystemExit(0 if complete else 1)
if __name__=='__main__': main()
