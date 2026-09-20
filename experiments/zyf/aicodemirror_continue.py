"""Continue only alternate-provider probes that were not service refusals."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.zyf.aicodemirror_probe import BASE, MODEL, PARENT
from experiments.zyf.blackai_continuation import read, atomic, export_video, backend, recorded_video_block
from methods.longemo.common import code_hash, file_hash, manifest
from evaluation.io_utils import load_records

NAME = "aicodemirror_gemini37_recovery_20260921"


def setup(args):
    parent = args.runtime / "runs" / PARENT
    out = args.runtime / "runs" / NAME
    out.mkdir(parents=True,exist_ok=True)
    if args.stage == "frontend":
        diagnostics = args.runtime / "diagnostics/aicodemirror_remaining_20260921"
        states = [read(p) for p in diagnostics.glob("*/result.json")]
        vids = sorted(r["video_id"] for r in states if r["status"] == "ok" or r.get("error_type") == "ValueError")
        assert vids == ["G2_V000067", "G2_V000108", "G2_V000115"]
        qs = [q for q in read(parent/"questions.json") if q["video_id"] in vids]
        config = {"protocol": NAME,"parent_run":PARENT,"method_hash":code_hash(),
            "selection_basis":"16 alternate-provider probes; select 2 valid windows and 1 schema-error response; exclude 13 explicit service refusals",
            "question_ids":[q["question_id"] for q in qs],"model":MODEL,"base_url":BASE,
            "media":{"window_seconds":20,"padding":2,"fps":1,"max_frames":16,"max_pixels":150528},
            "max_tokens":8192,"visual_thinking":"medium","audio_thinking":"low","temperature":1,
            "scope":"Mixed inherited BlackAI 3.8/3.7 windows plus new AICodeMirror 3.7 windows; no successful scores replayed",
            "audio_cache":"Reuse only exact media/prompt fingerprints; retain original provider provenance",
            "answer_judge":"Matrix gpt-6-astra","embedding":"OpenRouter google/gemini-embedding-2"}
        manifest(out/"experiment_manifest.json",config)
        atomic(out/"questions.json",qs)
        for vid in vids:
            source=parent/"memory"/vid;target=out/"memory"/vid
            atomic(out/"videos"/vid/"questions.json",[q for q in qs if q["video_id"]==vid])
            if target.exists():
                assert (target/"checkpoint_source.json").exists()
                continue
            old=read(source/"memory.json");assert not old.get("complete")
            files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
            assert not any(p.is_symlink() for p in source.rglob('*'))
            stage=target.with_name(vid+'.staging');assert not stage.exists()
            shutil.copytree(source,stage)
            assert files=={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
            archive=stage/"ancestry"/PARENT;archive.mkdir(parents=True)
            for name in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
                p=stage/name
                if p.exists():
                    dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);p.rename(dest)
            atomic(stage/'checkpoint_source.json',{'parent_run':PARENT,'files':files,'inherited_windows':old['completed_windows']})
            stage.rename(target)
    return out,read(out/'questions.json')


def frontend(args,out,questions):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LONGEMO_FFMPEG_COMPAT='1')
    env.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE',None)
    env['PATH']=str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
    setting=out/'perception_settings.json'
    atomic(setting,{'generationConfig':{'thinkingConfig':{'thinkingLevel':'medium'}}})
    state_path=out/'frontend_status.json'
    state=read(state_path) if state_path.exists() else {'videos':{}}
    todo=sorted({q['video_id'] for q in questions}-set(state['videos']))
    state.update(status='running',active_videos=todo,updated_unix=time.time());atomic(state_path,state)
    def work(vid):
        folder=out/'videos'/vid;memory=out/'memory'/vid;root=folder/'build_memory';root.mkdir(exist_ok=True)
        if not (root/vid).exists():(root/vid).symlink_to(memory,target_is_directory=True)
        command=[sys.executable,'-u','-m','experiments.zyf.aicodemirror_worker','build',
            '--data-path',folder/'questions.json','--videos-dir',args.runtime/'data/episode/videos',
            '--subtitles-dir',args.runtime/'data/prepared_subtitles','--output-dir',root,
            '--window-seconds',20,'--padding',2,'--fps',1,'--max-frames',16,'--max-pixels',150528,
            '--with-audio','--workers',1,'--model',MODEL,'--base-url',BASE,'--audio-model',MODEL,
            '--audio-base-url',BASE,'--credential-file',args.credential_file,'--thinking','default',
            '--timeout',240,'--tries',3,'--max-tokens',8192,'--temperature',1,'--config',setting]
        command=list(map(str,command));atomic(folder/'build_command.json',{'command':command})
        with (folder/'build.log').open('a') as log:rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        block=recorded_video_block(folder,memory)
        result={'video_id':vid,'returncode':rc,'status':'memory_complete' if rc==0 and read(memory/'memory.json').get('complete') else block['status'] if block else 'frontend_failed'}
        if block:result['rejection']=block['rejection']
        export_video(out,vid,result)
        return result
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(work,v) for v in todo]):
            result=f.result();state['videos'][result['video_id']]=result
            state.update(active_videos=[v for v in todo if v not in state['videos']],updated_unix=time.time());atomic(state_path,state)
            print(json.dumps(result),flush=True)
    state.update(status='finished',active_videos=[],updated_unix=time.time());atomic(state_path,state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['frontend','backend']);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--credential-file',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();assert 1<=a.workers<=3
    out,qs=setup(a)
    with (out/(a.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic(out/(a.stage+'_process.json'),{'pid':os.getpid(),'started_unix':time.time()})
        if a.stage=='frontend':frontend(a,out,qs)
        else:
            for name in ['matrix_gemini38_gpt6_full_v1','blackai_gemini38_matrix_gpt6_remaining_v1','blackai_gemini37_matrix_gpt6_compact_20260920']:
                scores=a.runtime/'runs'/name/'scores.jsonl'
                assert scores.exists()
                assert not {r['question_id'] for r in load_records(scores) if r.get('status')=='ok'}.intersection(q['question_id'] for q in qs)
            backend(a,out,qs)


if __name__=='__main__':main()
