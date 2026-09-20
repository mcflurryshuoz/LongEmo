"""One user-authorized checkpoint retry in an isolated run; no old scores change."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.zyf.blackai_continuation import read, atomic, export_video, backend
from experiments.zyf.azure_benchmark import recorded_video_block
from methods.longemo.common import file_hash, manifest

NAME = 'blackai_gemini38_matrix_gpt6_retry_20260920'


def setup(runtime, selection):
    selected = read(selection)
    parent = runtime/'runs'/selected['source_run']
    source_manifest = read(parent/'experiment_manifest.json')
    assert source_manifest['fingerprint'] == selected['source_fingerprint']
    qs = read(parent/'questions.json')
    ids = selected['question_ids']
    assert len(ids) == len(set(ids)) and set(ids) <= {q['question_id'] for q in qs}
    questions = [q for q in qs if q['question_id'] in ids]
    assert {q['video_id'] for q in questions} == set(selected['video_ids'])
    out = runtime/'runs'/NAME
    config = {**source_manifest['configuration'], 'protocol':NAME,
              'target_question_count':len(ids), 'selection_sha256':file_hash(selection),
              'parent_fingerprint':selected['source_fingerprint'],
              'retry_rule':'One additional build invocation per selected incomplete video; unchanged input/model/media; existing checkpoints reused.'}
    manifest(out/'experiment_manifest.json',config)
    atomic(out/'questions.json',questions)
    for vid in selected['video_ids']:
        atomic(out/'videos'/vid/'questions.json',[q for q in questions if q['video_id']==vid])
    return parent,out,questions,selected


def clone_checkpoint(parent,out,vid):
    receipt_path=out/'checkpoint_sources'/(vid+'.json')
    if receipt_path.exists():
        return read(receipt_path)
    source=parent/'memory'/vid
    assert not read(source/'memory.json').get('complete',False) if (source/'memory.json').exists() else True
    assert not (parent/'videos'/vid/'graph/predictions.jsonl').exists()
    assert not (parent/'videos'/vid/'accepted_scores.jsonl').exists()
    files={str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    assert files and all(not p.is_symlink() for p in source.rglob('*'))
    target=out/'memory'/vid
    stage=out/'checkpoint_staging'/vid
    if not target.exists():
        stage.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,stage,dirs_exist_ok=True)
        assert {str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}==files
        target.parent.mkdir(parents=True,exist_ok=True)
        stage.rename(target)
    assert {str(p.relative_to(target)):file_hash(p) for p in target.rglob('*') if p.is_file()}==files
    result={'source_run':parent.name,'video_id':vid,'files':files,'time_unix':time.time()}
    atomic(receipt_path,result)
    return result


def frontend(args,parent,out,questions,selected):
    source_states={**read(parent/'frontend_status.json')['videos'],
                   **read(parent/'recovery/frontend_status.json')['videos']}
    path=out/'frontend_status.json'
    state=read(path) if path.exists() else {'videos':{}}
    todo=[v for v in selected['video_ids'] if v not in state['videos']]
    for vid in todo:
        assert source_states[vid]['status'] in ('frontend_failed','recovery_exhausted')
        clone_checkpoint(parent,out,vid)
    compat = args.runtime/'tmp'/'ffmpeg_compat.py'
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if compat.exists():
        env['PATH'] = str(compat.parent) + os.pathsep + env.get('PATH','')
        env['LONGEMO_FFMPEG_COMPAT'] = '1'
        env['LONGEMO_ALLOW_AUDIO_CACHE_REUSE'] = '1'
        # The media loader resolves the executable by name.  Keep this
        # compatibility shim scoped to the retry environment on g450.
        shim = compat.parent/'ffmpeg'
        if not shim.exists():
            shim.symlink_to(compat)

    def work(vid):
        folder=out/'videos'/vid
        intent=folder/'retry_intent.json'
        if intent.exists():
            return {'video_id':vid,'status':'retry_submission_unresolved'}
        root=folder/'build_memory';root.mkdir(exist_ok=True)
        link = root/vid
        target = (out/'memory'/vid).resolve()
        if link.exists() or link.is_symlink():
            assert link.is_symlink() and link.resolve() == target
        else:
            link.symlink_to(target,target_is_directory=True)
        command=list(read(parent/'videos'/vid/'build_command.json')['command'])
        assert command[2:4]==['-m','experiments.zyf.native_worker']
        command[command.index('--output-dir')+1]=str(root)
        command[command.index('--data-path')+1]=str(folder/'questions.json')
        atomic(folder/'build_command.json',{'command':command,'time_unix':time.time()})
        atomic(intent,{'time_unix':time.time(),'attempts':1,'selection_sha256':file_hash(args.selection)})
        with (folder/'build.log').open('a') as log:
            rc=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        memory=out/'memory'/vid
        complete=(memory/'memory.json').exists() and read(memory/'memory.json').get('complete')
        block=recorded_video_block(folder,memory)
        result={'video_id':vid,'status':'memory_complete' if complete and rc==0 else
                block['status'] if block else 'frontend_failed','returncode':rc}
        if block: result['rejection']=block['rejection']
        export_video(out,vid,result)
        return result

    state.update(status='running',queued_videos=todo,updated_unix=time.time());atomic(path,state)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,v):v for v in todo}
        for f in as_completed(futures):
            vid=futures[f]
            try: result=f.result()
            except Exception as exc: result={'video_id':vid,'status':'orchestration_error','error_type':type(exc).__name__}
            state['videos'][vid]=result
            state.update(queued_videos=[v for v in todo if v not in state['videos']],updated_unix=time.time())
            atomic(path,state)
            print(result,flush=True)
    state.update(status='finished',queued_videos=[],updated_unix=time.time());atomic(path,state)
    atomic(out/'outbox/producer_done.json',{'video_ids':sorted(v for v in selected['video_ids'] if (out/'outbox'/(v+'.ready.json')).exists()),'time_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();assert 1<=args.workers<=8
    parent,out,questions,selected=setup(args.runtime,args.selection)
    with (out/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.stage=='frontend':frontend(args,parent,out,questions,selected)
        else:
            assert file_hash(parent/'scores.jsonl')==selected['source_scores_sha256']
            assert not any(q['question_id'] in selected['question_ids'] and q.get('status')=='ok'
                           for q in __import__('evaluation.io_utils',fromlist=['load_records']).load_records(parent/'scores.jsonl'))
            backend(args,out,questions)


if __name__=='__main__':main()
