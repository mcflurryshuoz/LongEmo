"""Fixed 45-question alternate transport/provider run after a successful exact-input probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from evaluation.io_utils import load_records
from experiments.zyf import blackai36_continuation as blackai
from experiments.zyf import aicodemirror_recharge as mirror
from experiments.zyf import aicodemirror_v106 as v106
from experiments.zyf import claude_gemini_continuation as claude
from experiments.zyf.aicodemirror_continue import prepare_question_files
from experiments.zyf.blackai_continuation import atomic, read, backend, export_video, recorded_video_block
from experiments.zyf.mirror_stream_probe import EXPECTED_REQUEST
from experiments.zyf.recover_blackai import transient_failure
from methods.longemo.common import code_hash, file_hash, manifest, fingerprint
from methods.longemo.media import window_input, subtitles
from methods.longemo.memory import apply_window
from methods.longemo.prompts import PERCEPTION

NAME = 'aicodemirror_stream_remaining_20260921'
PARENTS = {**{f'G2_V{x:06d}': mirror.NAME for x in [104,105,112,117]},
           'G2_V000106': v106.NAME,
           **{f'G2_V{x:06d}': blackai.NAME for x in [113,115,116]}}
WORKERS = [*mirror.WORKERS, 'aicodemirror_v106.py', 'mirror_stream_probe.py',
           'mirror_stream_client.py', 'mirror_stream_worker.py', 'mirror_stream_continuation.py', 'blackai_continuation.py']


def guard(runtime, questions):
    blackai.score_guard(runtime, questions)
    for name in [blackai.NAME, mirror.NAME, v106.NAME]:
        path = runtime/'runs'/name/'scores.jsonl'
        seen = {r['question_id'] for r in load_records(path) if r.get('status') == 'ok'} if path.exists() else set()
        assert not seen.intersection(q['question_id'] for q in questions), 'first valid score overlap'


def clone(runtime, out, vid):
    target = out/'memory'/vid
    if target.exists():
        assert read(target/'checkpoint_source.json')['parent_run'] == PARENTS[vid]
        return
    parent = runtime/'runs'/PARENTS[vid]
    state = read(parent/'frontend_status.json')
    assert vid not in state.get('active_videos', []) and state['videos'][vid]['status'] == 'frontend_failed'
    worker = read(parent/'videos'/vid/'worker_process.json')
    assert not Path('/proc', str(worker['pid'])).exists()
    source = parent/'memory'/vid
    assert source.is_dir() and not source.is_symlink() and not any(p.is_symlink() for p in source.rglob('*'))
    graph = read(source/'memory.json');assert not graph.get('complete')
    hashes = {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage = target.with_name(vid+'.staging');assert not stage.exists()
    stage.parent.mkdir(parents=True, exist_ok=True);shutil.copytree(source, stage)
    assert hashes == {str(p.relative_to(stage)): file_hash(p) for p in stage.rglob('*') if p.is_file()}
    history = stage/'ancestry'/PARENTS[vid];history.mkdir(parents=True, exist_ok=False)
    for name in ['manifest.json','checkpoint_source.json','calls.jsonl','audio/calls.jsonl']:
        p = stage/name
        if p.exists():
            dest = history/name;dest.parent.mkdir(parents=True, exist_ok=True);p.rename(dest)
    if PARENTS[vid] == blackai.NAME:
        for p in (stage/'audio').glob('W*.json'):
            if p.stem not in graph['completed_windows']:
                dest = history/'pending_audio'/p.name;dest.parent.mkdir(exist_ok=True);p.rename(dest)
    atomic(stage/'checkpoint_source.json', {'parent_run': PARENTS[vid], 'files': hashes,
           'inherited_windows': graph['completed_windows'], 'new_visual_model': 'claude-opus-5',
           'new_audio_model': 'gemini-2.5-pro', 'visual_transport': 'SSE',
           'audio_reuse': 'Keep pending audio only for the identical AICodeMirror Gemini 2.5 profile; validate input fingerprint in worker.'})
    assert hashes == {str(p.relative_to(source)): file_hash(p) for p in source.rglob('*') if p.is_file()}
    stage.rename(target)


def seed_probe(runtime, out):
    vid = 'G2_V000104';memory = out/'memory'/vid;dest = memory/'probe_seed_receipt.json'
    if dest.exists():return
    diag = runtime/'diagnostics/mirror_stream_v104_20260921'
    protocol = read(diag/'protocol.json');result = read(diag/'result.json')
    assert result['status'] == 'ok' and result['request_hash'] == EXPECTED_REQUEST
    assert read(diag/'audit.json')['parent_files_unchanged']
    assert read(diag/'transport.json')['message_stop_seen']
    source = runtime/'runs'/mirror.NAME/'memory'/vid
    assert protocol['parent_file_hashes'] == {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
    graph = read(memory/'memory.json');assert len(graph['completed_windows']) == 57 and 'W00058' not in graph['completed_windows']
    core = [1140.0,1160.0];interval = [1138.0,1162.0]
    media, metadata = window_input(runtime/'data/episode/videos'/(vid+'.mp4'), *interval,
        fps=1, max_frames=16, max_pixels=150528, with_audio=True,
        subtitle_rows=subtitles(runtime/'data/prepared_subtitles'/(vid+'.json')))
    audio_path = memory/'audio/W00058.json';audio = read(audio_path)
    media = [part for part in media if part['type'] != 'input_audio'] + [{'type':'text','text':
        'Timestamped audio observations from an independent audio model; these may contain errors. '
        'Match voice identity cautiously using the frames and dialogue. '+json.dumps(audio['result'],ensure_ascii=False)}]
    metadata.update(audio_representation='derived_timestamped_cues',audio_observer={
        'model':audio['model']['model'],'input_fingerprint':audio['input_fingerprint'],'source_sha256':file_hash(audio_path)})
    context = {'video_id':vid,'window_id':'W00058','core_interval':core,'media_interval':interval,
               'corrections_enabled':False,'cast':graph['entities'],'preceding_events':graph['events'][-3:]}
    messages = [{'role':'system','content':PERCEPTION},{'role':'user','content':[
        {'type':'text','text':json.dumps(context,ensure_ascii=False)}]+media}]
    assert fingerprint(messages) == EXPECTED_REQUEST
    payload = read(diag/'validated_payload.json')
    updated = apply_window(graph,payload,window_id='W00058',core=core,media=interval,metadata=metadata,allow_revisions=False)
    atomic(memory/'windows/W00058.json', {'input':context,'sampling':metadata,'perception':payload})
    atomic(memory/'memory.json', updated)
    atomic(dest, {'source':'diagnostics/mirror_stream_v104_20260921','request_hash':EXPECTED_REQUEST,
           'payload_sha256':file_hash(diag/'validated_payload.json'),'result':result,
           'new_api_call':False,'reason':'Reuse independently validated identical input; no repeat request.'})


def setup(args):
    out = args.runtime/'runs'/NAME;out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parent
    if not (out/'experiment_manifest.json').exists():
        assert args.stage == 'frontend'
        questions = [q for q in read(args.runtime/'runs'/blackai.NAME/'questions.json') if q['video_id'] in PARENTS]
        assert len(questions) == 45 and len({q['video_id'] for q in questions}) == 8
        guard(args.runtime, questions)
        cfg = read(args.runtime/'runs'/mirror.NAME/'experiment_manifest.json')['configuration'].copy()
        cfg.update(protocol=NAME, parent_runs=PARENTS, question_ids=[q['question_id'] for q in questions],
                   worker_hashes={n:file_hash(root/n) for n in WORKERS},
                   retry='One checkpoint attempt per video in this new protocol; no automatic outer recovery. Existing three schema attempts unchanged.',
                   scope='45 unscored questions, 8 stopped videos. SSE transport for five AICodeMirror HTTP failures; alternate provider for three BlackAI structure failures. V129 remains in the prior running experiment. Preserve all first scores and inherited windows.')
        cfg.pop('parent_run', None)
        cfg['visual'] = {**cfg['visual'], 'options':{'stream':True}, 'transport':'Anthropic SSE; final message_stop required'}
        manifest(out/'experiment_manifest.json',cfg);atomic(out/'questions.json',questions)
    cfg = read(out/'experiment_manifest.json')['configuration'];assert cfg['method_hash'] == code_hash()
    for n,h in cfg['worker_hashes'].items():assert file_hash(root/n) == h
    questions = read(out/'questions.json');assert len(questions)==45 and [q['question_id'] for q in questions]==cfg['question_ids']
    guard(args.runtime,questions);prepare_question_files(out,questions);(out/'outbox').mkdir(exist_ok=True)
    if args.stage == 'frontend':
        for vid in PARENTS:clone(args.runtime,out,vid)
        seed_probe(args.runtime,out)
    return out,questions


def run_video(args,out,vid):
    folder=out/'videos'/vid;path=folder/'attempts.json'
    if path.exists():
        raise RuntimeError('Stream protocol already attempted; audit before any recovery')
    cmd=claude.build_command(args,out,vid)
    cmd[cmd.index('experiments.zyf.claude_gemini_worker')]='experiments.zyf.mirror_stream_worker'
    atomic(folder/'build_command.json',{'command':cmd})
    atomic(path,{'attempts':1,'in_flight':True,'started_unix':time.time()})
    env=dict(os.environ,LONGEMO_FFMPEG_COMPAT='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE',None);env['PATH']=str(args.runtime/'tmp')+os.pathsep+env.get('PATH','')
    with (folder/'build.log').open('a') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(folder/'worker_process.json',{'pid':p.pid,'attempt':1,'started_unix':time.time()});rc=p.wait()
    atomic(path,{'attempts':1,'in_flight':False,'returncode':rc,'finished_unix':time.time()})
    complete=bool(read(out/'memory'/vid/'memory.json').get('complete'))
    block=recorded_video_block(folder,out/'memory'/vid)
    outcome={'video_id':vid,'status':'memory_complete' if complete else block['status'] if block else 'frontend_failed',
             'attempts':1,'outer_recovery_allowed':False,'transient_remaining':False if complete else transient_failure(out,vid)}
    if block:outcome['rejection']=block['rejection']
    export_video(out,vid,outcome);return outcome


def frontend(args,out,questions):
    path=out/'frontend_status.json';state=read(path) if path.exists() else {'videos':{}}
    queue=[v for v in PARENTS if v not in state['videos']]
    active={}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while queue or active:
            while queue and len(active)<args.workers:
                vid=queue.pop(0);active[pool.submit(run_video,args,out,vid)]=vid
            state.update(status='running',active_videos=sorted(active.values()),queued_videos=queue.copy(),question_count=45,updated_unix=time.time());atomic(path,state)
            done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                vid=active.pop(f)
                try:result=f.result()
                except Exception as exc:result={'video_id':vid,'status':'orchestration_error','error_type':type(exc).__name__}
                state['videos'][vid]=result;atomic(path,state);print(json.dumps(result),flush=True)
    state.update(status='finished',active_videos=[],queued_videos=[],updated_unix=time.time());atomic(path,state)
    atomic(out/'outbox/producer_done.json',{'video_ids':list(PARENTS),'time_unix':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['frontend','backend'])
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--credential-file',type=Path,required=True)
    p.add_argument('--workers',type=int,default=7);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    assert 1<=a.workers<=7
    out=a.runtime/'runs'/NAME;out.mkdir(parents=True,exist_ok=True)
    with (out/(a.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out,questions=setup(a)
        if a.prepare_only:return
        atomic(out/(a.stage+'_process.json'),{'pid':os.getpid(),'workers':a.workers,'started_unix':time.time()})
        if a.stage=='frontend':frontend(a,out,questions)
        else:backend(a,out,questions);guard(a.runtime,questions)


if __name__=='__main__':main()
