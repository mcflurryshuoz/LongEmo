"""Bounded checkpoint recovery for explicit transient HTTP failures, with versioned bundles."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from evaluation.io_utils import load_records, write_json
from experiments.zyf.blackai_continuation import (NAME, MATRIX, read, atomic, import_video, evaluate_video)
from experiments.zyf.azure_benchmark import aggregate, SOURCE_HASH
from methods.longemo.common import code_hash, file_hash

TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 522, 524}


def transient_failure(run, vid):
    latest = {}
    for p in (run/'memory'/vid/'calls.jsonl', run/'memory'/vid/'audio/calls.jsonl'):
        for row in load_records(p) if p.exists() else []:
            latest[row.get('purpose')] = row
    errors = [r for r in latest.values() if r.get('status') == 'error']
    # Unknown empty-model responses and explicit policy/authentication errors are not retried.
    return bool(errors) and all(r.get('http_status') in TRANSIENT and r.get('service_error_code')
                               not in ('content_filter', 'content_policy_violation') for r in errors)


def publish(run, vid, state, predecessor):
    dest = run/'recovery/outbox'; dest.mkdir(parents=True, exist_ok=True)
    archive = dest/(vid+'.tar.gz')
    if archive.exists(): return
    files = [p for p in (run/'memory'/vid).rglob('*') if p.is_file() and not p.is_symlink()]
    receipt = {'video_id': vid, 'outcome': state, 'recovery': True,
               'experiment_fingerprint': read(run/'experiment_manifest.json')['fingerprint'],
               'predecessor_files': predecessor['files'],
               'files': {str(p.relative_to(run)): file_hash(p) for p in files}}
    rp = dest/(vid+'.receipt.json'); write_json(rp, receipt)
    temp = archive.with_suffix('.tmp')
    with tarfile.open(temp, 'w:gz') as t:
        for p in files: t.add(p, arcname=str(p.relative_to(run)), recursive=False)
        t.add(rp, arcname='receipt.json', recursive=False)
    temp.replace(archive)
    atomic(dest/(vid+'.ready.json'), {'video_id': vid, 'archive': archive.name, 'bytes': archive.stat().st_size,
                                    'sha256': file_hash(archive), 'recovery': True})


def frontend(runtime, run, max_attempts, worker_module='experiments.zyf.native_worker'):
    recovery = run/'recovery'; outbox = recovery/'outbox'; outbox.mkdir(parents=True, exist_ok=True)
    path = recovery/'frontend_status.json'; state = read(path) if path.exists() else {'videos': {}, 'attempts': {}}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    while True:
        first = read(run/'frontend_status.json')
        def recovered_before_interruption(vid):
            p = run/'memory'/vid/'memory.json'
            return state['attempts'].get(vid, 0) > 0 and p.exists() and read(p).get('complete')
        ready = [v for v, r in first['videos'].items() if r['status'] == 'frontend_failed'
                 and v not in state['videos'] and (transient_failure(run, v) or recovered_before_interruption(v))]
        if not ready:
            if first['status'] in ('finished', 'paused'): break
            state.update(status='waiting_transient_failures', active_video=None, updated_unix=time.time()); atomic(path, state)
            time.sleep(15); continue
        vid = sorted(ready)[0]
        predecessor = read(run/'outbox'/(vid+'.receipt.json'))
        assert predecessor['outcome']['status'] == 'frontend_failed'
        command = read(run/'videos'/vid/'build_command.json')['command']
        assert command[2:4] == ['-m', worker_module]
        assert '--output-dir' in command and '--credential-file' in command
        count = state['attempts'].get(vid, 0)
        complete = recovered_before_interruption(vid)
        while count < max_attempts and transient_failure(run, vid):
            count += 1
            state['attempts'][vid] = count
            state.update(status='running', active_video=vid, updated_unix=time.time()); atomic(path, state)
            with (recovery/(vid+'.log')).open('a') as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            complete = (run/'memory'/vid/'memory.json').exists() and read(run/'memory'/vid/'memory.json').get('complete')
            if complete: break
            if count < max_attempts and transient_failure(run, vid): time.sleep(30)
        result = {'video_id': vid, 'status': 'memory_complete' if complete else 'recovery_exhausted',
                  'recovery_attempts': count, 'transient_remaining': transient_failure(run, vid)}
        publish(run, vid, result, predecessor)
        state['videos'][vid] = result
        state.update(active_video=None, updated_unix=time.time()); atomic(path, state)
        print(json.dumps(result), flush=True)
    state.update(status='finished', active_video=None, updated_unix=time.time()); atomic(path, state)
    atomic(outbox/'producer_done.json', {'video_ids': sorted(state['videos']), 'time_unix': time.time()})


def install_recovery(run, archive, marker):
    vid = marker['video_id']; registry = run/'recovery/imported'/(vid+'.json')
    if registry.exists():
        assert read(registry)['bundle_sha256'] == marker['sha256']
        return read(registry)['receipt']
    stage = run/'recovery/staging'/vid
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run/'experiment_manifest.json', stage/'experiment_manifest.json')
    receipt = import_video(archive, marker, stage, {vid})
    assert receipt.get('recovery') is True
    previous = read(run/'imports'/(vid+'.json'))
    assert previous['outcome']['status'] == 'frontend_failed'
    assert previous['files'] == receipt['predecessor_files']
    folder = run/'videos'/vid
    assert not (folder/'graph/predictions.jsonl').exists() and not (folder/'accepted_scores.jsonl').exists()
    assert read(run/'backend_status.json')['videos'][vid]['status'] == 'frontend_failed'
    for name, sha in previous['files'].items():
        p = run/name
        assert p.is_file() and not p.is_symlink() and file_hash(p) == sha
    current = {str(p.relative_to(run)) for p in (run/'memory'/vid).rglob('*') if p.is_file()}
    assert current == set(previous['files'])
    history = run/'recovery/history'/vid; history.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run/'imports'/(vid+'.json'), history/'original_receipt.json')
    (run/'memory'/vid).rename(history/'memory')
    (stage/'memory'/vid).rename(run/'memory'/vid)
    atomic(run/'imports'/(vid+'.json'), receipt)
    atomic(registry, {'bundle_sha256': marker['sha256'], 'receipt': receipt})
    return receipt


def aggregate_if_idle(run, questions):
    # The original backend aggregates recovered accepted_scores while it is alive.
    with (run/'backend.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: return
        aggregate(run, questions)


def backend(runtime, run, credential):
    recovery = run/'recovery'; inbox = recovery/'inbox'; inbox.mkdir(parents=True, exist_ok=True)
    path = recovery/'backend_status.json'; state = read(path) if path.exists() else {'videos': {}}
    questions = read(run/'questions.json'); expected = {q['video_id'] for q in questions}
    credentials = read(credential)
    env = dict(os.environ, MODEL_API_KEY=credentials['MODEL_API_KEY'], MODEL_BASE_URL=MATRIX,
               OPENROUTER_API_KEY=credentials['OPENROUTER_API_KEY'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    from types import SimpleNamespace
    args = SimpleNamespace(runtime=runtime, credential_file=credential)
    while True:
        for mp in sorted(inbox.glob('*.ready.json')):
            marker = read(mp); vid = marker['video_id']
            assert vid in expected and marker.get('recovery') is True
            if vid in state['videos']: continue
            original = read(run/'backend_status.json')['videos'].get(vid)
            if original is None: continue
            state.update(status='running', active_video=vid, updated_unix=time.time()); atomic(path, state)
            receipt = install_recovery(run, inbox/marker['archive'], marker)
            if receipt['outcome']['status'] == 'memory_complete':
                result = evaluate_video(args, run, questions, vid, env)
            else: result = receipt['outcome']
            state['videos'][vid] = result
            state.update(active_video=None, updated_unix=time.time()); atomic(path, state)
            aggregate_if_idle(run, questions)
            print(json.dumps(result), flush=True)
        done = list(inbox.glob('producer_done_*.json'))
        if done and set(read(done[-1])['video_ids']) <= set(state['videos']): break
        state.update(status='waiting_recovery_memory', active_video=None, updated_unix=time.time()); atomic(path, state)
        time.sleep(15)
    aggregate_if_idle(run, questions)
    state.update(status='finished', updated_unix=time.time()); atomic(path, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('frontend', 'backend'))
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--credential-file', type=Path)
    p.add_argument('--max-attempts', type=int, default=3)
    args = p.parse_args()
    assert 1 <= args.max_attempts <= 3 and code_hash() == SOURCE_HASH
    run = args.runtime/'runs'/NAME
    directory = run/'recovery'; directory.mkdir(exist_ok=True)
    with (directory/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'frontend': frontend(args.runtime, run, args.max_attempts)
        else: backend(args.runtime, run, args.credential_file)


if __name__ == '__main__': main()
