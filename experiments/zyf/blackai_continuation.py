"""E11: BlackAI on g450, immutable memory transfer, GPT-6 on AIStudio.

Targets only E10's unscored question IDs; E10 files are never modified.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import time

from evaluation.io_utils import load_questions, load_records, write_json, write_records
from experiments.zyf.azure_benchmark import SOURCE_HASH, aggregate, recorded_video_block
from experiments.zyf.finish_scoring import scoring_inventory
from methods.longemo.common import code_hash, file_hash, manifest

NAME = 'blackai_gemini38_matrix_gpt6_remaining_v1'
BLACKAI = 'https://www.blackaicoding.com/v1beta'
MATRIX = 'https://matrixllm.alipay.com/v1'


def read(p):
    return json.loads(p.read_text())


def atomic(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    write_json(tmp, value)
    tmp.replace(p)


def setup(runtime, selection):
    assert code_hash() == SOURCE_HASH, 'frozen method/evaluator changed'
    source = read(selection)
    all_questions = load_questions(runtime/'data/questions.json', 'episode')
    qmap = {q['question_id']: q for q in all_questions}
    ids = source['question_ids']
    assert len(ids) == len(set(ids)) and set(ids) <= set(qmap)
    assert len(all_questions) == source['full_question_count'] == 558
    questions = [qmap[qid] for qid in ids]
    out = runtime/'runs'/NAME
    out.mkdir(parents=True, exist_ok=True)
    config = {'protocol': NAME, 'source_hash': SOURCE_HASH, 'selection_sha256': file_hash(selection),
              'full_question_count': 558, 'target_question_count': len(ids),
              'questions_sha256': file_hash(runtime/'data/questions.json'),
              'data_revision': read(runtime/'data/manifest.json')['revision'],
              'perception': {'model': 'gemini-3.8-flash', 'base_url': BLACKAI,
                             'audio_thinking': 'low', 'visual_thinking': 'medium',
                             'visual_max_tokens': 32768, 'temperature': 1},
              'answer': {'model': 'gpt-6-astra', 'base_url': MATRIX, 'max_tokens': 8192},
              'embedding': {'model': 'google/gemini-embedding-2', 'base_url': 'https://openrouter.ai/api/v1'},
              'media': {'window_seconds': 20, 'padding': 2, 'fps': 1, 'max_frames': 24, 'max_pixels': 200704},
              'retrieval': {'top_k': 12, 'evidence_chars': 48000, 'max_inspections': 0},
              'scoring': 'unchanged official scorer, earliest valid score retained',
              'scope': 'E10 missing questions only; independent provider run, not a homogeneous full-benchmark score'}
    manifest(out/'experiment_manifest.json', config)
    write_json(out/'questions.json', questions)
    for vid in {q['video_id'] for q in questions}:
        write_json(out/'videos'/vid/'questions.json', [q for q in questions if q['video_id'] == vid])
    return out, questions, config


def invoke(folder, name, module, arguments, env):
    command = [sys.executable, '-u', '-m', module, *map(str, arguments)]
    write_json(folder/(name+'_command.json'), {'command': command, 'time_unix': time.time()})
    with (folder/(name+'.log')).open('a') as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def export_video(out, vid, state):
    box = out/'outbox'; box.mkdir(exist_ok=True)
    dest = box/(vid+'.tar.gz')
    if dest.exists():
        return
    memory = out/'memory'/vid
    files = [p for p in memory.rglob('*') if p.is_file() and not p.is_symlink()]
    receipt = {'video_id': vid, 'outcome': state, 'experiment_fingerprint': read(out/'experiment_manifest.json')['fingerprint'],
               'files': {str(p.relative_to(out)): file_hash(p) for p in files}}
    receipt_path = box/(vid+'.receipt.json'); write_json(receipt_path, receipt)
    tmp = dest.with_suffix('.tmp')
    with tarfile.open(tmp, 'w:gz') as t:
        for p in files:
            t.add(p, arcname=str(p.relative_to(out)), recursive=False)
        t.add(receipt_path, arcname='receipt.json', recursive=False)
    tmp.replace(dest)
    atomic(box/(vid+'.ready.json'), {'video_id': vid, 'archive': dest.name,
                                   'sha256': file_hash(dest), 'bytes': dest.stat().st_size})


def frontend(args, out, questions):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    setting = out/'perception_settings.json'
    write_json(setting, {'generationConfig': {'thinkingConfig': {'thinkingLevel': 'medium'}}})
    vids = sorted({q['video_id'] for q in questions})
    path = out/'frontend_status.json'
    status = read(path) if path.exists() else {'videos': {}}
    for vid, state in status['videos'].items():
        export_video(out, vid, state)

    def work(vid):
        folder = out/'videos'/vid
        root = folder/'build_memory'; root.mkdir(exist_ok=True)
        memory = out/'memory'/vid; memory.mkdir(parents=True, exist_ok=True)
        link = root/vid
        if not link.exists(): link.symlink_to(memory, target_is_directory=True)
        assert link.resolve() == memory.resolve()
        rc = invoke(folder, 'build', 'experiments.zyf.native_worker', ['build', '--data-path', folder/'questions.json',
            '--videos-dir', args.runtime/'data/episode/videos', '--subtitles-dir', args.runtime/'data/prepared_subtitles',
            '--output-dir', root, '--window-seconds', 20, '--padding', 2, '--fps', 1, '--max-frames', 24,
            '--max-pixels', 200704, '--with-audio', '--workers', 1, '--model', 'gemini-3.8-flash',
            '--base-url', BLACKAI, '--audio-model', 'gemini-3.8-flash', '--audio-base-url', BLACKAI,
            '--credential-file', args.credential_file, '--thinking', 'default', '--timeout', 240,
            '--tries', 3, '--max-tokens', 32768, '--temperature', 1, '--config', setting], env)
        complete = (memory/'memory.json').exists() and read(memory/'memory.json').get('complete')
        block = recorded_video_block(folder, memory)
        state = {'video_id': vid, 'status': 'memory_complete' if rc == 0 and complete else
                 block['status'] if block else 'frontend_failed', 'returncode': rc}
        if block: state['rejection'] = block['rejection']
        for p in (memory/'calls.jsonl', memory/'audio/calls.jsonl'):
            rows = load_records(p) if p.exists() else []
            if rows and rows[-1].get('http_status') in (401, 402, 403):
                state.update(status='blocked_provider', http_status=rows[-1]['http_status'])
        export_video(out, vid, state)
        return state

    todo = [v for v in vids if v not in status['videos']]
    stop = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        while todo or active:
            while todo and len(active) < args.workers and not stop:
                vid = todo.pop(0); active[pool.submit(work, vid)] = vid
            if not active: break
            done, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
            for f in done:
                vid = active.pop(f)
                try: state = f.result()
                except Exception as e: state = {'video_id': vid, 'status': 'orchestration_error', 'error_type': type(e).__name__}
                status['videos'][vid] = state
                if state['status'] in ('blocked_provider', 'orchestration_error'): stop = True
                print(json.dumps(state), flush=True)
            status.update(status='paused' if stop else 'running', active_videos=sorted(active.values()),
                          queued_videos=todo, updated_unix=time.time(), question_count=len(questions))
            atomic(path, status)
    status.update(status='paused' if todo else 'finished', active_videos=[], queued_videos=todo, updated_unix=time.time())
    atomic(path, status)


def import_video(archive, marker, out, expected_videos):
    vid = marker['video_id']
    assert vid in expected_videos and file_hash(archive) == marker['sha256']
    with tarfile.open(archive) as t:
        members = t.getmembers()
        assert len({m.name for m in members}) == len(members)
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
        receipt = json.load(t.extractfile('receipt.json'))
        assert receipt['video_id'] == vid
        assert receipt['experiment_fingerprint'] == read(out/'experiment_manifest.json')['fingerprint']
        assert set(receipt['files']) == {m.name for m in members if m.name != 'receipt.json'}
        import hashlib
        blobs = {}
        for name, sha in receipt['files'].items():
            assert name.startswith('memory/'+vid+'/')
            value = t.extractfile(name).read()
            assert hashlib.sha256(value).hexdigest() == sha
            blobs[name] = value
        for name, value in blobs.items():
            p = out/name
            assert not p.is_symlink()
            if p.exists(): assert p.read_bytes() == value
            else:
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value)
    atomic(out/'imports'/(vid+'.json'), receipt)
    return receipt


def evaluate_video(args, out, questions, vid, env):
    folder = out/'videos'/vid
    graph = folder/'graph'
    command = ['answer', '--data-path', folder/'questions.json', '--memory-dir', out/'memory',
        '--output-dir', graph, '--plans-dir', folder/'plans', '--retrieval', 'graph',
        '--videos-dir', args.runtime/'data/episode/videos', '--subtitles-dir', args.runtime/'data/prepared_subtitles',
        '--with-audio', '--max-inspections', 0, '--inspection-seconds', 60, '--workers', 2,
        '--top-k', 12, '--evidence-chars', 48000, '--embedding-backend', 'gemini',
        '--embedding-model', 'google/gemini-embedding-2', '--embedding-base-url', 'https://openrouter.ai/api/v1',
        '--embedding-cache-dir', args.runtime/'embedding_cache_e11', '--model', 'gpt-6-astra',
        '--base-url', MATRIX, '--thinking', 'default', '--timeout', 240, '--tries', 5,
        '--credential-file', args.credential_file, '--audio-model', 'gemini-3.8-flash',
        '--audio-base-url', BLACKAI, '--max-tokens', 8192]
    # A resume can reuse successful answers; known rejected requests are not replayed.
    if not recorded_video_block(folder, out/'memory'/vid):
        invoke(folder, 'graph', 'methods.longemo', command, env)
    subset = [q for q in questions if q['video_id'] == vid]
    item = scoring_inventory(out, subset)[vid]
    if item['pending']:
        qpath = folder/'pending_judgments.json'; write_json(qpath, item['pending'])
        invoke(folder, 'score', 'evaluation.eval', ['--data-path', qpath, '--predictions', graph/'predictions.jsonl',
            '-g', 'episode', '--model', 'gpt-6-astra', '--base-url', MATRIX, '--output-dir', folder/'scores',
            '--workers', 2, '--tries', 1, '--timeout', 180, '--max-tokens', 8192], env)
    item = scoring_inventory(out, subset)[vid]
    write_records(folder/'accepted_scores.jsonl', [item['scored'][q['question_id']] for q in subset if q['question_id'] in item['scored']])
    block = recorded_video_block(folder, out/'memory'/vid)
    return {'video_id': vid, 'status': 'complete' if len(item['scored']) == len(subset) else
            block['status'] if block else 'answer_or_judge_failed', 'n_scored': len(item['scored'])}


def backend(args, out, questions):
    credentials = read(args.credential_file)
    env = dict(os.environ, MODEL_API_KEY=credentials['MODEL_API_KEY'], MODEL_BASE_URL=MATRIX,
               OPENROUTER_API_KEY=credentials['OPENROUTER_API_KEY'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    inbox = out/'inbox'; inbox.mkdir(exist_ok=True)
    vids = {q['video_id'] for q in questions}
    path = out/'backend_status.json'; status = read(path) if path.exists() else {'videos': {}}

    def work(vid):
        return evaluate_video(args, out, questions, vid, env)

    active = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(status['videos']) < len(vids) or active:
            for marker_path in sorted(inbox.glob('*.ready.json')):
                marker = read(marker_path); vid = marker['video_id']
                if vid in status['videos'] or vid in active.values(): continue
                if len(active) >= args.workers: break
                receipt = import_video(inbox/marker['archive'], marker, out, vids)
                if receipt['outcome']['status'] == 'memory_complete': active[pool.submit(work, vid)] = vid
                else: status['videos'][vid] = receipt['outcome']
            if active:
                done, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                for f in done:
                    vid = active.pop(f)
                    try: result = f.result()
                    except Exception as e: result = {'video_id': vid, 'status': 'orchestration_error', 'error_type': type(e).__name__}
                    status['videos'][vid] = result; print(json.dumps(result), flush=True)
            elif len(status['videos']) < len(vids): time.sleep(10)
            metrics = aggregate(out, questions)
            status.update(status='running' if active else 'waiting_memory', active_videos=sorted(active.values()),
                          updated_unix=time.time(), question_count=len(questions), n_scored=metrics['overall_unweighted']['n_scored'])
            atomic(path, status)
        status.update(status='finished', active_videos=[], updated_unix=time.time()); atomic(path, status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('frontend', 'backend'))
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    assert args.workers > 0
    out, questions, config = setup(args.runtime, args.selection)
    with (out/(args.stage+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps({'stage': args.stage, 'target_questions': len(questions),
                          'target_videos': len({q['video_id'] for q in questions}), 'run': str(out)}), flush=True)
        if args.execute: globals()[args.stage](args, out, questions)


if __name__ == '__main__': main()
