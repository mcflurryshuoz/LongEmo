"""Frozen full-episode evaluation with per-video resume and unchanged official scoring.

Default: prepare manifests and inspect data only. --execute enables paid model calls.
Run from the repository root: python -m experiments.zyf.full_benchmark --help
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from evaluation.io_utils import load_questions, load_records, write_json, write_records
from evaluation.metrics import summarize, markdown_summary
from methods.longemo.common import code_hash, file_hash, manifest, git_revision
from methods.longemo.media import probe

MODEL = "openai/gpt-6-astra"
AUDIO = "google/gemini-3.8-flash"
EMBEDDING = "google/gemini-embedding-2"
SOURCE_HASH = "b7dee478e8dac8ee1de4de862ef37eaf1180401a7089d5d41b5b3a50dfdeb058"


class Blocked(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def successful_predictions(path, questions):
    if not path.exists():
        return False
    rows = load_records(path)
    return (len(rows) == len(questions)
            and {r['question_id'] for r in rows} == {q['question_id'] for q in questions}
            and all(r.get('status') == 'ok' and isinstance(r.get('prediction'), str)
                    and r['prediction'].strip() for r in rows))


def successful_scores(folder, questions, predictions):
    """Keep the earliest valid official judgment; retry failures, never low scores."""
    expected = {q['question_id']: q for q in questions}
    found = {}
    paths = list(folder.glob('run_*/scores.jsonl'))
    # Timestamped run names have non-numeric suffixes; file mtime orders retries.
    for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p))):
        for row in load_records(path):
            qid = row['question_id']
            if qid not in expected or qid in found or row.get('status') != 'ok':
                continue
            maximum = max(int(x) for x in expected[qid]['rubric']['scores'])
            if (row.get('prediction') != predictions[qid]
                    or row.get('max_score') != maximum
                    or not isinstance(row.get('score'), (int, float))
                    or not 0 <= row['score'] <= maximum
                    or not math.isclose(row.get('normalized_score', -1), row['score']/maximum)):
                raise ValueError('cached official judgment does not match its frozen prediction/rubric')
            found[qid] = {**row, 'score_source': str(path)}
    return found


def inventory(root, questions, output):
    """Use downloader SHA attestations; decoding/hash validation happens again at build."""
    data_manifest = read(root/'manifest.json')
    files = {r['path']: r for r in data_manifest['files']}
    old = read(output) if output.exists() else {}
    cached = old.get('videos', {})
    result = {'revision': data_manifest['revision'], 'videos': {}, 'missing': [], 'missing_subtitles': []}
    for vid in sorted({q['video_id'] for q in questions}):
        relative = f'episode/videos/{vid}.mp4'
        video, subtitle = root/relative, root/'prepared_subtitles'/f'{vid}.json'
        entry = files.get(relative)
        if not subtitle.exists():
            result['missing_subtitles'].append(vid)
        if not entry or not video.exists() or video.stat().st_size != entry['bytes']:
            result['missing'].append(vid)
            continue
        signature = [video.stat().st_size, video.stat().st_mtime_ns, entry['sha256']]
        item = cached.get(vid, {})
        if item.get('file_signature') != signature:
            meta = probe(video)
            item = {'duration_seconds': meta['duration'], 'windows': math.ceil(meta['duration']/20),
                    'bytes': entry['bytes'], 'sha256': entry['sha256'], 'file_signature': signature}
        result['videos'][vid] = item
    result.update(video_count=len(result['videos']), expected_videos=len({q['video_id'] for q in questions}),
                  question_count=len(questions), complete=not result['missing'] and not result['missing_subtitles'],
                  duration_hours=sum(v['duration_seconds'] for v in result['videos'].values())/3600,
                  windows=sum(v['windows'] for v in result['videos'].values()),
                  bytes=sum(v['bytes'] for v in result['videos'].values()), generated_unix=time.time())
    write_json(output, result)
    return result


def copy_checkpoints(source, destination, video_ids):
    """Copy compatible perception checkpoints; never mutate parent experiments."""
    imports = {}
    if not source:
        return imports
    source = Path(source).resolve()
    for vid in video_ids:
        src, dst = source/vid, destination/vid
        if not (src/'memory.json').exists() or dst.exists():
            continue
        if read(src/'manifest.json')['configuration']['code_hash'] != SOURCE_HASH:
            raise ValueError('perception checkpoint source hash differs from the frozen method')
        # _build_video checks the full model/media/input configuration before reuse.
        temp = destination/(vid+'.importing')
        if temp.exists():
            shutil.rmtree(temp)
        shutil.copytree(src, temp)
        temp.rename(dst)
        imports[vid] = {'source': str(src), 'memory_sha256': file_hash(src/'memory.json'),
                        'manifest_sha256': file_hash(src/'manifest.json')}
    return imports


def aggregate(out, questions):
    rows, predictions = [], []
    for q in questions:
        folder = out/'videos'/q['video_id']
        p = folder/'graph'/'predictions.jsonl'
        pred = {r['question_id']: r for r in load_records(p)} if p.exists() else {}
        if q['question_id'] in pred:
            predictions.append(pred[q['question_id']])
        scored = folder/'accepted_scores.jsonl'
        scores = {r['question_id']: r for r in load_records(scored)} if scored.exists() else {}
        rows.append(scores.get(q['question_id'], {
            'question_id': q['question_id'], 'video_id': q['video_id'], 'granularity': 'episode',
            'type': q['type'], 'max_score': max(int(x) for x in q['rubric']['scores']),
            'status': 'pending', 'normalized_score': None}))
    write_records(out/'predictions.jsonl', predictions)
    write_records(out/'scores.jsonl', rows)
    metrics = summarize(rows)
    write_json(out/'metrics.json', metrics)
    text = markdown_summary(metrics)
    if metrics['overall_unweighted']['coverage'] != 1:
        text = '# INCOMPLETE: do not report as a full benchmark score\n\n'+text
    (out/'summary.md').write_text(text)
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--embedding-cache-dir', type=Path, required=True)
    p.add_argument('--import-memory', type=Path)
    p.add_argument('--video-workers', type=int, default=3)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--minimum-credits', type=float, default=25.0,
                   help='Pause before another video stage when available account credit is below this floor; not a spend cap')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    if args.video_workers < 1 or args.workers < 1 or args.minimum_credits < 0:
        p.error('workers must be positive and credit floor nonnegative')
    if code_hash() != SOURCE_HASH:
        raise ValueError('method source changed; create a separately versioned experiment')
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Prevent simultaneous resumes from racing prediction/score/checkpoint files.
    import fcntl
    with (out/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        questions = load_questions(root/'questions.json', 'episode')
        data = read(root/'manifest.json')
        vids = sorted({q['video_id'] for q in questions})
        if len(questions) != data['question_count'] or len(vids) != data['video_count']:
            raise ValueError('full question inventory differs from pinned dataset manifest')
        config = {'protocol': 'full-episode-graph-v1', 'data_revision': data['revision'],
                  'question_sha256': file_hash(root/'questions.json'), 'question_count': len(questions),
                  'video_count': len(vids), 'code_hash': code_hash(), 'orchestrator_sha256': file_hash(__file__),
                  'git_revision': git_revision(), 'model': MODEL, 'judge_model': MODEL,
                  'audio_model': AUDIO, 'embedding_model': EMBEDDING, 'reasoning_effort': 'medium',
                  'window_seconds': 20.0, 'padding': 2.0, 'fps': 1.0, 'max_frames': 24,
                  'max_pixels': 200704, 'max_tokens': 8192, 'retrieval': 'graph',
                  'top_k': 12, 'evidence_chars': 48000, 'inspections': 0,
                  'scoring': 'unchanged official episode scorer; earliest successful judgment retained',
                  'pilot_question_ids': data['pilot_question_ids'], 'development_overlap_explicit': True}
        manifest(out/'experiment_manifest.json', config)
        memory = out/'memory'
        memory.mkdir(exist_ok=True)
        imports = copy_checkpoints(args.import_memory, memory, vids)
        import_file = out/'imported_checkpoints.json'
        write_json(import_file, {**(read(import_file) if import_file.exists() else {}), **imports})
        inv = inventory(root, questions, out/'inventory.json')
        status = {'status': 'prepared' if inv['complete'] else 'blocked_data', 'updated_unix': time.time(),
                  'question_count': len(questions), 'video_count': len(vids), 'available_videos': inv['video_count']}
        write_json(out/'status.json', status)
        metrics = aggregate(out, questions)
        if not args.execute or not inv['complete']:
            print(json.dumps(status), flush=True)
            return 0 if inv['complete'] else 2
        credentials = read(args.credential_file)
        env = os.environ.copy()
        env.update(OPENROUTER_API_KEY=credentials['OPENROUTER_API_KEY'],
                   MODEL_API_KEY=credentials['OPENROUTER_API_KEY'], MODEL_BASE_URL='https://openrouter.ai/api/v1')

        def credits():
            req = Request('https://openrouter.ai/api/v1/credits',
                          headers={'Authorization': 'Bearer '+credentials['OPENROUTER_API_KEY']})
            with urlopen(req, timeout=30) as response:
                account = json.load(response)['data']
            remaining = float(account['total_credits']) - float(account['total_usage'])
            if remaining < args.minimum_credits:
                raise Blocked(f'available credits USD {remaining:.4f} below stage floor {args.minimum_credits:.2f}')
            return remaining

        def run(folder, name, module, arguments):
            credits()
            command = [sys.executable, '-u', '-m', module] + list(map(str, arguments))
            write_json(folder/(name+'_command.json'), {'command': command, 'time_unix': time.time()})
            with (folder/(name+'.log')).open('a') as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                credits()  # Classify credit exhaustion without logging provider credentials/payloads.
                raise RuntimeError(f'{name} failed; checkpoints retained at {folder}')

        inference = ['--model', MODEL, '--thinking', 'default', '--timeout', '240', '--tries', '5',
                     '--credential-file', args.credential_file, '--audio-model', AUDIO, '--max-tokens', '8192']

        def video_run(vid):
            folder = out/'videos'/vid
            folder.mkdir(parents=True, exist_ok=True)
            subset = [q for q in questions if q['video_id'] == vid]
            qpath = folder/'questions.json'
            write_json(qpath, subset)
            try:
                # Isolate build_results.json per subprocess; actual video checkpoints
                # live in the common memory directory via a validated directory link.
                build_root = folder/'build_memory'
                build_root.mkdir(exist_ok=True)
                (memory/vid).mkdir(exist_ok=True)
                link = build_root/vid
                if not link.exists():
                    link.symlink_to(memory/vid, target_is_directory=True)
                if link.resolve() != (memory/vid).resolve():
                    raise ValueError('build checkpoint link points to an unexpected directory')
                # Always call build for full source/media/config validation. Completed
                # windows have no model calls, and successful predictions resume by ID.
                run(folder, 'build', 'methods.longemo', ['build', '--data-path', qpath,
                    '--videos-dir', root/'episode/videos', '--subtitles-dir', root/'prepared_subtitles',
                    '--output-dir', build_root, '--window-seconds', '20', '--padding', '2', '--fps', '1',
                    '--max-frames', '24', '--max-pixels', '200704', '--with-audio', '--workers', '1']+inference)
                run(folder, 'graph', 'methods.longemo', ['answer', '--data-path', qpath, '--memory-dir', memory,
                    '--output-dir', folder/'graph', '--plans-dir', folder/'plans', '--retrieval', 'graph',
                    '--embedding-model', EMBEDDING, '--embedding-cache-dir', args.embedding_cache_dir,
                    '--videos-dir', root/'episode/videos', '--subtitles-dir', root/'prepared_subtitles',
                    '--with-audio', '--max-inspections', '0', '--inspection-seconds', '60',
                    '--workers', args.workers, '--top-k', '12', '--evidence-chars', '48000']+inference)
                prediction_path = folder/'graph'/'predictions.jsonl'
                if not successful_predictions(prediction_path, subset):
                    raise ValueError('incomplete predictions cannot enter the scorer')
                scores_dir = folder/'scores'
                manifest(scores_dir/'input_manifest.json', {'questions_sha256': file_hash(qpath),
                    'predictions_sha256': file_hash(prediction_path), 'model': MODEL, 'source_hash': SOURCE_HASH})
                predictions = {r['question_id']: r['prediction'] for r in load_records(prediction_path)}
                scored = successful_scores(scores_dir, subset, predictions)
                pending = [q for q in subset if q['question_id'] not in scored]
                if pending:
                    pending_path = folder/'pending_judgments.json'
                    write_json(pending_path, pending)
                    # Even an interrupted/failed scorer may have completed valid rows.
                    try:
                        run(folder, 'score', 'evaluation.eval', ['--data-path', pending_path,
                            '--predictions', prediction_path, '-g', 'episode', '--model', MODEL,
                            '--output-dir', scores_dir, '--workers', args.workers, '--tries', '3',
                            '--timeout', '180', '--max-tokens', '8192'])
                    finally:
                        scored = successful_scores(scores_dir, subset, predictions)
                        write_records(folder/'accepted_scores.jsonl', [scored[q['question_id']] for q in subset if q['question_id'] in scored])
                if len(scored) != len(subset):
                    raise RuntimeError('official judgments incomplete')
                write_records(folder/'accepted_scores.jsonl', [scored[q['question_id']] for q in subset])
                return {'video_id': vid, 'status': 'complete', 'n_scored': len(scored)}
            except Blocked as exc:
                return {'video_id': vid, 'status': 'blocked_api_credits', 'reason': str(exc)}
            except Exception as exc:
                return {'video_id': vid, 'status': 'error', 'reason': str(exc)[:400]}

        try:
            remaining = credits()
        except Blocked as exc:
            status.update(status='blocked_api_credits', reason=str(exc))
            write_json(out/'status.json', status)
            print(json.dumps(status), flush=True)
            return 3
        status.update(status='running', initial_available_credits=remaining, videos={})
        write_json(out/'status.json', status)
        todo = iter(vids)
        stopped = False
        with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
            active = {pool.submit(video_run, vid): vid for vid in [next(todo, None) for _ in range(args.video_workers)] if vid}
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    vid = active.pop(future)
                    outcome = future.result()
                    status['videos'][vid] = outcome
                    if outcome['status'] != 'complete':
                        stopped = True
                    print(json.dumps(outcome), flush=True)
                status['updated_unix'] = time.time()
                write_json(out/'status.json', status)
                if not stopped:
                    while len(active) < args.video_workers:
                        vid = next(todo, None)
                        if vid is None:
                            break
                        active[pool.submit(video_run, vid)] = vid
        metrics = aggregate(out, questions)
        complete = metrics['overall_unweighted']['coverage'] == 1
        failures = [r['status'] for r in status['videos'].values() if r['status'] != 'complete']
        status.update(status='complete' if complete and not failures else
                      ('blocked_api_credits' if 'blocked_api_credits' in failures else 'error'),
                      n_scored=metrics['overall_unweighted']['n_scored'], updated_unix=time.time())
        write_json(out/'status.json', status)
        return 0 if status['status'] == 'complete' else 3

if __name__ == '__main__':
    raise SystemExit(main())
