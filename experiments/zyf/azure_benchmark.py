"""Azure full-episode evaluation with per-video resume and unchanged official scoring.

Default: prepare manifests and inspect data only. --execute enables paid model calls.
Run from the repository root: python -m experiments.zyf.azure_benchmark --help
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

MODEL = "gpt-6-astra"
BASE_URL = "https://yifanyang-foundry-eastus2.cognitiveservices.azure.com/"
AUDIO = "google/gemini-3.8-flash"
EMBEDDING = "google/gemini-embedding-2"
GEMINI_PERCEPTION = "google/gemini-3.8-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
NATIVE_GEMINI = "gemini-3.8-flash"
NATIVE_EMBEDDING = "gemini-embedding-2"
BLACKAI_URL = "https://www.blackaicoding.com/v1beta"
SOURCE_HASH = "250f434224e257325c3ab512a5cf91d2c0662cbbe91a6a2ef72e923020dfeff9"


class Blocked(Exception):
    pass


class EmbeddingUnavailable(Exception):
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


def copy_audio(source, destination, video_ids):
    """Reuse only question-free audio caches; perception is rebuilt separately.

    Every imported observation is checked against its source audio and model
    fingerprint by bridge_audio before use. Parent experiment files stay intact.
    """
    imports = {}
    if not source:
        return imports
    source = Path(source).resolve()
    for vid in video_ids:
        for src in sorted((source/vid/'audio').glob('W*.json')):
            dst = destination/vid/'audio'/src.name
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            imports[f'{vid}/{src.name}'] = {'source': str(src), 'sha256': file_hash(src)}
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


def execution_manifest(path, config):
    """Permit scheduler-only revisions while retaining immutable provenance.

    All method/evaluator source, prompts, providers, data and semantic settings
    must still match. Original manifests are never overwritten. Each actual
    orchestration source has its own immutable execution-revision manifest.
    """
    if path.exists():
        original = read(path)['configuration']
        compare = lambda d: {k:v for k,v in d.items() if k not in ('git_revision','orchestrator_sha256')}
        if compare(original) != compare(config):
            raise ValueError('semantic experiment configuration changed; use a new run')
    else:
        manifest(path,config)
    revision = path.parent/'execution_revisions'/config['orchestrator_sha256']/'manifest.json'
    manifest(revision,config)


def policy_rejection(folder, memory):
    """A known service rejection is a recorded missing result, never a retry target."""
    paths = [memory/'calls.jsonl', memory/'audio/calls.jsonl']
    paths += list((folder/'plans/calls').glob('*.jsonl'))
    paths += list((folder/'graph/calls').glob('*.jsonl'))
    for p in paths:
        if not p.exists():continue
        for r in load_records(p):
            if r.get('service_error_code') in ('content_policy_violation','content_filter'):
                return {'source':str(p),'purpose':r.get('purpose'),'service_error_code':r['service_error_code']}
    for p in (folder/'scores').glob('run_*/scores.jsonl'):
        for r in load_records(p):
            if r.get('status') != 'ok' and any(code in str(r.get('error','')) for code in ('content_policy_violation','content_filter')):
                return {'source':str(p),'question_id':r.get('question_id'),'service_error_code':'content_policy_violation'}
    return None


def provider_rejection(memory):
    """Authentication/credit errors on the perception provider stop admission."""
    for p in (memory/'calls.jsonl', memory/'audio/calls.jsonl'):
        if not p.exists():
            continue
        rows = load_records(p)
        if rows and rows[-1].get('http_status') in (401, 402, 403):
            return {'source': str(p), 'http_status': rows[-1]['http_status']}
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--embedding-cache-dir', type=Path, required=True)
    p.add_argument('--import-audio', type=Path)
    p.add_argument('--perception-model', choices=(MODEL, GEMINI_PERCEPTION, NATIVE_GEMINI), default=MODEL,
                   help='Change only video perception; planning, answers and official scoring stay on Azure GPT-6')
    p.add_argument('--video-workers', type=int, default=16)
    p.add_argument('--startup-videos', type=int,
                   help='Initial admission count; expand after one video is scored (or its memory completes with deferred embedding)')
    p.add_argument('--video-attempts', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--minimum-credits', type=float, default=0.25,
                   help='Pause before another video stage when available account credit is below this floor; not a spend cap')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--defer-answers', action='store_true',
                   help='Build native Gemini memories while its embedding model is unavailable; resume without this flag to score')
    args = p.parse_args(argv)
    native = args.perception_model == NATIVE_GEMINI
    audio_model = NATIVE_GEMINI if native else AUDIO
    if native and args.import_audio:
        p.error('native Gemini requires a fresh audio frontend; do not import another provider\'s audio cache')
    if args.defer_answers and not native:
        p.error('defer-answers is only supported for the independent native Gemini frontend')
    if args.video_workers < 1 or args.video_attempts < 1 or args.workers < 1 or args.minimum_credits < 0:
        p.error('workers must be positive and credit floor nonnegative')
    if args.startup_videos is not None and not 1 <= args.startup_videos <= args.video_workers:
        p.error('startup-videos must be between 1 and video-workers')
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
        config = {'protocol': 'azure-full-episode-graph-v1', 'base_url': BASE_URL,
                  'azure_transport': __import__('evaluation.azure_transport', fromlist=['configuration']).configuration(), 'data_revision': data['revision'],
                  'question_sha256': file_hash(root/'questions.json'), 'question_count': len(questions),
                  'video_count': len(vids), 'code_hash': code_hash(), 'orchestrator_sha256': file_hash(__file__),
                  'git_revision': git_revision(), 'model': MODEL, 'judge_model': MODEL,
                  'audio_model': AUDIO, 'embedding_model': EMBEDDING, 'reasoning_effort': 'medium',
                  'window_seconds': 20.0, 'padding': 2.0, 'fps': 1.0, 'max_frames': 24,
                  'max_pixels': 200704, 'max_tokens': 8192, 'retrieval': 'graph',
                  'top_k': 12, 'evidence_chars': 48000, 'inspections': 0,
                  'scoring': 'unchanged official episode scorer; earliest successful judgment retained',
                  'pilot_question_ids': data['pilot_question_ids'], 'development_overlap_explicit': True}
        if args.perception_model == GEMINI_PERCEPTION:
            config.update(protocol='gemini-perception-azure-full-episode-graph-v1',
                perception_model=GEMINI_PERCEPTION, perception_base_url=OPENROUTER_URL,
                perception_reasoning_effort='medium', perception_temperature=1.0)
        elif native:
            config.update(protocol='blackai-gemini-azure-full-episode-graph-v2',
                perception_model=NATIVE_GEMINI, perception_base_url=BLACKAI_URL,
                perception_reasoning_effort='medium', perception_temperature=1.0,
                perception_max_tokens=32768,
                audio_model=audio_model, audio_base_url=BLACKAI_URL,
                audio_reasoning_effort='low', embedding_model=NATIVE_EMBEDDING,
                embedding_base_url=BLACKAI_URL, embedding_backend='gemini-native',
                memory_continues_without_embedding_service=True)
        execution_manifest(out/'experiment_manifest.json', config)
        memory = out/'memory'
        memory.mkdir(exist_ok=True)
        imports = copy_audio(args.import_audio, memory, vids)
        import_file = out/'imported_audio.json'
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
        env['MODEL_BASE_URL'] = BASE_URL
        if native:
            env.pop('OPENROUTER_API_KEY', None)
        else:
            env['OPENROUTER_API_KEY'] = credentials['OPENROUTER_API_KEY']
        env.pop('MODEL_API_KEY', None)
        from evaluation.azure_transport import RateLimiter, access_token
        limiter = RateLimiter()
        limiter.limits()  # Validate operational settings before starting workers.
        access_token()  # Verify the existing CLI login without printing tokens.
        write_json(out/'launch_settings.json', {'video_workers': args.video_workers,
            'startup_videos': args.startup_videos or args.video_workers,
            'workers_per_video': args.workers, 'video_attempts': args.video_attempts,
            'azure_limits': limiter.limits(), 'started_unix': time.time(),
            'openrouter_credit_floor_usd': args.minimum_credits,
            'defer_answers':args.defer_answers,
            'openrouter_stages': [] if native else ['build','graph_embeddings']})

        def credits(enforce=True):
            req = Request('https://openrouter.ai/api/v1/credits',
                          headers={'Authorization': 'Bearer '+credentials['OPENROUTER_API_KEY']})
            try:
                with urlopen(req, timeout=30) as response:
                    account = json.load(response)['data']
                remaining = float(account['total_credits']) - float(account['total_usage'])
            except Exception as exc:
                if native:
                    raise Blocked(f'OpenRouter credit preflight unavailable ({type(exc).__name__})') from None
                raise
            if enforce and remaining < args.minimum_credits:
                raise Blocked(f'available credits USD {remaining:.4f} below stage floor {args.minimum_credits:.2f}')
            return remaining

        def run(folder, name, module, arguments):
            uses_openrouter = not native and name != 'score'
            if uses_openrouter:
                credits()
            if native and name == 'build':
                module = 'experiments.zyf.native_worker'
            command = [sys.executable, '-u', '-m', module] + list(map(str, arguments))
            write_json(folder/(name+'_command.json'), {'command': command, 'time_unix': time.time()})
            with (folder/(name+'.log')).open('a') as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                if uses_openrouter:
                    credits()  # Classify credit exhaustion without logging payloads.
                if native and name == 'graph':
                    import re
                    match = re.search(r'ServiceError: model service returned HTTP (401|402|403|404)', (folder/'graph.log').read_text()[-8000:])
                    if match:
                        raise EmbeddingUnavailable('native Gemini embedding service returned HTTP '+match.group(1))
                raise RuntimeError(f'{name} failed; checkpoints retained at {folder}')

        inference = ['--model', MODEL, '--base-url', BASE_URL, '--thinking', 'default', '--timeout', '240', '--tries', '5',
                     '--credential-file', args.credential_file, '--audio-model', audio_model, '--max-tokens', '8192']
        if native:
            inference += ['--audio-base-url', BLACKAI_URL]
        embedding_arguments = ['--embedding-model', NATIVE_EMBEDDING if native else EMBEDDING,
                               '--embedding-cache-dir', args.embedding_cache_dir]
        if native:
            embedding_arguments += ['--embedding-backend','gemini-native','--embedding-base-url',BLACKAI_URL]
        perception_inference = list(inference)
        if args.perception_model in (GEMINI_PERCEPTION, NATIVE_GEMINI):
            perception_inference[perception_inference.index('--model')+1] = args.perception_model
            perception_inference[perception_inference.index('--base-url')+1] = BLACKAI_URL if native else OPENROUTER_URL
            if native:
                perception_inference[perception_inference.index('--max-tokens')+1] = '32768'
            perception_config = out/'perception_request_settings.json'
            settings = {'generationConfig': {'thinkingConfig': {'thinkingLevel': 'medium'}}} if native else {'reasoning': {'effort': 'medium'}}
            write_json(perception_config, settings)
            perception_inference += ['--temperature', '1', '--config', perception_config]

        def video_run(vid):
            folder = out/'videos'/vid
            folder.mkdir(parents=True, exist_ok=True)
            subset = [q for q in questions if q['video_id'] == vid]
            qpath = folder/'questions.json'
            write_json(qpath, subset)
            stage = 'build'
            try:
                rejection = policy_rejection(folder, memory/vid)
                if rejection:
                    return {'video_id':vid,'status':'blocked_input_policy','rejection':rejection}
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
                    '--max-frames', '24', '--max-pixels', '200704', '--with-audio', '--workers', '1']+perception_inference)
                stage = 'graph'
                prediction_path = folder/'graph'/'predictions.jsonl'
                if not successful_predictions(prediction_path, subset):
                    if args.defer_answers:
                        raise EmbeddingUnavailable('answering deferred until native Gemini embedding access is available')
                    run(folder, 'graph', 'methods.longemo', ['answer', '--data-path', qpath, '--memory-dir', memory,
                    '--output-dir', folder/'graph', '--plans-dir', folder/'plans', '--retrieval', 'graph',
                    '--videos-dir', root/'episode/videos', '--subtitles-dir', root/'prepared_subtitles',
                    '--with-audio', '--max-inspections', '0', '--inspection-seconds', '60',
                    '--workers', args.workers, '--top-k', '12', '--evidence-chars', '48000']+embedding_arguments+inference)
                stage = 'score'
                if not successful_predictions(prediction_path, subset):
                    raise ValueError('incomplete predictions cannot enter the scorer')
                scores_dir = folder/'scores'
                manifest(scores_dir/'input_manifest.json', {'questions_sha256': file_hash(qpath),
                    'predictions_sha256': file_hash(prediction_path), 'model': MODEL, 'base_url': BASE_URL, 'source_hash': SOURCE_HASH})
                predictions = {r['question_id']: r['prediction'] for r in load_records(prediction_path)}
                scored = successful_scores(scores_dir, subset, predictions)
                pending = [q for q in subset if q['question_id'] not in scored]
                if pending:
                    pending_path = folder/'pending_judgments.json'
                    write_json(pending_path, pending)
                    # Even an interrupted/failed scorer may have completed valid rows.
                    try:
                        run(folder, 'score', 'evaluation.eval', ['--data-path', pending_path,
                            '--predictions', prediction_path, '-g', 'episode', '--model', MODEL, '--base-url', BASE_URL,
                            '--output-dir', scores_dir, '--workers', args.workers, '--tries', '1',
                            '--timeout', '180', '--max-tokens', '8192'])
                    finally:
                        scored = successful_scores(scores_dir, subset, predictions)
                        write_records(folder/'accepted_scores.jsonl', [scored[q['question_id']] for q in subset if q['question_id'] in scored])
                if len(scored) != len(subset):
                    raise RuntimeError('official judgments incomplete')
                write_records(folder/'accepted_scores.jsonl', [scored[q['question_id']] for q in subset])
                return {'video_id': vid, 'status': 'complete', 'n_scored': len(scored)}
            except EmbeddingUnavailable as exc:
                return {'video_id':vid, 'status':'memory_complete_pending_embedding_service', 'reason':str(exc)}
            except Blocked as exc:
                return {'video_id': vid, 'status': 'blocked_api_credits', 'reason': str(exc)}
            except Exception as exc:
                rejection = policy_rejection(folder, memory/vid)
                if rejection:
                    return {'video_id':vid,'status':'blocked_input_policy','rejection':rejection}
                rejection = provider_rejection(memory/vid) if native and stage == 'build' else None
                if rejection:
                    return {'video_id':vid, 'status':'blocked_perception_provider', 'rejection':rejection}
                return {'video_id': vid, 'status': 'error', 'reason': str(exc)[:400]}

        try:
            remaining = None if native else credits()
        except Blocked as exc:
            if native:
                remaining = None
                status['embedding_credit_preflight'] = str(exc)
            else:
                status.update(status='blocked_api_credits', reason=str(exc))
                write_json(out/'status.json', status)
                print(json.dumps(status), flush=True)
                return 3
        status.update(status='running', initial_available_credits=remaining, videos={})
        write_json(out/'status.json', status)
        # V16 is the existing development video. Other initial videos are
        # ordered by duration only, so a complete end-to-end result arrives early.
        todo = sorted(vids, key=lambda v: (v != 'G2_V000016', inv['videos'][v]['duration_seconds'], v))
        attempts = {v: 0 for v in vids}
        stopped = False
        permanent_failures = 0
        passed_startup = False
        with ThreadPoolExecutor(max_workers=args.video_workers) as pool:
            active = {}
            def schedule(limit=None):
                while todo and len(active) < (limit or args.video_workers):
                    vid = todo.pop(0)
                    rejection = policy_rejection(out/'videos'/vid, memory/vid)
                    if rejection:
                        status['videos'][vid] = {'video_id':vid,'status':'blocked_input_policy','rejection':rejection}
                        continue
                    attempts[vid] += 1
                    active[pool.submit(video_run, vid)] = vid
            schedule(args.startup_videos)
            while active:
                done, _ = wait(active, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    vid = active.pop(future)
                    outcome = {**future.result(), 'video_attempt': attempts[vid]}
                    status['videos'][vid] = outcome
                    from methods.longemo.common import append_json
                    append_json(out/'video_attempts.jsonl', outcome)
                    if outcome['status'] in ('complete','memory_complete_pending_embedding_service'):
                        passed_startup = True
                    if outcome['status'] in ('blocked_api_credits','blocked_perception_provider'):
                        stopped = True
                    elif outcome['status'] not in ('complete','blocked_input_policy','memory_complete_pending_embedding_service'):
                        if attempts[vid] < args.video_attempts:
                            todo.insert(0,vid)
                        else:
                            permanent_failures += 1
                            if permanent_failures >= 3:
                                stopped = True
                    print(json.dumps(outcome), flush=True)
                status.update(updated_unix=time.time(), active_videos=sorted(active.values()),
                    queued_videos=len(todo), scheduling_paused=stopped)
                metrics = aggregate(out, questions)
                status['n_scored'] = metrics['overall_unweighted']['n_scored']
                write_json(out/'status.json', status)
                if not stopped:
                    schedule(None if passed_startup else args.startup_videos)
        metrics = aggregate(out, questions)
        complete = metrics['overall_unweighted']['coverage'] == 1
        failures = [r['status'] for r in status['videos'].values() if r['status'] != 'complete']
        status.update(status='complete' if complete and not failures else
                      ('blocked_api_credits' if 'blocked_api_credits' in failures else
                       'blocked_perception_provider' if 'blocked_perception_provider' in failures else
                       'partial_embedding_service' if 'memory_complete_pending_embedding_service' in failures and not todo else
                       'partial_input_policy' if failures and all(f == 'blocked_input_policy' for f in failures) and not todo else 'error'),
                      n_scored=metrics['overall_unweighted']['n_scored'], updated_unix=time.time())
        write_json(out/'status.json', status)
        return 0 if status['status'] == 'complete' else 3

if __name__ == '__main__':
    raise SystemExit(main())
