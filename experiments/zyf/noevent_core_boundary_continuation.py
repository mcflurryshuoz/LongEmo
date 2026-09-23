"""One fixed three-video continuation after offline core-ownership projection.

Only the final recorded invalid visual response is considered. Drop observations
wholly outside the owned core and entire emotion cues depending on them. Never
move timestamps, invent evidence, weaken the original validator, or repeat the
exhausted request. Rebuild the exact initial and repair message hashes without
HTTP before importing one window. All later windows use the unchanged builder.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_length_continuation as guards
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf import noevent_retry_probe as probe
from experiments.zyf.matched_pilot import digest, freeze, read, records, sha, stopped_parent, memory_root

CASES = {
    'G2_V000051': ('three_level_full558_gemini38_20260922', 43, [165, 166, 167, 168]),
    'G2_V000075': ('three_level_full558_gemini38_20260922', 55, [268, 269, 270]),
    'G2_V000124': ('noevent_runtime24_continuation_20260923', 34, [487, 488, 489, 490, 491]),
}
FIRST_SCORES = 250
ERROR = 'observation must overlap the owned core interval'
_VERIFY, _PROVENANCE = base.verify, base.window_provenance


def project(payload, core, media):
    """Deletion only; preserve every retained field and reject malformed spans."""
    if not isinstance(payload, dict) or not isinstance(payload.get('observations'), list):
        raise ValueError('invalid observation structure')
    result = copy.deepcopy(payload)
    ids, removed = set(), []
    for item in payload['observations']:
        oid, span = item.get('id'), item.get('span')
        if not isinstance(oid, str) or not oid or oid in ids:
            raise ValueError('ambiguous observation IDs')
        ids.add(oid)
        if (not isinstance(span, list) or len(span) != 2 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in span) or
                span[1] < span[0] or span[0] < media[0] or span[1] > media[1]):
            raise ValueError('invalid or out-of-media span cannot be projected')
        if span[1] < core[0] or span[0] >= core[1]:
            removed.append(oid)
    if not removed or len(removed) == len(payload['observations']):
        raise ValueError('projection must retain supported core observations')
    cues = payload.get('emotion_cues', [])
    if not isinstance(cues, list):
        raise ValueError('invalid emotion cues')
    for cue in cues:
        refs = cue.get('evidence_ids')
        if not isinstance(refs, list) or not refs or any(x not in ids for x in refs):
            raise ValueError('unknown emotion evidence cannot be repaired')
    result['observations'] = [x for x in result['observations'] if x['id'] not in removed]
    result['emotion_cues'] = [x for x in cues if not set(x['evidence_ids']) & set(removed)]
    return result, {'removed_observation_ids': removed,
                    'removed_emotion_cues': len(cues) - len(result['emotion_cues']),
                    'retained_observations': len(result['observations'])}


def validate_projection(runner, memory, payload, window, core, interval, sampling):
    try:
        runner.apply_window(memory, payload, window_id=window, core=core, media=interval, metadata=sampling)
    except ValueError as exc:
        if str(exc) != ERROR:
            raise ValueError('source failure is not exclusively core ownership') from None
    else:
        raise ValueError('valid responses must not be reprocessed')
    normalized, details = project(payload, core, interval)
    updated = runner.apply_window(memory, normalized, window_id=window, core=core, media=interval, metadata=sampling)
    return normalized, updated, details


def reconstruct(source, video, repo, checkpoint):
    """Strictly reconstruct all three original request hashes; no API calls."""
    modules = probe.bind_repo(repo)
    runner, common = modules['methods.longemo.noevent_runner'], modules['methods.longemo.common']
    from evaluation.clients import Client
    task = read(base.regular(source / 'tasks/build' / ('noevent-' + video) / 'task.json'))
    command = task['command']
    if '--api-key' in command or any(x.startswith('--api-key=') for x in command):
        raise ValueError('inline credentials are forbidden')
    if '--' in command:
        if not any(x.endswith('/noevent_diagnostics_worker.py') for x in command):
            raise ValueError('unexpected source wrapper')
        args = runner.parser().parse_args(command[command.index('--') + 1:])
    else:
        args = probe.parse_original_command(task, repo, runner)
    folder = Path(checkpoint['source'])
    if Path(args.output_dir).absolute() / video != folder or args.tries != 3:
        raise ValueError('source arguments do not match the frozen checkpoint')
    memory = read(folder / 'memory.json')
    count = len(memory['completed_windows'])
    if count != CASES[video][1]:
        raise ValueError('source completed prefix differs from the audited case')
    window = f'W{count+1:05d}'
    core = [count * args.window_seconds, min((count+1)*args.window_seconds, memory['duration'])]
    interval = [max(0, core[0]-args.padding), min(memory['duration'], core[1]+args.padding)]
    ledger = folder / 'calls.jsonl'
    rows = [r for r in records(ledger) if r.get('purpose') == f'window_perception:{video}:{window}']
    if [r.get('attempt') for r in rows] != [1, 2, 3]:
        raise ValueError('exactly three exhausted source attempts are required')
    evidence = {}
    for row in rows:
        if (row.get('status') != 'error' or row.get('failure_stage') != 'validate' or
                row.get('error_type') != 'ValueError' or row.get('validation_error') != ERROR or
                row.get('service_error_code') or row.get('http_status') not in (None, 200)):
            raise ValueError('unknown failure or refusal cannot enter boundary projection')
        p = base.regular(folder / row['diagnostic_path'])
        if sha(p) != row['diagnostic_sha256'] or not p.is_relative_to(folder / 'diagnostics'):
            raise ValueError('recorded diagnostic hash or path changed')
        d = read(p)
        if (d['metadata']['request_hash'] != row['request_hash'] or d.get('content_redacted') is not False or
                hashlib.sha256(d['content'].encode()).hexdigest() != d['response_sha256'] or
                d['response_sha256'] != row['response_sha256']):
            raise ValueError('recorded final content is not intact')
        evidence[str(p)] = sha(p)
    if rows[-1].get('will_retry') is not False:
        raise ValueError('source budget has not terminated')
    visual, audio = runner.client_for(args), runner.audio_client_for(args)
    if visual.configuration() != checkpoint['parent_model'] or audio.configuration() != checkpoint['parent_audio_observer']:
        raise ValueError('source model configuration changed')
    if visual.configuration()['max_tokens'] != 8192:
        raise ValueError('fixed cases retain the 8192 visual budget')
    pending = base.regular(folder / 'audio' / (window + '.json'))
    if read(pending).get('model') != audio.configuration():
        raise ValueError('pending audio model changed')
    with patch.object(Client, 'generate', side_effect=AssertionError('offline projection cannot call any API')):
        media, sampling = runner.window_input(Path(args.videos_dir) / (video+'.mp4'), *interval,
            subtitle_rows=runner.subtitles(Path(args.subtitles_dir)/(video+'.json')), **runner._media_options(args))
        media, audio_trace = runner.bridge_audio(media, interval, audio, pending, args.tries)
    sampling.update(audio_representation='derived_timestamped_cues', audio_observer=audio_trace)
    context = runner.perception_context(memory, window_id=window, core=core, media=interval)
    messages = [{'role':'system','content':runner.PERCEPTION_NOEVENT}, {'role':'user','content':[
        {'type':'text','text':json.dumps(context, ensure_ascii=False)}] + media}]
    initial = common.fingerprint(messages)
    for row in rows:
        if row['initial_request_hash'] != initial or row['request_hash'] != common.fingerprint(messages):
            raise ValueError('reconstructed original/repair request differs')
        d = read(folder / row['diagnostic_path'])
        messages += [{'role':'assistant','content':d['content']}, {'role':'user',
            'content':'Repair the JSON to satisfy the schema. Validation error: '+row['validation_error']}]
    payload = common.json_object(d['content'])
    normalized, updated, details = validate_projection(runner, memory, payload, window, core, interval, sampling)
    if len(details['removed_observation_ids']) != 1:
        raise ValueError('each audited case must remove exactly one context-only observation')
    return {'window': window, 'payload': normalized, 'updated': updated, 'context': context,
            'sampling': sampling, 'details': details, 'request_hash': rows[-1]['request_hash'],
            'initial_request_hash': initial, 'diagnostic_hashes': evidence,
            'parent_memory_sha256': sha(folder/'memory.json'), 'pending_audio_sha256': sha(pending)}


def seed_window(run, video, checkpoint, artifact):
    folder = memory_root(run, 'noevent', video) / video
    path = folder / 'memory.json'
    if sha(base.regular(path)) != artifact['parent_memory_sha256']:
        raise ValueError('cloned memory changed before projection')
    window = artifact['window']
    if sha(base.regular(folder/'audio'/(window+'.json'))) != artifact['pending_audio_sha256']:
        raise ValueError('cloned pending audio changed')
    if (run/'tasks').exists() or (folder/'manifest.json').exists():
        raise ValueError('cannot seed a started run')
    seed.exclusive_json(folder/'windows'/(window+'.json'), {'input':artifact['context'],
        'sampling':artifact['sampling'], 'perception':artifact['payload']})
    seed.replace_known_memory(path, artifact['updated'], artifact['parent_memory_sha256'])
    receipt = {k: artifact[k] for k in ('window','details','request_hash','initial_request_hash','diagnostic_hashes',
                                       'parent_memory_sha256','pending_audio_sha256')}
    receipt.update(no_api_calls=True, policy='drop_context_only_observation_and_entire_dependent_emotion_cue',
        checkpoint_sha256=digest(checkpoint), window_sha256=sha(folder/'windows'/(window+'.json')),
        memory_sha256_after_import=sha(path), retained_payload_sha256=digest(artifact['payload']))
    freeze(run/'boundary_seeds'/(video+'.json'), receipt)
    return receipt


def verify(run, config):
    _VERIFY(run, config)
    execution = read(base.regular(run/'boundary_execution.json'))
    if sha(run/'boundary_execution.json') != read(run/'boundary_execution.sha256.json')['sha256']:
        raise ValueError('projection execution manifest changed')
    for p, expected in execution['protected_files'].items():
        if sha(base.regular(p)) != expected:
            raise ValueError('source state, code, or protected first score changed')
    for item in execution['claims']:
        if read(base.regular(item['path'])) != item['value']:
            raise ValueError('source recovery claim changed')
    for video, receipt in execution['seeds'].items():
        if read(run/'boundary_seeds'/(video+'.json')) != receipt:
            raise ValueError('projection receipt changed')
        folder = memory_root(run, 'noevent', video)/video
        window = receipt['window']
        if sha(base.regular(folder/'windows'/(window+'.json'))) != receipt['window_sha256']:
            raise ValueError('projected seed window changed')
        if window not in read(folder/'memory.json')['completed_windows']:
            raise ValueError('projected seed disappeared')
    return execution


def provenance(run, config, video):
    value = _PROVENANCE(run, config, video)
    receipt = verify(run, config)['seeds'][video]
    exact = copy.deepcopy(value)
    exact['windows'][receipt['window']].update(source='offline_core_ownership_projection',
        original_request_hash=receipt['request_hash'], no_api_calls=True, projection=receipt['details'])
    freeze(run/'lineage/windows'/(video+'.json'), exact)
    return value


@contextmanager
def hooks():
    base.verify, base.window_provenance = verify, provenance
    try:
        yield
    finally:
        base.verify, base.window_provenance = _VERIFY, _PROVENANCE


def launch(args):
    run, parent, repo = (Path(x).absolute() for x in (args.run, args.parent_run, args.core_repo))
    if run.exists():
        raise FileExistsError('one fixed run only; existing output requires audit')
    sources = {v: parent.parent / item[0] for v,item in CASES.items()}
    allq = read(parent/'questions.json')
    expected = sorted(f'G2_Q{x:06d}' for _,_,qs in CASES.values() for x in qs)
    if sorted(read(args.selection)['question_ids']) != expected or args.workers != 3 or args.question_workers != 2:
        raise ValueError('fixed 12 questions, three videos and two question workers required')
    protected = base.protected_scores(parent, allq, 195)
    related = sorted({str(Path(x).absolute()) for x in args.related_run})
    if str(parent) in related or str(run) in related:
        raise ValueError('invalid related run')
    extra = guards.score_snapshot(related, allq, set(expected), protected)
    if len(extra)+len(protected) != FIRST_SCORES:
        raise ValueError('all 250 first scores must be protected before launch')
    old = read(parent/'configuration.json')
    receipts, artifacts, protected_files, claims = {}, {}, {}, []
    original = base.checkpoint_receipt
    for source in sorted(set(sources.values())):
        with stopped_parent(source):
            for name in ('configuration.json','process.json'):
                protected_files[str(source/name)] = sha(base.regular(source/name))
            for video in [v for v,s in sources.items() if s==source]:
                q = [q for q in allq if q['question_id'] in expected and q['video_id']==video]
                base.ensure_no_backend_attempts(source, q)
                protected_files.update(guards.terminal_video(source,video))
                receipts[video] = original(source,repo,old,video)
                artifacts[video] = reconstruct(source,video,repo,receipts[video])
    if args.preflight_only:
        return {'continuation':{'preflight':'passed','no_api_calls':True,'protected_first_scores':FIRST_SCORES,
            'cases':{v:{'window':artifacts[v]['window'],'details':artifacts[v]['details'],
                        'source_completed_windows':receipts[v]['completed_windows'],
                        'request_hash':artifacts[v]['request_hash']} for v in CASES}}}
    for item in extra.values():
        protected_files[item['path']] = item['sha256']
        protected_files[item['source']] = item['source_sha256']
    for p in (Path(__file__), Path(guards.__file__), Path(seed.__file__), Path(probe.__file__)):
        protected_files[str(p.absolute())] = sha(p)
    for video, source in sources.items():
        claim_path = parent.parents[1]/'recovery_claims/noevent_core_boundary'/(digest(str(source))[:20]+'-'+video+'.json')
        claim = {'source':str(source), 'video':video, 'owner':str(run), 'driver_sha256':sha(__file__),
                 'selection_sha256':sha(args.selection), 'protocol':'offline_deletion_only_seed_one_build_task'}
        guards.claim_video(claim_path,claim)
        claims.append({'path':str(claim_path),'value':claim})
    with patch.object(base,'checkpoint_receipt',side_effect=lambda p,r,o,v: receipts[v]):
        _,config = base.prepare(args)
    # Every offline seed was validated before any target is modified or API scheduled.
    seeds = {v:seed_window(run,v,receipts[v],artifacts[v]) for v in CASES}
    freeze(run/'boundary_execution.json', {'created_at':datetime.now(timezone.utc).isoformat(),
        'protected_first_scores':FIRST_SCORES, 'protected_files':protected_files, 'claims':claims, 'seeds':seeds,
        'no_api_seed':True,'new_task_budget':'one unchanged build per video; no outer retry',
        'scope':'Only three final recorded responses are projected; all later windows use unchanged frozen core.'})
    freeze(run/'boundary_execution.sha256.json', {'sha256':sha(run/'boundary_execution.json')})
    with hooks():
        return base.execute(run,config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','parent-run','core-repo','selection','credential-file'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--related-run',action='append',default=[])
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    args.visual_max_tokens,args.protected_count,args.workers,args.question_workers=8192,195,3,2
    result=launch(args)
    print(json.dumps(result['continuation']))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
