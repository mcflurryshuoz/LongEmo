"""Score previously unjudged successful answers, then account for every question.

This orchestration-only pass never builds memory, generates answers, or retries
an existing judge attempt. A rejected sibling question cannot hide a valid answer.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluation.io_utils import load_questions, load_records, write_json, write_records
from experiments.zyf.azure_benchmark import aggregate, successful_scores
from methods.longemo.common import code_hash, file_hash, git_revision


def records(path):
    return load_records(path) if path.exists() else []


def scoring_inventory(run, questions):
    """Include valid answers across partially failed videos; never resample a judge."""
    groups = defaultdict(list)
    for q in questions:
        groups[q['video_id']].append(q)
    result = {}
    for vid, subset in groups.items():
        folder = run/'videos'/vid
        predictions = {r['question_id']: r for r in records(folder/'graph/predictions.jsonl')}
        valid = {key: r['prediction'] for key, r in predictions.items()
                 if r.get('status') == 'ok' and isinstance(r.get('prediction'), str)
                 and r['prediction'].strip()}
        eligible = [q for q in subset if q['question_id'] in valid]
        scored = successful_scores(folder/'scores', eligible, valid)
        attempts = [r for p in sorted((folder/'scores').glob('run_*/scores.jsonl')) for r in records(p)]
        attempted = {r['question_id'] for r in attempts}
        reserved = {r['question_id'] for p in (run/'scoring_completion').glob('*/selected.json')
                    for r in json.loads(p.read_text())}
        pending = [q for q in eligible if q['question_id'] not in scored
                   and q['question_id'] not in attempted | reserved]
        # An accepted score must still match the same first successful judgment.
        for row in records(folder/'accepted_scores.jsonl'):
            if row['question_id'] not in scored or row != scored[row['question_id']]:
                raise ValueError('accepted score no longer matches frozen judgment')
        result[vid] = dict(questions=subset, predictions=predictions, scored=scored,
                           attempts=attempts, pending=pending, reserved=reserved)
    return result


def audit_rows(run, inventory):
    video_status = json.loads((run/'status.json').read_text()).get('videos', {})
    rows = []
    for vid, item in inventory.items():
        folder = run/'videos'/vid
        for q in item['questions']:
            qid = q['question_id']
            row = {'question_id': qid, 'video_id': vid, 'type': q['type']}
            if qid in item['scored']:
                s = item['scored'][qid]
                row.update(status='scored', score=s['score'], max_score=s['max_score'],
                           normalized_score=s['normalized_score'])
            else:
                attempts = [r for r in item['attempts'] if r['question_id'] == qid]
                calls = [(stage, r) for stage, p in (
                    ('planning', folder/'plans/calls'/f'{qid}.jsonl'),
                    ('answer', folder/'graph/calls'/f'{qid}.jsonl')) for r in records(p)]
                rejected = [(stage, r) for stage, r in calls if r.get('service_error_code')
                            in ('content_filter', 'content_policy_violation')]
                state = video_status.get(vid, {})
                source = state.get('rejection', {}).get('source', '')
                if attempts:
                    policy = any(any(code in str(r.get('error', '')) for code in
                                     ('content_filter', 'content_policy_violation')) for r in attempts)
                    row.update(status='blocked_judge_policy' if policy else 'judge_failed', stage='judge')
                elif rejected:
                    row.update(status='blocked_question_policy', stage=rejected[0][0])
                elif source.startswith(str(run/'memory') + '/'):
                    row.update(status='blocked_upstream_policy' if state['status'] == 'blocked_input_policy'
                               else 'blocked_upstream_request_precondition',
                               stage='audio' if '/audio/' in source else 'perception',
                               purpose=state['rejection'].get('purpose'))
                elif qid in item['reserved']:
                    row.update(status='judge_submission_unresolved', stage='judge')
                elif any(qid == p['question_id'] for p in item['pending']):
                    row.update(status='ready_for_first_judgment', stage='judge')
                else:
                    row.update(status='unresolved', stage='answer')
            rows.append(row)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--credential-file', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args(argv)
    run = args.run.resolve()
    with (run/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((run/'experiment_manifest.json').read_text())['configuration']
        if code_hash() != config['code_hash'] or file_hash(args.data_root/'questions.json') != config['question_sha256']:
            raise ValueError('frozen method/evaluator or questions changed')
        questions = load_questions(args.data_root/'questions.json', 'episode')
        inventory = scoring_inventory(run, questions)
        selected = [q for item in inventory.values() for q in item['pending']]
        print(json.dumps({'ready_for_first_judgment': [q['question_id'] for q in selected]}), flush=True)
        if args.execute and selected:
            credentials = json.loads(args.credential_file.read_text())
            env = dict(os.environ, MODEL_API_KEY=credentials['MODEL_API_KEY'], MODEL_BASE_URL=config['base_url'])
            stamp = run/'scoring_completion'/str(time.time_ns())
            stamp.mkdir(parents=True)
            for name in ('status.json', 'scores.jsonl', 'metrics.json'):
                (stamp/name).write_bytes((run/name).read_bytes())
            provenance = {'time_unix': time.time(), 'git_revision': git_revision(),
                          'orchestrator_sha256': file_hash(__file__), 'source_hash': code_hash(),
                          'predictions_sha256': {vid: file_hash(run/'videos'/vid/'graph/predictions.jsonl')
                                                 for vid, item in inventory.items() if item['pending']},
                          'selection': 'Successful predictions with no prior judge attempt; independent of score.'}
            write_json(stamp/'provenance.json', provenance)
            # Record submission intent before calls; ambiguous crashes are not retried.
            write_json(stamp/'selected.json', selected)
            for vid, item in inventory.items():
                if not item['pending']:
                    continue
                folder = run/'videos'/vid
                qpath = stamp/(vid+'.json')
                write_json(qpath, item['pending'])
                command = [sys.executable, '-u', '-m', 'evaluation.eval', '--data-path', str(qpath),
                           '--predictions', str(folder/'graph/predictions.jsonl'), '-g', 'episode',
                           '--model', config['judge_model'], '--base-url', config['base_url'],
                           '--output-dir', str(folder/'scores'), '--workers', '2', '--tries', '1',
                           '--timeout', '180', '--max-tokens', str(config['max_tokens'])]
                write_json(stamp/(vid+'_command.json'), {'command': command})
                with (stamp/(vid+'.log')).open('w') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                print(json.dumps({'video_id': vid, 'judge_exit_code': result.returncode}), flush=True)
        inventory = scoring_inventory(run, questions)
        if args.execute:
            for vid, item in inventory.items():
                path = run/'videos'/vid/'accepted_scores.jsonl'
                ordered = [item['scored'][q['question_id']] for q in item['questions'] if q['question_id'] in item['scored']]
                if ordered and records(path) != ordered:
                    write_records(path, ordered)
            metrics = aggregate(run, questions)
        else:
            metrics = json.loads((run/'metrics.json').read_text())
        rows = audit_rows(run, inventory)
        assert len(rows) == len({r['question_id'] for r in rows}) == config['question_count']
        report = {'time_unix': time.time(), 'source_run': str(run), 'metrics': metrics,
                  'question_count': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
                  'task_status_counts': {task: dict(Counter(r['status'] for r in rows if r['type'] == task))
                                         for task in sorted({r['type'] for r in rows})},
                  'questions': rows,
                  'note': 'Upstream blocks were attempted at video level; their question answering/judging was not completed. Missing scores are not zero.'}
        write_json(run/'task_completion.json', report)
        print(json.dumps({k: v for k, v in report.items() if k not in ('questions', 'metrics')}, ensure_ascii=False), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
