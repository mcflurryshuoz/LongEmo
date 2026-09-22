"""Read-only audit/export of a matched pilot; never calls models or changes a run."""
from __future__ import annotations
import argparse
import collections
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket

CONDITIONS = ('base', 'method', 'noevent')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def rows(path, active=False):
    if not path.exists():
        return []
    text = path.read_text()
    if active and not text.endswith('\n'):
        text = text.rsplit('\n', 1)[0] if '\n' in text else ''
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def identity(record):
    p = Path('/proc') / str(record['pid']) / 'stat'
    actual = p.read_text().rsplit(')', 1)[1].split() if p.exists() else None
    return {'pid': record['pid'], 'start_ticks': record.get('start_ticks'),
            'alive': bool(actual and record.get('host') == socket.gethostname()
                          and str(record.get('start_ticks')) == actual[19] and actual[0] != 'Z')}

def stats(values, total):
    return {'scored': len(values), 'total': total,
            'mean_scored': sum(values) / len(values) if values else None}

def snapshot(run):
    config = read(run / 'configuration.json')
    questions = read(run / 'questions.json')
    assert digest(questions) == config['questions_sha256']
    qmap = {q['question_id']: q for q in questions}
    assert len(qmap) == len(questions) == config['question_count']
    integrity = {'questions_sha256': sha(run / 'questions.json'), 'configuration_sha256': sha(run / 'configuration.json'),
                 'source_files_verified': {}, 'source_commits': {}, 'accepted_scores_verified': True}
    for branch, folder in config['repos'].items():
        repo = Path(folder)
        deployment = read(repo / 'deployment.json')
        for name, expected in deployment['files'].items():
            assert sha(repo / name) == expected, f'deployed file changed: {branch}/{name}'
        integrity['source_files_verified'][branch] = len(deployment['files'])
        integrity['source_commits'][branch] = deployment['commit']
    coordinator = identity(read(run / 'process.json'))
    with (run / 'coordinator.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            coordinator['lock_held'] = False
        except BlockingIOError:
            coordinator['lock_held'] = True
    tasks = {str(p.parent.relative_to(run / 'tasks')): read(p) for p in run.glob('tasks/*/*/task.json')}
    branches = {}
    for branch in ('event', 'noevent'):
        entries = []
        for video in sorted(config['media']):
            task = tasks.get('build/' + branch + '-' + video, {})
            status = {'ok': 'complete', 'error': 'failed'}.get(task.get('status'), task.get('status', 'pending'))
            folder = run / branch / 'videos' / video / 'memory' / video
            memory = read(folder / 'memory.json') if (folder / 'memory.json').exists() else {}
            total = math.ceil(memory['duration'] / config['window_seconds']) if memory else None
            item = {'video_id': video, 'status': status, 'completed_windows': len(memory.get('completed_windows', [])),
                    'total_windows': total, 'complete': bool(memory.get('complete')), 'final_failure': None}
            if task.get('child'):
                item['process'] = identity(task['child'])
            if status == 'complete':
                assert item['complete'], f'{branch}/{video}: zero exit without complete memory'
            if status == 'failed':
                completed = set(memory.get('completed_windows', []))
                failures = [row for path in (folder / 'calls.jsonl', folder / 'audio/calls.jsonl')
                            for row in rows(path) if row.get('status') == 'error'
                            and row.get('purpose', '').split(':')[-1] not in completed]
                latest = max(failures, key=lambda x: x.get('time_unix', 0), default={})
                keys = ('purpose', 'attempt', 'error_type', 'http_status', 'service_error_code', 'validation_error',
                        'failure_stage', 'retry_reason', 'will_retry', 'time_unix', 'request_hash')
                failure = {k: latest[k] for k in keys if k in latest}
                code = failure.get('service_error_code')
                category = ('content_filter' if code in ('content_filter', 'content_policy_violation') else
                            'http_428_unclassified' if failure.get('http_status') == 428 else
                            'generate_runtime_error_unknown' if failure.get('error_type') == 'RuntimeError'
                            and failure.get('failure_stage') == 'generate' else 'other')
                failure['category'] = category
                item['final_failure'] = failure
            entries.append(item)
        branches[branch] = {state: sum(x['status'] == state for x in entries)
                            for state in ('complete', 'running', 'failed', 'pending')}
        branches[branch].update(completed_windows=sum(x['completed_windows'] for x in entries), videos=entries)
    accepted = {c: {} for c in CONDITIONS}
    for condition in CONDITIONS:
        for path in sorted((run / 'accepted' / condition).glob('*.json')):
            value = read(path); row = value['score']; qid = row['question_id']; q = qmap[qid]
            assert qid == path.stem and row['status'] == 'ok' and qid not in accepted[condition]
            source = Path(value['source']); assert source.is_relative_to(run / 'scores' / condition)
            assert sum(x == row for x in rows(source)) == 1
            prediction_path = run / 'answers' / condition / q['video_id'] / 'predictions.jsonl'
            matches = [x for x in rows(prediction_path) if x['question_id'] == qid and x['status'] == 'ok']
            assert len(matches) == 1 and matches[0]['prediction'] == row['prediction']
            assert digest(row['prediction']) == value['prediction_sha256']
            maximum = max(int(x) for x in q['rubric']['scores'])
            assert row['max_score'] == maximum and 0 <= row['score'] <= maximum
            assert math.isclose(row['normalized_score'], row['score'] / maximum)
            branch = 'noevent' if condition == 'noevent' else 'event'
            memory_path = run / branch / 'videos' / q['video_id'] / 'memory' / q['video_id'] / 'memory.json'
            memory_hash = read(run / branch / 'frozen_memories' / (q['video_id'] + '.json'))['memory_sha256']
            assert sha(memory_path) == memory_hash
            plans = run / branch / 'plans'; frozen = read(plans / 'frozen' / (q['video_id'] + '.json'))
            assert set(frozen) == {x['question_id'] for x in questions if x['video_id'] == q['video_id']}
            assert all(sha(plans / (name + '.json')) == expected for name, expected in frozen.items())
            answer_folder = run / 'answers' / condition / q['video_id']
            index_signature = read(answer_folder / 'embedding_indexes.json')[q['video_id']]['signature']
            trace_path = answer_folder / 'traces' / (qid + '.json')
            routing = read(trace_path).get('routing')
            if condition in ('base', 'method'):
                expected_route = 'graph' if condition == 'base' else 'progressive'
                assert routing['effective_retrieval'] == expected_route
                assert routing['progressive_routing'] == 'none' and routing['hybrid_graph_tasks'] == []
            accepted[condition][qid] = {'status': 'scored', 'score': row['score'], 'max_score': maximum,
                'normalized_score_100': 100 * row['normalized_score'], 'source': str(source), 'source_sha256': sha(source),
                'accepted_sha256': sha(path), 'prediction_sha256': value['prediction_sha256'],
                'memory_sha256': memory_hash, 'plan_sha256': frozen[qid], 'embedding_signature': index_signature,
                'trace_sha256': sha(trace_path), 'routing': routing}
    for qid in set(accepted['base']) & set(accepted['method']):
        assert all(accepted['base'][qid][key] == accepted['method'][qid][key]
                   for key in ('memory_sha256', 'plan_sha256', 'embedding_signature'))
    integrity['paired_event_resources_and_pure_routing_verified'] = True
    types = sorted({q['type'] for q in questions})
    conditions = {}
    for c in CONDITIONS:
        conditions[c] = stats([x['normalized_score_100'] for x in accepted[c].values()], len(questions))
        conditions[c]['by_type'] = {t: stats([value['normalized_score_100'] for qid, value in accepted[c].items()
            if qmap[qid]['type'] == t], sum(q['type'] == t for q in questions)) for t in types}
    paired_ids = sorted(set.intersection(*(set(accepted[c]) for c in CONDITIONS)))
    means = {c: sum(accepted[c][qid]['normalized_score_100'] for qid in paired_ids) / len(paired_ids)
             if paired_ids else None for c in CONDITIONS}
    paired = {'count': len(paired_ids), 'question_ids': paired_ids, 'means': means,
              'method_minus_base': means['method'] - means['base'] if paired_ids else None,
              'base_minus_noevent': means['base'] - means['noevent'] if paired_ids else None}
    per_question = []
    for q in questions:
        item = {k: q[k] for k in ('question_id', 'video_id', 'type')}; item['conditions'] = {}
        for condition in CONDITIONS:
            qid, video = q['question_id'], q['video_id']
            if qid in accepted[condition]:
                item['conditions'][condition] = accepted[condition][qid]; continue
            branch = 'noevent' if condition == 'noevent' else 'event'
            build = next(x for x in branches[branch]['videos'] if x['video_id'] == video)
            status = {'failed': 'perception_failed', 'running': 'perception_running', 'pending': 'not_started'}.get(build['status'], 'waiting_for_planning')
            if build['status'] == 'complete':
                for stage, key, wait_name in [('plan', branch, 'waiting_for_answer'), ('answer', condition, 'waiting_for_scoring'), ('score', condition, 'scoring_without_valid_score')]:
                    state = tasks.get(stage + '/' + key + '-' + video, {}).get('status')
                    if state == 'ok':status = wait_name
                    elif state in ('running', 'needs_audit', 'error'):
                        status = stage + '_' + state; break
                    else:break
            item['conditions'][condition] = {'status': status, 'normalized_score_100': None}
            if status == 'perception_failed':item['conditions'][condition]['reason'] = build['final_failure']['category']
        per_question.append(item)
    command = read(run / 'launch_process.json')['command']
    execution = {name.replace('-', '_'): int(command[command.index('--' + name) + 1])
                 for name in ('build-workers', 'video-workers', 'question-workers')}
    return {'as_of': datetime.now(timezone.utc).isoformat(), 'run': run.name, 'configuration': config, 'execution': execution,
        'gate': read(run / 'gate.json') if (run / 'gate.json').exists() else None, 'coordinator': coordinator,
        'conditions': conditions, 'paired': paired, 'branches': branches, 'questions': per_question,
        'integrity': integrity, 'notes': [
            '每题首次有效评分归一化后等权平均，缺失不计零分。',
            '三路同题比较仅使用首分交集；一视频两题不足以判断方法优劣，不与历史511题结果直接比较。',
            '完整视频验证之后存在阶段屏障：其他视频感知任务全部结束，才进入规划、回答和评分。',
            'generate阶段RuntimeError尚无法区分响应结构异常、输出截断或非文本内容；不等同于内容过滤。',
            'HTTP 428的服务方语义尚未确认；未按临时网络故障或内容过滤自动重试。']}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = snapshot(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp'); temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'); temporary.replace(args.output)
    print(json.dumps({'as_of': result['as_of'], 'conditions': result['conditions'], 'branches': {b: {k: v for k,v in x.items() if k != 'videos'} for b,x in result['branches'].items()}}, ensure_ascii=False))

if __name__ == '__main__':main()
