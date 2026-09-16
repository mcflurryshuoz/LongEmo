"""Offline aggregate scores and retrieval diagnostics; no inference or gold export."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

from evaluation.io_utils import load_records, write_json
from evaluation.metrics import aggregate
from experiments.zyf.report_results import ledger


def read(path):
    return json.loads(path.read_text())


def mean(xs):
    return statistics.mean(xs) if xs else None


def bounds(event):
    return min(s[0] for s in event['spans']), max(s[1] for s in event['spans'])


def trace_diagnostics(trace, duration):
    evidence = trace.get('retrieval', {})
    events, timeline = evidence.get('events', []), evidence.get('timeline', [])
    recall = trace.get('recall_trace', {})
    selected = {e['id'] for e in events}
    visible = selected | {e['id'] for e in timeline} | {o['id'] for e in events for o in e.get('observations', [])}
    semantic = {x['id'] for x in recall.get('semantic', [])}
    dense = {x['id'] for x in recall.get('dense', [])}
    end = max((x['span'][1] for x in timeline), default=None)
    coverage = evidence.get('coverage', {})
    return {'question_id': trace['question']['question_id'], 'video_id': trace['question']['video_id'],
            'type': trace['question']['type'], 'evidence_characters': trace.get('evidence_characters'),
            'candidate_events': coverage.get('candidate_events'), 'full_events': len(events),
            'timeline_events': len(timeline), 'timeline_last_second': end,
            'timeline_end_fraction': end/duration if end is not None and duration else None,
            'full_event_spans': [list(bounds(e)) for e in events],
            'semantic_dense_overlap': len(semantic & dense),
            'dense_only_full_events': len((dense-semantic) & selected),
            'semantic_only_full_events': len((semantic-dense) & selected),
            'graph_neighbor_full_events': len(set(recall.get('graph_neighbor_ids', [])) & selected),
            'citations_absent_from_supplied_context': sorted(set(trace.get('answer', {}).get('evidence_ids', []))-visible),
            'invalid_memory_citations': trace.get('invalid_evidence_ids', [])}


def bootstrap_video_ci(rows, samples=2000):
    groups = defaultdict(list)
    for r in rows:
        groups[r['video_id']].append(r['normalized_score'])
    videos = list(groups.values())
    if len(videos) < 2:
        return None
    rng = random.Random(20260916)
    means = []
    for _ in range(samples):
        picked = [videos[rng.randrange(len(videos))] for _ in videos]
        means.append(sum(sum(v) for v in picked)/sum(len(v) for v in picked)*100)
    means.sort()
    return {'low': means[int(.025*samples)], 'high': means[min(samples-1, int(.975*samples))],
            'unit': 'video-cluster bootstrap; question-weighted score', 'samples': samples, 'seed': 20260916,
            'limitation': 'Does not measure judge/model randomness or dependence between related source videos.'}


def summarize(run, embedding_cache):
    config = read(run/'experiment_manifest.json')['configuration']
    rows = load_records(run/'scores.jsonl')
    valid = [r for r in rows if r['status'] == 'ok']
    inventory = read(run/'inventory.json')
    result = {'status': read(run/'status.json'), 'configuration': config,
              'data': {k:v for k,v in inventory.items() if k != 'videos'},
              'metrics': read(run/'metrics.json'), 'scored_questions': [], 'retrieval': []}
    for row in rows:
        result['scored_questions'].append({k:row.get(k) for k in
            ('question_id', 'video_id', 'type', 'status', 'score', 'max_score', 'normalized_score')})
    pilot = set(config['pilot_question_ids'])
    result['development_overlap'] = {'pilot': aggregate([r for r in rows if r['question_id'] in pilot]),
                                     'non_pilot': aggregate([r for r in rows if r['question_id'] not in pilot])}
    buckets = defaultdict(list)
    for row in rows:
        duration = inventory['videos'].get(row['video_id'], {}).get('duration_seconds')
        if duration is not None:
            buckets['under_10_min' if duration < 600 else '10_to_30_min' if duration < 1800 else 'over_30_min'].append(row)
    result['duration_groups'] = {k:aggregate(v) for k,v in buckets.items()}
    complete = len(valid) == config['question_count'] and len(rows) == config['question_count']
    result['full_score_complete'] = complete
    result['video_bootstrap_95_percent_ci'] = bootstrap_video_ci(valid) if complete else None
    for path in sorted((run/'videos').glob('*/graph/traces/*.json')):
        vid = path.parents[2].name
        duration = inventory['videos'].get(vid, {}).get('duration_seconds')
        result['retrieval'].append(trace_diagnostics(read(path), duration))
    traces = result['retrieval']
    result['retrieval_summary'] = {'n_questions': len(traces),
        'mean_full_events': mean([r['full_events'] for r in traces]),
        'questions_with_unseen_citations': sum(bool(r['citations_absent_from_supplied_context']) for r in traces),
        'questions_near_48k_budget': sum((r['evidence_characters'] or 0) > 45600 for r in traces),
        'trajectory_timeline_ends_before_half': sum(r['type'] == 'emotion trajectory' and
            r['timeline_end_fraction'] is not None and r['timeline_end_fraction'] < .5 for r in traces),
        'interpretation': 'Coverage diagnostics, not automatic correctness or causal error attribution.'}
    groups = {'perception_including_imports': list((run/'memory').glob('*/calls.jsonl')),
              'audio_including_imports': list((run/'memory').glob('*/audio/calls.jsonl')),
              'planning': list((run/'videos').glob('*/plans/calls/*.jsonl')),
              'answer': list((run/'videos').glob('*/graph/calls/*.jsonl'))}
    if embedding_cache:
        groups['embedding'] = list((embedding_cache/'api').glob('calls.jsonl'))
    result['costs'] = {k:ledger(v) for k,v in groups.items()}
    baseline = run/'imported_costs.json'
    result['imported_costs_already_incurred'] = read(baseline) if baseline.exists() else None
    judge = [raw for path in (run/'videos').glob('*/scores/run_*/scores.jsonl')
             for row in load_records(path) for raw in row.get('judge_responses', [])]
    result['judge_costs'] = {'returned_attempts': len(judge),
        'provider_reported_cost_usd': sum((r.get('usage') or {}).get('cost') or 0 for r in judge),
        'missing_cost_attempts': sum((r.get('usage') or {}).get('cost') is None for r in judge)}
    return result


def markdown(result):
    overall = result['metrics']['overall_unweighted']
    lines = ['# Full episode evaluation', '',
             f"Status: **{result['status']['status']}**. Official score coverage: **{overall['n_scored']}/{overall['n_total']}**.", '']
    if not result['full_score_complete']:
        lines += ['**No complete benchmark score is available. Partial scored means must not be reported as full results.**', '']
    lines += ['| Task | Scored / total | Normalized (%) |', '|---|---:|---:|']
    for name, row in result['metrics']['tasks'].items():
        pct = f"{row['percent_score']:.2f}" if row['percent_score'] is not None else 'pending'
        lines.append(f"| {name} | {row['n_scored']}/{row['n_total']} | {pct} |")
    lines += ['', 'The main model and official judge are GPT-6 Astra; self-judge bias remains possible. Nine development-pilot questions are disclosed separately in the JSON. No baseline improvement claim is supported without a completed comparison.', '',
              'Diagnostic exports include duration groups, temporal evidence coverage, semantic/dense overlap, graph expansion, supplied-context citation checks, and low-scoring question IDs. These signals prioritize review; they do not establish which stage caused an error.', '',
              'Imported perception/audio costs were incurred in earlier runs; subtract the separate imported-cost snapshot when reporting incremental spending. Missing provider usage prevents complete billing reconciliation.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--embedding-cache', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.run, a.embedding_cache)
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output/'results.json', result)
    (a.output/'results.md').write_text(markdown(result))
    print(json.dumps({'status': result['status']['status'], 'full_score_complete': result['full_score_complete']}))
