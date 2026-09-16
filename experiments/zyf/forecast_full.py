"""Project full-run cost from actual durations and disclosed, small pilot ledgers."""
import argparse
import json
import math
from pathlib import Path

from evaluation.io_utils import write_json, load_records
from experiments.zyf.report_results import ledger


def forecast(full_run, pilot, smoke):
    inventory = json.loads((full_run/'inventory.json').read_text())
    memories = [json.loads(p.read_text()) for p in (pilot/'memory').glob('*/memory.json')]
    completed = sum(len(m['completed_windows']) for m in memories)
    perception = ledger(list((pilot/'memory').glob('*/calls.jsonl')))
    audio = ledger(list((pilot/'memory').glob('*/audio/calls.jsonl')))
    window_cost = (perception['provider_reported_cost_usd']+audio['provider_reported_cost_usd'])/completed
    planning = ledger(list((smoke/'shared_plans/calls').glob('*.jsonl')))
    answering = ledger(list((smoke/'graph/calls').glob('*.jsonl')))
    rows = load_records(smoke/'graph/predictions.jsonl')
    questions = sum(r.get('status') == 'ok' for r in rows)
    score_files = sorted((smoke/'graph/scores').glob('run_*/scores.jsonl'))
    raw = [r for p in score_files for row in load_records(p) for r in row.get('judge_responses', [])]
    judge_cost = sum((r.get('usage') or {}).get('cost') or 0 for r in raw)
    # All smoke embedding charges are attributed to the two questions, including
    # one video's event index, rather than pretending cached embeddings are free.
    launch = json.loads((smoke/'launch.json').read_text())['command']
    cache = Path(launch[launch.index('--embedding-cache-dir')+1])
    embed = ledger(list((cache/'api').glob('calls.jsonl')))
    qa_cost = (planning['provider_reported_cost_usd']+answering['provider_reported_cost_usd']+
               judge_cost+embed['provider_reported_cost_usd'])/questions
    total_windows = inventory['windows'] if inventory['complete'] else None
    base = (max(0,total_windows-completed)*window_cost + inventory['question_count']*qa_cost) if total_windows else None
    return {'data_complete': inventory['complete'], 'inventoried_videos': inventory['video_count'],
            'total_videos': inventory['expected_videos'], 'duration_hours': inventory['duration_hours'],
            'total_windows': total_windows, 'question_count': inventory['question_count'],
            'calibration_completed_windows': completed, 'calibration_answered_questions': questions,
            'observed_returned_cost_per_window_usd': window_cost,
            'observed_incremental_plan_embedding_answer_judge_per_question_usd': qa_cost,
            'estimated_remaining_linear_cost_usd': base,
            'suggested_allowance_25_to_50_percent_headroom_usd': [math.ceil(base*1.25/100)*100, math.ceil(base*1.5/100)*100] if base else None,
            'scope': 'One graph-only full episode pass, with GPT-6 judging; imports reused; no full direct/inspection/independent-judge run.',
            'limitations': ['Empirical extrapolation from only 37 perception windows and two questions, not a price quote or confidence interval.',
                'Failed calls without returned usage may be missing from accounting.',
                'Longer-video contexts, cast size, repairs, provider routing and price changes can raise costs.']}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-run',type=Path,required=True)
    p.add_argument('--pilot-run',type=Path,required=True)
    p.add_argument('--smoke-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=forecast(a.full_run,a.pilot_run,a.smoke_run)
    write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
