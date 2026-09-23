"""Read-only, no-credential first-score snapshot for the current noevent runs."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path('/root/longemo/runtime/runs')
BASE = ROOT / 'three_level_full558_gemini38_20260922'
NAMES = ['noevent_embedding_recovery_20260923_v2',
         'noevent_runtime24_continuation_20260923',
         'noevent_v31_seed_continuation_20260923',
         'noevent_length5_continuation_20260923',
         'noevent_length2_continuation_20260923',
         'noevent_v101_audio_seed_continuation_20260923',
         'noevent_audio429_G2_V000100_20260923',
         'noevent_audio429_G2_V000114_20260923',
         'noevent_audio429_G2_V000091_20260923',
         'noevent_v114_seed_length_continuation_20260923']

_RAW = {}

def raw(p):
    p = Path(p)
    assert p.is_absolute() and p.is_file() and not any(x.is_symlink() for x in (p, *p.parents)), 'non-regular snapshot source'
    if p not in _RAW:
        _RAW[p] = p.read_bytes()
    return _RAW[p]

def read(p):
    return json.loads(raw(p))

def sha(p):
    return hashlib.sha256(raw(p)).hexdigest()

def records(p):
    p = Path(p)
    value = ([json.loads(line) for line in raw(p).splitlines() if line.strip()]
             if p.suffix == '.jsonl' else read(p))
    assert isinstance(value, list)
    return value

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def validate_score(q, envelope, question):
    row = envelope['score']
    maximum = max(int(k) for k in question['rubric']['scores'])
    assert row['question_id'] == q and row['status'] == 'ok'
    assert row['video_id'] == question['video_id'] and row['type'] == question['type']
    assert maximum > 0 and type(row['max_score']) is int and row['max_score'] == maximum
    assert type(row['score']) in (int, float) and math.isfinite(row['score']) and 0 <= row['score'] <= maximum
    assert type(row['normalized_score']) in (int, float) and math.isfinite(row['normalized_score'])
    assert math.isclose(row['normalized_score'], row['score']/maximum)
    assert envelope['prediction_sha256'] == digest(row['prediction'])
    return row

out = Path(sys.argv[1]).absolute()
assert not any(x.is_symlink() for x in (out, *out.parents)), 'snapshot output cannot use symlink aliases'
out = out.resolve()
assert not out.is_relative_to(ROOT) and not ROOT.is_relative_to(out), 'snapshot output must be outside all runs'
base_config = read(BASE / 'configuration.json')
assert sha(BASE / 'configuration.json') == '55f0c2b0bfa8c6046a203e789f516698a5a8e77b21dc081f0174e80113e61b77'
full_questions = read(BASE / 'questions.json')
assert digest(full_questions) == base_config['questions_sha256']
questions = {q['question_id']: q for q in full_questions}
assert len(questions) == len(full_questions) == 558
protected = read(ROOT / NAMES[1] / 'configuration.json')['protected_scores']
assert len(protected) == 195
assert {p.stem: sha(p) for p in (BASE/'accepted/noevent').glob('*.json')} == {
    q: v['sha256'] for q, v in protected.items()}
all_rows = {}
for q, v in protected.items():
    assert v['envelope'] == read(BASE / 'accepted/noevent' / (q + '.json'))
    original = Path(v['envelope']['source'])
    assert sha(original) == v['official_source_sha256']
    row = validate_score(q, v['envelope'], questions[q])
    assert [r for r in records(original) if r.get('question_id') == q] == [row]
    all_rows[q] = row

result = {'as_of': datetime.now(timezone.utc).isoformat(), 'runs': {},
          'new_first_scores': {},
          'protected_original_first_scores': {q: v['sha256'] for q,v in protected.items()}}
for name in NAMES:
    run = ROOT/name
    if not (run/'configuration.json').exists():
        continue
    cfg = read(run/'configuration.json')
    selected = cfg.get('selection', {}).get('question_ids', cfg.get('question_ids'))
    assert isinstance(selected, list) and len(selected) == len(set(selected)) and set(selected) <= set(questions)
    chosen = read(run / 'questions.json')
    assert digest(chosen) == cfg['questions_sha256']
    assert len(chosen) == len(selected) and {q['question_id'] for q in chosen} == set(selected)
    assert all(q == questions[q['question_id']] for q in chosen)
    rows = {}
    for p in sorted((run/'accepted/noevent').glob('*.json')):
        v, q = read(p), p.stem
        assert q not in all_rows and q in selected
        question = questions[q]
        row = validate_score(q, v, question)
        original = Path(v['source'])
        video = question['video_id']
        assert original.is_absolute() and original == original.resolve()
        assert original.is_relative_to(run / 'scores/noevent' / video)
        assert [r for r in records(original) if r.get('question_id') == q] == [row]
        if name != NAMES[0] or 'source_sha256' in v:
            assert sha(original) == v['source_sha256']
        # The embedding recovery uses JSONL; later runs freeze a JSON list.
        input_files = [p for p in (run / 'score_inputs' / (video + '.json'),
                                   run / 'score_inputs/noevent' / (video + '.jsonl')) if p.exists()]
        assert len(input_files) == 1
        for artifact in (run / 'answers/noevent' / video / 'predictions.jsonl', input_files[0]):
            matches = [r for r in records(artifact) if r.get('question_id') == q]
            assert len(matches) == 1 and matches[0].get('status') == 'ok'
            assert matches[0].get('video_id') == video and matches[0].get('type') == question['type']
            assert matches[0].get('prediction') == row['prediction']
        rows[q] = row; all_rows[q] = row
        result['new_first_scores'][q] = {**{k: row[k] for k in
            ('question_id','video_id','type','status','score','max_score','normalized_score')},
            'run':str(run), 'source':str(original), 'source_sha256':sha(original),
            'accepted_sha256':sha(p), 'prediction_sha256':v.get('prediction_sha256')}
    tasks = collections.Counter()
    for p in (run/'tasks').glob('*/*/task.json'):
        d=read(p);tasks[p.parents[1].name+':'+d.get('status','unknown')]+=1
    pipelines=collections.Counter(read(p).get('status','unknown') for p in (run/'pipelines/noevent').glob('*.json'))
    launch = read(run/'launch_process.json') if (run/'launch_process.json').exists() else {}
    safe_launch = {key: launch[key] for key in ('pid', 'start_ticks', 'started_unix', 'configuration_sha256') if key in launch}
    if 'configuration_sha256' in safe_launch:
        assert safe_launch['configuration_sha256'] == sha(run/'configuration.json')
    profile = cfg.get('profile', {})
    safe_profile = {key: profile[key] for key in ('model', 'audio_model', 'embedding_model', 'window_seconds',
                    'padding', 'fps', 'max_frames', 'max_pixels', 'evidence_chars', 'stage_tries') if key in profile}
    safe_profile['visual'] = {key: profile.get('visual', {})[key] for key in ('model', 'max_tokens') if key in profile.get('visual', {})}
    length = cfg.get('length_recovery', {})
    result['runs'][name] = {'run':str(run), 'configuration_sha256':sha(run/'configuration.json'),
        'accepted_first_scores':len(rows), 'accepted_ids':sorted(rows),
        'tasks':dict(tasks), 'pipelines':dict(pipelines),
        'launch':safe_launch or None, 'selected_question_ids':selected,
        'selected_videos':sorted({q['video_id'] for q in chosen}), 'profile':safe_profile,
        'workers':cfg.get('workers'), 'question_workers':cfg.get('question_workers'),
        'completed_windows_preserved':sum(v.get('completed_windows',0) for v in cfg.get('checkpoints',{}).values()),
        'length_recovery': {key: length[key] for key in ('source_run','source_configuration_sha256','driver_sha256','comparison_scope') if key in length}}
result['combined'] = {'scored':len(all_rows), 'total':len(questions),
    'mean_scored':100*sum(r['normalized_score'] for r in all_rows.values())/len(all_rows),
    'new_scored':len(result['new_first_scores'])}
out.parent.mkdir(parents=True,exist_ok=True)
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps({'as_of':result['as_of'], 'combined':result['combined'],
                  'snapshot_sha256':sha(out)},ensure_ascii=False))
