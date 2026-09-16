"""Regression tests for full-run coverage, judgment reuse, and zero-call credit stop."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.io_utils import write_json, write_records
from experiments.zyf import full_benchmark as full


def question(qid):
    return {'question_id': qid, 'video_id': 'V1', 'granularity': 'episode',
            'type': 'emotion trajectory', 'question': 'Describe emotion changes.',
            'rubric': {'scores': {'0': 'none', '4': 'complete'}}}


def score(qid, value, prediction='answer'):
    return {'question_id': qid, 'video_id': 'V1', 'granularity': 'episode',
            'type': 'emotion trajectory', 'status': 'ok', 'score': value,
            'max_score': 4, 'normalized_score': value/4, 'prediction': prediction}


class FullRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_duplicate_predictions_cannot_be_scored(self):
        p = self.root/'predictions.jsonl'
        row = {'question_id': 'Q1', 'prediction': 'answer', 'status': 'ok'}
        write_records(p, [row, row])
        self.assertFalse(full.successful_predictions(p, [question('Q1'), question('Q2')]))

    def test_first_success_is_kept_even_when_retry_is_higher(self):
        write_records(self.root/'run_a'/'scores.jsonl', [score('Q1', 1), {'question_id': 'Q2', 'status': 'error'}])
        write_records(self.root/'run_b'/'scores.jsonl', [score('Q1', 4), score('Q2', 2)])
        found = full.successful_scores(self.root, [question('Q1'), question('Q2')], {'Q1': 'answer', 'Q2': 'answer'})
        self.assertEqual(found['Q1']['score'], 1)
        self.assertEqual(found['Q2']['score'], 2)

    def test_changed_prediction_rejects_cached_score(self):
        write_records(self.root/'run_a'/'scores.jsonl', [score('Q1', 1, 'old answer')])
        with self.assertRaisesRegex(ValueError, 'frozen prediction'):
            full.successful_scores(self.root, [question('Q1')], {'Q1': 'new answer'})

    def test_partial_scores_preserve_full_denominator(self):
        write_records(self.root/'videos'/'V1'/'accepted_scores.jsonl', [score('Q1', 4)])
        metrics = full.aggregate(self.root, [question('Q1'), question('Q2')])
        self.assertEqual(metrics['overall_unweighted']['n_total'], 2)
        self.assertEqual(metrics['overall_unweighted']['coverage'], 0.5)
        self.assertIn('INCOMPLETE', (self.root/'summary.md').read_text())

    def test_empty_run_has_no_invented_score(self):
        metrics = full.aggregate(self.root, [question('Q1'), question('Q2')])
        self.assertEqual(metrics['overall_unweighted']['coverage'], 0)
        self.assertIsNone(metrics['overall_unweighted']['percent_score'])

    def test_interrupted_official_scoring_resumes_only_failed_question(self):
        from types import SimpleNamespace
        data = self.root/'data'
        write_json(data/'questions.json', [question('Q1'), question('Q2')])
        write_json(data/'manifest.json', {'question_count': 2, 'video_count': 1,
                                        'revision': 'fixed', 'pilot_question_ids': ['Q1']})
        key = self.root/'private.json'
        write_json(key, {'OPENROUTER_API_KEY': 'unit-test-placeholder'})
        judged, invocations = [], [0]

        def fake_process(command, **kwargs):
            module = command[3]
            output = Path(command[command.index('--output-dir')+1])
            qpath = Path(command[command.index('--data-path')+1])
            questions = full.read(qpath)
            if module == 'methods.longemo' and command[4] == 'answer':
                write_records(output/'predictions.jsonl', [
                    {'question_id': q['question_id'], 'video_id': q['video_id'],
                     'prediction': 'answer', 'status': 'ok'} for q in questions])
            if module == 'evaluation.eval':
                invocations[0] += 1
                rows = []
                for q in questions:
                    qid = q['question_id']
                    judged.append(qid)
                    rows.append({'question_id': qid, 'status': 'error'}
                                if invocations[0] == 1 and qid == 'Q2' else score(qid, 0))
                write_records(output/f'run_{invocations[0]}'/'scores.jsonl', rows)
                return SimpleNamespace(returncode=int(any(r['status'] != 'ok' for r in rows)))
            return SimpleNamespace(returncode=0)

        def balance(*args, **kwargs):
            return io.BytesIO(json.dumps({'data': {'total_credits': 100, 'total_usage': 0}}).encode())

        command = ['--data-root', str(data), '--output-dir', str(self.root/'out'),
                   '--credential-file', str(key), '--embedding-cache-dir', str(self.root/'cache'), '--execute']
        with patch.object(full, 'code_hash', return_value=full.SOURCE_HASH), \
             patch.object(full, 'git_revision', return_value='test'), \
             patch.object(full, 'inventory', return_value={'complete': True, 'video_count': 1}), \
             patch.object(full, 'urlopen', side_effect=balance), \
             patch.object(full.subprocess, 'run', side_effect=fake_process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(full.main(command), 3)
            self.assertEqual(full.read(self.root/'out'/'metrics.json')['overall_unweighted']['coverage'], 0.5)
            self.assertEqual(full.main(command), 0)
        self.assertEqual(judged, ['Q1', 'Q2', 'Q2'])
        metrics = full.read(self.root/'out'/'metrics.json')['overall_unweighted']
        self.assertEqual(metrics['coverage'], 1)
        self.assertEqual(metrics['percent_score'], 0)

    def test_credit_block_makes_no_model_subprocess_calls(self):
        data = self.root/'data'
        write_json(data/'questions.json', [question('Q1')])
        write_json(data/'manifest.json', {'question_count': 1, 'video_count': 1,
                                        'revision': 'fixed', 'pilot_question_ids': ['Q1']})
        key = self.root/'private.json'
        write_json(key, {'OPENROUTER_API_KEY': 'unit-test-placeholder'})
        response = io.BytesIO(json.dumps({'data': {'total_credits': 10, 'total_usage': 9}}).encode())
        with patch.object(full, 'code_hash', return_value=full.SOURCE_HASH), \
             patch.object(full, 'git_revision', return_value='test'), \
             patch.object(full, 'inventory', return_value={'complete': True, 'video_count': 1}), \
             patch.object(full, 'urlopen', return_value=response), \
             patch.object(full.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            rc = full.main(['--data-root', str(data), '--output-dir', str(self.root/'out'),
                            '--credential-file', str(key), '--embedding-cache-dir', str(self.root/'cache'), '--execute'])
        self.assertEqual(rc, 3)
        run.assert_not_called()
        self.assertEqual(full.read(self.root/'out'/'status.json')['status'], 'blocked_api_credits')


if __name__ == '__main__':
    unittest.main()
