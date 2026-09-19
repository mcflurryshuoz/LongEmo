import tempfile
from pathlib import Path
import unittest

from evaluation.io_utils import write_json, write_records
from experiments.zyf.finish_scoring import scoring_inventory, audit_rows
from experiments.zyf.test_full_benchmark import question, score


class FinishScoringTest(unittest.TestCase):
    def test_valid_sibling_is_scored_once_and_rejected_siblings_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            qs = [question('Q'+str(i)) for i in range(1, 5)]
            vid = qs[0]['video_id']
            folder = run/'videos'/vid
            write_records(folder/'graph/predictions.jsonl', [
                {'question_id': 'Q'+str(i), 'status': 'ok' if i < 4 else 'error',
                 'prediction': 'answer' if i < 4 else None} for i in range(1, 5)])
            first = score('Q1', 0)
            write_records(folder/'scores/run_1/scores.jsonl', [first,
                {'question_id': 'Q3', 'status': 'error', 'error': 'HTTP 400 (content_filter)'}])
            write_records(folder/'graph/calls/Q4.jsonl', [
                {'service_error_code': 'content_filter', 'purpose': 'answer:Q4'}])
            write_json(run/'status.json', {'videos': {vid: {'status': 'blocked_input_policy'}}})
            inventory = scoring_inventory(run, qs)
            self.assertEqual([q['question_id'] for q in inventory[vid]['pending']], ['Q2'])
            self.assertEqual(inventory[vid]['scored']['Q1']['score'], 0)
            rows = {r['question_id']: r for r in audit_rows(run, inventory)}
            self.assertEqual(rows['Q3']['status'], 'blocked_judge_policy')
            self.assertEqual(rows['Q4']['status'], 'blocked_question_policy')
            write_json(run/'scoring_completion/attempt/selected.json', [qs[1]])
            inventory = scoring_inventory(run, qs)
            self.assertEqual(inventory[vid]['pending'], [])
            rows = {r['question_id']: r for r in audit_rows(run, inventory)}
            self.assertEqual(rows['Q2']['status'], 'judge_submission_unresolved')

    def test_upstream_block_does_not_claim_question_was_answered(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            q = question('Q1'); vid = q['video_id']
            write_json(run/'status.json', {'videos': {vid: {
                'status': 'blocked_request_precondition', 'rejection': {
                    'source': str(run/'memory'/vid/'audio/calls.jsonl'), 'purpose': 'audio:W1'}}}})
            rows = audit_rows(run, scoring_inventory(run, [q]))
            self.assertEqual(rows[0]['status'], 'blocked_upstream_request_precondition')
            self.assertEqual(rows[0]['stage'], 'audio')
            self.assertNotIn('score', rows[0])


if __name__ == '__main__':
    unittest.main()
