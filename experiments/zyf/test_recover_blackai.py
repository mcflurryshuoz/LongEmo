import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.io_utils import write_json, write_records
from experiments.zyf.blackai_continuation import export_video, import_video
from experiments.zyf.recover_blackai import transient_failure, publish, install_recovery, frontend
from methods.longemo.common import manifest


class RecoveryTest(unittest.TestCase):
    def test_alternate_worker_retries_524_then_stops_on_policy_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest(root/'experiment_manifest.json', {'protocol': 'test'})
            write_json(root/'memory/V1/memory.json', {'complete': False})
            ledger = root/'memory/V1/calls.jsonl'
            transient = {'purpose': 'perception:V1:W1', 'status': 'error', 'http_status': 524}
            write_records(ledger, [transient])
            export_video(root, 'V1', {'status': 'frontend_failed'})
            write_json(root/'frontend_status.json', {'status': 'finished', 'videos': {'V1': {'status': 'frontend_failed'}}})
            write_json(root/'videos/V1/build_command.json', {'command': ['python', '-u', '-m', 'experiments.zyf.aicodemirror_worker', '--output-dir', str(root), '--credential-file', '/private/test.json']})
            def rejected(*args, **kwargs):
                write_records(ledger, [transient, {**transient, 'http_status': 400, 'service_error_code': 'content_filter'}])
                return type('Result', (), {'returncode': 1})()
            with patch('experiments.zyf.recover_blackai.subprocess.run', side_effect=rejected) as run:
                frontend(root, root, 3, worker_module='experiments.zyf.aicodemirror_worker')
                self.assertEqual(run.call_count, 1)
            state = json.loads((root/'recovery/frontend_status.json').read_text())
            self.assertEqual(state['attempts'], {'V1': 1})
            self.assertFalse(state['videos']['V1']['transient_remaining'])
            self.assertTrue((root/'recovery/outbox/V1.ready.json').exists())

    def test_only_explicit_transients_can_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = root/'memory/V1/calls.jsonl'
            failure = {'purpose': 'perception:V1:W1', 'status': 'error', 'http_status': 524}
            write_records(p, [failure]); self.assertTrue(transient_failure(root, 'V1'))
            write_records(p, [failure, {'purpose': failure['purpose'], 'status': 'ok'}])
            self.assertFalse(transient_failure(root, 'V1'))
            for change in ({'http_status': None}, {'http_status': 403}, {'service_error_code': 'content_filter'}):
                write_records(p, [{**failure, **change}]); self.assertFalse(transient_failure(root, 'V1'))

    def prepare(self, root):
        source = root/'source'; target = root/'target'
        for p in (source, target): manifest(p/'experiment_manifest.json', {'protocol': 'test'})
        write_json(source/'memory/V1/memory.json', {'video_id': 'V1', 'complete': False})
        export_video(source, 'V1', {'status': 'frontend_failed'})
        previous = json.loads((source/'outbox/V1.receipt.json').read_text())
        old_marker = json.loads((source/'outbox/V1.ready.json').read_text())
        import_video(source/'outbox/V1.tar.gz', old_marker, target, {'V1'})
        write_json(target/'backend_status.json', {'videos': {'V1': {'status': 'frontend_failed'}}})
        write_json(source/'memory/V1/memory.json', {'video_id': 'V1', 'complete': True})
        publish(source, 'V1', {'status': 'memory_complete'}, previous)
        marker = json.loads((source/'recovery/outbox/V1.ready.json').read_text())
        return source, target, marker

    def test_verified_unused_failure_replaced_and_history_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            source, target, marker = self.prepare(Path(d))
            receipt = install_recovery(target, source/'recovery/outbox/V1.tar.gz', marker)
            self.assertEqual(receipt['outcome']['status'], 'memory_complete')
            self.assertTrue(json.loads((target/'memory/V1/memory.json').read_text())['complete'])
            self.assertFalse(json.loads((target/'recovery/history/V1/memory/memory.json').read_text())['complete'])
            # Restart does not repeat installation or erase accepted outputs.
            write_records(target/'videos/V1/accepted_scores.jsonl', [{'question_id': 'Q1', 'score': 0}])
            install_recovery(target, source/'recovery/outbox/V1.tar.gz', marker)
            self.assertTrue((target/'videos/V1/accepted_scores.jsonl').exists())

    def test_unknown_change_or_answered_cache_cannot_be_replaced(self):
        for mode in ('changed_memory', 'answered'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                source, target, marker = self.prepare(Path(d))
                if mode == 'changed_memory': (target/'memory/V1/memory.json').write_text('{}')
                else: write_records(target/'videos/V1/graph/predictions.jsonl', [{'question_id': 'Q1', 'status': 'ok'}])
                before = (target/'memory/V1/memory.json').read_bytes()
                with self.assertRaises(AssertionError): install_recovery(target, source/'recovery/outbox/V1.tar.gz', marker)
                self.assertEqual((target/'memory/V1/memory.json').read_bytes(), before)


if __name__ == '__main__': unittest.main()
