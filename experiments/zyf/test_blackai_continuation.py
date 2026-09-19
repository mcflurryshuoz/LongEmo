import json
from pathlib import Path
import tempfile
import unittest

from evaluation.io_utils import write_json
from experiments.zyf.blackai_continuation import export_video, import_video
from methods.longemo.common import manifest


class MemoryTransferTest(unittest.TestCase):
    def test_export_import_and_repeat_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'; target = Path(temp)/'target'
            for p in (source, target): manifest(p/'experiment_manifest.json', {'provider': 'test'})
            write_json(source/'memory/V1/memory.json', {'complete': True, 'video_id': 'V1'})
            export_video(source, 'V1', {'status': 'memory_complete'})
            archive = source/'outbox/V1.tar.gz'
            marker = json.loads((source/'outbox/V1.ready.json').read_text())
            receipt = import_video(archive, marker, target, {'V1'})
            self.assertEqual(receipt['outcome']['status'], 'memory_complete')
            import_video(archive, marker, target, {'V1'})
            self.assertEqual((source/'memory/V1/memory.json').read_bytes(), (target/'memory/V1/memory.json').read_bytes())
            (target/'memory/V1/memory.json').write_text('{}')
            with self.assertRaises(AssertionError): import_video(archive, marker, target, {'V1'})

    def test_wrong_experiment_or_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'; target = Path(temp)/'target'
            manifest(source/'experiment_manifest.json', {'provider': 'one'})
            manifest(target/'experiment_manifest.json', {'provider': 'another'})
            write_json(source/'memory/V1/memory.json', {'complete': True})
            export_video(source, 'V1', {'status': 'memory_complete'})
            archive = source/'outbox/V1.tar.gz'; marker = json.loads((source/'outbox/V1.ready.json').read_text())
            with self.assertRaises(AssertionError): import_video(archive, marker, target, {'V1'})
            with self.assertRaises(AssertionError): import_video(archive, {**marker, 'sha256': 'bad'}, target, {'V1'})


if __name__ == '__main__': unittest.main()
