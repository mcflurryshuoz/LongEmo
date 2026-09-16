import json
from pathlib import Path
import tempfile
import unittest

from evaluation.io_utils import write_json
from methods.longemo.common import file_hash
from experiments.zyf.refresh_imports import refresh


class RefreshImportsTest(unittest.TestCase):
    def setUp(self):
        import shutil
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/'run';self.run.mkdir()
        self.src=self.root/'source'/'V1'
        self.dst=self.run/'memory'/'V1'
        write_json(self.src/'manifest.json',{'configuration':{'source':'fixed'}})
        write_json(self.src/'memory.json',{'completed_windows':['W1']})
        write_json(self.src/'windows'/'W1.json',{'unchanged':True})
        write_json(self.src/'audio'/'W1.json',{'unchanged':True})
        shutil.copytree(self.src,self.dst)
        write_json(self.run/'imported_checkpoints.json',{'V1':{'source':str(self.src),
            'memory_sha256':file_hash(self.dst/'memory.json'), 'manifest_sha256':file_hash(self.dst/'manifest.json')}})
        write_json(self.src/'memory.json',{'completed_windows':['W1','W2']})
        write_json(self.src/'windows'/'W2.json',{'new':True})
        write_json(self.src/'audio'/'W2.json',{'new':True})

    def test_refresh_preserves_backup_and_reuses_new_windows(self):
        self.assertEqual(refresh(self.run),['V1'])
        self.assertEqual(file_hash(self.src/'memory.json'),file_hash(self.dst/'memory.json'))
        backups=list((self.run/'import_history').glob('*/V1/memory.json'))
        self.assertEqual(json.loads(backups[0].read_text())['completed_windows'],['W1'])
        self.assertEqual(refresh(self.run),[])

    def test_modified_source_prefix_is_rejected_before_replacement(self):
        before=file_hash(self.dst/'memory.json')
        write_json(self.src/'windows'/'W1.json',{'unchanged':False})
        with self.assertRaisesRegex(ValueError,'rewrote'):
            refresh(self.run)
        self.assertEqual(file_hash(self.dst/'memory.json'),before)

    def test_started_full_run_blocks_refresh(self):
        (self.run/'videos'/'V1').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,'already exist'):
            refresh(self.run)


if __name__=='__main__':
    unittest.main()
