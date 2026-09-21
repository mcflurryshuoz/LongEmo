import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.zyf.aicodemirror_recharge import clone, guard
from experiments.zyf.blackai36_continuation import NAME
from methods.longemo.common import file_hash


class Tests(unittest.TestCase):
    def test_clone_preserves_completed_and_archives_pending_audio(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);s=root/'s';(s/'audio').mkdir(parents=True)
            (s/'memory.json').write_text('{"completed_windows":["W1"]}')
            (s/'audio/W1.json').write_text('{"preserved":true}')
            (s/'audio/W2.json').write_text('{"pending":true}')
            (s/'calls.jsonl').write_text('{"status":"error"}\n')
            before={str(p.relative_to(s)):file_hash(p) for p in s.rglob('*') if p.is_file()}
            clone(s,root/'t');t=root/'t'
            self.assertTrue((t/'audio/W1.json').exists());self.assertFalse((t/'audio/W2.json').exists())
            self.assertTrue((t/'ancestry'/NAME/'pending_audio/W2.json').exists())
            self.assertEqual(before,{str(p.relative_to(s)):file_hash(p) for p in s.rglob('*') if p.is_file()})
            self.assertEqual(json.loads((t/'checkpoint_source.json').read_text())['new_visual_model'],'claude-opus-5')

    def test_guard_rejects_success_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            rt=Path(d);p=rt/'runs'/NAME/'scores.jsonl';p.parent.mkdir(parents=True)
            p.write_text('{"question_id":"Q","status":"ok"}\n')
            with patch('experiments.zyf.aicodemirror_recharge.blackai.score_guard'):
                with self.assertRaises(AssertionError):guard(rt,[{'question_id':'Q'}])
                guard(rt,[{'question_id':'other'}])


if __name__=='__main__':unittest.main()
