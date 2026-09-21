import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from experiments.zyf.blackai_continuation import ready_marker


class Tests(unittest.TestCase):
    def test_partial_marker_then_complete_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'x.ready.json';p.write_text('')
            self.assertIsNone(ready_marker(p))
            (root/'x.tar.gz').write_bytes(b'test')
            m={'video_id':'V','archive':'x.tar.gz','sha256':'a'*64,'bytes':4}
            p.write_text(json.dumps(m));self.assertEqual(m,ready_marker(p))

    def test_archive_still_uploading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'x.ready.json';p.write_text(json.dumps({'video_id':'V','archive':'x.tar.gz','sha256':'a'*64,'bytes':4}))
            self.assertIsNone(ready_marker(p));(root/'x.tar.gz').write_bytes(b'te')
            self.assertIsNone(ready_marker(p));(root/'x.tar.gz').write_bytes(b'test')
            self.assertEqual(4,ready_marker(p)['bytes'])

    def test_stale_broken_marker_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.ready.json';p.write_text('{');os.utime(p,(time.time()-600,)*2)
            with self.assertRaises(ValueError):ready_marker(p)

    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.ready.json';p.write_text(json.dumps({'video_id':'V','archive':'../x.tar.gz','sha256':'a'*64,'bytes':4}))
            with self.assertRaises(AssertionError):ready_marker(p)


if __name__=='__main__':unittest.main()
