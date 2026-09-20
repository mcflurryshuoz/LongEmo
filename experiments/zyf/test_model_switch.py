import json
from pathlib import Path
import tempfile
import unittest

from experiments.zyf.blackai_model_switch import clone_checkpoint
from methods.longemo.common import file_hash


class CheckpointTests(unittest.TestCase):
    def test_copy_retains_provenance_and_invalidates_pending_audio_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "source"; source.mkdir()
            (source / "audio").mkdir()
            (source / "memory.json").write_text(json.dumps({"completed_windows": ["W00001"]}))
            (source / "manifest.json").write_text('{"configuration":{"model":"old"}}')
            for window in ["W00001", "W00002"]:
                (source / "audio" / (window + ".json")).write_text('{"model":"old"}')
            before = {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()}
            target = root / "new"
            receipt = clone_checkpoint(source, target)
            self.assertEqual(before, receipt["files"])
            self.assertEqual(before, {str(p.relative_to(source)): file_hash(p) for p in source.rglob("*") if p.is_file()})
            self.assertFalse((target / "manifest.json").exists())
            self.assertEqual(file_hash(target / "inherited_manifest.json"), before["manifest.json"])
            self.assertTrue((target / "audio/W00001.json").exists())
            self.assertFalse((target / "audio/W00002.json").exists())
            self.assertTrue((target / "inherited_pending_audio/W00002.json").exists())
            self.assertEqual(receipt, clone_checkpoint(source, target))

    def test_unknown_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            with self.assertRaises(AssertionError): clone_checkpoint(p / "missing", p)

    def test_completed_memory_is_not_selected_for_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); source = p / "source"; source.mkdir()
            (source / "memory.json").write_text('{"complete":true}')
            with self.assertRaises(AssertionError): clone_checkpoint(source, p / "new")
            self.assertFalse((p / "new").exists())

    def test_failure_in_first_window_can_start_without_memory_file(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); source = p / "source"; source.mkdir()
            (source / "manifest.json").write_text('{}')
            receipt = clone_checkpoint(source, p / "new")
            self.assertEqual(receipt["inherited_windows"], [])
            self.assertFalse((p / "new/memory.json").exists())


if __name__ == "__main__": unittest.main()
