"""Offline guards for the fixed V26 backend continuation; no model calls."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.zyf.noevent_embedding_resume import isolated, verify_frozen_inputs, verify_helper_sources
from experiments.zyf.matched_pilot import digest, run_command


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class NoeventEmbeddingResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "independent"
        self.video = "G2_V000026"
        self.questions = [{"question_id": qid, "video_id": self.video, "question": "original text"}
                          for qid in ("Q1", "Q2")]
        self.imports = [{"signature": f"batch{i}", "source": f"source{i}", "source_sha256": f"sha{i}"}
                        for i in range(3)]
        self.values = {item["signature"]: {"signature": item["signature"], "vectors": [[1.0, 2.0]]}
                       for item in self.imports}
        write(self.run / "questions.json", self.questions)
        write(self.run / "questions" / (self.video + ".json"), self.questions)
        write(self.run / "embedding_imports.json", self.imports)
        for signature, value in self.values.items():
            write(self.run / "noevent/embeddings/api" / (signature + ".json"), value)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        verify_frozen_inputs(self.run, self.video, self.questions, self.imports, self.values)

    def test_valid_input_check_is_read_only(self):
        before = {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        self.verify()
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob("*") if p.is_file()})

    def test_both_question_files_reject_changed_text_or_question_id(self):
        for path in (self.run / "questions.json", self.run / "questions" / (self.video + ".json")):
            for field, value in (("question", "changed text"), ("question_id", "wrong question")):
                with self.subTest(path=str(path), field=field):
                    changed = [{**q} for q in self.questions]
                    changed[0][field] = value
                    write(path, changed)
                    with self.assertRaisesRegex(ValueError, "question selection changed"):
                        self.verify()
                    write(path, self.questions)

    def test_all_three_batches_fail_closed_on_missing_changed_or_symlink_cache(self):
        for signature, value in self.values.items():
            path = self.run / "noevent/embeddings/api" / (signature + ".json")
            with self.subTest(signature=signature):
                path.unlink()
                with self.assertRaisesRegex(ValueError, "missing or not a regular file"):
                    self.verify()
                write(path, {**value, "vectors": [[2.0, 1.0]]})
                with self.assertRaisesRegex(ValueError, "imported embedding batch changed"):
                    self.verify()
                target = self.root / (signature + "-external.json")
                write(target, value)
                path.unlink()
                path.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "missing or not a regular file"):
                    self.verify()
                path.unlink()
                write(path, value)

    def test_import_receipt_cannot_change_provenance_or_drop_a_batch(self):
        path = self.run / "embedding_imports.json"
        for changed in (self.imports[:2], [{**item, "source_sha256": "different"} for item in self.imports]):
            with self.subTest(changed=changed):
                write(path, changed)
                with self.assertRaisesRegex(ValueError, "import receipt changed"):
                    self.verify()

    def test_helper_hashes_cover_original_coordinator_and_both_support_files(self):
        core = self.root / "core"
        folder = core / "experiments/zyf"
        names = ("full_suite.py", "matched_pilot.py", "full_suite_inheritance.py")
        for name in names:
            write(folder / name, {"original": name})
        hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}
        config = {"suite_sha256": hashes["full_suite.py"],
                  "suite_support_hashes": {name: hashes[name] for name in names[1:]}}
        self.assertEqual(verify_helper_sources(core, config), hashes)
        for name in names:
            with self.subTest(name=name):
                path = folder / name
                previous = path.read_bytes()
                path.write_text("changed helper")
                with self.assertRaisesRegex(ValueError, "helper changed"):
                    verify_helper_sources(core, config)
                path.write_bytes(previous)

    def test_recorded_answer_or_judge_task_never_spawns_again(self):
        folder = self.root / "task"
        cwd = self.root / "core"
        command = ["python", "do-not-run.py"]
        signature = digest({"command": command, "cwd": str(cwd)})
        with patch("experiments.zyf.matched_pilot.subprocess.Popen", side_effect=AssertionError("duplicate worker")) as spawn:
            for state in ("ok", "error", "running", "needs_audit"):
                write(folder / "task.json", {"signature": signature, "status": state})
                actual = run_command(folder, command, cwd)
                self.assertEqual(actual["status"], "needs_audit" if state == "running" else state)
            spawn.assert_not_called()

    def test_continuation_directory_cannot_overlap_parent(self):
        parent = self.root / "parent"
        for run in (parent, parent / "child", self.root):
            with self.assertRaisesRegex(ValueError, "isolated"):
                isolated(run, parent)
        isolated(self.root / "separate", parent)


if __name__ == "__main__":
    unittest.main()
