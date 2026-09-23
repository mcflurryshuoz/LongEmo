import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.zyf import noevent_seed_continuation as launcher
from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf import noevent_retry_probe as probe


class Tests(unittest.TestCase):
    def fixture(self, root):
        root = root.resolve()
        parent, run, excluded = root / "parent", root / "new", root / "main24"
        parent.mkdir(); excluded.mkdir()
        selection = root / "selection.json"
        selection.write_text(json.dumps({"schema_version": 1, "condition": "noevent", "reason": "authorized seed", "question_ids": ["Q1", "Q2", "Q3"]}))
        (excluded / "configuration.json").write_text(json.dumps({"condition": "noevent", "parent_run": str(parent),
            "selection": {"question_ids": ["Q4"]}, "checkpoints": {"V2": {}}}))
        args = SimpleNamespace(stage="run", run=str(run), parent_run=str(parent), core_repo=str(root / "repo"),
            selection=str(selection), credential_file=str(root / "credentials.json"), probe_dir=str(root / "probe"),
            exclude_run=[str(excluded)], video_id="V1", visual_max_tokens=8192, protected_count=195,
            workers=1, question_workers=2, python="python", output=None)
        config = {"condition": "noevent", "execution_conditions": ["noevent"], "parent_run": str(parent),
            "selection": json.loads(selection.read_text()), "checkpoints": {"V1": {"completed_windows": 3}},
            "protected_scores": {"old-Q": {}}, "profile": {"visual": {"max_tokens": 8192}}}
        receipt = {"stage": "visual", "window_id": "W00004", "request_hash": "a" * 64,
            "model_configuration": {"model": "gemini-3.8-flash", "max_tokens": 8192},
            "probe_source_hashes": {}, "target_files_after": {"windows/W00004.json": "b" * 64}}
        return args, config, receipt

    def prepare(self, args, config, sequence):
        sequence.append("prepare")
        run = Path(args.run)
        (run / "configuration.json").write_text(json.dumps(config))
        (run / "questions.json").write_text(json.dumps([{"question_id": qid, "video_id": "V1"} for qid in config["selection"]["question_ids"]]))
        return run, config

    def apply_seed(self, args, receipt, sequence):
        sequence.append("seed")
        path = Path(args.run) / "probe_seeds/V1/receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
        return receipt

    def test_prepare_once_then_seed_then_original_execute_with_stage_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, receipt = self.fixture(Path(temporary))
            sequence, stages = [], []
            original_verify = Mock(side_effect=lambda *a: stages.append("original_verify"))
            def execute(run, actual):
                sequence.append("execute")
                base.verify(run, actual)
                base.verify(run, actual)
                return {"combined": {"scored": 195}}
            with patch.object(base, "prepare", side_effect=lambda a: self.prepare(a, config, sequence)) as prepare, \
                 patch.object(seed, "apply_seed", side_effect=lambda *a: self.apply_seed(args, receipt, sequence)) as apply, \
                 patch.object(seed, "verify_seed_receipt", return_value=receipt) as verify_seed, \
                 patch.object(base, "verify", original_verify), patch.object(base, "execute", side_effect=execute) as executor:
                result = launcher.launch(args)
                self.assertIs(base.verify, original_verify)
                self.assertEqual(sequence, ["prepare", "seed", "execute"])
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(apply.call_count, 1)
                self.assertEqual(executor.call_count, 1)
                self.assertEqual(verify_seed.call_count, 3)
                self.assertEqual(len(stages), 3)
                self.assertEqual(result["combined"]["scored"], 195)
            manifest = json.loads((Path(args.run) / "seed_execution.json").read_text())
            self.assertEqual(manifest["frozen_code"]["coordinator"]["sha256"], launcher.COORDINATOR_SHA256)
            self.assertEqual(set(manifest["frozen_code"]), {"launcher", "coordinator", "seed_helper", "probe"})
            self.assertEqual(manifest["parameters"]["visual_max_tokens"], 8192)

    def test_overlapping_question_or_video_is_rejected_before_prepare(self):
        for qids, video in ((["Q1"], "V2"), (["Q4"], "V1")):
            with self.subTest(qids=qids, video=video), tempfile.TemporaryDirectory() as temporary:
                args, config, receipt = self.fixture(Path(temporary))
                path = Path(args.exclude_run[0]) / "configuration.json"
                value = json.loads(path.read_text())
                value["selection"]["question_ids"] = qids
                value["checkpoints"] = {video: {}}
                path.write_text(json.dumps(value))
                with patch.object(base, "prepare") as prepare, self.assertRaises(ValueError):
                    launcher.launch(args)
                prepare.assert_not_called()
                self.assertFalse(Path(args.run).exists())

    def test_existing_launch_is_not_started_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, receipt = self.fixture(Path(temporary))
            Path(args.run).mkdir()
            with patch.object(base, "prepare") as prepare, self.assertRaises(FileExistsError):
                launcher.launch(args)
            prepare.assert_not_called()

    def test_selection_or_model_budget_cannot_expand(self):
        for update in ({"visual_max_tokens": 16384}, {"workers": 4}, {"protected_count": 197}):
            with self.subTest(update=update), tempfile.TemporaryDirectory() as temporary:
                args, _, _ = self.fixture(Path(temporary))
                for name, value in update.items():
                    setattr(args, name, value)
                with patch.object(base, "prepare") as prepare, self.assertRaises(ValueError):
                    launcher.launch(args)
                prepare.assert_not_called()

    def test_exact_seed_provenance_does_not_mutate_original_frozen_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, receipt = self.fixture(Path(temporary))
            run = Path(args.run); run.mkdir()
            (run / "seed_execution.json").write_text("{}")
            execution = {"video_id": "V1", "seed_receipt_sha256": "c" * 64}
            original = {"memory_sha256": "d" * 64, "windows": {"W00004": {"source": "continuation", "model": "nominal"}}}
            before = copy.deepcopy(original)
            result = launcher.exact_seed_provenance(run, config, execution, receipt, complete_memory=original)
            self.assertEqual(original, before)
            self.assertEqual(result["other_windows_provenance"]["W00004"]["source"], "validated_one_request_probe")
            self.assertEqual(result["model_configuration"]["max_tokens"], 8192)

    def test_code_or_exclusion_changes_fail_at_next_stage_and_guards_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, config, receipt = self.fixture(Path(temporary))
            sequence = []
            original_verify = Mock()
            def execute(run, actual):
                excluded = Path(args.exclude_run[0]) / "configuration.json"
                excluded.write_text("{}")
                with self.assertRaises(ValueError):
                    base.verify(run, actual)
                return {"combined": {"scored": 195}}
            with patch.object(base, "prepare", side_effect=lambda a: self.prepare(a, config, sequence)), \
                 patch.object(seed, "apply_seed", side_effect=lambda *a: self.apply_seed(args, receipt, sequence)), \
                 patch.object(seed, "verify_seed_receipt", return_value=receipt), \
                 patch.object(base, "verify", original_verify), patch.object(base, "execute", side_effect=execute):
                launcher.launch(args)
                self.assertIs(base.verify, original_verify)


if __name__ == "__main__":
    unittest.main()
