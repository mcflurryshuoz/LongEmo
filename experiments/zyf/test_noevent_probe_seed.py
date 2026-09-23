import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.zyf import noevent_probe_seed as seed, noevent_retry_probe as probe
from experiments.zyf import test_noevent_retry_probe as fixture_helpers
from evaluation.clients import Client
from methods.longemo import common, noevent_runner as runner


class Tests(unittest.TestCase):
    def fixture(self, root, stage="visual"):
        root = root.resolve()
        parent, source, repo, content = fixture_helpers.Tests().fixture(root, stage)
        with patch.object(common, "code_hash", return_value="frozen"), \
             patch.object(runner, "probe", return_value={"duration": 40.0}), \
             patch.object(runner, "window_input", return_value=(content, {"audio": True, "subtitle_ids": []})):
            prepared = probe.prepare(parent, "V", repo)
        payload = {"observations": []} if stage == "audio" else {
            "entities": [], "observations": [], "summary": "The same room later.", "emotion_cues": []}
        response = {"model": "gemini-3.8-flash", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
        diagnostic = root / "diagnostic"
        diagnostic.mkdir(mode=0o700)
        probe.execute_once(prepared, diagnostic, Mock(return_value=fixture_helpers.Response(json.dumps(response).encode())))
        continuation = root / "continuation"
        target = seed.target_folder(continuation, "V")
        target.mkdir(parents=True)
        committed = {"memory.json": probe.sha(source / "memory.json")}
        pending = {"audio/W00002.json": probe.sha(source / "audio/W00002.json")} if stage == "visual" else {}
        for name in {**committed, **pending}:
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, path)
        checkpoint = {"source": str(source), "files": {p.relative_to(source).as_posix(): probe.sha(p) for p in source.rglob("*") if p.is_file()},
            "parent_manifest_sha256": probe.sha(source / "manifest.json"), "committed_files": committed,
            "pending_audio_strict_validation": pending, "completed_windows": 1}
        cfg = {"condition": "noevent", "execution_conditions": ["noevent"], "parent_run": str(parent),
               "core_repo": str(repo), "checkpoints": {"V": checkpoint}, "profile": {"window_seconds": 20}}
        (continuation / "configuration.json").write_text(json.dumps(cfg))
        (continuation / "prepared.json").write_text(json.dumps({"configuration_sha256": probe.sha(continuation / "configuration.json")}))
        (continuation / "inheritance").mkdir()
        (continuation / "inheritance/V.json").write_text(json.dumps(checkpoint))
        return parent, continuation, repo, source, target, diagnostic, prepared, checkpoint

    def test_visual_import_advances_one_window_without_api_or_parent_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            before = {str(p): probe.sha(p) for p in parent.rglob("*") if p.is_file()}
            with patch.object(probe, "prepare", return_value=prepared), patch.object(Client, "generate") as generate:
                receipt = seed.apply_seed(parent, run, "V", diagnostic, repo)
                generate.assert_not_called()
                self.assertEqual(seed.apply_seed(parent, run, "V", diagnostic, repo), receipt)
            self.assertEqual(json.loads((target / "memory.json").read_text())["completed_windows"], ["W00001", "W00002"])
            self.assertTrue(json.loads((target / "memory.json").read_text())["complete"])
            self.assertEqual(receipt["model_configuration"]["max_tokens"], prepared["client"].max_tokens)
            self.assertEqual(receipt["source"], "validated_one_request_probe")
            self.assertFalse((target / "manifest.json").exists())
            self.assertEqual(before, {str(p): probe.sha(p) for p in parent.rglob("*") if p.is_file()})
            self.assertEqual(seed.verify_seed_receipt(run, "V", checkpoint), receipt)

    def test_original_builder_skips_the_imported_visual_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            with patch.object(probe, "prepare", return_value=prepared):
                seed.apply_seed(parent, run, "V", diagnostic, repo)
            task = json.loads((parent / "tasks/build/noevent-V/task.json").read_text())
            args = probe.parse_original_command(task, repo, runner)
            args.output_dir = str(target.parent)
            with patch.object(common, "code_hash", return_value="frozen"), \
                 patch.object(runner, "code_hash", return_value="frozen"), \
                 patch.object(runner, "probe", return_value={"duration": 40.0}), \
                 patch.object(runner, "window_input", side_effect=AssertionError("completed window must not be extracted again")), \
                 patch.object(Client, "generate", side_effect=AssertionError("completed window must not be called again")):
                result = runner._build_video("V", args, prepared["client"])
            self.assertEqual(result["status"], "ok")
            self.assertIsNotNone(seed.verify_seed_receipt(run, "V", checkpoint, allow_memory_growth=True))

    def test_audio_import_only_populates_exact_original_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary), "audio")
            before = (target / "memory.json").read_bytes()
            with patch.object(probe, "prepare", return_value=prepared):
                receipt = seed.apply_seed(parent, run, "V", diagnostic, repo)
            cache = json.loads((target / "audio/W00002.json").read_text())
            self.assertEqual(cache, {"input_fingerprint": prepared["artifact"]["audio_input_fingerprint"],
                                   "model": prepared["client"].configuration(), "result": {"observations": []}})
            self.assertEqual((target / "memory.json").read_bytes(), before)
            self.assertEqual(receipt["stage"], "audio")

    def test_tampered_probe_artifact_is_rejected_without_target_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            before = (target / "memory.json").read_bytes()
            path = diagnostic / "validated_payload.json"
            value = json.loads(path.read_text())
            value["payload"]["summary"] = "tampered"
            path.write_text(json.dumps(value))
            with patch.object(probe, "prepare", return_value=prepared), self.assertRaises(ValueError):
                seed.apply_seed(parent, run, "V", diagnostic, repo)
            self.assertEqual((target / "memory.json").read_bytes(), before)
            self.assertFalse((target / "windows/W00002.json").exists())

    def test_seed_refuses_already_claimed_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            task = run / "tasks/build/noevent-V/task.json"
            task.parent.mkdir(parents=True)
            task.write_text('{"status":"error"}')
            with patch.object(probe, "prepare") as prepare, self.assertRaises(ValueError):
                seed.apply_seed(parent, run, "V", diagnostic, repo)
            prepare.assert_not_called()

    def test_existing_unknown_window_or_audio_is_never_overwritten(self):
        for stage, relative in (("visual", "windows/W00002.json"), ("audio", "audio/W00002.json")):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary), stage)
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("unknown")
                with patch.object(probe, "prepare", return_value=prepared), self.assertRaises(ValueError):
                    seed.apply_seed(parent, run, "V", diagnostic, repo)
                self.assertEqual(path.read_text(), "unknown")

    def test_interrupted_seed_intent_blocks_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            pending = run / "probe_seeds/V"
            pending.mkdir(parents=True)
            (pending / "intent.json").write_text("{}")
            with patch.object(probe, "prepare") as prepare, self.assertRaises(ValueError):
                seed.apply_seed(parent, run, "V", diagnostic, repo)
            prepare.assert_not_called()

    def test_target_pending_audio_must_match_validated_probe_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent, run, repo, source, target, diagnostic, prepared, checkpoint = self.fixture(Path(temporary))
            (target / "audio/W00002.json").write_text("{}")
            with patch.object(probe, "prepare", return_value=prepared), self.assertRaises(ValueError):
                seed.apply_seed(parent, run, "V", diagnostic, repo)
            self.assertFalse((target / "windows/W00002.json").exists())


if __name__ == "__main__":
    unittest.main()
