"""Offline checks only: tiny fixtures and local Python PTYs, never SSH/SCP."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


installer = module("install_full_media")
relay = module("transfer_full_media")


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.stage = self.runtime / "transfers/full558_test"
        self.manifest = self.root / "manifest.json"
        self.payload = b"tiny video fixture"
        digest = hashlib.sha256(self.payload).hexdigest()
        subdigest = hashlib.sha256(b"[]\n").hexdigest()
        self.rows = {f"G2_V{i:06d}": {"video_sha256": digest, "video_bytes": len(self.payload),
                    "subtitles_sha256": subdigest, "subtitles_bytes": 3} for i in range(1, 142)}
        self.manifest.write_text(json.dumps({"schema_version": 1, "videos": self.rows}))
        self.manifest_sha = installer.sha(self.manifest)
        self.args = types.SimpleNamespace(runtime=str(self.runtime), stage_dir=str(self.stage),
                    manifest=str(self.manifest), manifest_sha256=self.manifest_sha)
        self.setup = installer.setup(self.args)
        self.destination, self.ready = self.setup[1:3]

    def test_existing_verified_media_skips_and_never_overwrites_conflict(self):
        video = "G2_V000001"
        target = self.destination / (video + ".mp4")
        target.write_bytes(self.payload)
        result = installer.inspect(*self.setup)
        self.assertEqual(result["verified"], [video])
        self.assertEqual(len(result["missing"]), 140)
        target.write_bytes(b"unknown original")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            installer.inspect(*self.setup)
        self.assertEqual(target.read_bytes(), b"unknown original")

    def test_valid_upload_publishes_then_deletes_only_own_temp(self):
        video = "G2_V000002"
        temp = "upload-" + uuid.uuid4().hex + "-" + video + ".mp4.part"
        (self.stage / temp).write_bytes(self.payload)
        result = installer.install(*self.setup, video, temp)
        self.assertEqual(result["status"], "installed")
        self.assertEqual((self.destination / (video + ".mp4")).read_bytes(), self.payload)
        self.assertTrue((self.ready / (video + ".json")).is_file())
        self.assertFalse((self.stage / temp).exists())
        self.assertEqual(installer.install(*self.setup, video, temp)["status"], "already_ready")

    def test_bad_upload_and_symlink_never_publish(self):
        video = "G2_V000003"
        temp = "upload-" + uuid.uuid4().hex + "-" + video + ".mp4.part"
        (self.stage / temp).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "hash or size"):
            installer.install(*self.setup, video, temp)
        self.assertFalse((self.ready / (video + ".json")).exists())
        target = self.destination / (video + ".mp4")
        target.symlink_to(self.stage / temp)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            installer.install(*self.setup, video, temp)
        self.assertTrue(target.is_symlink())

    def test_done_requires_all_actual_video_and_subtitle_hashes(self):
        command = ["finish", "--runtime", str(self.runtime), "--stage-dir", str(self.stage),
                   "--manifest", str(self.manifest), "--manifest-sha256", self.manifest_sha]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(installer.main(command), 2)
        self.assertFalse((self.stage / "producer_done.json").exists())
        subs = self.runtime / "data/prepared_subtitles"
        subs.mkdir(parents=True)
        for video in self.rows:
            (self.destination / (video + ".mp4")).write_bytes(self.payload)
            (subs / (video + ".json")).write_bytes(b"[]\n")
        (subs / "G2_V000141.json").write_bytes(b"bad")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(installer.main(command), 2)
        self.assertFalse((self.stage / "producer_done.json").exists())
        (subs / "G2_V000141.json").write_bytes(b"[]\n")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(installer.main(command), 0)
        done = json.loads((self.stage / "producer_done.json").read_text())
        self.assertEqual(done["media_manifest_sha256"], self.manifest_sha)
        self.assertTrue(done["complete"])
        self.assertEqual(done["video_count"], 141)


class RelayTests(unittest.TestCase):
    def test_pty_sends_only_newline_to_one_password_prompt(self):
        output = relay.pty_command([sys.executable, "-c", "import sys;print('password:',end='',flush=True);s=sys.stdin.readline();print('RECEIVED:'+repr(s),flush=True)"], timeout=5)
        self.assertIn("RECEIVED:'\\n'", output)

    def test_second_password_prompt_stops(self):
        with self.assertRaises(relay.TransferFailure) as result:
            relay.pty_command([sys.executable, "-c", "import sys;print('password:',end='',flush=True);sys.stdin.readline();print('password:',end='',flush=True);sys.stdin.readline()"], timeout=5)
        self.assertEqual(result.exception.category, "repeated_password_prompt")

    def test_auth_failure_is_not_retried_and_network_is_bounded(self):
        fake = types.SimpleNamespace(publish=lambda **kwargs: None)
        for kind, expected in (("authentication_or_host_key_rejected", 1), ("explicit_network_failure", 3)):
            calls = []
            def operation():
                calls.append(1)
                raise relay.TransferFailure(kind)
            with patch.object(relay.time, "sleep"), self.assertRaises(relay.TransferFailure):
                relay.Relay.bounded(fake, operation)
            self.assertEqual(len(calls), expected)

    def test_local_nonzero_network_evidence_classified(self):
        with self.assertRaises(relay.TransferFailure) as result:
            relay.pty_command([sys.executable, "-c", "print('Connection timed out',flush=True);raise SystemExit(255)"], timeout=5)
        self.assertEqual(result.exception.category, "explicit_network_failure")

    def test_exec_failure_exits_child_instead_of_running_parent_code(self):
        with self.assertRaises(relay.TransferFailure) as result:
            relay.pty_command(["/nonexistent/longemo-offline-test"], timeout=5)
        self.assertEqual(result.exception.category, "unclassified_command_failure")
        self.assertEqual(result.exception.code, 127)


if __name__ == "__main__":
    unittest.main()
