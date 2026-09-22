"""Single-worker native PTY SCP relay; reuses an already verified AIStudio tunnel."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shlex
import signal
import time
import uuid

MARKER = "LONGEMO_MEDIA_RESULT "
NETWORK = re.compile(r"connection (?:timed out|refused|reset|closed)|operation timed out|broken pipe|no route to host|network is unreachable|connection reset by peer|connection to .* port .* timed out", re.I)
AUTH = re.compile(r"permission denied|authentication failed|host key verification failed|remote host identification has changed|enter passphrase|verification code|one.time password", re.I)


class TransferFailure(RuntimeError):
    def __init__(self, category, code=None):
        self.category, self.code = category, code
        super().__init__(category)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pty_command(command, timeout=7200):
    """Only one empty password response per connection; no credential guessing."""
    pid, master = pty.fork()
    if pid == 0:
        try:
            os.execvp(command[0], command)
        except OSError:
            os._exit(127)
    started, output, tail, answered, status = time.monotonic(), "", "", False, None
    try:
        while True:
            if time.monotonic() - started > timeout:
                raise TransferFailure("operation_timeout")
            readable, _, _ = select.select([master], [], [], 0.25)
            if readable:
                try:
                    block = os.read(master, 65536)
                except OSError:
                    block = b""
                if block:
                    text = block.decode("utf-8", "replace")
                    output = (output + text)[-131072:]
                    tail = (tail + text)[-4096:]
                    if AUTH.search(tail):
                        raise TransferFailure("authentication_or_host_key_rejected")
                    if re.search(r"password:\s*$", tail, re.I):
                        if answered:
                            raise TransferFailure("repeated_password_prompt")
                        os.write(master, b"\n")
                        answered, tail = True, ""
                else:
                    _, status = os.waitpid(pid, 0)
                    break
            child, child_status = os.waitpid(pid, os.WNOHANG)
            if child:
                status = child_status
                # Drain terminal output after the child's exit.
                while select.select([master], [], [], 0)[0]:
                    try:
                        block = os.read(master, 65536)
                    except OSError:
                        break
                    if not block:
                        break
                    output = (output + block.decode("utf-8", "replace"))[-131072:]
                break
        code = os.waitstatus_to_exitcode(status)
        if code != 0:
            if AUTH.search(output):
                raise TransferFailure("authentication_or_host_key_rejected", code)
            if NETWORK.search(output):
                raise TransferFailure("explicit_network_failure", code)
            raise TransferFailure("unclassified_command_failure", code)
        return output
    finally:
        if status is None:
            try:
                os.kill(pid, signal.SIGTERM)  # This child only, never another tunnel.
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        os.close(master)


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    ssh = cfg["ssh"]
    if ssh.get("host") != "127.0.0.1" or type(ssh.get("port")) is not int or not 1 <= ssh["port"] <= 65535:
        raise ValueError("SSH must use the current loopback port from runtime config")
    if ssh.get("verified") is not True or not ssh.get("expected_hostname"):
        raise ValueError("a native SSH session must already verify the target hostname")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", ssh.get("user", "root")):
        raise ValueError("invalid SSH user")
    for key in ("remote_runtime", "remote_manifest", "remote_installer", "remote_stage_dir"):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", cfg[key]):
            raise ValueError("remote task paths must be absolute, without shell metacharacters")
    if not re.fullmatch(r"[a-f0-9]{64}", cfg["manifest_sha256"]):
        raise ValueError("manifest SHA256 must be explicit")
    return cfg


def ssh_options(cfg, scp=False):
    ssh = cfg["ssh"]
    return ["-P" if scp else "-p", str(ssh["port"]), "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + ssh.get("known_hosts", "/private/tmp/aistudio-pure-ssh-known-hosts"),
            "-o", "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "-o", "NumberOfPasswordPrompts=1",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]


def remote_command(cfg, mode, video=None, temp_name=None):
    argv = [cfg.get("remote_python", "python3"), cfg["remote_installer"], mode, "--runtime", cfg["remote_runtime"],
            "--manifest", cfg["remote_manifest"], "--manifest-sha256", cfg["manifest_sha256"], "--stage-dir", cfg["remote_stage_dir"]]
    if video:
        argv += ["--video-id", video, "--temp-name", temp_name]
    if cfg.get("remote_subtitles_dir"):
        argv += ["--subtitles-dir", cfg["remote_subtitles_dir"]]
    target = cfg["ssh"].get("user", "root") + "@127.0.0.1"
    return ["ssh", "-tt", *ssh_options(cfg), target, shlex.join(argv)]


class Relay:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.initial = load_config(self.config_path)
        self.state_dir = Path(self.initial["state_dir"]).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.status = {"stage": "starting", "pid": os.getpid(), "started_unix": time.time(), "verified_count": 0}

    def cfg(self):
        cfg = load_config(self.config_path)
        # The dynamic port and verified tunnel details can change; dataset and
        # task paths cannot silently change under a running relay.
        for key in ("local_videos", "manifest", "manifest_sha256", "remote_runtime", "remote_manifest", "remote_installer", "remote_stage_dir", "state_dir"):
            if cfg[key] != self.initial[key]:
                raise TransferFailure("runtime_configuration_changed")
        return cfg

    def publish(self, **values):
        self.status.update(values, as_of_unix=time.time())
        write(self.state_dir / "status.json", self.status)

    def request(self, mode, video=None, temp_name=None):
        cfg = self.cfg()
        output = pty_command(remote_command(cfg, mode, video, temp_name), cfg.get("operation_timeout", 7200))
        lines = [line.split(MARKER, 1)[1].strip() for line in output.splitlines() if MARKER in line]
        if not lines:
            raise TransferFailure("remote_receipt_missing")
        receipt = json.loads(lines[-1])
        if receipt.get("hostname") != cfg["ssh"]["expected_hostname"]:
            raise TransferFailure("remote_identity_mismatch")
        if receipt.get("media_manifest_sha256") != cfg["manifest_sha256"]:
            raise TransferFailure("remote_manifest_mismatch")
        if receipt.get("status") == "error":
            raise TransferFailure("remote_verification_failed")
        return receipt

    def bounded(self, operation):
        for attempt in range(1, 4):
            try:
                return operation()
            except TransferFailure as exc:
                self.publish(last_error=exc.category, network_attempt=attempt)
                if exc.category != "explicit_network_failure" or attempt == 3:
                    raise
                time.sleep(2 * attempt)

    def transfer(self, video, row):
        cfg = self.cfg()
        source = Path(cfg["local_videos"]) / (video + ".mp4")
        if source.is_symlink() or not source.is_file() or source.stat().st_size != row["video_bytes"] or sha(source) != row["video_sha256"]:
            raise TransferFailure("local_video_verification_failed")
        temp_name = "upload-" + uuid.uuid4().hex + "-" + video + ".mp4.part"
        destination = cfg["ssh"].get("user", "root") + "@127.0.0.1:" + cfg["remote_stage_dir"] + "/" + temp_name
        self.publish(stage="uploading", video_id=video, video_bytes=row["video_bytes"], temp_name=temp_name)
        pty_command(["scp", *ssh_options(cfg, scp=True), "--", str(source), destination], cfg.get("operation_timeout", 7200))
        write(self.state_dir / "pending.json", {"video_id": video, "temp_name": temp_name, "manifest_sha256": cfg["manifest_sha256"]})
        return temp_name

    def run(self):
        with (self.state_dir / "transfer.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise TransferFailure("another_transfer_worker_active") from None
            write(self.state_dir / "process.json", {"pid": os.getpid(), "started_unix": time.time(), "config": str(self.config_path)})
            try:
                cfg = self.cfg()
                if sha(cfg["manifest"]) != cfg["manifest_sha256"]:
                    raise TransferFailure("local_manifest_mismatch")
                rows = json.loads(Path(cfg["manifest"]).read_text())["videos"]
                if len(rows) != 141:
                    raise TransferFailure("expected_141_video_manifest")
                self.publish(stage="verifying_remote", expected_count=141, expected_bytes=sum(r["video_bytes"] for r in rows.values()))
                remote = self.bounded(lambda: self.request("status"))
                verified = set(remote["verified"])
                if verified | set(remote["missing"]) != set(rows) or verified & set(remote["missing"]):
                    raise TransferFailure("remote_inventory_mismatch")
                write(self.state_dir / "remote_inventory.json", remote)
                pending_path = self.state_dir / "pending.json"
                pending = json.loads(pending_path.read_text()) if pending_path.exists() else None
                if pending and pending["manifest_sha256"] != cfg["manifest_sha256"]:
                    raise TransferFailure("pending_upload_manifest_mismatch")
                queue = sorted(set(rows) - verified, key=lambda video: (rows[video]["video_bytes"], video))
                # Recover an already uploaded file first. Otherwise a new upload
                # would replace its only local receipt before it is installed.
                if pending and pending["video_id"] in queue:
                    queue.remove(pending["video_id"])
                    queue.insert(0, pending["video_id"])
                self.publish(stage="ready", verified_count=len(verified), remaining_count=len(queue))
                for video in queue:
                    if pending and pending["video_id"] == video:
                        temp_name = pending["temp_name"]
                        receipt = self.bounded(lambda: self.request("install", video, temp_name))
                        if receipt.get("status") == "temp_missing":
                            temp_name = self.bounded(lambda: self.transfer(video, rows[video]))
                            receipt = self.bounded(lambda: self.request("install", video, temp_name))
                    else:
                        temp_name = self.bounded(lambda: self.transfer(video, rows[video]))
                        self.publish(stage="installing", video_id=video)
                        receipt = self.bounded(lambda: self.request("install", video, temp_name))
                    if receipt.get("status") not in ("installed", "already_ready") or receipt.get("video_sha256") != rows[video]["video_sha256"]:
                        raise TransferFailure("video_ready_receipt_invalid")
                    write(self.state_dir / "receipts" / (video + ".json"), receipt)
                    verified.add(video)
                    pending_path.unlink(missing_ok=True)
                    pending = None
                    self.publish(stage="published", video_id=video, verified_count=len(verified), remaining_count=141-len(verified),
                                 verified_bytes=sum(rows[v]["video_bytes"] for v in verified))
                self.publish(stage="final_verification", verified_count=141)
                done = self.bounded(lambda: self.request("finish"))
                if done.get("complete") is not True or done.get("video_count") != 141:
                    raise TransferFailure("completion_receipt_invalid")
                write(self.state_dir / "producer_done.json", done)
                pending_path.unlink(missing_ok=True)
                self.publish(stage="complete", complete=True, remaining_count=0)
                return 0
            except Exception as exc:
                self.publish(stage="needs_attention", complete=False, error=exc.category if isinstance(exc, TransferFailure) else type(exc).__name__)
                return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    return Relay(args.config).run()


if __name__ == "__main__":
    raise SystemExit(main())
