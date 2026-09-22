"""Verified, non-overwriting publication of one LongEmo video at a time."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import uuid

MARKER = "LONGEMO_MEDIA_RESULT "


def sha(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("expected a regular file, not a link")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("file changed during verification")
    return h.hexdigest()


def regular_matches(path, expected_sha, size):
    path = Path(path)
    return (stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size == size and sha(path) == expected_sha)


def freeze_json(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        if not stat.S_ISREG(path.lstat().st_mode) or json.loads(path.read_text()) != value:
            raise ValueError("refusing to replace an unknown publication marker")
        return
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)  # Atomic publication; never replaces a target.
    finally:
        temporary.unlink()


def setup(args):
    runtime, stage = Path(args.runtime).absolute(), Path(args.stage_dir).absolute()
    if runtime.resolve() != runtime or stage.parent.resolve() != stage.parent:
        raise ValueError("runtime/staging path may not traverse symlinks")
    if not stage.is_relative_to(runtime / "transfers"):
        raise ValueError("staging must remain inside runtime/transfers")
    stage.mkdir(parents=True, exist_ok=True)
    if stage.resolve() != stage:
        raise ValueError("staging cannot be a symlink")
    manifest_path = Path(args.manifest)
    manifest_sha = sha(manifest_path)
    if manifest_sha != args.manifest_sha256:
        raise ValueError("media manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    videos = manifest["videos"]
    if manifest.get("schema_version") != 1 or len(videos) != 141:
        raise ValueError("expected the normalized 141-video manifest")
    for video, row in videos.items():
        if not re.fullmatch(r"G2_V\d{6}", video) or not re.fullmatch(r"[0-9a-f]{64}", row["video_sha256"]) or row["video_bytes"] <= 0:
            raise ValueError("invalid video manifest entry")
    destination = runtime / "data/episode/videos"
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination:
        raise ValueError("video destination cannot traverse symlinks")
    ready = stage / "ready"
    ready.mkdir(exist_ok=True)
    if ready.resolve() != ready:
        raise ValueError("ready directory cannot be a symlink")
    freeze_json(stage / "manifest_identity.json", {"media_manifest_sha256": manifest_sha, "video_count": 141})
    return stage, destination, ready, manifest_sha, videos


def ready_record(video, row, manifest_sha):
    return {"video_id": video, "ready": True, "video_sha256": row["video_sha256"],
            "video_bytes": row["video_bytes"], "media_manifest_sha256": manifest_sha}


def inspect(stage, destination, ready, manifest_sha, videos, selected=None):
    verified, missing = [], []
    for video in sorted(selected or videos):
        row, path = videos[video], destination / (video + ".mp4")
        if not path.exists() and not path.is_symlink():
            missing.append(video)
            continue
        if not regular_matches(path, row["video_sha256"], row["video_bytes"]):
            raise ValueError(f"existing video conflicts with manifest: {video}")
        freeze_json(ready / (video + ".json"), ready_record(video, row, manifest_sha))
        verified.append(video)
    return {"verified": verified, "missing": missing}


def install(stage, destination, ready, manifest_sha, videos, video, temp_name):
    if video not in videos or not re.fullmatch(r"upload-[0-9a-f]{32}-" + re.escape(video) + r"\.mp4\.part", temp_name or ""):
        raise ValueError("unexpected video or unique staging filename")
    existing = inspect(stage, destination, ready, manifest_sha, videos, [video])
    if existing["verified"]:
        return {"status": "already_ready", **ready_record(video, videos[video], manifest_sha)}
    source = stage / temp_name
    if not source.exists() and not source.is_symlink():
        return {"status": "temp_missing", "video_id": video}
    row = videos[video]
    if not regular_matches(source, row["video_sha256"], row["video_bytes"]):
        raise ValueError("uploaded file hash or size mismatch")
    target = destination / (video + ".mp4")
    temporary = destination / (".install-" + uuid.uuid4().hex + "-" + video + ".mp4")
    # This also works when the transfer directory and data volume differ.
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    try:
        if not regular_matches(temporary, row["video_sha256"], row["video_bytes"]):
            raise ValueError("installation copy verification failed")
        os.link(temporary, target)  # FileExistsError preserves any competing target.
    finally:
        temporary.unlink()
    freeze_json(ready / (video + ".json"), ready_record(video, row, manifest_sha))
    source.unlink()  # Only our unique, verified staged upload is removed.
    return {"status": "installed", **ready_record(video, row, manifest_sha)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("status", "install", "finish"))
    for name in ("runtime", "manifest", "manifest-sha256", "stage-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--video-id")
    p.add_argument("--temp-name")
    p.add_argument("--subtitles-dir")
    args = p.parse_args(argv)
    try:
        stage, destination, ready, manifest_sha, videos = setup(args)
        with (stage / "installer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.mode == "install":
                result = install(stage, destination, ready, manifest_sha, videos, args.video_id, args.temp_name)
            else:
                result = inspect(stage, destination, ready, manifest_sha, videos)
                result["status"] = "verified"
                if args.mode == "finish":
                    if result["missing"]:
                        raise ValueError("cannot finish before every video verifies")
                    subtitles = Path(args.subtitles_dir or (Path(args.runtime) / "data/prepared_subtitles"))
                    for video, row in videos.items():
                        if not regular_matches(subtitles / (video + ".json"), row["subtitles_sha256"], row["subtitles_bytes"]):
                            raise ValueError("subtitle does not match media manifest")
                    done = {"complete": True, "media_manifest_sha256": manifest_sha, "video_count": 141,
                            "video_bytes": sum(row["video_bytes"] for row in videos.values())}
                    freeze_json(stage / "producer_done.json", done)
                    result.update(status="complete", **done)
        result.update(hostname=socket.gethostname(), media_manifest_sha256=manifest_sha)
        print(MARKER + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        # No request/credential information is needed for remote diagnostics.
        print(MARKER + json.dumps({"status": "error", "error_type": type(exc).__name__,
                                   "reason": str(exc)[:300], "hostname": socket.gethostname()}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
