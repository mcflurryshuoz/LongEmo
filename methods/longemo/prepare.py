"""Download a pinned episode release and freeze a deterministic pilot subset."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib import request, error, parse

REPO = "mcflurryshuoz/LongEmoBench"


class SafeRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result and parse.urlparse(req.full_url).netloc != parse.urlparse(newurl).netloc:
            result.remove_header("Authorization")
        return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_url(url, token):
    headers = {"Authorization": "Bearer " + token} if token else {}
    return request.build_opener(SafeRedirect()).open(request.Request(url, headers=headers), timeout=90)


def tree(revision, path, token):
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{path}?limit=1000"
    result = []
    while url:
        with read_url(url, token) as response:
            result.extend(json.load(response))
            link = response.headers.get("Link", "")
        next_link = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = next_link.group(1) if next_link else None
    return result


def download(entry, revision, root, token):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe dataset path")
    destination = root / relative
    expected_sha = entry.get("lfs", {}).get("oid")
    if destination.exists() and destination.stat().st_size == entry["size"]:
        sha = digest_file(destination)
        if not expected_sha or sha == expected_sha:
            return {"path": str(relative), "bytes": entry["size"], "sha256": sha}
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{parse.quote(str(relative))}"
    for attempt in range(3):
        try:
            with read_url(url, token) as response, tmp.open("wb") as output:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
            sha = digest_file(tmp)
            if tmp.stat().st_size != entry["size"] or (expected_sha and sha != expected_sha):
                raise ValueError("dataset size/hash mismatch")
            tmp.replace(destination)
            return {"path": str(relative), "bytes": entry["size"], "sha256": sha}
        except (OSError, error.URLError, ValueError):
            if attempt == 2:
                raise RuntimeError(f"download failed: {relative}") from None
            time.sleep(attempt + 1)


def parse_srt(text):
    def seconds(value):
        hours, minutes, rest = value.replace(",", ".").split(":")
        return 3600 * int(hours) + 60 * int(minutes) + float(rest)

    rows = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.splitlines()
        index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if index is None:
            continue
        match = re.search(r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", lines[index])
        if not match:
            raise ValueError("invalid subtitle interval")
        start, end = map(seconds, match.groups())
        text = re.sub(r"<[^>]+>", "", " ".join(lines[index + 1:])).strip()
        if end < start:
            raise ValueError("subtitle end precedes start")
        if text:
            rows.append({"id": f"U{len(rows)+1}", "t": [start, end], "speaker": None, "text": text})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--token-file")
    p.add_argument("--revision")
    p.add_argument("--pilot-videos", type=int, default=3)
    p.add_argument("--seed", default="longemo-zyf-pilot-v1")
    p.add_argument("--all-videos", action="store_true")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.pilot_videos < 1:
        p.error("workers/pilot-videos must be positive")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    token = Path(args.token_file).read_text().strip() if args.token_file else os.environ.get("HF_TOKEN", "")
    if args.revision:
        revision = args.revision
    elif (root / "manifest.json").exists():
        revision = json.loads((root / "manifest.json").read_text())["revision"]
    else:
        with read_url(f"https://huggingface.co/api/datasets/{REPO}", token) as response:
            revision = json.load(response)["sha"]
    entries = tree(revision, "episode", token)
    videos = tree(revision, "episode/videos", token)
    subtitles = tree(revision, "episode/subtitles", token)
    question_entries = [x for x in entries if x["type"] == "file" and x["path"].endswith(".json")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(lambda item: download(item, revision, root, token), question_entries))
    questions = []
    for entry in question_entries:
        value = json.loads((root / entry["path"]).read_text())
        questions.extend(value if isinstance(value, list) else [value])
    ids = [q["question_id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate question IDs")
    video_ids = sorted({q["video_id"] for q in questions}, key=lambda v: hashlib.sha256((args.seed + v).encode()).hexdigest())
    selected = set(video_ids[:args.pilot_videos])
    pilot = [q for q in questions if q["video_id"] in selected]
    atomic_json(root / "questions.json", questions)
    atomic_json(root / "pilot_questions.json", pilot)
    manifest = {"repository": REPO, "revision": revision, "seed": args.seed,
                "question_count": len(questions), "video_count": len(video_ids),
                "task_counts": dict(Counter(q["type"] for q in questions)),
                "pilot_video_ids": sorted(selected), "pilot_question_ids": [q["question_id"] for q in pilot],
                "pilot_task_counts": dict(Counter(q["type"] for q in pilot)),
                "selection": "sha256(seed + video_id); all questions of selected videos; no answer/score-based selection",
                "video_bytes_total": sum(x["size"] for x in videos if x["type"] == "file"), "files": files}
    atomic_json(root / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}), flush=True)
    wanted = set(video_ids) if args.all_videos else selected
    media_entries = [x for x in videos + subtitles if x["type"] == "file" and Path(x["path"]).stem in wanted
                     and (not args.metadata_only or not x["path"].endswith(".mp4"))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda item: download(item, revision, root, token), media_entries):
            manifest["files"].append(result)
            atomic_json(root / "manifest.json", manifest)
            print("Downloaded", result["path"], result["bytes"], flush=True)
    for source in (root / "episode/subtitles").glob("*.srt"):
        atomic_json(root / "prepared_subtitles" / (source.stem + ".json"), parse_srt(source.read_text(encoding="utf-8-sig")))


if __name__ == "__main__":
    main()
