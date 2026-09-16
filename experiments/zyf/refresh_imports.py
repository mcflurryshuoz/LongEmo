"""Refresh pilot checkpoint copies only before any work in the full evaluation."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time

from evaluation.io_utils import write_json
from methods.longemo.common import file_hash
from experiments.zyf.report_results import ledger


def refresh(run):
    with (run/'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        imports_path = run/'imported_checkpoints.json'
        imports = json.loads(imports_path.read_text())
        memory = run/'memory'
        if (run/'videos').exists() and any((run/'videos').iterdir()):
            raise ValueError('full-run video stages already exist; do not replace their memories')
        existing = {p.name for p in memory.iterdir() if p.is_dir()}
        if existing != set(imports):
            raise ValueError('full-run memories are not solely the original imported checkpoints')
        changes = []
        # Validate every source/destination before the first mutation.
        for vid, record in imports.items():
            dst, src = memory/vid, Path(record['source'])
            if file_hash(dst/'memory.json') != record['memory_sha256']:
                raise ValueError('a destination checkpoint changed after import; refusing replacement')
            if file_hash(dst/'manifest.json') != file_hash(src/'manifest.json'):
                raise ValueError('source perception configuration changed')
            if file_hash(src/'memory.json') == record['memory_sha256']:
                continue
            before = json.loads((dst/'memory.json').read_text())
            after = json.loads((src/'memory.json').read_text())
            if not set(before['completed_windows']) <= set(after['completed_windows']):
                raise ValueError('source lost completed windows')
            for win in before['completed_windows']:
                for relative in (Path('windows')/(win+'.json'), Path('audio')/(win+'.json')):
                    if file_hash(dst/relative) != file_hash(src/relative):
                        raise ValueError('source rewrote an already imported window')
            changes.append((vid, src, dst))
        if not changes:
            return []
        backup = run/'import_history'/str(time.time_ns())
        backup.mkdir(parents=True)
        shutil.copy2(imports_path, backup/imports_path.name)
        cost_path = run/'imported_costs.json'
        if cost_path.exists():
            shutil.copy2(cost_path, backup/cost_path.name)
        for vid, src, dst in changes:
            temp = memory/(vid+'.refreshing')
            if temp.exists():
                raise ValueError('unfinished refresh directory exists; inspect before retrying')
            shutil.copytree(src, temp)
            dst.rename(backup/vid)
            temp.rename(dst)
            imports[vid].update(memory_sha256=file_hash(dst/'memory.json'),
                                manifest_sha256=file_hash(dst/'manifest.json'), refreshed_unix=time.time())
            write_json(imports_path, imports)
        write_json(cost_path, {
            'perception': ledger(list(memory.glob('*/calls.jsonl'))),
            'audio': ledger(list(memory.glob('*/audio/calls.jsonl'))),
            'note': 'Updated imported-only costs before full-run work; original snapshots retained in import_history.'})
        return [vid for vid, _, _ in changes]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps({'refreshed_videos':refresh(args.run)}))
