"""Change scheduler capacity while adopting, rather than cancelling, media workers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from experiments.zyf import blackai36_continuation as base
from experiments.zyf.blackai_continuation import atomic, read
from methods.longemo.common import file_hash


def process(pid):
    root = Path('/proc') / str(pid)
    try:
        raw = (root / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        command = (root / 'cmdline').read_bytes().decode().split('\0')
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': int(fields[19]), 'exit_status': int(fields[49]),
                'command': [x for x in command if x]}
    except FileNotFoundError:
        return None


def running(identity):
    current = process(identity['pid'])
    return bool(current and current['start_ticks'] == identity['start_ticks']
                and current['state'] not in ('Z', 'X'))


def children(pid):
    found = set()
    for task in (Path('/proc') / str(pid) / 'task').iterdir():
        found.update(int(x) for x in (task / 'children').read_text().split())
    return sorted(found)


def wait_stopped(identity):
    for _ in range(100):
        current = process(identity['pid'])
        assert current and current['start_ticks'] == identity['start_ticks']
        if current['state'] == 'T':
            return
        time.sleep(.05)
    raise RuntimeError('scheduler did not stop')


def snapshot_stopped(out, stage, parent):
    # Refuse a handoff in the middle of publishing a parent-owned atomic file.
    incomplete = list(out.glob('*.json.tmp')) + list((out / 'outbox').glob('*.tmp'))
    for n in ['attempts.json.tmp', 'worker_process.json.tmp', 'build_command.json.tmp']:
        incomplete.extend((out / 'videos').glob('*/' + n))
    for p in (out / 'outbox').glob('*.tar.gz'):
        if not p.with_name(p.name[:-7] + '.ready.json').exists():
            incomplete.append(p)
    assert not incomplete, 'scheduler was publishing; resume and try after completion'
    state = read(out / (stage + '_status.json'))
    child_ids = children(parent['pid'])
    if stage == 'backend':
        assert not child_ids and not state.get('active_videos'), 'backend must drain before switching'
        return {'state': state, 'workers': {}}
    active = {}
    for pid in child_ids:
        item = process(pid)
        assert item and item['ppid'] == parent['pid']
        if item['state'] in ('Z', 'X'):
            matches = [p for p in (out / 'videos').glob('*/worker_process.json') if read(p)['pid'] == pid]
            assert len(matches) == 1
            vid = matches[0].parent.name
        else:
            cmd = item['command']
            assert 'experiments.zyf.blackai36_worker' in cmd and '--data-path' in cmd
            question_file = Path(cmd[cmd.index('--data-path') + 1])
            assert question_file.parent.parent == out / 'videos' and question_file.name == 'questions.json'
            vid = question_file.parent.name
        assert vid not in active and vid not in state['videos']
        active[vid] = {'process': item, 'attempt': read(out / 'videos' / vid / 'attempts.json')['attempts']}
    for p in (out / 'videos').glob('*/attempts.json'):
        a = read(p)
        if a.get('in_flight') and p.parent.name not in active:
            identity = read(p.parent / 'worker_process.json')
            assert process(identity['pid']) is None, 'unaccounted live worker'
            active[p.parent.name] = {'process': None, 'attempt': a['attempts'], 'exit_already_observed': True}
    return {'state': state, 'workers': active}


def validate_delegation(out, snapshot, selection):
    if not selection:
        return
    assert selection['destination_run'] == 'aicodemirror_recharge_parallel_20260921'
    vids = selection['video_ids']
    assert len(vids) == len(set(vids))
    for vid in vids:
        assert vid in snapshot['state']['queued_videos']
        assert vid not in snapshot['workers'] and vid not in snapshot['state']['videos']
        assert not (out / 'videos' / vid / 'attempts.json').exists(), 'already dispatched video'


def handoff(out, stage, pid, audit, delegation=None):
    parent = process(pid)
    assert parent and pid == read(out / (stage + '_process.json'))['pid']
    cmd = parent['command']
    assert 'experiments.zyf.blackai36_continuation' in cmd and stage in cmd
    assert cmd[cmd.index('--runtime') + 1] == str(out.parent.parent)
    assert parent['state'] not in ('T', 'Z', 'X'), 'unexpected old scheduler state'
    os.kill(pid, signal.SIGSTOP)  # Only the scheduler; media children keep running.
    terminated = False
    try:
        wait_stopped(parent)
        snap = snapshot_stopped(out, stage, parent)
        validate_delegation(out, snap, delegation)
        record = {'old_scheduler': parent, 'stage': stage, 'snapshot': snap,
                  'created_unix': time.time(), 'operation': 'adopt_live_children_no_request_cancellation',
                  'delegation': delegation}
        assert not audit.exists()
        atomic(audit, record)
        # The stopped coordinator cannot dispatch another worker. Its already
        # verified children survive and are monitored by identity below.
        current = process(pid)
        assert current['start_ticks'] == parent['start_ticks'] and current['state'] == 'T'
        os.kill(pid, signal.SIGKILL)
        terminated = True
        for _ in range(100):
            if not running(parent):
                return record
            time.sleep(.05)
        raise RuntimeError('old scheduler did not exit')
    finally:
        if not terminated and running(parent):
            os.kill(pid, signal.SIGCONT)


def finish_adoption(out, vid, adoption):
    identity = adoption['process']
    while identity and running(identity):
        time.sleep(2)
    p = out / 'videos' / vid / 'attempts.json'
    attempt = read(p)
    assert attempt['attempts'] == adoption['attempt'], 'attempt changed during adoption'
    receipt = out / 'concurrency' / 'adopted' / (vid + '.json')
    if not receipt.exists():
        atomic(receipt, {'worker': adoption, 'exit_observed_unix': time.time(),
                        'policy': 'No media request resubmitted before original process exit.'})
    if attempt.get('in_flight'):
        attempt.update(in_flight=False, finished_unix=time.time(), returncode=None,
                       adopted_exit_receipt=str(receipt))
        atomic(p, attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['frontend', 'backend'])
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--credential-file', type=Path, required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--takeover-pid', type=int)
    p.add_argument('--handoff-file', type=Path, required=True)
    p.add_argument('--delegate-selection', type=Path)
    a = p.parse_args()
    assert 1 <= a.workers <= 16
    out = a.runtime / 'runs' / base.NAME
    with (out / (a.stage + '_capacity.lock')).open('a') as capacity_lock:
        fcntl.flock(capacity_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Validate every frozen method/profile hash before touching a live process.
        cfg = read(out / 'experiment_manifest.json')['configuration']
        assert cfg['method_hash'] == base.code_hash()
        for name, sha in cfg['worker_hashes'].items():
            assert file_hash(Path(base.__file__).parent / name) == sha
        if a.stage == 'backend':
            base.score_guard(a.runtime, read(out / 'questions.json'))
        if a.takeover_pid:
            delegation = read(a.delegate_selection) if a.delegate_selection else None
            assert not delegation or a.stage == 'frontend'
            record = handoff(out, a.stage, a.takeover_pid, a.handoff_file, delegation)
        else:
            record = read(a.handoff_file)
            assert record['stage'] == a.stage and not running(record['old_scheduler'])
        with (out / (a.stage + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            out, questions = base.setup(a)
            atomic(out / (a.stage + '_process.json'), {'pid': os.getpid(), 'started_unix': time.time(),
                   'workers': a.workers, 'handoff_file': str(a.handoff_file),
                   'launcher_sha256': file_hash(Path(__file__))})
            atomic(out / 'concurrency' / (a.stage + '_capacity.json'),
                   {'workers': a.workers, 'updated_unix': time.time(), 'method_and_profile_unchanged': True})
            if a.stage == 'frontend':
                delegation = record.get('delegation')
                if delegation:
                    state_path = out / 'frontend_status.json'
                    state = read(state_path)
                    for vid in delegation['video_ids']:
                        outcome = {'video_id': vid, 'status': 'delegated_provider',
                                   'destination_run': delegation['destination_run']}
                        assert vid not in state['videos'] or state['videos'][vid] == outcome
                        base.export_video(out, vid, outcome)
                        state['videos'][vid] = outcome
                    atomic(state_path, state)
                original = base.run_video
                def adopted(args, folder, vid):
                    item = record['snapshot']['workers'].get(vid)
                    if item:
                        finish_adoption(folder, vid, item)
                    return original(args, folder, vid)
                base.run_video = adopted
                base.frontend(a, out, questions)
            else:
                base.backend(a, out, questions)
                base.score_guard(a.runtime, questions)


if __name__ == '__main__':
    main()
