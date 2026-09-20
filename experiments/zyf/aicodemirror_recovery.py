"""Reuse bounded transient recovery and immutable replacement for AICodeMirror."""
import argparse
import fcntl
import os
from pathlib import Path
import time

from experiments.zyf.aicodemirror_continue import NAME
from experiments.zyf.blackai_continuation import atomic, read
from experiments.zyf.recover_blackai import frontend, backend
from methods.longemo.common import code_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['frontend', 'backend'])
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    args = parser.parse_args()
    run = args.runtime / 'runs' / NAME
    experiment = read(run / 'experiment_manifest.json')
    config = experiment['configuration']
    assert config['method_hash'] == code_hash()
    directory = run / 'recovery'
    directory.mkdir(exist_ok=True)
    with (directory / (args.stage + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(directory / (args.stage + '_process.json'), {
            'pid': os.getpid(), 'started_unix': time.time(), 'max_extra_attempts': 3})
        if args.stage == 'frontend':
            with (run / 'frontend.lock').open('a') as primary_lock:
                fcntl.flock(primary_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert read(run / 'frontend_status.json')['status'] == 'finished'
                os.environ['LONGEMO_FFMPEG_COMPAT'] = '1'
                os.environ.pop('LONGEMO_ALLOW_AUDIO_CACHE_REUSE', None)
                os.environ['PATH'] = str(args.runtime / 'tmp') + os.pathsep + os.environ.get('PATH', '')
                frontend(args.runtime, run, 3, worker_module='experiments.zyf.aicodemirror_worker')
        else:
            backend(args.runtime, run, args.credential_file)


if __name__ == '__main__':
    main()
