import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from experiments.zyf.blackai36_scale import process, running, finish_adoption, snapshot_stopped, validate_delegation


class Tests(unittest.TestCase):
    def test_delegate_only_untouched_queued_video(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); snap={'state':{'queued_videos':['V'],'videos':{}},'workers':{}}
            selection={'destination_run':'aicodemirror_recharge_parallel_20260921','video_ids':['V']}
            validate_delegation(root,snap,selection)
            p=root/'videos/V/attempts.json';p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaises(AssertionError):validate_delegation(root,snap,selection)

    def test_delegated_video_cannot_be_adopted_worker(self):
        with tempfile.TemporaryDirectory() as d:
            snap={'state':{'queued_videos':['V'],'videos':{}},'workers':{'V':{}}}
            with self.assertRaises(AssertionError):
                validate_delegation(Path(d),snap,{'destination_run':'aicodemirror_recharge_parallel_20260921','video_ids':['V']})

    def test_identity_and_reused_pid(self):
        p = process(os.getpid())
        self.assertTrue(running(p))
        self.assertFalse(running({**p, 'start_ticks': p['start_ticks'] + 1}))

    def test_wait_for_real_worker_before_clearing_inflight(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'videos/V'; folder.mkdir(parents=True)
            p = folder / 'attempts.json'; p.write_text(json.dumps({'attempts': 1, 'in_flight': True}))
            done = root / 'done'
            child = subprocess.Popen([sys.executable, '-c', 'import time,pathlib,sys;time.sleep(.3);pathlib.Path(sys.argv[1]).write_text("done")', str(done)])
            ident = process(child.pid)
            finish_adoption(root, 'V', {'process': ident, 'attempt': 1})
            child.wait(timeout=3)
            self.assertTrue(done.exists())
            self.assertFalse(json.loads(p.read_text())['in_flight'])
            self.assertTrue((root / 'concurrency/adopted/V.json').exists())

    def test_attempt_change_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'videos/V/attempts.json';p.parent.mkdir(parents=True)
            p.write_text('{"attempts":2,"in_flight":true}')
            with self.assertRaises(AssertionError):
                finish_adoption(root,'V',{'process':None,'attempt':1})
            self.assertTrue(json.loads(p.read_text())['in_flight'])

    def test_busy_backend_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'outbox').mkdir();(root/'backend_status.json').write_text('{"videos":{},"active_videos":["V"]}')
            with patch('experiments.zyf.blackai36_scale.children',return_value=[]):
                with self.assertRaises(AssertionError):snapshot_stopped(root,'backend',{'pid':1})

    def test_incomplete_publication_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'outbox').mkdir();(root/'outbox/V.tar.gz').write_bytes(b'partial')
            with self.assertRaises(AssertionError):snapshot_stopped(root,'frontend',{'pid':1})

    def test_coordinator_only_signal_preserves_child(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); pidfile=root/'pid'; done=root/'done'
            script='import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,"-c","import time,pathlib,sys;time.sleep(.6);pathlib.Path(sys.argv[1]).write_text(chr(49))",sys.argv[2]]);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)'
            parent=subprocess.Popen([sys.executable,'-c',script,str(pidfile),str(done)])
            try:
                limit=time.monotonic()+3
                while not pidfile.exists() and time.monotonic()<limit:time.sleep(.02)
                child=process(int(pidfile.read_text()))
                os.kill(parent.pid,signal.SIGSTOP);os.kill(parent.pid,signal.SIGKILL);parent.wait(timeout=3)
                self.assertTrue(running(child))
                limit=time.monotonic()+3
                while not done.exists() and time.monotonic()<limit:time.sleep(.02)
                self.assertTrue(done.exists())
            finally:
                if parent.poll() is None:parent.kill();parent.wait()


if __name__=='__main__':unittest.main()
