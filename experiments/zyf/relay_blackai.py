"""Relay immutable E11 memory bundles over an already verified AIStudio SSH port."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from experiments.zyf.stage_data import scp_once, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--source', required=True, help='Authorized g450:/absolute/outbox/')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--known-hosts', type=Path, required=True)
    p.add_argument('--destination', required=True)
    p.add_argument('--expected', type=int, required=True)
    p.add_argument('--source-ssh', default='ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=none -o ServerAliveInterval=15 -o ServerAliveCountMax=3',
                   help='SSH command used only for the authorized source rsync; target SCP stays unchanged')
    p.add_argument('--producer-done', action='store_true', help='Use producer_done.json for a dynamically sized recovery pass')
    a = p.parse_args()
    a.local.mkdir(parents=True, exist_ok=True)
    path = a.local/'relay_status.json'
    status = json.loads(path.read_text()) if path.exists() else {'sent': {}}
    inbox = a.local/'outbox'; inbox.mkdir(exist_ok=True)
    failures = 0
    while a.producer_done or len(status['sent']) < a.expected:
        r = subprocess.run(['rsync', '-a', '--exclude=*.tmp', '-e',
            a.source_ssh,
            a.source, str(inbox)+'/'], timeout=180, stdout=subprocess.DEVNULL)
        if r.returncode:
            failures += 1
            if failures >= 3: raise RuntimeError('three source transfer failures')
            time.sleep(20); continue
        for marker_path in sorted(inbox.glob('*.ready.json')):
            marker = json.loads(marker_path.read_text()); vid = marker['video_id']
            if vid in status['sent']: continue
            source = inbox/marker['archive']
            if not source.exists() or source.stat().st_size != marker['bytes']: continue
            if digest(source) != marker['sha256']: raise ValueError('local bundle integrity mismatch')
            tag = uuid.uuid4().hex
            upload = a.local/(tag+'.tar.gz'); upload.write_bytes(source.read_bytes())
            ready = a.local/(tag+'.ready.json')
            ready.write_text(json.dumps({**marker, 'archive': upload.name})+'\n')
            command = ['scp', '-q', '-P', str(a.port), '-o', 'ConnectTimeout=20', '-o', 'ControlMaster=no',
                '-o', 'ControlPath=none', '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile='+str(a.known_hosts),
                str(upload), str(ready), '127.0.0.1:'+a.destination+'/']
            rc = scp_once(command, a.local/'scp.log', timeout=300)
            if rc:
                failures += 1
                if failures >= 3: raise RuntimeError('three target transfer failures; verify tunnel')
                break
            failures = 0
            status['sent'][vid] = {'sha256': marker['sha256'], 'time_unix': time.time()}
            status.update(updated_unix=time.time(), expected=a.expected, port=a.port)
            tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(status, indent=2)+'\n'); tmp.replace(path)
            upload.unlink(); ready.unlink()
            print(json.dumps({'sent': vid, 'count': len(status['sent']), 'expected': a.expected}), flush=True)
        done = inbox/'producer_done.json'
        if a.producer_done and done.exists() and set(json.loads(done.read_text())['video_ids']) <= set(status['sent']):
            unique = a.local/('producer_done_'+uuid.uuid4().hex+'.json')
            unique.write_bytes(done.read_bytes())
            command = ['scp', '-q', '-P', str(a.port), '-o', 'ConnectTimeout=20', '-o', 'ControlMaster=no',
                       '-o', 'ControlPath=none', '-o', 'StrictHostKeyChecking=yes',
                       '-o', 'UserKnownHostsFile='+str(a.known_hosts), str(unique), '127.0.0.1:'+a.destination+'/']
            if scp_once(command, a.local/'scp.log', timeout=90) == 0:
                unique.unlink()
                status.update(status='finished', producer_done_uploaded=True, updated_unix=time.time())
                tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(status, indent=2)+'\n'); tmp.replace(path)
                return
        time.sleep(15)


if __name__ == '__main__': main()
