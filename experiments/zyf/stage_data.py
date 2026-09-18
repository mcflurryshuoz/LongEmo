"""Stage pinned video files through an already verified AIStudio SCP connection.

No API credentials are needed. Sender never overwrites an upload filename; the
receiver verifies manifest size/SHA256 before atomically admitting each video.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import pty
import select
import signal
import shutil
import time
import uuid


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def pinned(manifest):
    return {Path(x['path']).name:x for x in json.loads(manifest.read_text())['files']
            if x['path'].startswith('episode/videos/') and x['path'].endswith('.mp4')}


def scp_once(command, log_path, timeout):
    """AIStudio's verified SSH workflow uses one empty password submission.

    A PTY is required for the native SSH prompt. Never retry an authentication
    rejection; network failures return a status for bounded transport retries.
    """
    pid,fd=pty.fork()
    if pid==0:
        os.execvp(command[0],command)
    sent=False;tail=b'';deadline=time.monotonic()+timeout
    try:
        with log_path.open('ab') as log:
            while True:
                if time.monotonic()>=deadline:
                    os.killpg(pid,signal.SIGTERM)
                    os.waitpid(pid,0)
                    return 124
                readable,_,_=select.select([fd],[],[],1)
                if readable:
                    try:data=os.read(fd,8192)
                    except OSError:data=b''
                    if not data:
                        _,status=os.waitpid(pid,0)
                        if b'permission denied' in tail.lower():
                            raise RuntimeError('SSH authentication rejected; user interaction required')
                        return os.waitstatus_to_exitcode(status)
                    log.write(data);log.flush();tail=(tail+data)[-2048:]
                    if sent and (b'permission denied' in tail.lower() or b'password:' in tail.lower()):
                        os.killpg(pid,signal.SIGTERM)
                        os.waitpid(pid,0)
                        raise RuntimeError('SSH authentication rejected; user interaction required')
                    if b'password:' in tail.lower() and not sent:
                        os.write(fd,b'\n');sent=True;tail=b''
    finally:
        os.close(fd)


def receiver(args):
    files=pinned(args.manifest)
    args.videos.mkdir(parents=True,exist_ok=True)
    args.uploads.mkdir(parents=True,exist_ok=True)
    accepted={}
    # Existing bootstrap files must pass the same pinned manifest check.
    for name,entry in files.items():
        p=args.videos/name
        if p.is_file() and not p.is_symlink() and p.stat().st_size==entry['bytes'] and digest(p)==entry['sha256']:
            accepted[name]=entry['sha256']
    while True:
        for marker in sorted(args.uploads.glob('*.ready.json')):
            try:
                row=json.loads(marker.read_text())
                name=row['video'];upload=row['upload']
                if name not in files or not re.fullmatch(r'[a-f0-9]{32}\.part',upload):
                    raise ValueError('unexpected upload declaration')
                entry=files[name];source=args.uploads/upload;destination=args.videos/name
                if not source.is_file() or source.is_symlink():raise ValueError('missing or unsafe upload')
                if source.stat().st_size!=entry['bytes'] or digest(source)!=entry['sha256']:
                    raise ValueError('pinned file integrity mismatch')
                if destination.exists():
                    if not destination.is_file() or destination.is_symlink() or digest(destination)!=entry['sha256']:
                        raise ValueError('unexpected existing destination; refusing overwrite')
                    source.unlink()
                else:
                    os.replace(source,destination)
                accepted[name]=entry['sha256'];marker.unlink()
                print(json.dumps({'accepted':name,'count':len(accepted),'expected':len(files)}),flush=True)
            except json.JSONDecodeError:
                continue  # SCP may still be writing a small marker.
            except Exception as e:
                print(json.dumps({'marker':marker.name,'error_type':type(e).__name__,'error':str(e)}),flush=True)
                marker.rename(marker.with_suffix('.failed'))
        status={'verified':len(accepted),'expected':len(files),'verified_bytes':sum(files[n]['bytes'] for n in accepted),
                'complete':len(accepted)==len(files),'updated_unix':time.time(),'accepted':sorted(accepted)}
        temporary=args.status.with_suffix('.tmp');temporary.write_text(json.dumps(status,indent=2));os.replace(temporary,args.status)
        if status['complete']:return
        time.sleep(5)


def sender(args):
    files=pinned(args.manifest)
    state=json.loads(args.status.read_text()) if args.status.exists() else {'sent':{}}
    # Bootstrap V16 is validated independently by the receiver.
    for name in args.already_staged:
        if name not in files:raise ValueError('unknown bootstrap video')
        state['sent'][name]='bootstrap_receiver_verifies'
    scp=['scp','-q','-P',str(args.port),'-o','ConnectTimeout=20','-o','ControlMaster=no','-o','ControlPath=none',
         '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(args.known_hosts)]
    failures=0
    while len(state['sent'])<len(files):
        ready=[name for name,x in files.items() if name not in state['sent'] and (args.videos/name).is_file()
               and (args.videos/name).stat().st_size==x['bytes']]
        for name in sorted(ready,key=lambda n:(files[n]['bytes'],n)):
            tag=uuid.uuid4().hex;source=args.videos/name
            # rsync publishes final names atomically; also verify before upload.
            if digest(source)!=files[name]['sha256']:raise ValueError('local pinned file integrity mismatch')
            target=f'root@127.0.0.1:{args.remote_uploads}/'
            log=args.status.with_suffix('.scp.log')
            rc=scp_once(scp+[str(source),target+tag+'.part'],log,7200)
            if rc:
                failures+=1
                if failures>=3:raise RuntimeError('three SCP failures; tunnel availability must be checked')
                time.sleep(10);continue
            marker=args.status.parent/(tag+'.ready.json');marker.write_text(json.dumps({'video':name,'upload':tag+'.part'}))
            rc=scp_once(scp+[str(marker),target+marker.name],log,90)
            if rc:
                failures+=1
                print(json.dumps({'video':name,'stage':'commit_marker','transport_returncode':rc,
                                  'retry_with_new_upload':failures<3}),flush=True)
                if failures>=3:raise RuntimeError('three SCP failures; tunnel availability must be checked')
                # Never re-upload to a possibly partial remote filename. The
                # next attempt gets new media/marker names; receiver deduplicates.
                time.sleep(10)
                continue
            marker.unlink();failures=0
            state['sent'][name]={'bytes':files[name]['bytes'],'time_unix':time.time()}
            temporary=args.status.with_suffix('.tmp')
            temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,args.status)
            print(json.dumps({'sent':name,'count':len(state['sent']),'expected':len(files)}),flush=True)
        if not ready:time.sleep(10)


def sender_batched(args):
    """Copy media + completion markers in order over one SCP connection per batch."""
    files=pinned(args.manifest)
    state=json.loads(args.status.read_text()) if args.status.exists() else {'sent':{}}
    for name in args.already_staged:
        if name not in files:raise ValueError('unknown bootstrap video')
        state['sent'].setdefault(name,'bootstrap_receiver_verifies')
    command=['scp','-q','-P',str(args.port),'-o','ConnectTimeout=20','-o','ControlMaster=no','-o','ControlPath=none',
             '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(args.known_hosts)]
    failures=0
    while len(state['sent'])<len(files):
        ready=[n for n,e in files.items() if n not in state['sent'] and (args.videos/n).is_file()
               and (args.videos/n).stat().st_size==e['bytes']]
        ready=sorted(ready,key=lambda n:(files[n]['bytes'],n))[:args.batch_size]
        if not ready:time.sleep(10);continue
        folder=args.status.parent/('upload_batch_'+uuid.uuid4().hex);folder.mkdir(mode=0o700)
        paths=[]
        try:
            for name in ready:
                source=args.videos/name
                if digest(source)!=files[name]['sha256']:raise ValueError('local pinned file integrity mismatch')
                tag=uuid.uuid4().hex;part=folder/(tag+'.part');marker=folder/(tag+'.ready.json')
                os.link(source,part)
                marker.write_text(json.dumps({'video':name,'upload':part.name}))
                paths.extend([str(part),str(marker)])
            batch={'status':'uploading','videos':ready,'bytes':sum(files[n]['bytes'] for n in ready),
                   'started_unix':time.time(),'attempt':failures+1}
            args.status.with_suffix('.batch.json').write_text(json.dumps(batch,indent=2))
            print(json.dumps(batch),flush=True)
            rc=scp_once(command+paths+[f'root@127.0.0.1:{args.remote_uploads}/'],args.status.with_suffix('.scp.log'),12*3600)
            if rc:
                failures+=1
                batch.update(status='transport_failed',returncode=rc)
                args.status.with_suffix('.batch.json').write_text(json.dumps(batch,indent=2))
                if failures>=3:raise RuntimeError('three batch SCP failures; check tunnel and reconcile receiver before retry')
                time.sleep(10);continue
            for name in ready:state['sent'][name]={'bytes':files[name]['bytes'],'time_unix':time.time()}
            temporary=args.status.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,args.status)
            failures=0;batch.update(status='uploaded',finished_unix=time.time())
            args.status.with_suffix('.batch.json').write_text(json.dumps(batch,indent=2));print(json.dumps(batch),flush=True)
        finally:
            shutil.rmtree(folder)  # Task-owned hard links and markers, never original media.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('send','receive'))
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--status',type=Path,required=True);p.add_argument('--uploads',type=Path)
    p.add_argument('--port',type=int);p.add_argument('--known-hosts',type=Path)
    p.add_argument('--remote-uploads');p.add_argument('--already-staged',nargs='*',default=[])
    p.add_argument('--batch-size',type=int,default=1,choices=range(1,142))
    args=p.parse_args()
    if args.mode=='receive':receiver(args)
    elif args.batch_size>1:sender_batched(args)
    else:sender(args)

if __name__=='__main__':main()
