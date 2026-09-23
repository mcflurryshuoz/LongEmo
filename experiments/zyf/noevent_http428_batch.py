"""One fixed diagnostic batch, four cases at a time, no evaluation retry loop."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.zyf import noevent_http428_diagnosis as diag
from experiments.zyf import noevent_continuation as base


def write(path,value):
    base.write(path,value)


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]


def stop_service(row):
    return row.get('http_status') in (401,402,403,429) or row.get('status') in ('transport_error','invalid_response_json')


def batch(mode,spec_path,output):
    spec=base.read(spec_path)
    for case in spec['cases']:
        diag.load_case(spec_path,case['video_id'])
    output=Path(output).absolute()
    if output.exists():
        raise ValueError('batch directory already exists; inspect it, never auto restart')
    output.mkdir(parents=True,mode=0o700)
    with (output/'lock').open('x') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        code_root=Path(__file__).absolute().parents[2]
        hashes={str(p):base.sha(p) for p in sorted((code_root/'experiments/zyf').glob('*.py'))}
        write(output/'process.json',{'pid':os.getpid(),'start_ticks':ticks(os.getpid()),'argv':sys.argv,'started_at':datetime.now().astimezone().isoformat(),'code_hashes':hashes})
        state={'stage':mode,'status':'running','maximum_parallel_cases':4,'maximum_http_requests_per_case':1,'results':{},'question_count':sum(len(c['question_ids']) for c in spec['cases']),'spec_sha256':base.sha(spec_path)}
        write(output/'status.json',state)
        def one(case):
            vid=case['video_id']; task=output/'tasks'/vid;task.mkdir(parents=True,mode=0o700)
            env=os.environ.copy();env['PYTHONPATH']=str(code_root)+':'+case['core_repo'];env['PYTHONDONTWRITEBYTECODE']='1'
            args=[sys.executable,'-B','-u','-m','experiments.zyf.noevent_http428_diagnosis',mode,'--specs',str(spec_path),'--video-id',vid,'--output-dir',str(task/'result')]
            diag.verify_files(hashes)
            with (task/'private.log').open('xb') as log:
                p=subprocess.Popen(args,cwd=case['core_repo'],env=env,stdout=log,stderr=subprocess.STDOUT)
                write(task/'child.json',{'pid':p.pid,'start_ticks':ticks(p.pid),'argv':args})
                rc=p.wait()
            receipt=task/'result'/('diagnosis.json' if mode=='probe' else 'preflight.json')
            if rc==0 and receipt.exists():
                raw=base.read(receipt)
                row={k:raw.get(k) for k in ('status','classification','http_status','window_id','wire_sha256','request_hash')}
            else:
                row={'status':'local_preflight_or_audit_failed','exit_code':rc}
            write(task/'outcome.json',row)
            return vid,row
        with ThreadPoolExecutor(max_workers=4) as pool:
            cases=spec['cases']
            for start in range(0,len(cases),4):
                for vid,row in pool.map(one,cases[start:start+4]):
                    state['results'][vid]=row
                    write(output/'status.json',state)
                if mode=='probe' and any(stop_service(r) or r.get('status')=='local_preflight_or_audit_failed' for r in state['results'].values()):
                    state['status']='stopped_for_service_or_audit';break
        if state['status']=='running':state['status']='finished'
        state['ended_at']=datetime.now().astimezone().isoformat()
        write(output/'status.json',state);write(output/'finished.json',state)
        return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','probe']);p.add_argument('--specs',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();print(json.dumps(batch(a.mode,Path(a.specs),a.output_dir)))
