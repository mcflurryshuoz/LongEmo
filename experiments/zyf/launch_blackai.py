"""Prepare/launch BlackAI native Gemini perception, audio and embedding with an Azure GPT-6 judge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,default=Path('/mnt/data1/zyf/LongEmo-runtime'))
    p.add_argument('--video-workers',type=int,default=16)
    p.add_argument('--defer-answers',action='store_true',help='Keep building memories until native embedding access is enabled')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    runtime=args.runtime.resolve();repo=Path(__file__).resolve().parents[2]
    run=runtime/'runs/blackai_gemini38_gpt6_full_v2';run.mkdir(parents=True,exist_ok=True)
    launch=run/'launch.json'
    if launch.exists():
        previous=json.loads(launch.read_text())
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:pass
        else:raise SystemExit('existing supervisor is still live; inspect before resuming')
    env=os.environ.copy()
    env.update(AZURE_CONFIG_DIR='/home/azureuser/.azure',AZURE_OPENAI_API_VERSION='2025-01-01-preview',
        AZURE_RATE_LIMIT_DB=str(runtime/'limits/azure_gpt6.sqlite'),
        AZURE_RATE_LIMIT_CONFIG=str(runtime/'limits/azure_gpt6.json'),
        AZURE_TRANSPORT_LEDGER_DIR=str(run/'azure_transport'),
        PATH=str(runtime/'venv/bin')+':'+env['PATH'],TMPDIR=str(runtime/'tmp'),PYTHONPATH=str(repo),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    command=[str(runtime/'embedding-venv/bin/python'),'-u','-m','experiments.zyf.azure_benchmark',
        '--data-root',str(runtime/'data'),'--output-dir',str(run),
        '--credential-file',str(runtime/'api_config.json'),
        '--embedding-cache-dir',str(runtime/'cache/blackai_gemini38_gpt6_full_v2'),
        '--perception-model','gemini-3.8-flash','--video-workers',str(args.video_workers),
        '--startup-videos','1','--workers','2','--video-attempts','2']
    if args.defer_answers:command.append('--defer-answers')
    if not args.execute:return subprocess.run(command,cwd=repo,env=env).returncode
    command.append('--execute')
    with (run/'orchestrator.log').open('a') as log:
        process=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={'pid':process.pid,'command':command,'cwd':str(repo),'started_unix':time.time(),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'defer_answers':args.defer_answers,
        'operational_environment':{k:env[k] for k in ('AZURE_CONFIG_DIR','AZURE_OPENAI_API_VERSION',
            'AZURE_RATE_LIMIT_DB','AZURE_RATE_LIMIT_CONFIG','AZURE_TRANSPORT_LEDGER_DIR','TMPDIR',
            'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}}
    if launch.exists():
        history=run/'launch_history';history.mkdir(exist_ok=True)
        launch.rename(history/f'{time.time_ns()}.json')
    launch.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record));return 0


if __name__=='__main__':raise SystemExit(main())
