"""Continue only the five validated HTTP428 probes, reusing their exact caches.

The original full/pilot memories and 262 first scores are immutable. Four audio
seeds and one visual seed are imported offline with original validators. Each
video then receives one separately recorded build task, with no outer retry.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_probe_seed as seed
from experiments.zyf import noevent_retry_probe as probe
from experiments.zyf import noevent_http428_diagnosis as diagnosis

VIDEOS={'G2_V000007','G2_V000062','G2_V000070','G2_V000078','G2_V000080'}


def successful_cases(spec_path,batch):
    spec=base.read(spec_path);batch=Path(batch);state=base.read(base.regular(batch/'finished.json'))
    if state['status']!='finished' or set(state['results'])!=diagnosis.CASES:
        raise ValueError('diagnostic batch is not fully audited')
    found={v for v,r in state['results'].items() if r.get('classification')=='validated_payload_not_yet_imported'}
    if found!=VIDEOS:raise ValueError('fixed successful diagnostic selection changed')
    result={}
    for vid in sorted(found):
        _,case=diagnosis.load_case(spec_path,vid)
        folder=batch/'tasks'/vid/'result';receipt=base.read(base.regular(folder/'diagnosis.json'))
        if (receipt.get('status')!='validated' or receipt.get('classification')!='validated_payload_not_yet_imported'
                or receipt['request_hash']!=case['failed']['request_hash'] or receipt['parent_files_unchanged'] is not True):
            raise ValueError('selected window did not validate against the original request')
        result[vid]={**case,'probe_dir':str(folder)}
    qids=[q for c in result.values() for q in c['question_ids']]
    if len(qids)!=len(set(qids)):raise ValueError('duplicate selected questions')
    diagnosis.verify_files(spec['protected_first_score_files'])
    protected_q={Path(p).stem for p in spec['protected_first_score_files'] if Path(p).parent.name=='noevent' and Path(p).parent.parent.name=='accepted'}
    if len(protected_q)!=262 or set(qids)&protected_q:raise ValueError('selection overlaps first scores')
    return spec,result


@contextmanager
def reconstructed_source(prepared,parent,repo,video):
    """Route only this offline import to a byte-identical original pilot source."""
    original=probe.prepare
    def exact(actual_parent,actual_video,actual_repo):
        if (Path(actual_parent)!=Path(parent) or Path(actual_repo)!=Path(repo) or actual_video!=video):
            raise ValueError('offline seed attempted an unexpected target')
        return prepared
    probe.prepare=exact
    try:yield
    finally:probe.prepare=original


def import_one(run,spec_path,batch,video):
    spec,cases=successful_cases(spec_path,batch);case=cases[video]
    run=Path(run);cfg=base.read(base.regular(run/'configuration.json'))
    with base.stopped_parent(Path(case['source_run'])):
        prepared=probe.prepare(case['source_run'],video,case['core_repo'])
        if prepared['failed']!=case['failed']:raise ValueError('source failure changed')
        diagnosis.verify_files({**case['source_files'],**case['source_root_hashes'],**spec['protected_first_score_files']})
        # apply_seed independently checks checkpoint content, HTTP body, input
        # fingerprint, model, media, original validator and empty output paths.
        with reconstructed_source(prepared,cfg['parent_run'],cfg['core_repo'],video):
            receipt=seed.apply_seed(cfg['parent_run'],run,video,case['probe_dir'],cfg['core_repo'])
    if receipt['request_hash']!=case['failed']['request_hash']:raise ValueError('seed differs from successful probe')
    return receipt


def implementation():
    root=Path(__file__).absolute().parent
    return {str(p):base.sha(base.regular(p)) for p in root.glob('*.py')}


def prepare(args):
    run=Path(args.run).absolute()
    if run.exists():raise ValueError('new fixed run already exists; audit without preparing again')
    spec,cases=successful_cases(args.specs,args.batch)
    parent=Path(args.parent_run);core=Path(args.core_repo)
    if parent.name!='three_level_full558_gemini38_20260922':raise ValueError('full558 parent required')
    for other in parent.parent.glob('noevent*'):
        cfg=other/'configuration.json'
        if cfg.exists() and set(base.read(cfg).get('checkpoints',{}))&VIDEOS:
            raise ValueError('another continuation already owns a selected video')
    protected=dict(spec['protected_first_score_files'])
    for case in cases.values():protected.update(case['source_files']);protected.update(case['source_root_hashes'])
    protected.update(implementation());protected[str(Path(args.specs).absolute())]=diagnosis.SPECS_SHA
    protected[str(Path(args.batch)/'finished.json')]=base.sha(Path(args.batch)/'finished.json')
    claims={}
    for vid,case in cases.items():
        p=parent.parents[1]/'recovery_claims/noevent_http428_seed'/f'{vid}.json'
        value={'video_id':vid,'source_run':case['source_run'],'owner_run':str(run),'question_ids':case['question_ids'],
               'request_hash':case['failed']['request_hash'],'driver_sha256':base.sha(__file__)}
        seed.exclusive_json(p,value);claims[str(p)]=base.sha(p)
    selected=sorted(q for c in cases.values() for q in c['question_ids'])
    run.mkdir(mode=0o700)
    selection=run/'selection.json';seed.exclusive_json(selection,{'schema_version':1,'condition':'noevent','question_ids':selected,
        'reason':'Fixed five successful one-request HTTP428 diagnostics; reuse seeds, original budgets, first scores preserved.'})
    old=base.read(parent/'configuration.json')
    original=SimpleNamespace(run=str(run),parent_run=str(parent),core_repo=str(core),selection=str(selection),credential_file=old['credential_file'],
        visual_max_tokens=8192,protected_count=195,workers=5,question_workers=2,python=args.python)
    _,cfg=base.prepare(original)
    for vid,case in cases.items():
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONPATH']=str(Path(__file__).absolute().parents[2])+':'+case['core_repo']
        env['PATH']=str(parent.parents[1]/'bin')+':'+env['PATH']
        command=[args.python,'-B','-m','experiments.zyf.noevent_http428_continuation','seed','--run',str(run),'--specs',args.specs,'--batch',args.batch,'--video-id',vid]
        with (run/f'seed_{vid}.private.log').open('xb') as log:
            result=subprocess.run(command,cwd=case['core_repo'],env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise ValueError(f'offline seed import needs audit: {vid}')
    receipts={vid:base.sha(run/'probe_seeds'/vid/'receipt.json') for vid in VIDEOS}
    value={'schema_version':1,'configuration_sha256':base.sha(run/'configuration.json'),'cases':cases,'selected_question_ids':selected,
        'protected_files':protected,'claims':claims,'seed_receipts':receipts,'implementation':implementation(),'first_scores':262,
        'execution_conditions':['noevent'],'workers':5,'question_workers':2,'no_api_during_prepare':True}
    seed.exclusive_json(run/'http428_execution.json',value)
    seed.exclusive_json(run/'http428_execution.sha256.json',{'sha256':base.sha(run/'http428_execution.json')})
    verify_extra(run,cfg)
    return {'run':str(run),'videos':5,'questions':len(selected),'seed_stages':{v:base.read(run/'probe_seeds'/v/'receipt.json')['stage'] for v in VIDEOS}}


def verify_extra(run,cfg):
    path=run/'http428_execution.json';value=base.read(base.regular(path))
    if base.sha(path)!=base.read(run/'http428_execution.sha256.json')['sha256'] or value['configuration_sha256']!=base.sha(run/'configuration.json'):
        raise ValueError('frozen seed execution changed')
    if (set(cfg['checkpoints'])!=VIDEOS or cfg['selection']['question_ids']!=value['selected_question_ids']
            or cfg['workers']!=5 or cfg['question_workers']!=2 or cfg['profile']['visual']['max_tokens']!=8192):
        raise ValueError('fixed video selection, concurrency or model budget changed')
    diagnosis.verify_files({**value['protected_files'],**value['claims'],**value['implementation']})
    for vid,expected in value['seed_receipts'].items():
        if base.sha(run/'probe_seeds'/vid/'receipt.json')!=expected:raise ValueError('seed receipt changed')
        receipt=seed.verify_seed_receipt(run,vid,cfg['checkpoints'][vid],allow_memory_growth=True)
        if receipt['request_hash']!=value['cases'][vid]['failed']['request_hash']:raise ValueError('seed request changed')
    return value


@contextmanager
def guards(run,cfg):
    original_verify,original_provenance=base.verify,base.window_provenance
    def verify(target,actual):
        if Path(target)!=run:raise ValueError('unexpected run')
        original_verify(target,actual);verify_extra(run,actual)
    def provenance(target,actual,video):
        result=original_provenance(target,actual,video)
        receipt=seed.verify_seed_receipt(run,video,actual['checkpoints'][video],allow_memory_growth=True)
        base.freeze(run/'noevent/http428_seed_provenance'/f'{video}.json',{'receipt':receipt,'completed_memory_sha256':result['memory_sha256'],
            'interpretation':'The seed payload was reused offline; its audio or visual stage was not requested again.'})
        return result
    base.verify,base.window_provenance=verify,provenance
    try:yield
    finally:base.verify,base.window_provenance=original_verify,original_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','seed','run','report']);p.add_argument('--run',required=True)
    for n in ('specs','batch','parent-run','core-repo','video-id'):p.add_argument('--'+n)
    p.add_argument('--python',default=sys.executable);a=p.parse_args();run=Path(a.run).absolute()
    if a.stage=='prepare':result=prepare(a)
    elif a.stage=='seed':result=import_one(run,a.specs,a.batch,a.video_id)
    else:
        cfg=base.read(run/'configuration.json')
        with guards(run,cfg):
            base.verify(run,cfg)
            if a.stage=='run':
                seed.exclusive_json(run/'dispatch_claim.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]})
                result=base.execute(run,cfg)
            else:result=base.report(run)
    print(json.dumps({'status':'ok','stage':a.stage,'run':str(run)}))

if __name__=='__main__':main()
