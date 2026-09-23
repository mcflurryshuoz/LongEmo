"""Fixed unclassified HTTP428 cases: at most one unchanged-input HTTP each.

This is a diagnostic only. No builder, planner, answerer or judge is started.
Previously confirmed policy/DLP denials are outside the fixed specification.
Old attempts and 262 first scores remain immutable. Successful audio/visual
payloads are saved for a separate validated cache import, never re-requested.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.zyf import noevent_continuation as base
from experiments.zyf import noevent_retry_probe as probe

SPECS_SHA='fffb325f37f8e70adeef5014c0ebc3f61cdabbfa2a8cfef0d76ee39df342e3c1'
CASES={f'G2_V{n:06d}' for n in (5,7,8,13,20,21,38,42,50,57,58,62,64,68,70,78,80,94,97,136,137,139,140)}


def load_case(path,video):
    if base.sha(base.regular(path))!=SPECS_SHA:
        raise ValueError('fixed diagnostic specification changed')
    spec=base.read(path)
    if (len(spec['cases'])!=23 or {x['video_id'] for x in spec['cases']}!=CASES or video not in CASES or
            spec.get('protected_first_score_count')!=262 or spec.get('maximum_http_requests_per_case')!=1):
        raise ValueError('fixed diagnostic scope changed')
    case=next(x for x in spec['cases'] if x['video_id']==video)
    row=case['failed']
    probe.check_eligible(row)
    if row.get('http_status')!=428 or row.get('service_error_code') not in (None,'','428'):
        raise ValueError('only unclassified HTTP428 is eligible')
    return spec,case


def verify_files(expected):
    if any(base.sha(base.regular(p))!=h for p,h in expected.items()):
        raise ValueError('protected source, code, or first score changed')


def no_prior_dispatch(diagnostics,case):
    # Request intents survive a crash and are stronger than success summaries.
    for p in Path(diagnostics).rglob('request_started.json'):
        row=base.read(base.regular(p))
        if row.get('request_hash')==case['failed']['request_hash']:
            raise ValueError('this exact request already has a diagnostic dispatch')


def classify(summary,response):
    if summary.get('status')=='validated' and summary.get('reusable') is True:
        return 'validated_payload_not_yet_imported'
    if summary.get('status')=='content_filter':
        return 'explicit_content_filter'
    if response and response.get('http_status')==428 and response.get('body_redacted') is False:
        body=response.get('body_text','')
        if hashlib.sha256(body.encode()).hexdigest()!=response.get('body_sha256'):
            raise ValueError('response evidence hash changed')
        try:raw=json.loads(body)
        except ValueError:return 'http428_unclassified'
        # Only evidence from this exact response permits a DLP label.
        def strings(value):
            if isinstance(value,str):yield value
            elif isinstance(value,dict):
                for x in value.values():yield from strings(x)
            elif isinstance(value,list):
                for x in value:yield from strings(x)
        if any('Data leak protection rejected' in x for x in strings(raw)):
            return 'explicit_dlp_rejection'
        return 'http428_unclassified'
    return summary.get('status','unknown_response')


def run(stage,spec_path,video,output):
    spec,case=load_case(spec_path,video)
    source,repo=Path(case['source_run']),Path(case['core_repo'])
    output=Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('diagnostic output must be new and ordinary')
    if any(output==p or output.is_relative_to(p) for p in (source,repo)):
        raise ValueError('diagnostic may not write to old run or core')
    protected={**case['source_files'],**case['source_root_hashes'],**spec['protected_first_score_files'],
        case['task_path']:case['task_sha256'],str(Path(spec_path).absolute()):SPECS_SHA,
        str(Path(__file__).absolute()):base.sha(__file__),str(Path(probe.__file__).absolute()):base.sha(probe.__file__)}
    verify_files(protected)
    registry=source.parents[1]/'recovery_claims/noevent_http428_once'
    registry.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Short lock only while checking the ended source and rebuilding exact input.
    # A batch may diagnose different source videos concurrently after preflight.
    with (registry/'preflight.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with base.stopped_parent(source):
            no_prior_dispatch(source.parents[1]/'diagnostics/noevent_resume_20260923',case)
            prepared=probe.prepare(source,video,repo)
    if prepared['failed']!=case['failed'] or prepared['artifact']['window_id']!=case['window_id']:
        raise ValueError('reconstructed source request differs from frozen selection')
    prepared['protected'].update(protected)
    verify_files(prepared['protected'])
    wire=json.dumps(prepared['adapter'].build_payload(prepared['client'],prepared['messages']),allow_nan=False).encode()
    receipt={'schema_version':1,'status':'preflight_passed','no_api_calls':True,'video_id':video,
        'source_run':str(source),'core_repo':str(repo),'window_id':case['window_id'],
        'request_hash':case['failed']['request_hash'],'wire_sha256':hashlib.sha256(wire).hexdigest(),
        'protected_first_scores':262,'question_ids':case['question_ids'],'spec_sha256':SPECS_SHA,
        'driver_sha256':base.sha(__file__),'artifact':prepared['artifact'],'protected_files':prepared['protected']}
    if stage=='preflight':
        output.mkdir(parents=True,mode=0o700)
        probe.write_private(output/'preflight.json',receipt)
        return {'status':'preflight_passed','video_id':video,'wire_sha256':receipt['wire_sha256']}
    with (registry/'claim.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        no_prior_dispatch(source.parents[1]/'diagnostics/noevent_resume_20260923',case)
        claim=registry/(case['failed']['request_hash']+'.json')
        probe.write_private(claim,{'source_run':str(source),'video_id':video,'window_id':case['window_id'],
            'request_hash':case['failed']['request_hash'],'owner':str(output),'maximum_http_requests':1,
            'claimed_unix':time.time(),'pid':os.getpid(),'driver_sha256':base.sha(__file__)})
        output=probe.claim_output(source,video,output)
    probe.write_private(output/'preflight.json',receipt)
    summary=probe.execute_once(prepared,output)
    response=base.read(output/'response.json') if (output/'response.json').exists() else None
    result={'video_id':video,'window_id':case['window_id'],'question_ids':case['question_ids'],
        'status':summary['status'],'classification':classify(summary,response),
        'request_hash':case['failed']['request_hash'],'maximum_http_requests':1,
        'http_status':summary.get('http_status'),'parent_files_unchanged':summary['parent_files_unchanged'],
        'summary_sha256':base.sha(output/'summary.json'),
        'response_sha256':base.sha(output/'response.json') if response else None,
        'wire_sha256':receipt['wire_sha256'],'official_score':False}
    probe.write_private(output/'diagnosis.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','probe'))
    p.add_argument('--specs',required=True);p.add_argument('--video-id',choices=sorted(CASES),required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args()
    result=run(a.stage,a.specs,a.video_id,a.output_dir)
    print(json.dumps(result))
    return 0


if __name__=='__main__':
    sys.dont_write_bytecode=True
    raise SystemExit(main())
