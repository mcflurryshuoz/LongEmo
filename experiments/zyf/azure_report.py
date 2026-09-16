"""Export a sanitized live progress report without questions, answers or raw media."""
import argparse
from collections import Counter
import json
from pathlib import Path

from evaluation.io_utils import write_json
from experiments.zyf.azure_status import snapshot, records


def export(run, output):
    result=snapshot(run)
    result['initial_configuration']=json.loads((run/'experiment_manifest.json').read_text())
    result['execution_revisions']=[json.loads(p.read_text()) for p in sorted((run/'execution_revisions').glob('*/manifest.json'))]
    result['operational_changes']=list(records([run/'operational_changes.jsonl'])) if (run/'operational_changes.jsonl').exists() else []
    result['metrics']=json.loads((run/'metrics.json').read_text())
    result['full_score_complete']=result['status']['status']=='complete' and result['metrics']['overall_unweighted']['coverage']==1
    p=run/'resources_latest.json'
    result['resources']=json.loads(p.read_text()) if p.exists() else None
    p=run/'imported_audio.json'
    result['imported_question_free_audio_caches']=len(json.loads(p.read_text())) if p.exists() else 0
    result['video_outcome_counts']=dict(Counter(v['status'] for v in result['status'].get('videos',{}).values()))
    overall=result['metrics']['overall_unweighted'];azure=result['azure']
    lines=['# Azure E06 progress','',f"Status: {result['status']['status']}. Scored: {overall['n_scored']}/{overall['n_total']}. Perception windows: {result['completed_windows']}/8342.",'',
        'This is a progress snapshot, not a complete benchmark result. Incomplete/service-rejected videos stay in the full denominator.', '',
        f"Azure HTTP outcomes: {azure['http_statuses']}. Returned tokens: {azure['returned_total_tokens']}. Azure billing: unknown (no billing export).",'',
        f"Video outcomes: {result['video_outcome_counts']}. Active video workers at snapshot: {len(result['status'].get('active_videos',[]))}.",'',
        'The queue was launched with 16 video workers and two answer/judge workers per video. Current operational Azure limits are 500k tokens/minute, 120 requests/minute and 24 maximum in-flight requests. Rejected content is recorded and excluded from automatic retries. The first wave gates later scheduling on one completed official-scoring result.', '',
        'The model/evaluator source is unchanged across the recorded scheduler revision. The initial manifest and every actual execution revision are included in JSON. The benchmark uses GPT-6 both to answer and judge, so self-judge bias remains possible.']
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'progress.json',result)
    (output/'progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'windows':result['completed_windows'],'scored':overall['n_scored'],'status':result['status']['status'],'http':azure['http_statuses']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.run,a.output)
