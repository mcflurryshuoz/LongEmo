"""Audit BlackAI Pro first scores against the protected 506-score snapshot."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from evaluation.io_utils import load_records
from evaluation.metrics import summarize
from experiments.zyf.blackai_continuation import read, atomic
from experiments.zyf.blackai31_continuation import NAME, SCORE_HASHES, guard
from experiments.zyf.blackai36_report import make_report as old_report
from experiments.zyf.finish_scoring import scoring_inventory
from experiments.zyf.series_scores import LABELS
from methods.longemo.common import file_hash


def make_report(runtime, output):
    run=runtime/'runs'/NAME;questions=read(run/'questions.json');guard(runtime,questions)
    output.mkdir(parents=True,exist_ok=True)
    old_report(runtime,output/'pre_switch_reference')
    prior=read(output/'pre_switch_reference/report.json')
    prior_scored={q['question_id']:q for q in prior['questions'] if q['status']=='scored'}
    assert len(prior_scored)==506
    target_ids={q['question_id'] for q in questions};assert not target_ids.intersection(prior_scored)
    baseline=load_records(runtime/'runs'/next(iter(SCORE_HASHES))/'scores.jsonl')
    raw={q['question_id']:dict(q) for q in baseline};accepted={}
    sources=[]
    for name in [*SCORE_HASHES,NAME]:
        p=runtime/'runs'/name/'scores.jsonl'
        if not p.exists():continue
        rows=load_records(p)
        for q in rows:
            if q.get('status')!='ok':continue
            qid=q['question_id'];assert qid not in accepted
            if name==NAME:assert qid in target_ids
            accepted[qid]=dict(q,source_run=name)
        sources.append({'run':name,'scores_sha256':file_hash(p),'metrics':summarize(rows)})
    for qid,q in prior_scored.items():
        assert all(accepted[qid].get(k)==q.get(k) for k in ['score','max_score','normalized_score','source_run'])
    state=read(run/'backend_status.json') if (run/'backend_status.json').exists() else {'status':'not_started','videos':{}}
    dispositions=[];metrics=[];series=defaultdict(list)
    for old in prior['questions']:
        row=dict(old);qid=row['question_id'];vid=row['video_id']
        if qid in accepted:
            metric=accepted[qid];row={k:row[k] for k in ['question_id','video_id','type','series']}
            row.update({k:metric.get(k) for k in ['score','max_score','normalized_score','source_run']});row['status']='scored'
        else:
            metric={**raw[qid],'status':'pending','normalized_score':None}
            if qid in target_ids:
                row.update(status=state.get('videos',{}).get(vid,{}).get('status','pending_memory_or_score'),source_run=NAME)
                for k in ['graph_outcome','experiment_fingerprint','graph_complete','previous_blackai_status']:row.pop(k,None)
                receipt=run/'imports'/(vid+'.json')
                if receipt.exists():
                    r=read(receipt);row.update(graph_outcome=r['outcome'],experiment_fingerprint=r['experiment_fingerprint'])
                memory=run/'memory'/vid/'memory.json'
                if memory.exists():row['graph_complete']=bool(read(memory).get('complete'))
        dispositions.append(row);metrics.append(metric);series[row['series']].append(metric)
    assert len(dispositions)==len({q['question_id'] for q in dispositions})==558
    pending=sum(len(v['pending']) for v in scoring_inventory(run,questions).values())
    data={'as_of':datetime.now(timezone.utc).isoformat(timespec='seconds'),
          'aggregation':'First valid scores, equal question weights, missing excluded. Mixed providers/models/inherited windows; not a homogeneous full benchmark.',
          'protected_first_506_unchanged':True,'runs':sources,'combined':summarize(metrics),
          'series':{k:summarize(v) for k,v in sorted(series.items())},'questions':dispositions,
          'question_disposition_counts':dict(Counter(q['status'] for q in dispositions)),
          'new_run':NAME,'new_run_metrics':summarize([m for m in metrics if m['question_id'] in target_ids]),
          'backend':state,'successful_answers_awaiting_first_judge':pending}
    atomic(output/'report.json',data)
    def pct(m):return '—' if m.get('percent_score') is None else f"{m['percent_score']:.2f} ({m['n_scored']}/{m['n_total']})"
    lines=['# BlackAI Pro 接替 AICodeMirror 补评','',f"核验时间：{data['as_of']}；后端：{state['status']}。",'',
           f"累计混合来源：**{pct(data['combined']['overall_unweighted'])}**；本轮：**{pct(data['new_run_metrics']['overall_unweighted'])}**。原 506 条首次评分保持不变。",'',
           '感知请求模型为 BlackAI `gemini-3.1-pro-preview`，返回的上游模型别名单独记入账本。继承原图谱窗口与来源；每视频首个未完成窗口仅一次验证，成功结果直接保留续跑。GPT-6 规划、答题、官方评分及 OpenRouter Gemini Embedding 2 不变。缺失题不计零分，不能视为同一配置的全量成绩。','',
           '| 任务 | 分数 /100（已评分/总题数） |','|---|---:|']
    for name,m in data['combined']['tasks'].items():lines.append(f'| {name} | {pct(m)} |')
    lines+=['','| 剧名 | 强度比较 | 情感轨迹 | 情感推理（等题权） |','|---|---:|---:|---:|']
    for name,m in data['series'].items():
        cells=[pct(m['tasks'].get(t,{})) for t in ['emotional intensity comparison','emotion trajectory','emotional reasoning']]
        lines.append('| '+('剧名未标注' if name=='unclassified' else LABELS.get(name,name))+' | '+' | '.join(cells)+' |')
    lines+=['',f"成功答案待首次评分：{pending}；旧 Matrix 拒绝的三题仍单独保留。",'',
            '[558 题首分、来源与缺失状态](report.json)。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return {'combined':data['combined']['overall_unweighted'],'new':data['new_run_metrics']['overall_unweighted'],'pending':pending}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(make_report(a.runtime,a.output),ensure_ascii=False))
