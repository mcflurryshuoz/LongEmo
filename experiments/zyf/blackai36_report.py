"""Report first valid scores, provenance, and unresolved questions without API calls."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
from evaluation.metrics import summarize
from evaluation.io_utils import load_records
from experiments.zyf.blackai36_continuation import NAME,SCORE_HASHES,score_guard
from experiments.zyf.blackai_continuation import read,atomic
from experiments.zyf.finish_scoring import scoring_inventory
from experiments.zyf.series_scores import source_series,LABELS
from methods.longemo.common import file_hash


def make_report(runtime,output):
    run=runtime/'runs'/NAME;target=read(run/'questions.json');score_guard(runtime,target)
    questions=read(runtime/'data/questions.json');assert len(questions)==558
    target_ids={q['question_id'] for q in target};accepted={};runs=[];baseline={}
    for index,name in enumerate([*SCORE_HASHES,NAME]):
        p=runtime/'runs'/name/'scores.jsonl'
        if not p.exists():continue
        rows=load_records(p)
        if index==0:baseline={r['question_id']:r for r in rows}
        for row in rows:
            if row.get('status')!='ok':continue
            qid=row['question_id'];assert qid not in accepted
            accepted[qid]={**row,'source_run':name}
        runs.append({'run':name,'scores_sha256':file_hash(p),'metrics':summarize(rows)})
    state=read(run/'backend_status.json') if (run/'backend_status.json').exists() else {'status':'not_started','videos':{}}
    inventory=scoring_inventory(run,target)
    pending=sum(len(v['pending']) for v in inventory.values())
    flat=[];series=defaultdict(list);dispositions=[]
    for q in questions:
        qid=q['question_id'];vid=q['video_id'];s=source_series(q.get('source',{}).get('from','')) or 'unclassified'
        row={'question_id':qid,'video_id':vid,'type':q['type'],'series':s}
        if qid in accepted:
            metric=accepted[qid];row.update({k:metric.get(k) for k in ['score','max_score','normalized_score','source_run']});row['status']='scored'
        else:
            metric={**baseline[qid],'status':'pending','normalized_score':None}
            if qid in ['G2_Q000259','G2_Q000260']:
                row.update(status='blocked_question_policy',stage='answer',provider='Matrix GPT-6')
            elif qid=='G2_Q000350':row.update(status='blocked_judge_policy',stage='judge',provider='Matrix GPT-6')
            else:
                assert qid in target_ids,'unclassified missing question'
                video_state=state.get('videos',{}).get(vid,{})
                row.update(status=video_state.get('status','pending_memory_or_score'),source_run=NAME)
                receipt=run/'imports'/(vid+'.json')
                if receipt.exists():
                    item=read(receipt);row['graph_outcome']=item['outcome'];row['experiment_fingerprint']=item['experiment_fingerprint']
                if (run/'memory'/vid/'memory.json').exists():row['graph_complete']=bool(read(run/'memory'/vid/'memory.json').get('complete'))
        flat.append(metric);series[s].append(metric);dispositions.append(row)
    data={'as_of':datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'aggregation':'First valid scores; equal question weights; missing excluded. Mixed models/providers and inherited windows, not a homogeneous complete benchmark.',
        'runs':runs,'combined':summarize(flat),'series':{s:summarize(rows) for s,rows in sorted(series.items())},
        'question_disposition_counts':dict(Counter(x['status'] for x in dispositions)),
        'questions':dispositions,'backend':state,'successful_answers_awaiting_first_judge':pending,'protected_score_files_unchanged':True}
    output.mkdir(parents=True,exist_ok=True);atomic(output/'report.json',data)
    pct=lambda x:'—' if x.get('percent_score') is None else f"{x['percent_score']:.2f} ({x['n_scored']}/{x['n_total']})"
    combined=data['combined']['overall_unweighted'];new=next(r for r in runs if r['run']==NAME)['metrics']['overall_unweighted'] if any(r['run']==NAME for r in runs) else {}
    lines=['# BlackAI Gemini 3.6 补评','',f"核验时间：{data['as_of']}；后端状态：{state['status']}。",'',
        f"新实验：{pct(new)}；累计混合来源：{pct(combined)}。原 486 道首次评分文件哈希保持不变。",'',
        '本轮范围为 69 题、13 视频，音视频新窗口使用 BlackAI Gemini 3.6，父图谱观察保留来源。Matrix GPT-6 规划/答题/官方评分，OpenRouter Gemini Embedding 2。缺失题不计零分，不能作为同一配置的全量成绩。','',
        '| 任务 | 分数 /100（已评分/总题数） |','|---|---:|']
    for task,m in data['combined']['tasks'].items():lines.append(f'| {task} | {pct(m)} |')
    lines+=['','| 剧名 | 强度比较 | 情感轨迹 | 情感推理（等题权） |','|---|---:|---:|---:|']
    for s,m in data['series'].items():
        cells=[pct(m['tasks'].get(task,{})) for task in ['emotional intensity comparison','emotion trajectory','emotional reasoning']]
        lines.append('| '+('剧名未标注' if s=='unclassified' else LABELS.get(s,s))+' | '+' | '.join(cells)+' |')
    lines+=['',f"未评分 {558-combined['n_scored']} 题；已有成功答案待首次评分 {pending} 题。",'',
        '逐题状态：`'+json.dumps(data['question_disposition_counts'],ensure_ascii=False)+'`。',
        '旧有 2 道回答过滤与 1 道评分过滤保留；新感知拒绝、暂时故障和待执行状态按逐题来源区分。',
        '', '[逐题分数、来源、哈希和状态](report.json)。']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return {'path':str(output),'new':new,'combined':combined,'pending_judgments':pending,'backend_status':state['status']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(make_report(a.runtime,a.output),ensure_ascii=False))
