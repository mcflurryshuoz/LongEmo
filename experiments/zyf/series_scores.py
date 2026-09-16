"""Offline series/episode aggregation of an immutable score snapshot; no API calls."""
import argparse
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import re

from evaluation.metrics import summarize, aggregate
from evaluation.io_utils import write_json

ALIASES={'jiayouernv':'Home with Kids'}
LABELS={'Frasier':'欢乐一家亲','Friends':'老友记','Modern Family':'摩登家庭',
        'Malcolm in the Middle':'Malcolm in the Middle','Home with Kids':'家有儿女'}


def source_series(source):
    value=re.fullmatch(r'(.+?)\s+[sS]\d+[eE]\d+',source.strip())
    if not value:return None
    series=value.group(1)
    return ALIASES.get(series,series)


def export(data_root, run, out):
    questions=json.loads((data_root/'questions.json').read_text())
    raw=(run/'scores.jsonl').read_bytes()
    rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    qmap={q['question_id']:q for q in questions}
    scores={r['question_id']:r for r in rows}
    assert len(scores)==len(rows)==len(questions) and set(scores)==set(qmap)
    statuses=json.loads((run/'status.json').read_text()).get('videos',{})
    by_series=defaultdict(list);by_episode=defaultdict(list);unclassified=[]
    for q in questions:
        source=q.get('source',{}).get('from','')
        series=source_series(source)
        if series is None:
            unclassified.append(q['question_id']);continue
        by_series[series].append(q['question_id'])
        by_episode[(series,source,q['video_id'])].append(q['question_id'])
    result={'as_of':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
            'run':run.name,'score_snapshot_sha256':hashlib.sha256(raw).hexdigest(),
            'question_file_sha256':hashlib.sha256((data_root/'questions.json').read_bytes()).hexdigest(),
            'grouping':'Explicit source.from series + SxxExx only; URL-only sources remain unclassified.',
            'aliases':ALIASES,'judge':'gpt-6-astra; same model family as generation',
            'comparison_note':'Paired baseline comparisons are reported separately and must use identical question IDs.',
            'aggregation':'Official per-question normalized mean over successful scores. Missing scores are not zero. Coverage remains explicit.',
            'series':{},'unclassified':{'videos':len({qmap[q]['video_id'] for q in unclassified}),
                'questions':len(unclassified),'metrics':summarize([scores[q] for q in unclassified])}}
    for series,qids in sorted(by_series.items()):
        videos={qmap[q]['video_id'] for q in qids};episodes=[]
        for (group,source,vid),ids in sorted(by_episode.items()):
            if group!=series:continue
            metrics=summarize([scores[q] for q in ids]);overall=metrics['overall_unweighted']
            episodes.append({'source':source,'video_id':vid,'status':statuses.get(vid,{}).get('status','running_or_queued'),
                'question_ids':ids,'fully_scored':overall['coverage']==1,'metrics':metrics,
                'question_scores':[{k:scores[q].get(k) for k in ('question_id','type','status','score','max_score','normalized_score')} for q in ids]})
        complete=[e for e in episodes if e['fully_scored']]
        complete_qids=[q for e in complete for q in e['question_ids']]
        result['series'][series]={'label':LABELS.get(series,series),'video_count':len(videos),
            'fully_scored_videos':len(complete),'partly_scored_videos':sum(0<e['metrics']['overall_unweighted']['coverage']<1 for e in episodes),
            'policy_blocked_videos':sum(statuses.get(v,{}).get('status')=='blocked_input_policy' for v in videos),
            'metrics':summarize([scores[q] for q in qids]),'episodes':episodes,
            'complete_episode_comparison_subset':{'question_ids':complete_qids,
                'video_ids':[e['video_id'] for e in complete],
                'selection':'All fully-scored episodes, selected by availability, not by score; service-filter selection bias remains.',
                'metrics':summarize([scores[q] for q in complete_qids]) if complete_qids else {'overall_unweighted':aggregate([]),'tasks':{}}}}
    total_named=sum(s['metrics']['overall_unweighted']['n_total'] for s in result['series'].values())
    assert total_named+len(unclassified)==len(questions)
    out.mkdir(parents=True,exist_ok=True);write_json(out/'series_scores.json',result)
    lines=['# 按剧集汇总的当前成绩','',f"快照时间：{result['as_of']}。成绩来自现有首次成功评分，无新增 API 调用。",'',
        '按题目 source.from 中明确的剧名与集号分组。仅含 URL 的来源保留为未分类，不推测剧名。所有均分只针对已评分题目，不能视为整部剧的完整成绩。','',
        '| 剧集 | 完整评分集数／总集数 | 已评分题数／总题数 | 当前均分／100 | 强度比较 | 情感轨迹 | 情感推理（等题权） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def pct(metrics):
        value=metrics.get('percent_score');return '—' if value is None else f'{value:.2f}'
    for name,s in result['series'].items():
        m=s['metrics'];o=m['overall_unweighted'];t=m['tasks']
        lines.append(f"| {s['label']} ({name}) | {s['fully_scored_videos']}/{s['video_count']} | {o['n_scored']}/{o['n_total']} | {pct(o)} | {pct(t.get('emotional intensity comparison',{}))} | {pct(t.get('emotion trajectory',{}))} | {pct(t.get('emotional reasoning',{}))} |")
    lines+=['','情感推理的 60/40 加权子指标另存 JSON；上表与总体均分采用题目等权。评分与回答同用 GPT-6，存在自评偏差。单独提供的 S01E08 两题已有直答预测补评只能做同题参考，不能拿不同覆盖范围的整剧均分比较方法优劣。','',
        '## 已评分的具体集','', '| 剧集／集号 | 视频 ID | 已评分／总题数 | 当前均分／100 | 完整性 |', '|---|---|---:|---:|---|']
    for s in result['series'].values():
        for e in s['episodes']:
            o=e['metrics']['overall_unweighted']
            if not o['n_scored']:continue
            lines.append(f"| {e['source']} | {e['video_id']} | {o['n_scored']}/{o['n_total']} | {pct(o)} | {'完整' if e['fully_scored'] else '部分'} |")
    lines+=['','可用于同题对照的已完整评完集及 question_id 列表已写入 JSON 的 complete_episode_comparison_subset。此子集存在服务过滤带来的选择偏差；比较时两种方法必须使用相同题号、judge 和评分口径，并单独记录视觉/音频预算。']
    (out/'series_scores.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'as_of':result['as_of'],'series':{name:{'video_count':s['video_count'],'fully_scored_videos':s['fully_scored_videos'],
        'policy_blocked_videos':s['policy_blocked_videos'],'metrics':s['metrics'],
        'complete_episode_subset':s['complete_episode_comparison_subset']['metrics']['overall_unweighted']} for name,s in result['series'].items()},
        'unclassified_questions':len(unclassified)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.data_root,a.run,a.output)
