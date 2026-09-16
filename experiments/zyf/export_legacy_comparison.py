"""Sanitized comparison of frozen legacy answers and existing graph scores."""
import argparse,json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from evaluation.io_utils import load_records,write_json
from evaluation.metrics import summarize


def export(run,output):
    config=json.loads((run/'comparison_manifest.json').read_text())
    baseline=load_records(run/'accepted_scores.jsonl');graph=load_records(run/'graph_original_scores.jsonl')
    ids=config['configuration']['question_ids']
    for rows in (baseline,graph):
        assert len(rows)==len(ids) and {r['question_id'] for r in rows}==set(ids)
        assert all(r['status']=='ok' for r in rows)
    b={r['question_id']:r for r in baseline};g={r['question_id']:r for r in graph}
    bm=summarize(baseline);gm=summarize(graph)
    result={'as_of':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'episode':'Friends S01E08','manifest':config,
        'baseline_inputs_verified':{'model':'gemini-2.5-flash','frames':128,'audio':'raw source audio','subtitles':True,
            'verification':'Actual modelVersion and media from saved prediction records; subtitle text present in original messages.'},
        'baseline_metrics':bm,'graph_metrics':gm,
        'overall_delta_percentage_points':gm['overall_unweighted']['percent_score']-bm['overall_unweighted']['percent_score'],
        'pairs':[{'question_id':qid,'type':g[qid]['type'],'max_score':g[qid]['max_score'],
            'direct_score':b[qid]['score'],'graph_score':g[qid]['score'],
            'direct_normalized':b[qid]['normalized_score'],'graph_normalized':g[qid]['normalized_score']} for qid in ids],
        'interpretation':'Tie on both questions. Different generation models, visual budgets and audio processing; not a controlled graph ablation.',
        'judge_findings_paraphrased':{'G2_Q000271':'Both answers identify the affectionate gesture but omit the key causal realization about the pattern of maternal criticism.',
            'G2_Q000272':'Both answers omit required stages of the emotional trajectory; the graph answer also misses the final reassurance followed by renewed indignation.'}}
    transport=[json.loads(line) for p in (run/'azure_transport').glob('*.jsonl') for line in p.read_text().splitlines()]
    result['new_judging_usage']={'attempts':len(transport),'returned_total_tokens':sum(r.get('usage',{}).get('total_tokens',0) for r in transport),
        'actual_models':sorted({r.get('response_model') for r in transport if r.get('response_model')}),'billing_usd':None}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'comparison.json',result)
    lines=['# 《老友记》S01E08：已有直答预测与事件图检索对照','',
        f"时间：{result['as_of']}。视频 G2_V000076，共两题，双方覆盖率均为 2/2。",'',
        'Gemini 2.5 Flash 直答答案来自之前 pilot_v2 的成功预测，本次仅补齐首次官方评分；GPT-6 图方法复用 E06 的首次成功评分。共同采用 GPT-6 Astra 和原始官方 rubric、提示词与归一化公式。','',
        '| 题目 | Gemini 2.5 Flash 直接作答 | GPT-6 事件图检索 |','|---|---:|---:|']
    for r in result['pairs']:
        label='情感推理解释' if r['type']=='emotional reasoning' else '情感轨迹'
        lines.append(f"| {r['question_id']} {label} | {r['direct_score']}/{r['max_score']} | {r['graph_score']}/{r['max_score']} |")
    lines += [f"| 归一化均分（百分制） | {bm['overall_unweighted']['percent_score']:.2f} | {gm['overall_unweighted']['percent_score']:.2f} |",'',
        f"当前差值：{result['overall_delta_percentage_points']:+.2f} 个百分点。这两题没有观察到提分。",'',
        '判分理由显示：解释题双方都描述了示好行为，但缺少母女批评模式引发自我意识这一关键原因；轨迹题都遗漏了所需阶段，图方法还未覆盖结尾短暂获得安慰后再度不满的变化。答案较长并未自动补齐评分要求。','',
        '比较限制：生成模型不同；直答使用 128 帧、原始音频和字幕，图方法使用分窗感知及 Gemini 音频观察，视觉与音频处理预算不同。样本只有两道开发题，且 GPT-6 同时用于图方法和评分，因此不能把差值归因于图检索，也不能代表整部剧。题目选择依据是现有预测可用性，而非分数。','',
        f"本次新增评分调用 {len(transport)} 次，返回 token 共 {result['new_judging_usage']['returned_total_tokens']}；Azure 实际账单金额未知。"]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'episode':result['episode'],'direct':bm['overall_unweighted']['percent_score'],'graph':gm['overall_unweighted']['percent_score'],'delta':result['overall_delta_percentage_points']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.run,a.output)
