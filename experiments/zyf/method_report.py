"""Present current method scores consistently while retaining raw run provenance."""
import argparse
import json
from pathlib import Path

from experiments.zyf.series_scores import LABELS

TASKS = [('emotional intensity comparison','情感强度比较'),('emotion trajectory','情感轨迹'),('emotional reasoning','情感推理')]
START = '<!-- LONGEMO_METHOD_RESULTS:START -->'
END = '<!-- LONGEMO_METHOD_RESULTS:END -->'


def pct(metric):
    if metric.get('percent_score') is None:return '—'
    return f"{metric['percent_score']:.2f} ({metric['n_scored']}/{metric['n_total']})"


def score_table(data):
    rows=['| 指标 | 分数 /100 | 已评分 / 总题数 |','|---|---:|---:|']
    values=[('总体',data['combined']['overall_unweighted'])]
    values += [(label,data['combined']['tasks'][task]) for task,label in TASKS]
    for label,m in values:
        score='—' if m.get('percent_score') is None else f"{m['percent_score']:.2f}"
        rows.append(f"| {label} | {score} | {m['n_scored']}/{m['n_total']} |")
    return '\n'.join(rows)


def series_score_table(data):
    rows = [
        '#### 分剧成绩',
        '',
        '单元格为“分数 /100（已评分 / 该剧该类总题数）”。',
        '',
        '| 剧名 | 情感强度比较 | 情感轨迹 | 情感推理 |',
        '|---|---:|---:|---:|',
    ]
    for name, metrics in data['series'].items():
        label = '剧名未标注' if name == 'unclassified' else LABELS.get(name, name)
        cells = []
        for task, _ in TASKS:
            metric = metrics['tasks'].get(task, {})
            cells.append(pct(metric))
        rows.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def render_method_report(data, *, json_link='report.json', current_link=None):
    overall=data['combined']['overall_unweighted'];missing=overall['n_total']-overall['n_scored']
    lines=['# 当前方法评测结果','',
           '**方法：情感事件记忆图谱 + 结构化语义／Embedding 双路检索。** 先从视频、音轨与字幕构建事件图谱，再检索事件证据，由 GPT-6 回答并按官方规则评分。','']
    if current_link:lines += [f'[统一成绩入口]({current_link})。','']
    lines += [f"当前总体分数 **{overall['percent_score']:.2f}/100**，覆盖 **{overall['n_scored']}/{overall['n_total']} 题（{overall['coverage']*100:.2f}%）**。尚有 **{missing} 题未取得有效评分**。",'',
              score_table(data),'',
              '## 分剧成绩','',
              '单元格为“分数 /100（已评分 / 该剧该类总题数）”。','',
              '| 剧名 | 情感强度比较 | 情感轨迹 | 情感推理 |','|---|---:|---:|---:|']
    for name,m in data['series'].items():
        label='剧名未标注' if name=='unclassified' else LABELS.get(name,name)
        lines.append('| '+label+' | '+' | '.join(pct(m['tasks'].get(task,{})) for task,_ in TASKS)+' |')
    lines += ['', '## 评测口径','',
              '- 评测范围为 LongEmoBench episode：141 个视频、558 道题。每题先除以该题最高分，再对已评分题等权平均；缺失题不计零分。',
              '- 每题保留首次有效评分。情感推理列为该类题目的等题权均分；结果／解释的 60/40 指标另见机器可读报告。',
              '- 当前方法的事件图组织、双路检索和 GPT-6 答题／官方评分流程保持一致。感知环节包含不同模型、提供方及采样预算，保留逐窗口与逐题来源；这里汇总当前方法的已有成绩，不代表单一感知配置的全量结果。',
              '',f'[逐题分数、覆盖情况与来源]({json_link}) · [实验过程与历史快照](../../progress_history.md)。',
              '',f"数据核验时间：{data['as_of']}。"]
    return '\n'.join(lines)+'\n'


def publish(source, repo):
    raw=Path(source).read_bytes();data=json.loads(raw)
    questions=data['questions'];overall=data['combined']['overall_unweighted']
    assert len(questions)==len({q['question_id'] for q in questions})==558
    assert sum(q['status']=='scored' for q in questions)==overall['n_scored']
    root=Path(repo);out=root/'experiments/zyf/results/current_method';out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_bytes(raw)
    (out/'report.md').write_text(render_method_report(data))
    for name in ['blackai36_remaining_20260921','blackai31_after_mirror_20260921']:
        target=root/'experiments/zyf/results'/name/'report.md'
        assert target.is_file() and not target.is_symlink()
        target.write_text(render_method_report(data,json_link='../current_method/report.json',current_link='../current_method/report.md'))
    readme=root/'README.md';text=readme.read_text();assert text.count(START)==text.count(END)==1
    summary=(START+'\n\n'+score_table(data)+'\n\n'+series_score_table(data)+'\n\n'
        '[完整分剧成绩、覆盖率与计分口径](experiments/zyf/results/current_method/report.md)。\n\n'+END)
    before,rest=text.split(START);_,after=rest.split(END);readme.write_text(before+summary+after)
    missing=overall['n_total']-overall['n_scored']
    progress=['# LongEmo 方法进度','',
      '[当前方法评测结果](results/current_method/report.md) · [方法整体流程](../../README.md#zyf-分支当前方法整体流程) · [研究方案](plan.md) · [历史实验记录](progress_history.md)','',
      '已实现情感事件记忆图谱、结构化语义与 Embedding 双路检索、图关系扩展，以及 GPT-6 证据驱动回答和官方评分。','',
      score_table(data),'',
      f"覆盖率 **{overall['coverage']*100:.2f}%**；剩余 **{missing} 题**尚未取得有效评分。分剧成绩、逐题状态与来源统一见[方法成绩报告](results/current_method/report.md)。",'',
      '当前目标是完成剩余题目的图谱和评分，并在相同题集、模型与媒体预算下建立 direct 视频对照。已有小样本比较保留在历史记录，尚不能据此断言图方法优于 direct。','',
      '主报告按当前方法统一汇总。运行批次、提供方切换、断点处理和故障诊断保存在[历史实验记录](progress_history.md)及逐题来源中；保留首次有效评分，缺失题不计零分。','',
      f"数据核验时间：{data['as_of']}。"]
    (root/'experiments/zyf/progress.md').write_text('\n'.join(progress)+'\n')
    return {'scored':overall['n_scored'],'total':overall['n_total'],'percent_score':overall['percent_score'],'output':str(out)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.source,a.repo),ensure_ascii=False))
