# LongEmo 本轮材料核验记录

日期：2026-09-16。此文件区分已读取材料与尚未获得的内容，避免把旧 plan 的陈述当成代码事实。

## 本地材料

| 材料 | 路径 | 状态 |
|---|---|---|
| 旧方法设计 | `/Users/yingmanji/Desktop/emo/plan_old.md` | 173,026 字节；重点审阅总体流程、实体/对白接口、观察、事件整合、状态图、检索、修订、实验与评分章节 |
| API 资源说明 | `/Users/yingmanji/Desktop/emo/key.md` | 335 字节；脱敏查看 Gemini 代理配置和 OpenRouter 资源，未运行内容/调用模型 |
| 当前工作目录 | `/Users/yingmanji/Documents/ChatGPT/emo` | 本轮开始时除 `.git` 外没有项目代码 |
| 旧 plan 的配套文档 | `MEMORY_FIELDS.md`、`GRAPH_DESIGN.md`、`LOCALIZATION_PROMPTS.md`、`MEMORY_ORGANIZATION_EXAMPLE.md`、`METHOD_EXAMPLES.md` | 不在上述 Desktop 目录 |
| 旧 plan 引用的 S1/S2 代码 | `../../code/vebench/seg_pipeline/...` | 对应相对位置未找到；不宣称已检查实现 |

## 远程材料

### GitHub

目标：[mcflurryshuoz/LongEmo](https://github.com/mcflurryshuoz/LongEmo)。

初次公开网页、raw 和连接器访问未取得正文。随后用户提供 GitHub token，已通过 GitHub API 成功只读访问，确认仓库为私有仓库。没有修改远程仓库。

| 核验项目 | 结果 |
|---|---|
| 默认分支 | `main` |
| 固定提交 | `b29b70a361bf6d7b08d4907167720f288fdbc2fe` |
| 目录树 | 完整递归树，未截断；未发现 AGENTS.md |
| 本地快照 | `/Users/yingmanji/Documents/ChatGPT/emo/sources/LongEmo`；29 个重点文件，不是完整 clone |
| 读取范围 | 中文 README、pyproject、评分/指标/judge、推理及 API 适配器、Agentic、LongEmo 方法占位 |
| 完整性 | 29 个下载文件与固定树的 Git blob SHA 全部相符 |
| 实现验证 | 23 项离线合成检查；真实 scorer + 内存 mock judge，无网络/计费调用 |
| 已发现阻塞 | Agentic 导入不存在的 file_hash；parser 缺 tries/workers；详见 evaluation_contract.md |

默认分支、当前评分和接口已核实；真实 benchmark 题目、baseline 分数和错误样例仍未取得。`methods/longemo` 只有命名空间占位，没有旧 plan 所述图谱流水线实现。凭据只用于授权请求，未写入源码、文档或下载清单；授权下载进程已退出。

### Hugging Face

目标：[LongEmoBench episode](https://huggingface.co/datasets/mcflurryshuoz/LongEmoBench/tree/main/episode)。

通过公开 API 成功获得元信息和文件目录，未下载视频：

| 字段 | 返回值 |
|---|---|
| dataset ID | `mcflurryshuoz/LongEmoBench` |
| 元信息 revision | `aba29b08a2742efc1a00b2420dcfcf9f0cc63763` |
| private / gated | `false` / `auto` |
| card languages | `en`, `zh` |
| configs | `clip`（默认）、`episode` |
| 已声明 splits | 两个 config 都只有 `test` |
| episode JSON 文件 | 558 个，`G2_Q000001.json` 至 `G2_Q000558.json` |
| episode JSON 目录文件总字节 | 1,354,922 |
| episode 视频文件 | 141 个 |
| episode 视频目录大小总和 | 68,375,210,073 字节，约 68.38 GB / 63.68 GiB |

公开 API 来源：[元信息](https://huggingface.co/api/datasets/mcflurryshuoz/LongEmoBench)、[episode 清单](https://huggingface.co/api/datasets/mcflurryshuoz/LongEmoBench/tree/main/episode)、[视频清单](https://huggingface.co/api/datasets/mcflurryshuoz/LongEmoBench/tree/main/episode/videos)。

目录响应含 560 个条目：558 个 JSON 和 `subtitles`、`videos` 两个目录，响应头未发现 next page 链接。视频目录返回 141 个条目。读取 `G2_Q000001.json` 时返回访问受限/需要有权限的登录。

**未核实：** HF JSON 实际内容是否符合当前代码 schema、单文件题目条数、题型分布、答案、视频时长、有效样本数、字幕正文与说话人标签。代码声明的 schema 与正式评分现已核实，详见 evaluation_contract.md。元信息与各目录是分次对 `main` 请求的快照，正式实验需固定同一个 revision 重新校验，不能将这次目录清点替代实验 manifest。GitHub 授权不等同于 HF 门禁授权。

## 旧 plan 中必须统一的接口

1. §② 约第 1160–1307 行：状态放在事件内部，同一人物整段描述，不另存顶层状态；取消情绪对象字段和精确引用等内容。
2. §③ 约第 2025–2075 行：再次出现独立状态、情绪对象、精确时段及依据引用。
3. §② 语义更新示例以 S 标记语义事实，§③ 以 S 标记情绪状态。
4. 概览称不引入独立事情索引，后文状态图又使用事情 T。
5. 总体调用表与逐窗语义更新示例的实际模型调用数不一致。

新版的处理是设计建议：唯一状态真源保存在事件内，粒度改成可定位的原子状态，代码生成只读索引；显式保留情绪对象和来源；语义摘要使用 F ID 并按需更新。

## 本次产物与未执行事项

- `plan.md`：两阶段方法、统一 schema、检索与修订、数据协议、模型资源验证、基线消融、实施顺序。
- `research.md`：长视频推理与情感方法的原始来源、最接近工作的差异和取舍。
- `research_access.md`：本文件。
- `evaluation_contract.md`：基于固定代码提交的任务、评分、提交协议、实现问题和 plan 修订依据。
- `review_checks/check_evaluation.py` 与 `results.json`：23 项离线检查及结果。Agentic 后续逻辑检查使用临时内存 stub 绕过已记录的 import 错误，不表示原入口可以运行。

未调用付费模型，未修改 benchmark 或评分器，未生成 benchmark 预测，未宣称已经取得新分数。原始 key 未复制进文档。
