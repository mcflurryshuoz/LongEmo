# 三层消融 pilot 启动与断点记录

`matched_pilot.py` 串联两路感知、独立规划、三套回答及官方评分。先完成服务可用性验证，再启动本脚本；脚本不自动探测 API。

| 条件 | 记忆与索引 | 回答配置 |
|---|---|---|
| base | event 分支构建的冻结图；与 method 共用 Embedding 缓存和冻结 plans | `graph` |
| method | 同一张冻结图、同一索引和 plans | `progressive --progressive-routing none` |
| noevent | 独立时间窗口记录、索引和 plans | 窗口 BM25＋Embedding |

默认感知配置为 Matrix GPT-6 视觉、BlackAI Gemini 3.8 Flash 音频、20 秒窗口＋2 秒边界、1 fps、24 帧、200704 像素；提供方和模型均可通过命令行覆盖并冻结到运行配置。规划、回答和 judge 为同一 GPT-6，Embedding 为 OpenRouter Gemini Embedding 2。三路使用相同的 `--evidence-chars 48000` 预算参数；method 的覆盖信息与超大记录可能使实际证据超出该参数，不能将其写成严格 JSON 上限。实际输入字符和累计 token 按 trace／调用账本另行统计，不声称总推理预算相同。

## 启动

在部署了脚本的 noevent checkout 执行，路径按部署填写，凭证放在仓库外：

```bash
common=(
  --method-repo /root/longemo/LongEmo-method-168b272
  --noevent-repo /root/longemo/LongEmo-noevent
  --runtime /root/longemo/runtime
  --questions /root/longemo/runtime/data/questions.json
  --credential-file /root/longemo/runtime/private/answer.json
  --build-workers 12 --video-workers 6 --question-workers 2
)

# 一个视频，使用前50题中属于该视频的全部问题贯通所有阶段。
python -m experiments.zyf.matched_pilot prepare "${common[@]}" \
  --mode smoke --video-id G2_V000001 --run-name three_level_smoke
python -u -m experiments.zyf.matched_pilot all "${common[@]}" \
  --mode smoke --video-id G2_V000001 --run-name three_level_smoke

# smoke通过后，新目录启动固定前50题／21视频，不是仅设置answer --limit。
python -m experiments.zyf.matched_pilot prepare "${common[@]}" \
  --mode pilot --run-name three_level_pilot50
nohup python -u -m experiments.zyf.matched_pilot all "${common[@]}" \
  --mode pilot --run-name three_level_pilot50 \
  > /root/longemo/runtime/three_level_pilot50.log 2>&1 &
```

`prepare` 只做本地验证与清单冻结，不调用 API。它按输入顺序固定前 50 题，并断言包含 21 个 episode 视频；smoke 再从该集合裁剪一个视频的问题。总题单和逐视频题单都写入运行目录，build、plan、answer、score 全部使用这些真实裁剪文件。代码、媒体 SHA、模型与采样配置改变必须换运行目录；密钥不写入清单。

`--build-workers` 是两路感知合计的视频并发，默认 **12**。队列按每个视频的 event／noevent 交错排列，使两路同步开始，初始目标约每路 6 个；任务时长不同会使实际分配变化。同一提供方的限制须按两路请求合计遵守。后续最多 `--video-workers × --question-workers` 个题目请求；每个 event 视频先跑 base，再跑 method，避免同时构建同一向量索引。base 与 method 使用感知完成后预先生成并校验 SHA 的共同 plans，noevent 独立规划。

## 分阶段与恢复

也可以把 `all` 换成 `build`、`plan`、`answer`、`score`，按顺序运行同一组参数。已完成的 task 会复用，尚未启动的 task 可以继续。所有请求使用 `tries=1`：脚本不自动重发明确拒绝，也不自动扩大尝试预算。失败 task 保留错误与调用账本；进程中断且请求结果不明的 task 标记需要审计，不自动重跑。不要删除 task.json 或评分文件来“恢复”，核查后为必要的新尝试建立独立配置与运行记录。

运行锁与 `process.json` 防止重复协调器；`tasks/<stage>/<condition-video>/task.json` 记录子进程 PID、Linux start ticks、命令、开始／结束时间及返回码，日志位于相邻 `output.log`。每视频独立构建目录，避免各子进程覆盖公共 `build_results.json`。子进程退出码为 0 后仍核验记忆完整性和冻结 plans 的全部题目覆盖；校验失败的 task 写为 error，正常视频继续完成并汇总，对应阶段返回非零。

结果入口为 `status.json` 与 `base/method/noevent/accepted_scores.jsonl`。分阶段结束后聚合当前成绩；运行中还可查看 task、memory 的完成窗口和 calls 账本。官方 scorer 每次生成独立目录，协调器只启动尚未尝试的评分 task，并把每题首次有效评分冻结到 `accepted/<condition>/<qid>.json`；0 分同样保留。答案或首次评分改变会报错。没有预测的题保留缺失，失败视频不阻止另一表示的独立视频继续执行。

`status` 在协调器退出后重新核验和聚合；运行中直接读取 `status.json` 和 task 文件，避免启动第二个协调器。三个条件均覆盖全部冻结题单时才标记 `complete=true`。smoke 与 pilot 是不同运行，不能将两者成绩混合；调用用量分别保留以计入成本。
