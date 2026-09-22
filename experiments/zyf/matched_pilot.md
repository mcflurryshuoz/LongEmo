# 三层消融 pilot 启动与断点记录

`matched_pilot.py` 串联两路感知、独立规划、三套回答及官方评分。先贯通一个完整视频，再启动其余视频；脚本不自动探测 API。

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
  --build-tries 3 --plan-tries 3 --answer-tries 3
  --audio-base-url https://matrixllm.alipay.com/v1
)

# 固定50题／21视频；先完成V1的全部13窗口和三套首分，成功后自动展开剩余视频。
python -m experiments.zyf.matched_pilot prepare "${common[@]}" \
  --mode pilot --gate-video G2_V000001 --run-name three_level_pilot50
nohup python -u -m experiments.zyf.matched_pilot all "${common[@]}" \
  --mode pilot --gate-video G2_V000001 --run-name three_level_pilot50 \
  > /root/longemo/runtime/three_level_pilot50.log 2>&1 &
```

`prepare` 只做本地验证与清单冻结，不调用 API。它按输入顺序固定前 50 题，并断言包含 21 个 episode 视频。总题单和逐视频题单都写入运行目录，build、plan、answer、score 全部使用这些真实裁剪文件。代码、媒体 SHA、模型、采样和尝试次数改变必须换运行目录；密钥不写入清单。

`--gate-video` 保留完整50题清单，先只调度该视频的两种记忆构建、规划、三套回答和官方评分；三种条件均有该视频全部问题的首次有效评分才放行其余视频。`G2_V000001` 是这21视频中最短的一个，约245秒、13个窗口。任何一步失败都写 `gate.json` 并暂停尚未启动的队列；重新运行不会清除失败或重置预算。成功视频的记忆、答案及首分在同一运行内复用。未通过门槛时必须使用 `all`，不能通过单独阶段绕过门槛。

若只需独立完整视频实验，可改为 `--mode smoke --video-id G2_V000001` 并使用另一个运行目录；它从固定50题集合裁剪该视频的问题。脚本不提供截短窗口的“完整视频”模式，单窗口服务成功不算贯通。

`--build-workers` 是两路感知合计的视频并发，默认 **12**。队列按每个视频的 event／noevent 交错排列，使两路同步开始，初始目标约每路 6 个；任务时长不同会使实际分配变化。同一提供方的限制须按两路请求合计遵守。后续最多 `--video-workers × --question-workers` 个题目请求；每个 event 视频先跑 base，再跑 method，避免同时构建同一向量索引。base 与 method 使用感知完成后预先生成并校验 SHA 的共同 plans，noevent 独立规划。

## 分阶段与恢复

门槛通过后，也可以把 `all` 换成 `build`、`plan`、`answer`、`score`，按顺序运行同一组参数。感知、规划、回答默认最多3次结构尝试，分别由 `--build-tries`、`--plan-tries`、`--answer-tries` 控制并冻结；明确的服务拒绝仍由客户端提前停止。官方评分固定 `tries=1`，没有提高评分次数的参数。已完成的 task 会复用，尚未启动的 task 可以继续；失败 task 不自动重开，进程中断且请求结果不明的 task 标记需要审计。不要删除 task.json 或评分文件来“恢复”。

本次修正后的独立协议可通过 `--parent-run /root/longemo/runtime/runs/three_level_pilot50_matched_20260922` 导入父运行已完成的窗口，同时必须使用新的 `--run-name`。父运行须是同一固定50题协议，协调器锁可取，协调器与所有已记录子进程的 PID／start ticks 均证明已退出；跨主机身份或缺失的活跃进程记录会阻止导入。父运行出现任何评分行或已接受首分时，本导入器拒绝执行，须另行审计首分来源；它不会重抽已评分题。

导入核验视频／字幕 SHA、完整模型 client 配置、采样、连续完成窗口，以及逐窗口重放得到的记忆。仅复制已提交的 `memory.json`、窗口产物及这些窗口对应的已核验音频；新构建器生成自己的 manifest 和 build fingerprint。旧 manifest、调用账本和未提交窗口的音频进入 `inheritance/history`，不作为新配置缓存使用。每个文件的来源、SHA和处理决定保存在 `inheritance/manifest.json`；旧目录不修改，也不覆盖未知新文件。

同一提供方已明确拒绝的媒体，两种表示均继承 `blocked_content`，避免仅换表示后重复请求；未知 HTTP 428 仅阻断原失败表示，等待诊断。已记录的结构／JSON错误和 `RemoteDisconnected` 可以在新协议中获得一次断点任务，仍遵守冻结的请求次数。其他未知错误、认证或额度故障保留阻断，不增加外层重启循环。父运行已完成窗口及拒绝证据均保留，即使该视频继续阻断。

运行锁与 `process.json` 防止重复协调器；`tasks/<stage>/<condition-video>/task.json` 记录子进程 PID、Linux start ticks、命令、开始／结束时间及返回码，日志位于相邻 `output.log`。每视频独立构建目录，避免各子进程覆盖公共 `build_results.json`。子进程退出码为 0 后仍核验记忆完整性和冻结 plans 的全部题目覆盖；校验失败的 task 写为 error，正常视频继续完成并汇总，对应阶段返回非零。

结果入口为 `status.json` 与 `base/method/noevent/accepted_scores.jsonl`。每个视频任务结束即在线程锁内聚合并原子更新状态，包含任务状态、已结束构建的完成窗口和评分覆盖；无需等全阶段结束。聚合器跳过仍活跃子进程的输出，活跃构建的窗口数暂为 `null`，可单独查看它原子发布的 memory；这样不会读取尚未写完的 JSONL。已结束文件出现损坏会报错，不会悄悄丢弃记录。官方 scorer 每次生成独立目录，协调器只启动尚未尝试的评分 task，并把每题首次有效评分冻结到 `accepted/<condition>/<qid>.json`；0 分同样保留。答案或首次评分改变会报错。门槛通过后，一个失败视频不阻止独立视频继续执行。

`status` 在协调器退出后重新核验和聚合；运行中直接读取 `status.json` 和 task 文件，避免启动第二个协调器。三个条件均覆盖全部冻结题单时才标记 `complete=true`。smoke 与 pilot 是不同运行，不能将两者成绩混合；调用用量分别保留以计入成本。
