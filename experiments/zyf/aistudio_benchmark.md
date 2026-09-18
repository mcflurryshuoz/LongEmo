# E10：AIStudio / Matrix 全量 episode 实验

用户授权在实验 `62910175` 配置并启动实验。约 800 GiB 内存、96 CPU、初始 795 GiB 可用磁盘；模型通过 API 调用，不加载本地大模型，不占 GPU 显存。

## 固定配置

| 阶段 | 模型 | 服务 |
|---|---|---|
| 音频观察、视频感知 | Gemini 3.8 Flash | Matrix chat |
| 事件／问题向量 | Gemini Embedding 2，3072 维 | OpenRouter embeddings |
| 检索规划、答案、官方 judge | GPT-6 Astra | Matrix chat |

这是新实验，不能视为 E09 的续跑。BlackAI 在容器的 TLS 握手超时；Matrix 原生 Gemini 路径返回 404，chat 图像／音频接口返回 200。独立合成语音正确转写；一秒静音测试出现幻听，诊断原文保留。检查证明接口可用，不代表任务质量。Matrix 返回模型别名，未提供更细后端版本。

方法仍为：不看问题构建事件图 → 语义与真实 embedding 双路召回 → 图扩展和时序覆盖 → GPT-6 回答。20 秒窗口、2 秒 padding、1 fps、24 帧上限、每帧 200704 像素；感知 medium、32768 输出上限；音频 low、4096 输出上限；回答／judge medium、8192 输出上限。top-k 12、证据 48000 字符、不回看。官方 rubric、评分归一化不变，保留首次成功判断，不重抽低分。

## 数据与运行

固定 HF revision `bb1933541008571883fa1eec1e7bd44d94b6ad2b`，141 视频、558 题、8342 窗口，视频 68,375,210,073 字节。项目 `/root/longemo/LongEmo-zyf`；数据 `/root/longemo/runtime/data`；运行 `/root/longemo/runtime/runs/matrix_gemini38_gpt6_full_v1`。凭证仅在 `/root/longemo/runtime/private`，不入 Git。新实验不导入旧图、音频、答案或评分。

g450 → 本地 4 路分片下载 → AIStudio SCP。为减少频繁新建连接，后续按批次在同一 SCP 连接内依次发送多个视频和各自完成标记；接收端仍逐视频校验后入队。`stage_data.py` 使用唯一上传名，接收端按 manifest 大小和 SHA-256 验证、原子发布，拒绝覆盖未知目标。保留完整 558 题分母，`--wait-for-data` 等待未到数据；V16 完成答题与评分后扩至 16 视频，每视频最多 2 个问答／评分 worker。本地中转依赖本机与 ssctl 在线；断线可恢复，已完成的校验与模型检查点保留。

```bash
export PATH=/root/longemo/runtime/bin:$PATH
cd /root/longemo/LongEmo-zyf
python3 -u -m experiments.zyf.azure_benchmark \
  --data-root /root/longemo/runtime/data \
  --output-dir /root/longemo/runtime/runs/matrix_gemini38_gpt6_full_v1 \
  --credential-file /root/longemo/runtime/private/answer.json \
  --embedding-cache-dir /root/longemo/runtime/embedding_cache_e10 \
  --perception-model gemini-3.8-flash --matrix-perception \
  --answer-base-url https://matrixllm.alipay.com/v1 \
  --embedding-backend gemini --wait-for-data \
  --startup-videos 1 --video-workers 16 --workers 2 --video-attempts 2 --execute
```

沿用历史模块名 `azure_benchmark`，此命令不登录或调用 Azure。首次 OpenRouter 余额约 $20.22，仅向量阶段使用该账户；余额低于 $0.25 时暂停新增 embedding 阶段，此阈值不是支出上限。实际配置、调用用量、覆盖率与首次成功评分分别保存。

[API 预检](results/aistudio_62910175/)与[进度记录](progress.md)。尚未完成的评测不报告为全量成绩。

## 续跑与单视频服务拒绝

首次运行在完成 66/558 题、1038 个窗口后触发三次构建失败阈值，于 2026-09-18 16:29 暂停。V7/V9/V14/V20 返回 HTTP 428，具体服务原因未知；V3/V11/V22/V38 返回明确的 content_filter。已完成 23 个视频，部分样本归一化均分 54.55%。

用户要求继续后，仅修订调度器：每个请求最近一次记录若为 HTTP 428，将所属视频记作 `blocked_request_precondition`，保留缺失结果，不再提交该请求，也不阻止其他视频调度。明确的内容拒绝继续单独记作 `blocked_input_policy`。两类视频都保留在完整 558 题分母内。认证／额度错误与其他连续构建失败仍会停止调度。

感知、检索、回答、官方 judge 的代码和配置不变，方法哈希仍为 `15f2bb63a975c33657bf354293c20a412ff7cc8fb447bf26050196158a288a88`。原实验 manifest 保留；调度器修订另记 execution revision。完成的窗口通过原校验后复用，答案和首次成功评分保留，不重新抽取低分结果。

续传先按接收端 SHA-256 校验记录对齐 53 个已到视频，再发送其余文件；g450 下载重开四路并启用 SSH keepalive。隧道仍需保持在线。

## 用户请求的独立 GPU 负载

按用户追加要求启动了独立 GPU 活动作业，使用原本空闲的 0 号 L20X，约 1.1 GiB 显存，目标 50% 计算占空比，最长 12 小时（截至 2026-09-19 03:10 CST）。实际利用率随采样窗口变化。PID 65954，脚本 `/root/longemo/runtime/ops/gpu_load.py`，状态 `/root/longemo/runtime/ops/gpu_load_status.json`。超过 82°C 或可用显存不足 8 GiB 时停止。

此作业不参与模型推理或评分，资源开销单独记录；不能把它当作事件图方法的 GPU 需求或训练计算。[配置记录](results/aistudio_62910175/gpu_activity.json)。
