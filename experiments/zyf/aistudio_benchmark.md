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

g450 → 本地 4 路分片下载 → AIStudio SCP。`stage_data.py` 使用唯一上传名，接收端按 manifest 大小和 SHA-256 验证、原子发布，拒绝覆盖未知目标。保留完整 558 题分母，`--wait-for-data` 等待未到数据；V16 完成答题与评分后扩至 16 视频，每视频最多 2 个问答／评分 worker。本地中转依赖本机与 ssctl 在线；断线可恢复，已完成的校验与模型检查点保留。

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
