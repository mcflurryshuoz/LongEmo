# noevent-v2：渐进式窗口检索评测结果

2026-09-24 03:32 CST 队列结束；最终审计完成。

**244/558题，已评分均分64.45；完整缓存候选279题，候选覆盖率87.46%。** 缺失排除，真零分保留。

| 类型 | 已评分 | 均分 |
|---|---:|---:|
| emotion trajectory | 98 | 62.50 |
| emotional intensity comparison | 86 | 55.81 |
| emotional reasoning | 60 | 80.00 |

| 剧集／来源 | 已评分 | 均分 |
|---|---:|---:|
| Frasier | 43 | 60.85 |
| Friends | 59 | 69.63 |
| Home with Kids | 5 | 33.33 |
| Malcolm in the Middle | 17 | 69.12 |
| Modern Family | 49 | 63.78 |
| unclassified | 71 | 63.85 |

## 未完成范围

- no_complete_observation_cache: 272题
- answer_output_json_error: 8题
- excluded_prior_backend_refusal: 7题
- plan_content_filter: 14题
- plan_invalid_parameter: 5题
- embedding_http429: 7题
- answer_content_filter: 1题

## 方法与可比性

复用同一份完整纯窗口感知缓存，不重新感知视频；没有事件节点、状态链或关系边。初始最多4窗口，最多3轮追加语义＋Embedding检索，先按相关性选择，再按时间展示。仅引用本题已披露证据。Matrix GPT-6规划、回答和官方评分；OpenRouter Gemini Embedding 2。

每包48000字符、累计96000字符；协议token计量为cl100k_base加消息开销，单次输入24000、累计80000，最多4次输出各8192。实际服务usage另记。

缓存分组：219题原8192来源、17题继承后16K、8题固定边界预处理；不是单一感知配置的全集结果。

原50题pilot的80%运行成功率门槛未通过（39/50），原gate保留。其余229道未尝试题经源文件、预算、引用与首分保护审计后另行执行；不重跑pilot失败题，不按分数选择。

旧single-pass noevent成绩和旧method成绩未并入。method-v2共享观察上的图分支尚未评测，不能据此计算事件图收益。

三次执行均已结束，成功答案漏评和有效judge漏归档均为0。代码未推送；本目录只发布安全结果，不含参考答案、原始预测或密钥。
