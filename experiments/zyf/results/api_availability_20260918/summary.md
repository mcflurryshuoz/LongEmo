# g450 API 可用性检查

检查时间：2026-09-18 11:45（北京时间）。仅执行最小合成输入探测与账户额度查询；benchmark 保持暂停，未切换服务。

| 接口 | 本次实测 |
|---|---|
| Azure GPT-6 Astra | 两次均 HTTP 429，rate_limit_exceeded；第二次返回 token 限额 1,000,000/分钟、剩余额度 -63,218，Retry-After 30 秒。当前受 token 速率限制，尚未取得成功回复。 |
| BlackAI Gemini 3.8 Flash | 合成图片＋1 秒音频请求 HTTP 404：当前账户组无配置账号支持此模型。模型目录仍列出该名字，但不代表实际可调用。 |
| BlackAI Gemini Embedding 2 | 原生 batchEmbedContents HTTP 404：当前账户组无配置账号支持此模型。 |
| OpenRouter Gemini Embedding 2 | HTTP 200，返回一个有效的 3072 维向量；本次报告费用 0.0000033 美元。 |

OpenRouter 账户接口报告累计额度 70 美元、累计使用 49.779445405 美元、剩余约 20.22 美元。这里只验证了 Embedding，不能据此保证整个 benchmark 的用量预算或其他模型可用性。

本次未重测 Matrix。原始探测结果见 [请求结果](1789703034879330380.json)、[额度与目录](service_metadata.json)、[Azure 重试及限流头](azure_retry.json)。
