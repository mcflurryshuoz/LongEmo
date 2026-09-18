# AICodeMirror API 验证

2026-09-18，g450 实测。仅使用合成文本、16×16 红色图片与 1 秒音调，未发送 benchmark 内容，未修改运行配置或恢复实验。用户提供的 key 只保留在探测进程内存中，未写入凭证、源码或结果文件。

| 能力 | 结果 |
|---|---|
| Claude Sonnet 4.6 | HTTP 200，返回 OK. |
| Gemini 3.8 Flash 图片 | HTTP 200，正确返回 Red；响应未提供实际 modelVersion。 |
| Gemini 3.8 Flash 音频 | HTTP 200，描述为短促、高音调的电子声；响应未提供实际 modelVersion。 |
| GPT-6 Astra | Codex Responses 流式请求 HTTP 200，response.completed，返回 OK，响应 model=gpt-6-astra。 |
| Gemini Embedding 2 | batchEmbedContents 返回 HTTP 503、model_not_found：账户 Gemini 通道没有可用模型，未取得向量。 |

已验证的服务路径：

- Claude：`https://api.aicodemirror.ai/api/claudecode/v1/messages`
- Gemini：`https://api.aicodemirror.ai/api/gemini/v1beta/models/gemini-3.8-flash:generateContent`
- GPT-6：`https://api.aicodemirror.ai/api/codex/backend-api/codex/responses`（stream=true，store=false）
- 本次鉴权使用 `Authorization: Bearer <API_KEY>`。

路径依据为 [CC Switch Claude 配置](https://github.com/farion1231/cc-switch/blob/main/src/config/claudeProviderPresets.ts)、[Gemini 配置](https://github.com/farion1231/cc-switch/blob/main/src/config/geminiProviderPresets.ts)、[Codex 配置](https://github.com/farion1231/cc-switch/blob/main/src/config/codexProviderPresets.ts)，调用成功以本次保存的响应为准。网站控制台的直接读取返回 403，不影响上面的 API 调用成功。

这只是小请求可用性验证，尚未验证长视频窗口、完整 JSON 输出、大并发或可支撑全量实验的余额；不能由返回的模型别名确认不可变权重版本。该平台 GPT-6 使用 Responses 流接口，当前 Azure Chat Completions 适配需另行调整，不能只替换 base URL。Gemini 采用原生请求格式，但正式接入仍需验证授权头、长媒体预算和输出校验，并记录新的服务配置与缓存来源。

原始结果：

- [1789704266290725348.json](1789704266290725348.json)：GET https://api.aicodemirror.ai/api/claudecode/v1/models，HTTP 200。
- [1789704301971828423.json](1789704301971828423.json)：POST https://api.aicodemirror.ai/api/claudecode/v1/messages，HTTP 200。
- [1789704325040378849.json](1789704325040378849.json)：GET https://api.aicodemirror.ai/api/gemini/v1beta/models，HTTP 200。
- [1789704380860177728.json](1789704380860177728.json)：POST https://api.aicodemirror.ai/api/gemini/v1beta/models/gemini-3.8-flash:generateContent，HTTP 200。
- [1789704422949961977.json](1789704422949961977.json)：POST https://api.aicodemirror.ai/api/gemini/v1beta/models/gemini-embedding-2:batchEmbedContents，HTTP 503。
- [1789704541900668378.json](1789704541900668378.json)：POST https://api.aicodemirror.ai/api/gemini/v1beta/models/gemini-3.8-flash:generateContent，HTTP 200。
- [1789704556079437548.json](1789704556079437548.json)：POST https://api.aicodemirror.ai/api/codex/backend-api/codex/responses，HTTP 200。
