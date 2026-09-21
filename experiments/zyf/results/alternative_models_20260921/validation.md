# 剩余失败窗口的 API 与模型对照

2026-09-21，g450 实测。对两个视觉失败窗口（V71 W4、V105 W1）及全部四个音频失败窗口（V48 W61、V115 W24、V129 W4、V137 W60）进行了有限验证，共 18 次生成请求。不是完整图谱，也不产生 benchmark 分数；现有 **476/558、57.65/100** 保持不变。

| 服务 / 模型 | 视觉窗口 | 音频窗口 |
|---|---|---|
| AICodeMirror / Claude Opus 5 | V105 通过原图谱校验；V71 两次 HTTP 524 | 当前 Anthropic 适配器不接收原始音频，未提交 |
| AICodeMirror / Claude Sonnet 5 | V105 返回 JSON，但观察引用了未声明人物；V71 非 JSON 响应，原因未确诊 | 未测试 |
| AICodeMirror / Gemini 2.5 Pro | V71、V105 均明确 `content_filter` | V48、V115、V129 通过；V137 返回内容，但 `voice` 为空，未通过校验 |
| AICodeMirror / Gemini 3.1 Pro | V71、V105 均明确 `content_filter` | V48 通过；V137 同样缺少有效 `voice` |
| BlackAI / Gemini 2.5 Pro | V71、V105 均 HTTP 404 | V48 HTTP 404 |

模型目录包含相关别名，不等于生成端点可用。BlackAI 的 404 不能解释为内容过滤。Claude Opus 5 的 V71 因明确 524 只额外原样尝试一次，仍失败后停止。Sonnet 的非 JSON 响应不自动认定为内容拒绝或暂时网络错误。

所有模型对同一窗口使用完全一致的消息指纹、帧、字幕、可用音频观察与已有记忆上下文。保持 20 秒窗口、2 秒 padding、最多 16 帧和 150528 像素；原 `apply_window` 和音频时间／voice／cue 校验不变。Gemini 2.5 使用其支持的 thinkingBudget=1024，3.1 请求 thinkingLevel=low；服务方实际返回 `gemini-3.1-pro-high`，该差异已记入逐请求记录。模型别名及返回值不保证不可变权重版本。

## 对下一批实验的建议

有实证支持的候选是 **Claude Opus 5 负责视觉图谱、Gemini 2.5 Pro 负责音频观察**。Claude 已通过一个原 Gemini 拒绝的视觉窗口，Gemini Pro 已通过三个原音频失败窗口，说明“换模型”有实际收益；仍需解决 V71 网关超时和 V137 音频 schema 问题，并验证其余视觉窗口及完整视频连续处理，不能宣称 15 个视频都能完成。

这一组合在接口层面可行：[Claude 支持多图输入](https://platform.claude.com/docs/en/build-with-claude/vision)，[Gemini 2.5 Pro 支持音频、图像及视频输入](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro)。下一批应使用独立运行目录，继承窗口保留来源，答题／评分仍用 Matrix GPT-6，Embedding 仍用 OpenRouter Gemini Embedding 2。现有成功评分不重抽，新模型补评单独报告。

本轮只做接口和结构验证，没有运行这个组合的完整图谱构建或新评分。代码的 8 项相关测试通过；密钥与实际媒体留在私有 runtime。

[逐请求配置、结果与相同输入校验](validation.json)。
