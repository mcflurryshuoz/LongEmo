# BlackAI Gemini 3.8 Flash 复测

2026-09-19：AIStudio 62910175 的合成图片和语音请求均连接超时，未收到 HTTP 响应。g450 原生接口 `https://www.blackaicoding.com/v1beta/models/gemini-3.8-flash:generateContent` 返回 HTTP 200 / STOP，图片颜色识别正确；合成单音被描述为钟声，因此该测试只证明可调用，不能证明音频质量。

随后在 g450 对开发视频 G2_V000016 的第一窗做独立真实前端测试：core 0–20 秒、媒体 0–22 秒，与正式流程相同的音频观察、视频采样和 JSON 校验。27.98 秒内成功完成，得到 10 条音频观察、3 个实体和 2 个事件，窗口结果通过现有图谱校验。此测试不产生 benchmark 分数、不复用为完整视频记忆。

返回未提供 modelVersion，只确认所请求模型别名可调用，不确认实际权重版本。响应未给出余额或已结算费用，不能推断全量预算。继续采用分机执行：g450 感知，AIStudio GPT-6 答题／评分。

[AIStudio 请求结果](synthetic_aistudio.json) · [g450 合成请求](synthetic_g450.json) · [真实窗口验证](real_window_g450.json)。
