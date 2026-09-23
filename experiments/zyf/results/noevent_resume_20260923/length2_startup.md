# noevent 两视频启动记录

2026-09-23 **13:40:12 CST**实际启动，固定V117、V138共11题；源失败请求均有HTTP200、`finish_reason=length`证据。复用已经停止的检查点，原运行与首分保持不变。

| 项目 | 配置 |
|---|---|
| 继承窗口 | V117 10窗＋V138 56窗，共66窗 |
| 并发 | 2视频、每视频2题 |
| 视觉输出上限 | 16384 token |
| 感知 | Matrix Gemini 3.8 Flash，24帧，20秒核心＋2秒边界 |
| 后端 | Matrix GPT-6规划／回答／官方评分，OpenRouter Gemini Embedding 2 |
| 启动身份 | PID `44279`，start_ticks `8209837` |

13:41:23已观察到6次真实API请求，V117完成11窗、V138完成57窗，两者均通过原截断窗口；尚无新评分。先前3个音频任务出现上游429 `Resource exhausted`，故本批采用两路并发；不据此判断充值不足或内容过滤。目前不提供确定完成时间。

运行：`/root/longemo/runtime/runs/noevent_length2_continuation_20260923`。复用已验证部署：`/root/longemo/runtime/staging/length5_deploy_20260923_122602`。

- 驱动SHA：`0ff112b61676fc9341d520e1f0d5132bed26df163fa9651567eb16eca6138081`
- 配置SHA：`a1f8f80fc9b8bbfdb436d4ea3e6b065d626c2e0236dfaadcc7711375bb9ff47c`
- 题单SHA：`1af11f078cfa84d46005f6b58326b5d122a3c82ae8fab729fb3e38d08b3d1b7d`

[启动JSON与11题范围](length2_startup.json) · [当前235题报告](report.md)。该队列尚未贡献新分数；当前235题已包含227题原预算和8题继承／16384来源。
