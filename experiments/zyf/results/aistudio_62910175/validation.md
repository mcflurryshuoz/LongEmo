# AIStudio 62910175 接入验证

96 CPU、800 GiB 内存、初始约 795 GiB 可用磁盘，Python 3.10.15 / NumPy 可用。无需本地模型显存。FFmpeg 与 FFprobe 采用静态 7.0.2 构建；安装包来源 https://johnvansickle.com/ffmpeg/ ，FFprobe SHA-256 为 `4f231a1960d83e403d08f7971e271707bec278a9ae18e21b8b5b03186668450d`。

- Matrix GPT-6 chat：HTTP 200，返回 `gpt-6-astra`，OK。
- Matrix Gemini 3.8 chat 图像＋音频：HTTP 200；合成语音正确转写，返回音频 token。短静音样本被错误描述为有声，原始输出保留；不把接入检查等同于质量验证。
- Matrix 原生 Gemini endpoint：HTTP 404；本实验明确使用 chat 适配器。
- OpenRouter Gemini Embedding 2：HTTP 200，3072 维有限实数；预检余额约 $20.22。
- BlackAI、HF CDN：TLS 握手超时；HF 主站直连失败。数据采用已授权的 SSH/SCP 中转。

AIStudio 上冻结提交 `82b3b205fdf3d0522baf686e7bb901962ac4e58f` 的 33 项相关离线测试全部通过，包括 Matrix 路由、延迟数据入队、哈希校验、SCP 空密码交互和认证拒绝停止。方法提示词与原官方评分不变。新源码与协议单独冻结，不合并旧评分。
