# V101 独立队列启动记录

2026-09-23 **13:59:54 CST**实际启动；14:00:26确认进程身份存活，1个构建任务运行，尚无新增评分。当前240题评分快照仍为13:51，全部首分及来源保持不变。

| 项目 | 配置 |
|---|---|
| 固定题单 | V101：Q368、Q369、Q370，共3题 |
| 并发 | 1视频、每视频2题 |
| 继承记忆 | 主队列已校验的63/75完整窗口 |
| 音频断点 | 导入已验证的W64音频缓存，没有重新发送audio W64 |
| 输出上限 | 视觉8192、音频4096 token，均不变 |
| 当前进展 | 完整窗口仍63，visual W64请求在途，不将音频成功算作整窗完成 |
| 进程身份 | PID `55578`，start_ticks `8328001` |
| 验证 | 本地41项相关检查＋5项独立review，远端19项通过 |

prepare阶段没有API请求；真实bridge_audio测试确认缓存命中时零重复音频调用，运行账本也显示audio W64重发0次。上游曾失败的那次音频成功不等于整集完成，不预填评分，也不提供确定完成时间。

运行：`/root/longemo/runtime/runs/noevent_v101_audio_seed_continuation_20260923`。部署：`/root/longemo/runtime/staging/v101_seed_deploy_20260923_135813`。

- 驱动SHA：`08612b6b48cca7efb10440c1ba248170ff63cace0949a505aace46e066cf3e82`
- 配置SHA：`146b954fa3a709df18e092d1b3dad3234e0d7ecd594f2206d0e6d8fc5441c32e`
- 安全启动快照SHA：`5709bdddcfabf6b6b7550cd8f6379a1b40a1cf0b59325ca532f8b04718e34c8b`

[原样安全启动JSON](v101_startup.json) · [当前评分与执行状态](report.md) · [原音频探针摘要](v101_audio500_probe_summary.json)。
