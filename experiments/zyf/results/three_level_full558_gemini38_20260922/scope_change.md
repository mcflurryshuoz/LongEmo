# 后续仅评 noevent 与 method

用户在 2026-09-23 明确缩减执行范围：继续 noevent 和 method；base 使用已有成绩，不再新启动 base 作答或评分。历史 base 与当前方法若配置不同，只作为参考，不宣称同配置对照。

## 已完成与尚未完成

- 定时目标已更新为 noevent → method。
- 新调度支持不可变的 `execution_scope.json`，原实验配置、方法、首次评分、失败预算及媒体等待起点保持不变；事件阶段仅运行 method。
- 本地调度检查共 32 项通过，包括禁止新增 base、保留 base 首分、base 缺分不影响两个执行条件完成。
- **尚未部署到 AIStudio。** 最近 01:38 的远端核验显示旧协调器仍处于 noevent，event 阶段尚未启动；已有 base 13 条均为试跑继承。随后 SSH 失联，一次新隧道请求在平台 API 的 TLS 握手阶段超时。本轮没有向协调器发信号，也没有修改远端冻结文件。

## 恢复连接后的交接

1. 核验旧协调器 PID、启动时间、阶段、锁与 SIGINT 是否被忽略；保存完整原启动参数及所有首次评分清单。
2. 若仍在 noevent 且 SIGINT 可投递，仅向协调器 PID 发送一次 SIGINT，让已经提交的视频流水线完整结束。不能向进程组发信号或再次中断在途请求。
3. 核验协调器与全部任务子进程均已退出、锁已释放，保存停止审计。若状态不明确，先审计，不重启。
4. 把新协调器部署到独立目录，原 f8f9392 部署保持不动；在原 run 下冻结 `execution_scope.json`，绑定原 configuration SHA、新协调器路径及 SHA、用户缩减范围和旧进程身份。
5. 用新 `full_suite.py` 的绝对路径及原启动参数恢复，只追加 `--execution-scope <run>/execution_scope.json`；`--noevent-repo` 仍指向原部署，不能用新目录的启动脚本替换该路径。
6. 核验只出现 noevent/method 新任务，base 首分清单和原 configuration SHA 未变；报告保留 base 参考来源，但不再把 base 未评分题当作待执行任务。

新协调器 SHA256：`a8b93c69b89e179e764d90501b05bb8343f0b0e9c12a3864e5724d310ce0d175`。原配置 SHA256：`55f0c2b0bfa8c6046a203e789f516698a5a8e77b21dc081f0174e80113e61b77`。

这份记录表示执行范围与交接方案已准备，不代表远端切换已经完成。分数继续以各题首次有效官方评分为准。
