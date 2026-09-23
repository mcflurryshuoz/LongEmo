# LongEmo 评测进度

2026-09-23 **16:03 CST**：noevent **243/558，25.65分**，原243条首分与来源保持不变。method保留97/558、59.45分；base保留13/558、63.46分，均不新增请求。

三个原音频429窗口各一次诊断通过并复用，但后续V100 W49音频与V91 W28音频明确DLP拒绝，分别停在48/76、27/66，不重发。V114在59/64窗遇W60视觉8192输出截断，已启动独立16384长度任务处理固定7题；当前60/64窗，原截断窗口通过，W60音频重发0次，暂无新增分数。新任务单视频、每视频2题；旧三任务均已结束并审计，原首分与尝试保持不变。

已评分汇总仍为230题原8192视觉预算＋13题继承原窗口后以16384继续构建，属于混合预算。初轮共同成功68题仍为noevent32.60、method62.13，未并入后续48题，不能据不同题集的总体均分推断全集提升。noevent缺315题，method缺461题；明确拒绝保留，不重判、不重抽低分。

[当前报告](results/noevent_resume_20260923/report.md) · [三视频结束审计](results/noevent_resume_20260923/audio429_end_audit.md) · [V114启动记录](results/noevent_resume_20260923/v114_length_startup.md) · [V101结束审计](results/noevent_resume_20260923/v101_end_audit.md) · [初轮68题比较](results/three_level_full558_gemini38_20260922/completion_audit.md)
