# 全集顺序评测

`full_suite.py` 冻结完整题单，按 **noevent → event（base / method 共图）** 顺序评测。默认校验558题／141视频；实际数据不符时停止，不静默裁剪。每个视频构建结束立即规划、回答和官方评分，无需等待其余视频完成感知。事件图只构建一次；base 与 method 共用记忆、plans 和 Embedding 缓存，依次答题和评分。

感知、音频、GPT-6 后端、Embedding、采样、每阶段尝试数均继承父 pilot 的冻结配置。新增全集脚本位于 `experiments/zyf`，不会改变只覆盖 `methods/longemo` 和 `evaluation` 的方法源码哈希；新增脚本及继承 helper 的 SHA 单独冻结。不要修改活跃 pilot 源码。

## 准备与启动

媒体 manifest 必须先覆盖全集，即使视频尚未全部上传：

```json
{
  "schema_version": 1,
  "videos": {
    "G2_V000001": {
      "video_sha256": "64位SHA256",
      "subtitles_sha256": "64位SHA256",
      "video_bytes": 43736589,
      "subtitles_bytes": 3
    }
  }
}
```

真实文件填满141视频。视频和准备好的字幕分别落在 `data/episode/videos/<id>.mp4`、`data/prepared_subtitles/<id>.json`。上传者应先写唯一临时文件、验证后原子改名；协调器发现最终文件 SHA／大小不符会保存 `media_integrity_failed`，不会当作尚未上传而反复使用。

在包含新脚本的 noevent 部署中执行；路径按实际部署填写：

```bash
cd '/root/longemo/LongEmo-noevent-full-<commit>'
export PATH=/root/longemo/runtime/bin:$PATH
common=(
  --parent-run /root/longemo/runtime/runs/three_level_pilot50_gemini38_perception_20260922
  --method-repo /root/longemo/LongEmo-method-3a6c0af
  --noevent-repo '/root/longemo/LongEmo-noevent-full-<commit>'
  --runtime /root/longemo/runtime
  --run-name three_level_full558_gemini38_20260922
  --questions /root/longemo/runtime/data/full558_20260922/questions.json
  --media-manifest /root/longemo/runtime/data/full558_20260922/media_manifest.json
  --producer-done /root/longemo/runtime/transfers/full558_20260922/producer_done.json
  --credential-file /root/longemo/runtime/private/answer.json
  --python /opt/conda/bin/python
  --workers 12 --question-workers 2
  --parent-wait-timeout 86400 --media-wait-timeout 86400
)

# 仅冻结数据、代码和配置，可与父pilot及媒体上传同时进行，不调用API。
/opt/conda/bin/python -m experiments.zyf.full_suite prepare "${common[@]}"

# 父pilot仍活跃时只等待；父进程/锁核验结束且继承审计通过后才发出新请求。
nohup /opt/conda/bin/python -u -m experiments.zyf.full_suite run "${common[@]}" \
  >> /root/longemo/runtime/three_level_full558_gemini38_20260922.log 2>&1 &
```

`--workers` 为当前一种感知表示的逐视频流水并发，默认12；每视频问答／评分最多2路。同一时刻不同时扩展 noevent 和 event 队列。外部接口总并发仍包括该阶段的感知、规划、回答及评分请求，不能把视频数当成总请求数。

生产者结束后原子写入标记，SHA针对上述完整 manifest 文件原始字节：

```json
{"complete": true, "media_manifest_sha256": "media_manifest.json的SHA256"}
```

缺视频／字幕时不会占用 API worker，也不会立即永久失败；已到且校验通过的视频可以先处理。只有该标记有效或冻结等待时间耗尽，仍缺文件的视频才记录 `media_missing`。每阶段等待起点持久化，重启不重置等待预算。阶段失败、缺失仍进入报告，不能把队列结束说成558题成功。

也可在新部署直接使用 `bash experiments/zyf/run_full_suite.sh prepare`，随后后台启动 `nohup bash experiments/zyf/run_full_suite.sh run >> /root/longemo/runtime/three_level_full558_gemini38_20260922.log 2>&1 &`。脚本默认使用上述全集元数据、运行名称、12个视频 worker 和24小时媒体等待；路径与并发可通过脚本中的 `LONGEMO_*` 环境变量覆盖。

## 父结果与首分保护

父协调器、锁及所有已记录子进程须证明已退出。取锁后的父配置、题单、源码、媒体哈希、完整记忆及窗口产物、计划、预测、首评逐一校验并复制到新运行 `inheritance/parent`；快照记录所有文件 SHA，原目录不修改。重入时父文件变化会报错。等待父运行结束后，以及每个新阶段启动前，重新核验新部署的方法源码、协调器和继承工具 SHA，防止等待期间代码漂移。未能确认结果的旧尝试保留阻断，不重置预算。

完整同配置记忆、成功 plans 和 Embedding 缓存复用。父pilot只含某视频的部分问题，因此计划、成功预测及首分逐题继承；旧 answer manifest 保留为历史，新增问题使用独立题单和目录。成功预测未评分时直接提交首次判断，不再回答。某题旧规划／回答／评分失败仅锁定该题对应阶段，不影响同视频的新题；旧构建失败阻断该表示的视频，不能为其新问题再次触发同一失败构建。

父 Embedding 文档请求若仍失败或结果不明，相关继承视频的新答题保留 `embedding_guard` 阻断，避免新增问题重置共享索引请求预算。账本无法精确映射视频时，保守限制在父运行该表示已经尝试过答题的旧视频；未尝试的新视频继续执行，已有成功答案仍可首次评分。

感知／规划／回答保持父配置的有限尝试数，官方 judge 固定一次。确认所有旧子进程停止后，恢复流程会离线导入 scorer 已写盘但尚未归档的有效首分，task 的不明状态仍保留，不重新调用 API。新运行中已启动过的失败或不明 task 不自动重开，0分与任何其他首分同样保留；没有外层无限恢复。每题首次成功答案与评分来源可在继承索引、各阶段题单、调用账本及 `accepted/<condition>/<qid>.json` 追溯。

## 进度

`status.json` 每个视频及条件评分结束更新，包含三条件覆盖和已评分均分、两阶段逐视频状态。完整输出：

- `configuration.json`、`questions.json`、`media_manifest.json`：冻结全集和配置。
- `parent_status.json`、`inheritance/parent_index.json`：父运行等待与继承证据。
- `tasks/<stage>/<condition-video>/task.json`：子进程 PID、start ticks、命令和一次执行结果。
- `stage_questions/`：本次实际调用的规划／答题／首评问题子集。
- `noevent/base/method/accepted_scores.jsonl`：父首分与新首分汇总；缺失不计零分。
- `pipelines/`、`phases/`：逐视频流水状态与阶段结束标记。

运行中直接读取这些文件；停止后可用相同参数执行 `status`。再次执行 `run` 只推进从未启动的任务，活跃或身份未知的旧子进程会阻止新请求；不要删除 task 文件、重写首分或修改配置来恢复。

媒体中转工具为 `transfer_full_media.py`（本地单进程）和 `install_full_media.py`（远端逐文件校验及发布）。使用已验证的原生 SSH 隧道，动态端口来自本地配置；已有文件 SHA 相同才跳过。每次 SCP 使用唯一临时名，完成后原子发布最终视频；只有全部141视频和字幕通过实际校验，才发布上述 producer_done。认证失败立即停止；明确网络错误最多3次。
