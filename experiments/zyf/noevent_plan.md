# noevent 消融方案

`noevent` 用来测量事件图本身的贡献。它保留相同的视频、20 秒核心窗口＋两侧 2 秒上下文、抽帧与音频观察、GPT-6 规划／回答、Gemini Embedding 2、问题集合和官方评分器；唯一改变是感知输出不再组织成事件、情感状态或关系。

## 1. 记录单位

每个窗口只保存一个不可变的 `window_record`：

- `window_id`、绝对 `core_span`、媒体／字幕／模型输入指纹；
- 人物索引和窗口内观察（视觉、音频、字幕、原始时间戳）；
- 一个面向检索的自然语言摘要，以及窗口内明确出现的动作、对象、评分、数量和引用；
- 不保存 `event_id`、`state_id`、`continues_event`、因果／变化关系或跨窗口合并结果。

窗口之间只按绝对时间排序。相邻窗口不表示同一事件，也不允许通过写入隐含关系来恢复事件图。

## 2. 感知协议

新增 `noevent` 感知提示词，明确禁止输出事件、状态、关系和跨窗口连续性判断，只要求描述当前窗口可观察内容。感知输入只有当前媒体和人物身份索引（ID、姓名、外观／声音描述），不提供前序窗口摘要、观察或人物的窗口出现记录。`emotion_cues` 是窗口内的情感观察，不构成持久状态节点。音频观察仍由独立模型产生，并以带时间戳的证据附在窗口记录中。每个窗口原子校验和保存；失败窗口不产生部分记录。

## 3. 检索与回答

问题规划器仍输出人物、对象、查询词和可选时间范围，但不再生成图模式或关系扩展请求。检索器对窗口摘要、观察、动作／对象和显式信号做 BM25 与 Gemini Embedding 2 双路召回，RRF 合并后返回：

1. 相关窗口的完整记录，以及所引用观察的完整正文、时间／模态和对应人物身份索引；
2. 按视频时间排序的轻量窗口索引；
3. 每条证据的窗口 ID、绝对时间和模态来源。

回答模型可以基于已披露窗口证据进行跨窗口轨迹、因果、比较与计数推理，也可判断两个观察是否属于同一持续过程；它不接收预先构建的事件、状态或关系边，不能把这些图标注当作已知事实。媒体重叠不能重复计数，缺失证据要说明不确定性。

默认预算为 **48000 字符**，约束实际发送的完整 evidence JSON（包含观察正文、人物、时间索引与统计字段）。整窗装入后才计为已披露；放不下的窗口不截断、不保留悬空证据 ID，所有候选均超预算时显式失败。问题、system 提示与协议封装开销在 trace 单独记录，模型用量按调用账本统计；多轮 method 的累计输入另计，单包预算相同不代表总推理 token 相同。

## 4. 公平比较

`method` 与 `noevent` 必须使用相同的：

- 感知模型、采样帧数、音频配置、字幕和输入视频；
- GPT-6 版本、规划／回答提示词预算和重试规则；
- Gemini Embedding 2 版本、前缀、缓存分片和 RRF 常数；
- 题单、缺失题处理、官方 judge 和评分公式。

证据包使用相同的 48000 字符上限；严格按最终序列化内容核验，不以摘要长度代替真实上下文长度。

两种方法使用不同的运行目录、记忆缓存和向量缓存。报告总体、三题型、分剧、覆盖率、窗口数、token、延迟和失败归因；只比较同题有效评分，不把未完成窗口或服务拒绝按零分计入。

## 5. 当前实现与实施顺序

1. `methods/longemo/noevent_memory.py` 已实现窗口记录 schema、绝对时间校验、模态证据校验和原子提交；输出不包含 `events`、`states`、`relations` 或 `continues_event`。
2. `methods/longemo/noevent_retrieval.py` 已实现窗口摘要／观察的 BM25 与 Gemini Embedding 2 双路召回、RRF 合并及按时间排序的全局覆盖；`noevent_runner.py` 提供独立 build/answer CLI。
3. `methods/longemo/tests/test_noevent.py` 覆盖无事件字段／前序窗口泄漏、失败原子性、完整观察与人物披露、真实 JSON 预算及超大窗口处理。
4. 下一步用与 `method` 相同的冻结媒体、采样、模型、题单和 judge 同步启动 50 题 pilot，核验窗口覆盖、证据 ID、调用账本和公平性，再扩展到 520 题，最后才考虑 558 题全集。

启动示例（凭证必须位于仓库外）：

```bash
python -m methods.longemo.noevent_runner build --data-path pilot_questions.json \
  --videos-dir episode/videos --subtitles-dir prepared_subtitles --output-dir runs/noevent/memory \
  --credential-file runtime/private/answer.json --model gpt-6-astra \
  --base-url https://matrixllm.alipay.com/v1 --with-audio \
  --audio-model google/gemini-3.8-flash --workers 4
python -m methods.longemo.noevent_runner answer --data-path pilot_questions.json \
  --memory-dir runs/noevent/memory --output-dir runs/noevent/answers \
  --credential-file runtime/private/answer.json --model gpt-6-astra \
  --base-url https://matrixllm.alipay.com/v1 --embedding-backend gemini \
  --embedding-model google/gemini-embedding-2 --evidence-chars 48000 --workers 16
```

## 6. 预期解释

如果 noevent 在轨迹题明显下降，说明跨窗口事件归并和状态变化关系有效；如果强度比较接近 method，瓶颈更可能在感知对具体动作／显式评分的捕获；如果推理题下降，说明关系和事件级去重提供了跨时间证据。结果只支持消融范围内的结论，不把窗口检索当作另一种完整事件图方法。
