# LongEmoBench

[English](README.md)

本目录提供统一开放问答格式的推理与评估代码。视频、纯字幕文本与纯音频推理均支持 `clip` 和 `episode`，须使用支持对应输入的模型服务。第二粒度视频使用覆盖完整视频的采样帧。评估同时接收基础模型和外部方法（包括 agent）的预测。

目录只包含代码、提示词和合成测试；不包含题库、视频、字幕、审核平台或历史实验结果。

## 代码组织

```text
evaluation/
  io_utils.py           # 题目读取与筛选、结果保存
  clients.py            # 共用 HTTP 发送与请求校验
  judge_prompts.py      # 评估提示词与内嵌四个示例
  metrics.py            # 标签指标与成绩汇总
  eval.py               # 评估入口
  inference/
    run.py              # 视频／文本／音频推理统一入口
    runner.py           # 运行参数、断点续跑、重试
    prompts.py          # 推理提示词与回答解析
    video_loader.py     # 读取视频或采样带时间戳的帧
    adapters/
      __init__.py       # 协议选择与请求配置
      openai.py         # Chat Completions／Responses 编码与解析
      anthropic.py      # Messages 编码与解析
      gemini.py         # generateContent 编码与解析
      model_settings.py # 各模型的生成参数差异
      message_content.py # 共用文本、图片和音频内容检查
      qwen_audio.py     # Qwen2-Audio 有序音频切分
      affectgpt.py      # AffectGPT 独立入口
      emotion_llama.py  # Emotion-LLaMA 独立入口
      r1_omni.py        # R1-Omni 独立入口
      emollm_runner.py  # 情绪模型共用的批处理辅助函数
    emollm/
      AffectGPT/        # AffectGPT 官方源码
      Emotion-LLaMA/    # Emotion-LLaMA 官方源码
      R1-Omni/          # R1-Omni 官方源码
      sources.json      # 上游版本与来源信息
      README.md         # 情绪模型配置说明
subtitles/              # G2 字幕：<video_id>.json
docs/
  data_format.md        # 数据与预测格式
  model_input_limits.md # 模型输入、容量与适配支持情况
tests/                  # 合成数据与本地接口测试
```

数据加载负责读取已准备好的题目，并按粒度、题号和数量筛选。四个 few-shot 直接写入 `judge_prompts.py` 的评估 System Prompt。

推理代码按以下顺序阅读：

1. [run.py](evaluation/inference/run.py)：命令行参数、数据选择和推理主流程。
2. [runner.py](evaluation/inference/runner.py) 与 [adapters/__init__.py](evaluation/inference/adapters/__init__.py)：服务配置和协议选择；模型参数差异集中在 `model_settings.py`。
3. [prompts.py](evaluation/inference/prompts.py) 与 [video_loader.py](evaluation/inference/video_loader.py)：题目消息和媒体输入的组装。
4. [clients.py](evaluation/clients.py)：共用请求校验和 HTTP 发送；`openai.py`、`anthropic.py`、`gemini.py` 负责对应协议的请求编码与响应解析。


## 环境与数据

Python 3.10 及以上，Python 运行时代码仅使用标准库。模型由独立服务提供，脚本不加载模型权重。抽帧、音轨提取和原生视频静音处理要求 `PATH` 中可用的 `ffmpeg` 和 `ffprobe`。纯字幕输入无需这两个工具；直接转发未经处理的原生视频无需解码媒体。以下命令在本目录执行；也可通过 `python -m pip install .` 安装 `longemobench-infer`、`longemobench-infer-transformers` 和 `longemobench-score` 命令。

每次运行通过 `-g` / `--granularity` 选择一个粒度，两个粒度独立保存数据与成绩。`--data-path` 接受包含题目对象或数组的 JSON 文件、JSONL 文件，或平铺的 JSON/JSONL 题目目录；视频按 `<video_id>.mp4` 命名。省略 `--videos-dir` 时，默认使用 `--data-path` 目录下的 `videos/`；若传入题目文件，则使用该文件所在目录下的 `videos/`。显式指定 `--videos-dir` 可覆盖默认值。输入须符合[统一数据格式](docs/data_format.md)，并包含 `answer_details` 字段。

## 推理

`--model` 填写服务中的实际模型名称，程序自动识别模型系列和输入形式。Qwen、Gemini、GPT、Claude、GLM、Doubao-Seed、InternVL 和 DeepSeek 共用同一个推理入口，实际模态依具体型号确定。

| 模型系列 | 接口与服务 | clip 默认视频输入 | 音频 |
|---|---|---|---|
| Qwen-VL | Chat Completions，须显式提供服务地址 | `video_url` 原生 MP4 | 仅输入画面，移除音轨。 |
| Qwen-Omni | Chat Completions，须显式提供服务地址 | `video_url` 原生 MP4 | 本地 Qwen3-Omni-30B 分别传入视频与音轨；托管接口按其支持的模态组合处理。 |
| Gemini | 原生 Gemini，`https://generativelanguage.googleapis.com/v1beta` | 内联 MP4 | 音频随视频传入。 |
| GPT | Responses，`https://api.openai.com/v1` | 均匀采样并带时间戳的视频帧 | 不发送音频。 |
| Claude | Anthropic Messages，`https://api.anthropic.com/v1` | 均匀采样并带时间戳的视频帧 | 不发送音频。 |
| GLM 视觉型号 | Chat Completions，服务地址见下表 | 带时间位置的图片帧 | 不发送音频。 |
| Doubao-Seed 视觉型号 | Chat Completions，服务地址见下表 | 默认原生视频；`--sample-frames` 时使用带时间位置的图片帧 | 仅已确认支持音频的具体版本保留音轨。 |
| InternVL3.5 | OpenAI-compatible 服务，须显式提供地址 | 带时间位置的图片帧 | 不发送音频。 |
| 支持视觉的 DeepSeek Flash | Chat Completions，服务地址见下表 | 带时间位置的图片帧 | 不发送音频。 |
| Qwen2-Audio-7B-Instruct | OpenAI-compatible 服务，须显式提供地址 | 使用 `--modality audio`，不支持图片 | 从视频提取并按时间排列的音频段。 |

各模型支持的媒体组合、接口与上下文限制，以及官方能力与当前适配器的差异，见[模型输入能力与限制](docs/model_input_limits.md)。

用 `--base-url` 或 `MODEL_BASE_URL` 指定本地或托管服务。已配置官方地址的模型可省略 URL；Qwen 权重、InternVL 及未知模型名须显式提供服务地址，不隐式连接本地服务。

[runner.py](evaluation/inference/runner.py) 中，`setup_api()` 配置模型系列、服务地址和密钥，并调用 `adapters/__init__.py` 的 `prepare_request()`。协议适配器按接口组织：`openai.py` 处理 Chat Completions 和 Responses，`anthropic.py` 处理 Messages，`gemini.py` 处理 generateContent。GPT、Qwen、GLM、Seed、DeepSeek 和 Gemini 的参数差异集中在 `model_settings.py`；`qwen_audio.py` 仅负责音频切分。

推理入口统一构造 `messages`。[clients.py](evaluation/clients.py) 将请求编码和响应解析委托给对应适配器，负责共用 HTTP 发送与校验。完整端点按原地址发送，各模型系列共用这些协议实现。官方地址集中定义于 `OFFICIAL_API_URLS`。另配置以下 Chat Completions 服务：

| 服务 | 官方 base URL | 选用方式 |
|---|---|---|
| Qwen 托管 API | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 通过 `--base-url` 显式指定；Qwen 权重名称没有默认服务地址。 |
| MiniMax | `https://api.minimax.io/v1` | MiniMax 模型名默认使用；国内地址为 `https://api.minimaxi.com/v1`。 |
| Kimi | `https://api.moonshot.ai/v1` | Kimi／Moonshot 模型名默认使用；国内地址为 `https://api.moonshot.cn/v1`。 |
| GLM | `https://open.bigmodel.cn/api/paas/v4` | GLM 模型名默认使用。 |
| Doubao-Seed | `https://ark.cn-beijing.volces.com/api/v3` | Seed／Doubao 模型名默认使用；视频默认传原生视频，使用 `--sample-frames` 才切换为抽帧。 |
| DeepSeek | `https://api.deepseek.com/v1` | DeepSeek 模型名默认使用。 |
| OpenRouter | `https://openrouter.ai/api/v1` | 显式指定 `--base-url`，模型名填写完整的 `provider/model` ID。 |

显式指定的 `--base-url` 优先。MiniMax、Kimi 的地址和协议路由不代表所有模型的媒体输入与生成参数均已验证。MiniMax 官方文本 API 不接受本脚本默认的温度 `0`，运行时需指定 `--temperature 1`；Kimi 参数要求依具体模型确定。[Qwen 地址说明](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/base-url)、[MiniMax API](https://platform.minimax.io/docs/api-reference/text-openai-api)、[MiniMax 区域地址](https://platform.minimax.io/docs/token-plan/cursor)、[Kimi 国际 API](https://platform.kimi.ai/docs/overview)、[Kimi 国内 API](https://platform.kimi.com/docs/get-api-key)。

通过 `--api-key API_KEY` 传入密钥。除 OpenRouter 外，省略时优先使用 `MODEL_API_KEY`，否则按模型系列选择标准密钥环境变量，未匹配系列时按接口格式选择（`GEMINI_API_KEY`、`ANTHROPIC_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`、`MINIMAX_API_KEY`、`MOONSHOT_API_KEY`、`ZHIPUAI_API_KEY`、`ARK_API_KEY`、`DEEPSEEK_API_KEY`）。运行配置不保存密钥。

OpenRouter 沿用 `evaluation.inference.run` 推理入口和 Chat Completions 适配器。通过 `OPENROUTER_API_KEY` 单独提供其密钥；该变量优先于通用的 `MODEL_API_KEY`，不复用各厂商或其他中转服务的密钥。显式提供的 `--api-key` 仍然优先。[OpenRouter 接入说明](https://openrouter.ai/docs/quickstart)。

GLM、Kimi、MiniMax 的 OpenRouter Debug 配置使用以下支持图片输入的型号，并显式指定 `--sample-frames`：

| 调试型号 | 完整模型 ID | 目录声明的输入模态 |
|---|---|---|
| [GLM 5.3 Flash](https://openrouter.ai/z-ai/glm-5.3-flash) | `z-ai/glm-5.3-flash` | 文本、图片、视频 |
| [Kimi K3](https://openrouter.ai/moonshotai/kimi-k3) | `moonshotai/kimi-k3` | 文本、图片、视频 |
| [MiniMax M3](https://openrouter.ai/minimax/minimax-m3) | `minimax/minimax-m3` | 文本、图片、视频 |

三条路径均以正式 urllib 客户端成功返回 HTTP 200。Debug 采用本轮成功样本中的最高配置：

| OpenRouter 调试型号 | FPS | 最大帧数 | 单帧像素上限 | 总像素上限 | 实测尺寸 |
|---|---:|---:|---:|---:|---|
| GLM | 2 | 64 | 100,352 | 6,422,528 | 420×224 |
| Kimi | 2 | 32 | 100,352 | 3,211,264 | 420×224 |
| MiniMax | 2 | 32 | 100,352 | 3,211,264 | 420×224 |

这些值仅覆盖 OpenRouter Debug 配置，不是厂商上限。更大请求遇到传输失败，最大容量与重复运行稳定性尚未确认；GLM 的 64 帧请求曾发生 TLS 错误，相同配置再次单独发送后成功。Kimi 与 MiniMax 后续均跑通 32 帧，40／48／64 帧测试在连接、上传或等待响应时失败，未收到模型拒绝。测试中的“直连”仅表示关闭显式 HTTP 代理，不代表已绕过 VPN／TUN。现有 Debug 已填入提供的 OpenRouter 密钥；命令行仍可使用 `OPENROUTER_API_KEY`。[实测记录](docs/model_input_limits.md#45-openrouter-实测)。

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
python -m evaluation.inference.run --data-path /dataset/clip \
  --model z-ai/glm-5.3-flash --base-url https://openrouter.ai/api/v1 \
  --sample-frames --max-frames 64 --frame-max-pixels 100352 --total-pixels 6422528
```

### 第一粒度视频推理

```bash
common=(--data-path /dataset/clip/questions -g clip
        --modality video --videos-dir /dataset/clip/videos)

python evaluation/inference/run.py "${common[@]}" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/video/qwen3-vl

python evaluation/inference/run.py "${common[@]}" \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/video/qwen3-omni

python evaluation/inference/run.py "${common[@]}" \
  --model gemini-3-flash-preview --api-key API_KEY \
  --output-dir output/clip/video/gemini3-flash

python evaluation/inference/run.py "${common[@]}" \
  --model GPT_MODEL_NAME --base-url https://api.openai.com/v1 --api-key API_KEY \
  --output-dir output/clip/video/gpt

python evaluation/inference/run.py "${common[@]}" \
  --model CLAUDE_MODEL_NAME --base-url https://api.anthropic.com/v1 --api-key API_KEY \
  --output-dir output/clip/video/claude
```

将 `GPT_MODEL_NAME`、`CLAUDE_MODEL_NAME` 替换为账户可用且支持图片输入的实际模型名，`API_KEY` 替换为服务密钥。Qwen 示例要求指定地址已部署对应权重；请将 `--base-url` 替换为实际服务地址。

视频推理**默认不附加字幕**，使用 `--with-subtitle` 开启，字幕位于媒体之后、问题之前。G1 读取题目 JSON 的 `subtitles` 字段；G2 读取与 `evaluation/` 同级的 `subtitles/<video_id>.json`，每个文件保存该视频的字幕列表。路径按代码所在目录定位，不受终端工作目录影响，无需额外指定字幕路径。纯文本推理要求字幕非空；G1 视频推理没有可用字幕时不添加字幕区块。

`clip` 默认按模型和接口选择原生视频或抽帧输入；Seed 默认使用原生视频；`episode` 视频模式统一使用抽帧。添加 `--sample-frames` 可使 `clip` 也使用抽帧。JPEG 帧在完整视频范围内均匀采样，并附带时间戳。默认设置如下：

| 模型系列 | FPS | 最大帧数 | 单帧像素上限 | 采样像素总量上限 |
|---|---:|---:|---:|---:|
| GPT | 1 | 128 | 262,144 | 33,554,432 |
| Claude | 2 | 80 | 262,144 | 8,028,160 |
| DeepSeek | 2 | 600 | 602,112 | 361,267,200 |
| 其他模型 | 2 | 768 | 602,112 | 下述 Qwen 预算公式 |

显式 CLI 参数可覆盖默认值。实际帧数还受集中配置的模型上限约束：GPT 为 1500，Claude、DeepSeek 为 600，Seed 为 1280。默认值不等同于所有服务的输入上限。先按 `round(视频时长 × fps)` 计算帧数，再由最大帧数、源视频可用帧数和像素预算限制；选定数量后，在完整视频范围内均匀采样。以上参数仅用于抽帧输入，原生视频由服务端解码。官方来源与 API 实测见[模型输入限制](docs/model_input_limits.md)。

对于 10 秒、120 秒和 600 秒视频，GPT 默认分别约取 10、120 和 128 帧；Claude 分别约取 20、80 和 80 帧；DeepSeek 分别约取 20、240 和 600 帧；其他模型未应用服务专属上限时分别约取 20、240 和 768 帧。结果中的 `media` 保存目标、源视频及实际平均采样帧率，以及帧数和实际时间戳。

DeepSeek 官方 API 已接收 600 帧、每帧 1008×560 的输入，请求体为 46.583 MiB，返回 HTTP 200；请求模型为 `deepseek-v4-flash`，响应模型名称为 `deepseek-flash`。Claude `claude-sonnet-4-6` 通过当前中转成功处理 80 帧、每帧 420×224：请求体 1.915 MiB，HTTP 200，用时 90.81 秒。同一请求复测时连接中断，更大配置发生传输失败或 HTTP 524。默认 80 帧是本轮成功样本的最高数量，不保证重复运行稳定，也不替代官方 600 张图片上限。这些结果不代表答案准确率已验证。[实测说明](docs/model_input_limits.md)。

采样采用 [Qwen-Omni 的 FPS 与最大帧数联合控制思路](https://github.com/QwenLM/Qwen2.5-Omni/blob/d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py#L149)，由现有 FFmpeg 解码器提取。通用 HTTP 输入允许单帧，不要求 Qwen 模型专用的最小帧数或偶数帧对齐。派生帧为临时文件，原 MP4 保持不变。原生视频始终保持原有音轨；抽帧输入默认不带音频，只有提供 `--with-audio` 才另传音轨。降低图片采样密度不会截短源音轨。

分辨率按像素面积控制，不固定最长边或正方形尺寸：当前宽屏样本使用 GPT 默认值时生成 672×364 图片；Claude 达到 80 帧时生成 420×224 图片；DeepSeek 默认生成 1008×560 图片。Claude 的短视频采样帧数较少时，可分配更高的单帧预算，最高为 262,144 像素。解码器近似保持宽高比例，并将宽高按 28 的倍数对齐，这是本地预处理方式。GPT、Claude 和 DeepSeek 的总预算严格限制所有采样图片的像素总和；增加帧数会降低单帧预算，到达 100,352 像素的最低预算后会减少帧数。`--total-pixels` 可覆盖总预算。像素限制不保证固定的 JPEG／请求体大小，也不保证识别准确率。

其他模型保留 Qwen-Omni 工具默认值：每帧最少 100,352、最多 602,112 像素，视频预算参数为 90,316,800。未指定 `--total-pixels` 时，Qwen 原公式将该预算乘以 2 后分配到各帧；触及最小尺寸后继续增加帧数会增加总像素量。小图可能放大，最小尺寸与网格取整也可能使此默认模式下的实际像素数略超较紧的单帧目标。单独降低 `--frame-max-pixels` 不会增加帧数。结果记录请求与实际预算、缩放后的宽高及实际像素总和。

原生视频始终直接发送原文件，包括其中的内置音轨；抽帧输入默认不带音频，只有提供 `--with-audio` 才另传音轨。

抽帧需要音频时添加 `--with-audio` 另传音轨；原生视频始终保留原有音轨。此开关与字幕开关及模型文字输出设置相互独立。

抽帧模式下的独立音轨统一提取为 16 kHz、单声道、PCM16 WAV，与视频帧使用相同的时间起点和范围；音轨晚于画面开始时保留其延迟。媒体放在字幕和问题之前。派生文件临时生成，原视频保持不变。`media.audio` 与 `media.audio_details` 记录实际采用的音频形式及提取参数。原生模式直接发送包含内置音轨的原视频。

```bash
# 在原生 Gemini 或支持组合输入的 Qwen 推理命令后添加：
--sample-frames
# 抽帧并另传音频：
--sample-frames --with-audio
```

DashScope Qwen3-Omni-Flash 支持有声视频，但不接受图片加独立音频；当前 GPT Responses、Claude Messages 路径只传画面。本地 Qwen3-Omni-30B 的 vLLM `video_url` 路径不启用 `use_audio_in_video`，因此另传音轨。[Qwen 服务示例](https://github.com/QwenLM/Qwen3-Omni#vllm-serve-usage)、[DashScope 模态组合](https://help.aliyun.com/en/model-studio/qwen-omni)、[Gemini 多模态内容](https://ai.google.dev/api/generate-content#Content)。

原生输入将已准备的 MP4 编码为 Base64。DashScope 要求编码后的视频字符串小于 10 MB，托管的 `qwen3-omni-flash` 还要求单个视频不超过 150 秒。Gemini 内联输入适用于较小片段；当前实现不通过 Gemini Files API 上传视频。可根据实验需要选择抽帧输入，原生请求须满足服务端限制。[Qwen 限制](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)、[Gemini 视频输入](https://ai.google.dev/gemini-api/docs/video-understanding)。

### 第二粒度视频推理

```bash
python evaluation/inference/run.py \
  --data-path /dataset/episode/questions -g episode --modality video \
  --videos-dir /dataset/episode/videos \
  --model gemini-3-flash-preview \
  --output-dir output/episode/video/gemini3-flash
```

第二粒度沿用同一入口，自动使用带时间戳的采样帧，无需添加 `--sample-frames`。回答要求由题目指定，不额外添加 G2 类别模板。抽帧需要音频时使用 `--with-audio`；可选字幕通过 `--with-subtitle` 启用，读取前述 G2 字幕文件。

第二粒度沿用上述模型默认值。对于 20 分钟和 40 分钟视频，GPT 的 128 帧分别约为 0.107 FPS 和 0.053 FPS；Claude 的 80 帧分别约为 0.067 FPS 和 0.033 FPS；DeepSeek 的 600 帧分别约为 0.5 FPS 和 0.25 FPS；其他模型默认最多 768 帧，未应用服务专属上限时分别约为 0.64 FPS 和 0.32 FPS。帧在完整视频范围内均匀分布。抽帧可能遗漏短暂表情或反应；完整请求仍须满足服务的体积、时长和上下文限制。采样与像素预算应按实验调整，请求成功不代表识别准确率已验证。

向方舟或 DeepSeek 官方端点发送前，客户端检查包含内联媒体在内的实际 JSON 编码字节数：方舟上限为 64,000,000 字节（将官方 64 MB 按十进制保守处理），DeepSeek 为 48 × 1024 × 1024 字节（48 MiB）。超限时在本地报错，提示调低 `--max-frames` 或 `--frame-max-pixels`；不调用 API，也不截掉视频后半部分。上下文及其他服务限制仍需满足。

### 两个粒度的纯字幕推理

```bash
python evaluation/inference/run.py \
  --data-path /dataset/episode/questions \
  --granularity episode --modality text \
  --model TEXT_MODEL_NAME \
  --base-url http://localhost:8000/v1 \
  --output-dir output/episode/text/model
```

第一粒度改用 `--granularity clip`。文本模式始终输入字幕，无需 `--with-subtitle`；只发送按原顺序排列的字幕文本和问题，不发送视频、说话人标注、字幕编号或时间戳。字幕缺失或为空时记录输入错误。代码不自动截断字幕，输入容量由模型服务决定。

文本模式直接通过 `prompts.text_messages()` 构造消息，与视频、音频共用四种接口传输。第二粒度视频流程使用采样帧；其他长视频方法也可自行推理后提交预测文件。

### InternVL 与纯音频推理

`OpenGVLab/InternVL3_5-241B-A28B` 使用 vLLM 或 LMDeploy 单独部署后，由通用入口调用，无需再放入 `emollm` 建立独立入口。模型名须与部署服务实际提供的名称一致：

```bash
python evaluation/inference/run.py \
  --data-path /dataset/clip/questions -g clip --modality video \
  --model OpenGVLab/InternVL3_5-241B-A28B \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/video/internvl3.5

python evaluation/inference/run.py \
  --data-path /dataset/clip/questions -g clip --modality audio \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/audio/qwen2-audio
```

音频模式沿用 `--videos-dir` 的目录规则，从对应视频提取音轨，不发送画面。Qwen2-Audio 的音轨按不超过 30 秒分段，全部音频段与时间位置放入同一个请求，由模型一次回答。vLLM 的 `--limit-mm-per-prompt '{"audio": N}'` 须允许至少最长输入的分段数；上下文或接口容量不足时仍会报错，不丢弃后续音轨。原始 `Qwen-Audio-Chat` 自定义代码模型不在本次通用适配范围内。原生 Gemini 和已支持的本地 Qwen-Omni 也可使用音频模式。[Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)、[InternVL 部署说明](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B#deployment)。

GLM、Seed 视频推理须选支持视觉输入的具体型号。核对日期的 `deepseek-flash` 支持图片，已完成上述官方 API 实测；`deepseek-v4-pro` 为文本型号。其他新增接入除明确记录的实测外，仅完成本地请求检查，未下载权重或执行 GPU 推理。InternVL 服务端还须允许所配置的图片项数量。[具体能力与验证范围](docs/model_input_limits.md)。

### 共用设置

通用指令和回答格式模板统一使用英文，题目内容按原文传入。

| 参数 | 用途 |
|---|---|
| `--modality video\|text\|audio` | 视频、纯字幕或从视频提取的纯音频输入，默认 `video`。 |
| `--api-key KEY` | 所选模型服务的密钥。 |
| `--prompt-mode system` | 独立发送通用指令，视频、文本和音频模式均默认使用。 |
| `--prompt-mode user` | 将相同通用指令放在 User 消息中。 |
| `--qid ID` | 选择题号，可重复指定。 |
| `--limit N` | 先按粒度和题号筛选，再取前 N 题；未指定时运行全部筛选结果。 |
| `--output-dir DIR` | 推理结果的输出目录。 |
| `--force` | 重新推理所有选中记录，包括已有成功预测的题目。 |
| `--workers N` | 并发数，默认 1。 |
| `--tries N` | 每次请求最多尝试次数，默认 3。 |
| `--max-tokens N` | 输出长度上限，默认 16384（16K）。 |
| `--temperature N` | 采样温度，默认 `0`；官方 Seed 2.0 Pro／Lite 260215 接口使用其固定值 `1`。 |
| `--timeout N` | 请求超时秒数，默认 180。 |
| `--sample-frames` | 强制第一粒度视频使用抽帧；第二粒度视频始终使用抽帧。 |
| `--with-audio` | 抽帧时附加独立音轨。 |
| `--fps N` | 目标采样帧率，须为有限正数，模型默认值见上表。 |
| `--max-frames N` | 请求的正整数帧数上限，受模型及像素预算限制，默认值见上表。 |
| `--frame-max-pixels N` | 每帧像素面积上限，最低 100,352，实际还受像素总预算限制。 |
| `--total-pixels N` | 采样图片像素总和的严格上限；GPT、Claude、DeepSeek 默认值见上表，其他模型省略时沿用 Qwen 预算公式。 |

快速测试时，在上面的推理命令中添加 `--limit 10 --output-dir output/debug/test`。目录输入按文件名排序读取，JSON 数组和 JSONL 按记录原顺序读取。

普通运行无需额外配置文件。需要传入个别服务附加参数时，可用 `--config FILE` 指定 JSON 文件。

`--thinking default` 沿用服务的思考设置。Qwen3-VL Plus／Flash 和 Qwen3-Omni Flash 支持 `on`／`off`；Instruct 与 Thinking 使用不同权重，不兼容的开关会报错。Qwen-Omni 默认只输出文本。通用入口统一采用非流式请求，一次接收完整 JSON 响应，不提供流式开关；仅支持流式响应的服务不适用于此入口。Gemini 3 的 `on` 映射为 high，`off` 会报错；Flash 可在 `--config` 中设置 minimal，但这不保证完全停止思考。自定义 GPT／Claude 推理参数也通过 `--config` 设置，并保留 `--thinking default`。

`predictions.jsonl` 保存预测、原始回答、输入关联与状态。视频记录包含输入形式与音频状态，抽帧记录还包含采样设置和时间戳。推理会复用同一问题编号的成功记录；添加 `--force` 后重新运行所有选中记录，题目范围仍由 `--qid` 和 `--limit` 控制。参考答案、答案细节和 rubric 不进入被测模型输入。

## 评估

三个情绪模型的官方源码统一放在 `evaluation/inference/emollm/`，独立入口和通用模型适配代码统一放在 `evaluation/inference/adapters/`。各模型使用独立依赖环境，权重、配置及命令见[官方模型接入说明](evaluation/inference/emollm/README.md)。公共 `run.py` 仍用于通用模型服务。官方源码保持原样，外层适配器负责题目输入与预测输出。安装包不包含上游源码树或权重；使用安装后的适配器时，通过 `repo_dir` 指向已准备的官方源码目录。

```bash
python evaluation/eval.py \
  --data-path /dataset/clip/questions --granularity clip \
  --predictions output/clip/video/model/predictions.jsonl \
  --model JUDGE_MODEL_NAME --api-key API_KEY \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/scores/model
```

每次启动评估均处理 `--data-path` 中所选粒度的全部题目，不按题号筛选。

是否提供 `rubric` 决定评分路径：

- **`rubric` 为 `null`：**代码计算标签 Precision、Recall、F1、EM。Transition 前后分别评分后取平均。重复标签去重，词表外标签保留为错误预测。
- **提供 `rubric`：**由 LLM 根据本题的参考答案、`answer_details` 和 rubric 评价预测回答，允许的分值直接读取本题 rubric。

标注时，所有需要 LLM 评分的题目均须填写 rubric，包括 G2 的解释题和结果题；仅标签题填写 `null`。

裁判输入为问题、参考答案及 `answer_details`、模型回答和 rubric。`answer_details` 说明必要内容或其他可接受答案，本身不分配分值。四个示例已放入 System Prompt，输出仅含 `score` 和 `reason`。

仅评估标签题时不创建 API 客户端，无需 `--model`、`--base-url` 或密钥；只有需要 LLM 评分时才配置裁判客户端。AffectGPT、Emotion-LLaMA 和 R1-Omni 的独立入口已输出统一预测格式；其他外部方法或 agent 的预测至少包含：

```json
{"question_id": "Q000001", "prediction": "3"}
```

预测仅通过 `question_id` 对应 `--granularity` 筛选后的题库；预测中的 `granularity`、`video_id`、`status`、`error` 均忽略。优先读取非 `null` 且非空白的 `prediction`，否则读取 `pred_answer`。开放答案使用文本；标签答案支持列表，Transition 支持 `before`／`after` 两组标签。完整约定见[数据格式](docs/data_format.md)。

### 成绩与输出

每次评估均在 `--output-dir` 下新建以时间戳命名的子目录，保存以下文件，并打印实际保存位置。历次评估结果分别保留；评估始终重新处理所选粒度的全部题目，无需 `--force`。

| 文件 | 内容 |
|---|---|
| `scores.jsonl` | 逐题成绩、评估输入、原始裁判回答及错误。 |
| `metrics.json` | 按任务 `type` 汇总的完整指标及评分覆盖率。 |
| `summary.md` | 便于阅读的各任务 `type` 成绩。 |

标签指标逐题平均；轨迹同时报告 0–4 原始平均分、除以 4 的归一化成绩和乘以 25 的百分制成绩；原因解释报告 0–3 原始平均分及归一化、百分制成绩。结果题报告 ACC。

G2 情绪推理类别内保留 0/1 结果题与多档解释题的单独成绩，并按既定权重计算 `0.6 × 结果题 ACC + 0.4 × 解释题归一化平均分`。每道 LLM 评分题按本题 rubric 的最高分归一化；不加权成绩对全部有效题目等权平均。只有一部分有有效成绩时使用该部分成绩；无成绩为 `null`。整体不加权成绩使用标签 F1 与归一化后的 LLM 分数。

报告直接按题目的 `type` 分组，不另设答案类型分组或映射文件，也不根据分数档位改变题目类别。

两个答案字段都没有内容时不参与评分，包括字段缺失、`null`、空字符串或仅含空白字符的情况。缺失预测与评估失败单独报告状态与覆盖率，不计为答错。裁判格式错误或分数越界先重试，再记录评估异常。存在未完成题目时返回非零退出码。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用合成题目和本地 HTTP 服务，验证输入组装、回答解析、评分分流、成绩汇总、错误处理和重复运行。不调用真实模型服务，也不替代正式题库核验；实际运行时需确认所用服务支持对应的模型参数与媒体输入。

### Transformers 本地推理

对于直接加载 Qwen checkpoint 的 Transformers 推理，使用独立入口，不经过 HTTP 部署入口：

```bash
python -m evaluation.inference.transformers \
  --data-path /path/to/questions \
  --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --videos-dir /path/to/videos \
  --output-dir output/transformers
```

该入口会根据模型名选择原生处理方式：Qwen2.5/Qwen3 Omni 使用
`qwen_omni_utils`，默认开启 `use_audio_in_video=True`；Qwen2/Qwen2.5/Qwen3 VL
使用 `qwen_vl_utils` 处理原生视频；Qwen2-Audio 从视频提取临时 WAV 后交给官方
音频处理器；InternVL 使用官方的抽帧和 `model.chat` 流程。它与
`evaluation.inference.run` 的 HTTP/API 入口分开。运行环境需
安装所选 checkpoint 官方要求的 `torch`、`transformers` 以及对应的媒体处理工具。
