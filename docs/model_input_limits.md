# 模型输入能力与容量限制

核对日期：2026-09-10；GPT、Claude、DeepSeek 及 OpenRouter 请求探测与采样配置更新：2026-09-12。

本文记录 LongEmoBench 推理中涉及的通用模型、相关音频接口及三个专用情绪模型，说明输入形式、容量限制和当前代码的适配范围。官方规格、预算估算和 API 实测分别记录；当前请求测试见第 4 章。

## 1. 输入形式与容量的含义

| 项目 | 含义 |
|---|---|
| 视频文件 | 向视频接口提交 MP4 等文件，由服务或模型处理器解码、采样。是否处理音轨取决于具体模型及接口。 |
| 独立图片帧 | 抽帧后，将每帧作为独立图片发送，并附时间位置。当前通用入口的 `frames` 使用这种形式。 |
| 专用视频帧列表 | 接口将一组图片作为一个视频输入处理。它的帧数限制与 token 消耗可能不同于独立多图。 |
| 图片张数 | 接口或部署配置允许的图片项数量；降低分辨率不能绕过此限制。 |
| 上下文 | 图像、声音、文字等转换后的 token 总量，并需为模型输出保留空间。表中“上下文”与厂商明确给出的“输入 token 上限”分别标明。 |
| 请求体积 | 实际 HTTP 请求大小，包含媒体编码和其他字段。Base64 通常使二进制体积增加约三分之一。 |

以下容量对应所列模型与接口，不能根据模型家族名称直接套用到其他版本或转发服务。

## 2. 通用模型

### 2.1 本地 Qwen

视频文件由本地工具链处理后送入模型。实际输入容量还受权重配置、服务启动参数及显存约束。

| 模型 | 支持的媒体 | 上下文依据 | 当前使用说明 |
|---|---|---|---|
| Qwen3-VL-8B-Instruct / Thinking | 图片、视频；不处理音频 | Instruct 发布配置的 `text_config.max_position_embeddings` 为 **262,144**。 | 不能仅按帧数判断能否容纳输入；其他 checkpoint 以其实际配置为准。 |
| Qwen3-Omni-30B-A3B-Instruct / Thinking | 图片、音频、视频；官方 vLLM 示例支持图片＋音频 | Instruct 发布配置的 `thinker_config.text_config.max_position_embeddings` 为 **65,536**。 | 不将 VL 的帧数设置直接套用到 Omni；部署可能进一步降低 `max-model-len`。 |
| Qwen2-Audio-7B-Instruct | 音频、文字；不接收图片 | 音频预处理配置为 **16 kHz、30 秒窗口**；全请求仍受服务上下文限制。 | 使用 `--modality audio`，从对应视频提取音轨，按不超过 30 秒分段。 |

来源：[Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)、[8B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json)、[Qwen3-Omni 官方输入示例](https://github.com/QwenLM/Qwen3-Omni#vllm-serve-usage)、[30B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json)。

本地服务还可以通过 `--limit-mm-per-prompt` 限制每个请求的媒体项数量。核对时的 vLLM 文档默认每种模态为 999 项；发送 1,024 张独立图片时需要确认部署已允许该数量。该参数是服务配置，增大它不代表模型上下文或显存同步增加。[vLLM 参数说明](https://docs.vllm.ai/en/latest/configuration/engine_args/#--limit-mm-per-prompt)

Qwen2-Audio 的全部音频段按时间顺序放入**同一个请求**，附各段时间位置，由模型一次回答完整问题；不分别推理各段后拼接答案。vLLM 的 `--limit-mm-per-prompt '{"audio": N}'` 须允许至少实际分段数，`N` 按最长输入确定。分段不保证长音频能装入上下文；容量不足时保留错误，不截断音轨。原始 `Qwen-Audio-Chat` 使用另一套官方自定义代码，本次通用入口适配的是 `Qwen/Qwen2-Audio-7B-Instruct`。[官方 Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)、[音频预处理配置](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/raw/main/preprocessor_config.json)

### 2.2 百炼 Qwen API

| 托管模型 | 独立图片／专用视频帧列表 | 视频文件及组合要求 |
|---|---|---|
| Qwen3-VL-8B | 独立多图：URL 最多 **256 张**、Base64 最多 **250 张**；专用视频帧列表 **4–512 帧**。 | 视频 **2 秒–20 分钟**；URL 文件不超过 **2 GB**，Base64 编码串小于 **10 MB**；不理解音轨。 |
| Qwen3-VL-Plus / Flash，以及文档列出的 235B 型号 | 专用视频帧列表可到 **2,000 帧**。独立图片仍按多图限制处理。 | 所列型号的视频时长可到 **1 小时**；这些额度不能套用到 8B。 |
| Qwen3-Omni-Flash | 专用视频帧列表 **2–128 帧**。 | 原生视频最长 **150 秒**；URL 文件不超过 **256 MB**，Base64 串小于 **10 MB**。允许文本搭配一种非文本输入形式，有声视频可同时含画面与声音；不按开源 30B 的方式混合独立图片和音频。 |
| Qwen3.5-Omni 系列 | 支持混合媒体；专用视频帧列表 **2–2,048 帧**。 | 文档给出的上限为视频 **1 小时**、音频 **3 小时**。属于另一个模型系列，不替代 Qwen3-Omni-30B 的结果。 |

来源：[百炼图像与视频限制](https://www.alibabacloud.com/help/en/model-studio/vision)、[百炼 Qwen-Omni](https://help.aliyun.com/zh/model-studio/qwen-omni)。Qwen3-Omni-Flash 的独立音频时长在文档总述与细则中存在不一致，本文不将视频的 150 秒直接作为独立音频上限。

当前代码发送的逐帧 `image_url` 属于独立多图，不能使用专用 `video` 帧列表的更高额度。

### 2.3 Gemini

| 模型 ID | 支持的输入 | 输入 token 上限 |
|---|---|---:|
| `gemini-3-flash-preview` | 文字、图片、视频、音频；原生接口支持组合媒体。 | 1,048,576 |
| `gemini-2.5-flash` | 文字、图片、视频、音频。 | 1,048,576 |
| `gemini-2.5-pro` | 文字、图片、视频、音频、PDF。 | 1,048,576 |

来源：[Gemini 3 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3-flash-preview)、[Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash)、[Gemini 2.5 Pro](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-pro)。

图片文档给出的单请求上限为 **3,600 张**。原生视频默认按 **1 FPS** 处理，可通过视频元数据设置 FPS；视频文档对 1M 上下文给出低分辨率约 3 小时、高分辨率约 1 小时的参考范围。音频约 **32 token／秒**，音频文档列出的单请求总时长上限为 **9.5 小时**。这些媒体数量和时长仍受实际 token 总量、上传方式及服务限制约束。[图片](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)、[视频](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding)、[音频](https://ai.google.dev/gemini-api/docs/generate-content/audio)

Google 的 OpenAI-compatible 接口与原生 `generateContent` 分开处理。兼容文档展示图片和音频输入，不据此假定网关支持原生视频或任意混合媒体请求。[兼容接口说明](https://ai.google.dev/gemini-api/docs/openai)

### 2.4 GPT 与相关音频接口

| 模型 | 图片 | 音频 | 原始视频文件 | 上下文 token |
|---|---|---|---|---:|
| [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1) | 支持 | 不支持 | 不支持 | 1,047,576 |
| [GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1) | 支持 | 不支持 | 不支持 | 400,000 |
| [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2) | 支持 | 不支持 | 不支持 | 400,000 |
| [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4) | 支持 | 不支持 | 不支持 | 1,050,000 |
| [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5) | 支持 | 不支持 | 不支持 | 1,050,000 |
| [GPT-4o](https://developers.openai.com/api/docs/models/gpt-4o) | 支持 | 不支持 | 不支持 | 128,000 |
| [gpt-audio](https://developers.openai.com/api/docs/models/gpt-audio) | 不支持 | 支持 | 不支持 | 128,000 |
| [gpt-4o-audio-preview](https://developers.openai.com/api/docs/models/gpt-4o-audio-preview) | 不支持 | 支持 | 不支持 | 128,000 |
| [gpt-realtime](https://developers.openai.com/api/docs/models/gpt-realtime) | 支持 | 支持 | 不作为 MP4 文件输入 | 32,000 |

核对时，OpenAI 图片接口文档规定每请求最多 **1,500 张图片、512 MB 总请求体**，同时须满足具体模型的上下文与图像处理要求。该数字替代旧资料中的 500 张／50 MB，使用时仍应对应实际服务版本。[图片输入要求](https://developers.openai.com/api/docs/guides/images-vision#image-input-requirements)

`gpt-audio` 等音频模型不等于支持图片＋音频的视觉模型。`gpt-realtime` 属于独立的 Realtime 会话接口，文档列出会话最长 60 分钟；当前通用批量入口没有实现该协议。[Realtime 会话说明](https://developers.openai.com/api/docs/guides/realtime-conversations)

### 2.5 Claude

| 模型 | 支持的输入 | 上下文 | 官方请求限制 |
|---|---|---:|---|
| `claude-sonnet-4-6` | 文字、图片；不直接接收视频或音频。 | 1M tokens | 最多 **600 张图片**；标准接口总请求体 **32 MB**。 |

来源：[Sonnet 4.6 模型说明](https://platform.claude.com/docs/en/models/sonnet-4-6/overview)、[视觉输入限制](https://platform.claude.com/docs/en/build-with-claude/vision#request-limits)。

视觉文档对 200K 上下文的 Claude 型号另列 **100 张**上限。多于 20 张图片时，官方建议各边不超过 **2,000 像素**，以满足不同平台的多图尺寸限制。768 或 1,024 张独立图片超过 Sonnet 4.6 所列的 600 张额度，降低分辨率无法消除该限制。

当前调试配置使用 `claude-sonnet-4-6`。本轮中转最高成功样本为 **80 帧、420×224**，但相同配置复测时连接中断；更大请求发生传输失败或 HTTP 524。当前预算按这一成功样本设置，不代表稳定上限或模型上限，完整结果见第 4.4 节。

### 2.6 GLM 与 Doubao-Seed

| 模型或系列 | 已核对能力 | 当前入口 |
|---|---|---|
| `glm-4.6v` / `glm-4.6v-flash` | 图片、视频、文字；官方概览列 **128K** 上下文，本轮未查到独立多图数量上限。`glm-4.6` 是不同的纯文本模型。 | Chat Completions；视频以带时间位置的独立图片帧输入，不另传音频。 |
| 支持视觉输入的 Doubao-Seed 型号 | 原生视频接口按视频 token、FPS 和模型版本限制；强制改为独立图片帧时，当前策略上限为 **1,280** 帧。 | 火山方舟 Chat Completions 默认传 `video_url`；`--sample-frames` 才使用多图帧。仅已确认的 Seed 2.0 Mini/Lite `260428` 版本保留视频音频，其他版本默认移除音轨。 |

来源：[GLM 模型概览](https://docs.bigmodel.cn/cn/guide/start/model-overview)、[GLM-4.6V 多图请求](https://docs.bigmodel.cn/cn/guide/models/vlm/glm-4.6v)、[GLM-4.6 文本模型](https://docs.bigmodel.cn/cn/guide/models/text/glm-4.6)、[方舟图片输入及限制](https://www.volcengine.com/docs/82379/1362931?lang=zh)、[方舟视频理解](https://www.volcengine.com/docs/82379/1895586?lang=zh)、[音视频联合理解版本说明](https://www.volcengine.com/docs/6492/2165228?lang=en)。同一家族包含不同能力的型号；已实现通用请求格式不等于任意 GLM 或 Seed 型号都支持图片或音频。

官方 `doubao-seed-2-0-pro-260215` 和 `doubao-seed-2-0-lite-260215` 的温度固定为 **1**，接口忽略其他手动值；适配器按此值发送并记录。其他模型仍保留通用默认温度 **0**。[方舟生成参数](https://www.volcengine.com/docs/82379/1298454?lang=zh)

### 2.7 InternVL3.5

本次核对的型号为 **`OpenGVLab/InternVL3_5-241B-A28B`**，与用户提供的 ModelScope 型号一致。作者发布的是通用视觉语言模型，提供多图和视频示例，并支持通过 vLLM 或 LMDeploy 部署。因此接入通用推理入口即可，不需要另建 `emollm` 下的专用入口。本文核验了作者官方模型卡及其指向的 ModelScope 记录；ModelScope 页面本轮直接读取超时。[官方模型卡](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B)、[ModelScope 型号](https://modelscope.cn/models/OpenGVLab/InternVL3_5-241B-A28B)

当前向已部署的 OpenAI-compatible 服务发送带时间位置的图片帧，不发送音频。LMDeploy 官方模型卡给出 `api_server` 接口及多图用法，并注明该 241B-A28B 型号使用 `tp=8` 的部署示例。服务版本、并行配置、实际上下文及图片项数量须按部署确定；示例不代表本机已具备运行该权重的资源。没有下载模型权重或复制官方源码，也没有进行 GPU 推理。[作者部署说明](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B#deployment)、[vLLM 支持列表](https://docs.vllm.ai/en/latest/models/supported_models/)、[LMDeploy InternVL](https://lmdeploy.readthedocs.io/en/latest/multi_modal/internvl.html)

### 2.8 DeepSeek

| 官方模型名 | 本次核对能力 | 当前用法 |
|---|---|---|
| `deepseek-flash` | 当前指向 DeepSeek-V4.1-Flash，支持文字与图片，**1M** 上下文。 | `--modality video` 传图片帧，或 `--modality text` 传字幕；不传音频。 |
| `deepseek-v4-pro` | 核对日期的官方表列为文字模型，不支持视觉。 | 用于纯字幕推理或文本裁判，不能据此配置视频帧输入。 |

官方 Chat Completions 地址为 `https://api.deepseek.com`。Flash 的图片请求限制为最多 **600 张图片、48 MiB 请求体**；单张 Base64 图片不超过 **32 MiB**，总图片大小不超过 **64 MiB**（不使用 Files API 时）。本项目采用内联图片，未接入 DeepSeek Files API。当前调试配置保留用户指定的 `deepseek-v4-flash`；官方说明该旧别名现已转到 **DeepSeek-V4.1-Flash**。实验同时记录请求中的名称与 `raw_response` 返回的模型名称，不将旧别名当作固定的历史版本。[官方接口说明](https://api-docs.deepseek.com/)、[型号与能力](https://api-docs.deepseek.com/quick_start/pricing/)、[图片输入及限制](https://api-docs.deepseek.com/guides/vision)

请求包含 15 张及以上图片时，各边最多 **4,096 像素**。本轮官方接口已成功接收 **600 帧、1008×560、46.583 MiB** 的请求，达到图片张数上限，但请求体已接近 48 MiB；其他视频在相同像素设置下可能产生更大的 JPEG，仍须按实际请求体检查。

### 2.9 OpenRouter：GLM、Kimi 与 MiniMax

以下调试配置共用 `https://openrouter.ai/api/v1`，通过 Chat Completions 发送带时间位置的图片帧。现有 Debug 已配置 OpenRouter 密钥；命令行也支持通过 `OPENROUTER_API_KEY` 提供。

| 模型 ID | 当前配置状态 |
|---|---|
| `z-ai/glm-5.3-flash` | 64 帧、420×224 已通过 urllib 返回 HTTP 200；Parasail 路由。 |
| `moonshotai/kimi-k3` | 32 帧、420×224 已通过 urllib 返回 HTTP 200；DigitalOcean 路由。 |
| `minimax/minimax-m3` | 32 帧、420×224 已通过 urllib 返回 HTTP 200；GMICloud 路由。 |

接口格式依据 [OpenRouter 官方接入说明](https://openrouter.ai/docs/quickstart)；模型记录见 [GLM 5.3 Flash](https://openrouter.ai/z-ai/glm-5.3-flash)、[Kimi K3](https://openrouter.ai/moonshotai/kimi-k3)、[MiniMax M3](https://openrouter.ai/minimax/minimax-m3)。实际容量受所路由的供应商约束，不能直接套用其他厂商的 600 张限制。Debug 按成功样本配置：GLM 为 64 帧、单帧上限 100,352、总像素上限 6,422,528；Kimi 与 MiniMax 为 32 帧、单帧上限 100,352、总像素上限 3,211,264。三项均为 2 FPS，不改变各模型系列的全局默认值，也不是已测得的硬上限。详见第 4.5 节。

## 3. 帧数、分辨率与 token 预算

### 3.1 当前采样方式

通用入口先按 `round(时长 × fps)` 计算目标帧数，再受 `max_frames` 和源视频可用帧数限制；确定帧数后，在完整视频范围内均匀采样，并根据像素预算调整每帧尺寸。

| 已确定的模型配置 | FPS | 最大帧数 | 单帧像素上限 | 总像素上限 |
|---|---:|---:|---:|---:|
| GPT | 1 | 128 | 262,144 | 33,554,432 |
| Claude Sonnet 4.6，当前中转 | 2 | 80 | 262,144 | 8,028,160 |
| DeepSeek Flash | 2 | 600 | 602,112 | 361,267,200 |

显式 CLI 参数优先；表中的最大帧数不表示每个视频都取满该数量，例如 53.22 秒视频在 DeepSeek 的 2 FPS 配置下实际取约 106 帧。Claude 的总预算为 `80 × 100352`，80 帧时当前宽屏视频输出 **420×224**，较少帧时可使用更大的单帧尺寸。增加最大帧数时，也须配置足够的总像素预算，并满足接口的请求体和上下文限制。

- 指定 `total_pixels` 时严格限制采样图片像素总和，不使用 Qwen 的乘 2 公式。更多帧共享总预算，单帧预算随之降低；到达 **100,352** 像素的最低预算后，会限制实际帧数。帧仍覆盖完整视频范围，不截取前 N 帧。降低单帧上限不会自动增加帧数。
- **262,144 = 512×512** 是面积预算，不是固定输出尺寸；当前宽屏视频对应约 **672×364**，602,112 像素上限对应约 **1008×560**。宽高按 **28** 的倍数对齐来自本地解码器，不是各模型的统一官方要求。结果记录请求与实际帧数、缩放宽高、实际像素总和。
- 未单独设置模型采样配置时，通用默认值为 **2 FPS、768 帧、每帧 602,112 像素**。省略 `--total-pixels` 时沿用 Qwen 工具的预算及乘 2 公式，触及最小尺寸后再增加帧数会增加总像素量。Seed 抽帧的数量上限为 **1,280**。原生视频采用服务端预算。
- 像素预算不等于模型 token 预算，也不保证固定的 JPEG／HTTP 请求大小。较小的人脸和短暂表情可能丢失；当前 GPT 路径不传音频，增加像素或帧数不能补回声音语气。

最大帧数、单帧尺寸、请求体和上下文需要共同满足，不能分别取各项最大值后假定请求有效。DeepSeek 官方会放大像素面积低于约 544×544 的图片，因此较小的本地帧主要减少上传体积，不能据此推断 token 同比例减少。[DeepSeek 图像处理说明](https://api-docs.deepseek.com/guides/vision/#token-usage)

实现：[采样与像素预算](../evaluation/inference/video_loader.py)、[推理入口](../evaluation/inference/run.py)。

### 3.2 Gemini 3 的独立图片与视频帧

以下是官方给出的近似 token 预算。`media_resolution` 是模型侧媒体处理档位，与本地 JPEG 宽高分开控制。

| 媒体分辨率 | 每张独立图片 | 每帧原生视频 |
|---|---:|---:|
| 默认 | 1,120 | 70 |
| low | 280 | 70 |
| medium | 560 | 70 |
| high | 1,120 | 280 |

按上述预算，仅计算独立图片：

| 输入 | 估算图像 token |
|---|---:|
| 768 张，默认档 | 860,160 |
| 1,024 张，默认档 | 1,146,880 |
| 1,024 张，medium | 573,440 |
| 1,024 张，low | 286,720 |

1,024 张默认档图片的估算已超过 Gemini 3 Flash 的输入上限。768 张仍需为时间标记、题目及启用时的字幕和音频留出空间。Gemini 2.5 的媒体处理预算不同，不套用此表。[官方媒体分辨率说明](https://ai.google.dev/gemini-api/docs/generate-content/media-resolution)

### 3.3 GPT 的图片预算示例

GPT-5.5 使用 32×32 patch，图片 token 约为 `ceil(ceil(宽/32) × ceil(高/32) × 1.2)`；以下尺寸在默认处理下无需服务端缩放。表中仅估算图片，不含提示词、时间标记、字幕和输出，实际计费可能有取整差异。

| 每帧尺寸 | 每帧估算 token | 128 帧估算 token |
|---|---:|---:|
| 1008×560 | 692 | 88,576 |
| 672×364 | 303 | 38,784 |
| 476×252 | 144 | 18,432 |

旧设置的跨型号估算保留如下，**不代表新的 GPT 默认配置**：

| 模型 | 768 张，每张 644×336 | 1,024 张，每张 560×308 |
|---|---:|---:|
| GPT-4.1 / GPT-4o | 326,400 tokens | 435,200 tokens |
| GPT-5.1 | 268,800 tokens | 358,400 tokens |
| GPT-5.2 / GPT-5.4 | 213,504 tokens | 221,184 tokens |

不同 GPT 型号采用不同的图块或 patch 规则，不能直接套用 GPT-5.5 的估算。默认 128 帧和像素预算是本项目的采样选择；未找到 GPT-5.5 官方推荐的固定 FPS 或视频帧数。官方归档视频示例使用每秒 1 帧并允许调整，不构成情绪识别精度依据。[官方图片处理与计费规则](https://developers.openai.com/api/docs/guides/images-vision#calculating-costs)、[官方视频抽帧示例](https://developers.openai.com/cookbook/examples/gpt4o/introduction_to_gpt4o)。

### 3.4 Qwen 的部署预算

Qwen3-VL 官方说明图像预算约按 32×32 像素一个视觉 token 计算，视频处理另有时间压缩。按此尺度粗估，1,024 张 560×308 的独立图片约为 **17 万视觉 token**。这说明配置需要核验上下文预算；是否超过 Qwen3-Omni 的容量，须用 Omni 的实际 processor 计数后判断，不能将 VL 的粗估当成 Omni 精确计数。[Qwen3-VL 像素与 token 说明](https://github.com/QwenLM/Qwen3-VL#using--transformers-to-chat)

## 4. 请求体积、音轨与转发平台

### 4.1 音轨不会随抽帧变短

当前通用入口独立提取 **16 kHz、单声道、PCM16 WAV**。仅按音频数据计算，20 分钟约 38.4 MB，40 分钟约 76.8 MB；转为 Base64 后约 51.2 MB 和 102.4 MB，还未加上图片和请求字段。降低采样 FPS 或图片尺寸不会缩短这段音频。[提取实现](../evaluation/inference/video_loader.py)

### 4.2 Gemini 上传限制的文档差异

核对时，Google 的[文件输入方法](https://ai.google.dev/gemini-api/docs/generate-content/file-input-methods)写内联请求 100 MB，而同一接口的[图片](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding)和[音频](https://ai.google.dev/gemini-api/docs/generate-content/audio)说明仍写总请求 20 MB。不能将这一不一致自行解释为“旧接口／新接口”或“编码前／编码后”的区别。

[Files API 通用文档](https://ai.google.dev/gemini-api/docs/generate-content/files)列出 2 GB／文件、20 GB／项目、保留 48 小时；视频页的部分表格存在不同额度，本文不据此承诺更高上传容量。实际运行需确认所用接口的有效限制。当前代码使用内联 Base64，尚未实现 Gemini Files 上传与引用。

### 4.3 AICodeMirror

本地与转发服务均通过 `--base-url` 或 `MODEL_BASE_URL` 指定地址。`adapters/__init__.py` 选择协议：`openai.py` 处理 Chat Completions 与 Responses，`anthropic.py` 处理 Messages，`gemini.py` 处理 generateContent；模型参数差异集中在 `model_settings.py`。推理入口统一构造消息，共用客户端委托协议适配器完成编码与解析，并负责 HTTP 发送和校验。Qwen 权重、InternVL 及未知模型名没有默认本地地址，须显式指定。路由适配不能证明转发层保留官方的模型版本、上下文、图片数量或请求体积额度。

2026-09-12，以 `G1_Q000001` 对应的 53.22 秒视频执行以下 GPT-5.5 请求测试。代理／直连区分是否使用显式 HTTP 代理；“直连”不保证绕过系统 VPN／TUN，且仍使用 AICodeMirror，不代表调用 OpenAI 官方端点。体积为编码后的请求体，耗时为单次请求观察值。请求结果保存在 `output/debug/gpt_probe/`。

| 图片帧数 | 单帧尺寸 | 请求体积（MiB） | 网络路径 | 耗时（秒） | 结果 |
|---:|---|---:|---|---:|---|
| 16 | 1008×560 | 1.269 | 代理 | 67.11 | HTTP 200 |
| 20 | 1008×560 | 1.593 | 直连 | 29.23 | HTTP 200 |
| 32 | 1008×560 | 2.473 | 直连 | 56.43 | HTTP 200 |
| 64 | 672×364 | 2.868 | 直连 | 73.90 | HTTP 200 |
| 128 | 672×364 | 5.781 | 直连 | 97.04 | HTTP 200 |
| 256 | 476×252 | 7.162 | 直连 | 141.57 | HTTP 200 |
| 384 | 476×252 | 10.763 | 直连 | 162.99 | 上传时 Broken pipe，未收到 HTTP 响应 |
| 512 | 476×252 | 14.356 | 直连 | 161.85 | 上传时 Broken pipe，未收到 HTTP 响应 |

256 帧探测使用 8 FPS 目标率以获得足够帧数，总预算为 33,554,432 像素，实际总量为 30,707,712；这不是默认 FPS。此前 106 帧、8.225 MiB 的较大图片请求发生上传超时，不能据此判定 API 拒绝该图片张数。

HTTP 200 表示该次请求成功返回；上传中断不能认定为模型拒绝图片张数。上述探测尚未定位真实帧数／体积硬上限，也未评估抽帧设置对准确率的影响。不同网络路径和单次请求耗时不足以比较稳定性能。官方容量仍不能直接视为 AICodeMirror 的保证值。

正式推理入口使用新的 GPT 默认值完成 4 道真实题目，结果与媒体参数见 `output/debug/gpt_defaults/predictions.jsonl`，均成功返回且标签格式解析正常。第一题实际取 53 帧、672×364，请求体 2.41 MiB；原设置为 8.23 MiB。第 2、4 题首次分别遇到远端断开和 HTTP 524，保持配置重试后成功；首次记录另存于 `output/debug/gpt_probe/defaults_first_attempt.jsonl`。虚拟评估题 `G1_QTEST001` 没有对应视频，未纳入本次视频测试。此结果验证推理流程，不代表预测答案已通过人工准确性审核。

本次 Codex 转发接口的返回记录还包含额外的 Codex 指令，并将 `temperature` 记为 1、`max_output_tokens` 记为 null；本地发送值分别为 0 和 16,384。正式实验应保留此配置差异，不能将请求参数视为已经得到服务端执行。

### 4.4 DeepSeek 与 Claude 容量实测

2026-09-12，继续使用 `G1_Q000001` 的完整 53.22 秒视频。600 帧测试采用 12 FPS 目标率，以实际取满 600 帧；没有重复图片凑数，也没有截取视频前段。测试记录分别保存于 `output/debug/deepseek_capacity/` 和 `output/debug/claude_capacity/`。

| 模型与接口 | 图片帧数 | 单帧尺寸 | 请求体积（MiB） | 耗时（秒） | 结果 |
|---|---:|---|---:|---:|---|
| DeepSeek Flash，官方接口 | 600 | 672×364 | 27.059 | 95.77 | HTTP 200 |
| DeepSeek Flash，官方接口 | 600 | 1008×560 | 46.583 | 100.25 | HTTP 200 |
| Claude Sonnet 4.6，AICodeMirror | 16 | 672×364 | 0.739 | 12.36 | HTTP 200 |
| Claude Sonnet 4.6，AICodeMirror | 40 | 420×224 | 0.985 | 35.70 | HTTP 200 |
| Claude Sonnet 4.6，AICodeMirror | 64 | 420×224 | 1.528 | 63.85 | HTTP 200 |
| Claude Sonnet 4.6，AICodeMirror | 80 | 420×224 | 1.915 | 90.81／132.62 | 首次 HTTP 200；相同请求复测时连接中断 |
| Claude Sonnet 4.6，AICodeMirror | 88 | 420×224 | 2.125 | 173.31 | 上传时 Broken pipe，无 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 96 | 420×224 | 2.293 | 108.64 | 上传时 Broken pipe，无 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 128 | 420×224 | 3.075 | 151.09 | urllib 超时设置 240 秒，服务返回 HTTP 524 |
| Claude Sonnet 4.6，AICodeMirror | 200 | 420×224 | 4.791 | 150.30 | 上传中断，未收到 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 256 | 420×224 | 6.134 | 162.18 | 上传中断，未收到 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 256 | 476×252 | 7.167 | 5.02／169.36 | 两次分别为 SSL EOF、Broken pipe，均无 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 600 | 420×224 | 14.394 | 166.65–167.89 | 代理与直连均上传中断，无 HTTP 响应 |
| Claude Sonnet 4.6，AICodeMirror | 600 | 672×364 | 27.068 | 172.34 | 上传时 Broken pipe，未收到 HTTP 响应 |

DeepSeek 已验证达到官方 600 张图片上限的两组尺寸。1008×560 是本轮验证的较大尺寸，不代表任意视频都可在该尺寸下发送 600 帧；其请求体仅余约 1.4 MiB，画面复杂度变化可能导致超限。

Claude 本轮最高成功样本为 **80 帧、420×224**，记录见 `output/debug/claude_capacity/frames80_pixels100352.json`。相同配置复测失败，不能称为稳定上限。88、96 帧未收到 HTTP 响应；128 帧以 240 秒客户端超时重试后，在 151.09 秒收到 HTTP 524。更大请求的传输错误不能证明图片数量或模型容量超限；中转、上传速度和网络波动仍影响结果。当前默认采用 80 帧及对应总像素预算，官方 600 张图片限制保持独立。HTTP 200 仅证明请求成功返回，不代表答案准确率已审核。

此前临时 curl 诊断的 128 帧请求只收到 `100 Continue`，90 秒内上传 1,966,080 字节后因客户端时限停止；这既不是推理成功，也不是模型拒绝。正式实现保持 urllib，未因这些测试切换传输库。

### 4.5 OpenRouter 实测

同样使用 `G1_Q000001` 的完整视频，以下成功请求均通过正式实现所用的 urllib、Chat Completions 请求组装与响应解析。

| 模型 | 图片帧数与尺寸 | 请求体积（MiB） | 响应供应商 | 网络 | 耗时（秒） | 结果记录 |
|---|---|---:|---|---|---:|---|
| `z-ai/glm-5.3-flash` | 16 帧，672×364 | 0.738 | Wafer | 代理 | 41.22 | `output/debug/glm_capacity/connectivity_16.json` |
| `z-ai/glm-5.3-flash` | 64 帧，420×224 | 1.527 | Parasail | 直连 | 143.16 | `output/debug/glm_capacity/frames64_pixels100352_serial.json` |
| `moonshotai/kimi-k3` | 16 帧，672×364 | 0.738 | DigitalOcean | 直连 | 238.12 | `output/debug/kimi_capacity/connectivity_16_direct.json` |
| `minimax/minimax-m3` | 16 帧，672×364 | 0.738 | GMICloud | 直连 | 47.12 | `output/debug/minimax_capacity/connectivity_16_direct.json` |
| `moonshotai/kimi-k3` | 32 帧，420×224 | 0.766 | DigitalOcean | 直连 | 70.07 | `output/debug/kimi_capacity/serial2_frames32_pixels100352.json` |
| `minimax/minimax-m3` | 32 帧，420×224 | 0.766 | GMICloud | 直连 | 77.41 | `output/debug/minimax_capacity/serial2_frames32_pixels100352.json` |

以上均返回 HTTP 200 和实际答案。GLM 的 256 帧、Kimi 的 128／256 帧、MiniMax 的 128／256 帧测试出现 SSL EOF、写入超时或断管，未获得模型拒绝图片数量的 HTTP 响应；部分较小请求也因网络失败。GLM 的 64 帧请求曾发生 TLS 错误，相同配置再次单独发送后成功，不能将此前失败解释为固定帧数上限。MiniMax 的另一次 16 帧 curl 请求成功仅作传输诊断，未修改正式客户端。不同路由与网络下的耗时不可直接用来比较模型速度。

后续逐个请求复测中，Kimi 与 MiniMax 的 32 帧均成功，40／48／64 帧均未获得模型回答：

| 帧数／请求体积 | Kimi | MiniMax |
|---|---|---|
| 40／0.984 MiB | 上传时断管，180.44 秒。 | TLS 连接阶段失败，5.02 秒，尚未发送请求体。 |
| 48／1.159 MiB | 63.74 秒完成上传，155.58 秒时连接关闭，无 HTTP 响应。 | 上传时断管，22.52 秒。 |
| 64／1.527 MiB | 直连在 TLS 阶段失败；显式代理能够连接，但上传时断管。 | urllib 在 TLS 阶段失败；curl HTTP/1.1 完整上传 1,600,728 字节后收到空响应。 |

以上错误发生在不同传输阶段，均未收到模型对帧数或体积的拒绝。32 帧是本轮最高成功样本，不是已经定位的容量上限。补充请求记录使用各模型目录下的 `serial2_frames<N>_pixels100352` 文件名前缀。

本节“直连”仅指禁用显式 HTTP 代理。本轮 DNS 将 `openrouter.ai` 解析为 `198.18.0.177`，提示仍可能经过 TUN／fake-IP 路由；没有证据表明这些请求绕过了代理软件。网络路径与模型请求容量须分别判断。

当前 Debug 采用 GLM 64 帧、Kimi 与 MiniMax 32 帧的成功配置，实际采样覆盖完整视频。这些测试尚未确定三款模型的容量上限或重复运行稳定性，也未审核答案准确率。

## 5. 专用情绪模型

三个独立入口使用各自的官方模型代码和预处理方式，当前均接收 `clip` 数据。下表记录的是现有适配路径，不能作为模型家族的理论硬上限；通用入口的 `--max-frames` 等参数不作用于这些入口。

| 模型 | 当前视觉输入 | 当前音频输入 | 入口 |
|---|---|---|---|
| AffectGPT | 当前 raw-frame 配置均匀取 **8 帧、224×224**。 | 分布在音轨中的 **8 段、每段 2 秒**，16 kHz。 | [affectgpt.py](../evaluation/inference/adapters/affectgpt.py) |
| Emotion-LLaMA | 当前 demo 路径使用**首帧、448×448**；时序特征槽位不提供额外视频变化信息。 | 完整音轨经 HuBERT 编码和时间维均值池化。 | [emotion_llama.py](../evaluation/inference/adapters/emotion_llama.py) |
| R1-Omni | 按 checkpoint 的帧数与图像 processor 处理，帧数缺省时回退为 **8 帧**。 | 使用 Whisper 配置的音频窗口，通常为 **30 秒**；没有长音频分块汇总。 | [r1_omni.py](../evaluation/inference/adapters/r1_omni.py) |

版本及环境见[专用模型说明](../evaluation/inference/emollm/README.md)和[固定源码版本](../evaluation/inference/emollm/sources.json)。官方来源：[AffectGPT](https://github.com/zeroQiaoba/AffectGPT)、[Emotion-LLaMA](https://github.com/ZebangCheng/Emotion-LLaMA)、[R1-Omni](https://github.com/HumanMLLM/R1-Omni)。本轮没有加载这些模型权重执行推理。

预处理依据：[AffectGPT raw-frame 配置](https://github.com/zeroQiaoba/AffectGPT/blob/fffca794c6792023234565370e3f69f0488aede9/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml)、[Emotion-LLaMA demo 配置](https://github.com/ZebangCheng/Emotion-LLaMA/blob/e69faadfd3824b94dd334f463cc5bbdb27229c6d/eval_configs/demo.yaml)、[Emotion-LLaMA 媒体编码](https://github.com/ZebangCheng/Emotion-LLaMA/blob/e69faadfd3824b94dd334f463cc5bbdb27229c6d/minigpt4/conversation/conversation.py)、[R1-Omni 媒体预处理](https://github.com/HumanMLLM/R1-Omni/blob/17cafcae0b0a1c454896f4eb7a1c006bf13e1f4d/humanomni/mm_utils.py)。

## 6. 当前实现与验证范围

| 项目 | 当前实现 |
|---|---|
| G1 视频 | 按模型和接口选择视频文件或图片帧；`--sample-frames` 强制抽帧。 |
| G2 视频 | `-g episode --modality video` 自动使用完整视频范围内的图片帧。 |
| 纯字幕 | `--modality text` 支持两个粒度，不发送画面和音频。 |
| 纯音频 | `--modality audio` 从视频提取音轨、不传画面；Qwen2-Audio 按 30 秒分段后在同一请求中传入全部片段。支持音频输入的原生 Gemini、本地 Qwen-Omni 也可使用此模式。 |
| GLM、InternVL、DeepSeek 视觉型号 | 共用 Chat Completions 及图片帧输入；无已确认支持的音频组合时不传音频。 |
| Doubao-Seed 视觉型号 | `clip` 默认通过 Chat Completions 传原生视频；`episode` 或 `--sample-frames` 使用图片帧。Seed 2.0 Mini/Lite 的 `260428` 版本已确认可做音视频联合理解，其他版本不默认读取音轨。 |
| 音频 | 原生视频始终保留原文件中的音轨；抽帧路径默认不带音频，提供 `--with-audio` 后另传音轨。 |
| 模型侧图片档位 | GPT 的 `detail` 和 Gemini 的 `media_resolution` 默认由接口决定，普通运行无需额外配置；需要调整 Gemini 生成配置时，可用 `--config FILE` 提供 JSON 参数。 |
| 容量预检 | DeepSeek 抽帧数量不超过 600。发送到官方端点前，按包含内联媒体的实际 JSON 编码字节数检查请求体：方舟上限为 64,000,000 字节（官方 64 MB 按十进制保守处理），DeepSeek 为 48 × 1024 × 1024 字节（48 MiB）。超限时提示调低 `--max-frames` 或 `--frame-max-pixels`，不调用 API，也不截掉视频后半部分。其他服务的请求大小与 token 预算尚未统一预检；GLM、Seed 未添加未经确认的固定帧数上限。 |
| 相关音频接口 | `gpt-audio` 等仅列为能力对照；当前音频路由不为它们提取音轨。Realtime 尚未接入。 |

实现位置：[媒体路由](../evaluation/inference/run.py)、[请求编码](../evaluation/clients.py)、[采样和音频提取](../evaluation/inference/video_loader.py)。

新增接入包含本地请求组装、路由和响应解析测试；GPT-5.5、DeepSeek Flash、Claude Sonnet 4.6 及 OpenRouter 三款模型的 API 实测见第 4.3–4.5 节。未加载 InternVL、Qwen2-Audio 等权重进行真实模型或 GPU 测试，未据这些请求验证识别准确率。

### 本地长视频采样记录

验证样本为 `G2_V000002.mp4`，时长约 22 分 36 秒。以下使用原有 Qwen 像素预算，是本地预处理记录，无模型 API 调用；不代表新的 GPT 默认设置：

| 图片数 | 尺寸 | 平均采样间隔 | 抽帧耗时 | 图片 Base64 总体积 |
|---|---|---|---:|---:|
| 32 | 1008×560 | 约 43.75 秒 | 4.27 秒 | 约 2.31 MB |
| 768 | 644×336 | 约 1.77 秒 | 54.0 秒 | 约 29.35 MB |
| 1,024 | 560×308 | 约 1.33 秒 | 72.6 秒 | 约 34.15 MB |

三次均覆盖 0 至 1356.222 秒，源视频哈希保持不变，输出中未包含音频。图片体积未计入 JSON、提示词等开销；结果只验证全片采样与尺寸变化，不代表服务端已接收请求或模型能正确回答题目。
