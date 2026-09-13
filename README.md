# LongEmoBench

[中文说明](README_zh.md)

Inference and evaluation for the unified open-question format. Video, subtitle-only and audio-only inference support both `clip` and `episode` with compatible model services. Episode video input uses frames sampled across the complete video. Evaluation supports predictions from these baselines and external methods, including agents.

This directory contains code, prompts and synthetic tests. Questions, videos, subtitles, annotation tools and historical experiment outputs are not included.

## Layout

```text
evaluation/
  io_utils.py           # Load and select questions, save results
  clients.py            # Shared HTTP sending and request validation
  judge_prompts.py      # Judge system/user prompts and four inline examples
  metrics.py            # Label metrics and score aggregation
  eval.py               # Evaluation entry point
  inference/
    run.py              # Video/text/audio inference entry point
    runner.py           # Runtime options, checkpoints and retries
    prompts.py          # Inference prompts and answer parsing
    video_loader.py     # Read videos or sample timestamped frames
    adapters/
      __init__.py       # Protocol selection and request configuration
      openai.py         # Chat Completions and Responses encoding/parsing
      anthropic.py      # Messages encoding/parsing
      gemini.py         # generateContent encoding/parsing
      model_settings.py # Model-specific generation settings
      message_content.py # Shared text, image and audio content checks
      qwen_audio.py     # Ordered Qwen2-Audio chunks
      affectgpt.py      # Independent AffectGPT entry
      emotion_llama.py  # Independent Emotion-LLaMA entry
      r1_omni.py        # Independent R1-Omni entry
      emollm_runner.py  # Batch utilities for specialist models
    emollm/
      AffectGPT/        # Official AffectGPT repository
      Emotion-LLaMA/    # Official Emotion-LLaMA repository
      R1-Omni/          # Official R1-Omni repository
      sources.json      # Upstream versions and notices
      README.md         # Emotion-model setup guide
subtitles/              # Episode subtitles: <video_id>.json
docs/
  data_format.md        # Question and prediction formats
  model_input_limits.md # Model inputs, capacity and adapter support
tests/                  # Synthetic local HTTP integration tests
```

Data loading reads prepared questions and selects them by granularity, question ID and count. The four judge examples are included directly in the evaluation System Prompt in `judge_prompts.py`.

Read the inference code in this order:

1. [run.py](evaluation/inference/run.py): CLI options, data selection and the inference loop.
2. [runner.py](evaluation/inference/runner.py) and [adapters/__init__.py](evaluation/inference/adapters/__init__.py): service setup and protocol selection; `model_settings.py` holds model-specific parameters.
3. [prompts.py](evaluation/inference/prompts.py) and [video_loader.py](evaluation/inference/video_loader.py): question messages and media preparation.
4. [clients.py](evaluation/clients.py): shared request validation and HTTP sending; `openai.py`, `anthropic.py` and `gemini.py` build protocol payloads and parse responses.


## Requirements and data

Python 3.10 or later; Python runtime code uses the standard library. A separately running model service is required for inference and LLM evaluation. The scripts do not load model weights. Frame sampling and audio extraction require `ffmpeg` and `ffprobe` on `PATH`. Subtitle-only input does not use them; forwarding an unchanged native video does not decode media.

Run the commands below from this directory. Optional installation with `python -m pip install .` provides `longemobench-infer`, `longemobench-infer-transformers`, and `longemobench-score` commands.

Each run selects one granularity with `-g` / `--granularity`. Keep its questions, media and results separate from the other granularity. Set `--data-path` to a JSON file containing an object or array, a JSONL file, or a flat directory of JSON/JSONL files. Prepared videos are named `<video_id>.mp4`. If `--videos-dir` is omitted, videos are read from the `videos/` subdirectory of `--data-path` when it is a directory, or of its parent when it is a file; an explicit `--videos-dir` overrides this default.

The loader requires the unified format documented in [data_format.md](docs/data_format.md), including the `answer_details` field. Reference answers and scoring rubrics stay outside the tested model's input.

## Inference

Set `--model` to the service's actual model name; the model family and input representation are selected automatically. Qwen, Gemini, GPT, Claude, GLM, Doubao-Seed, InternVL and DeepSeek use the same inference entry point, subject to the selected model's supported modalities.

| Family | API and service | Default clip video input | Audio |
|---|---|---|---|
| Qwen-VL | Chat Completions; explicit service URL required | Native MP4 via `video_url` | The embedded audio track is sent with the video; use depends on endpoint support. |
| Qwen-Omni | Chat Completions; explicit service URL required | Native MP4 via `video_url` | The embedded audio track is sent with the video. |
| Gemini | Native Gemini, `https://generativelanguage.googleapis.com/v1beta` | Inline MP4 | Embedded audio travels with the video. |
| GPT | Responses, `https://api.openai.com/v1` | Uniform timestamped frames | No audio is sent. |
| Claude | Anthropic Messages, `https://api.anthropic.com/v1` | Uniform timestamped frames | No audio is sent. |
| GLM vision models | Chat Completions; see service URLs below | Timestamped frames | No audio is sent. |
| Doubao-Seed vision models | Chat Completions; see service URLs below | Native video by default; timestamped frames with `--sample-frames` | Native video includes its embedded audio track. |
| InternVL3.5 | OpenAI-compatible service; explicit URL required | Timestamped frames | No audio is sent. |
| DeepSeek Flash with vision | Chat Completions; see service URLs below | Timestamped frames | No audio is sent. |
| Qwen2-Audio-7B-Instruct | OpenAI-compatible service; explicit URL required | Use `--modality audio`; images are not supported | Ordered audio chunks extracted from the source video. |

See [model input capabilities and limits](docs/model_input_limits.md) for supported media combinations, API and context limits, and differences between published model capabilities and the current adapters.

Use `--base-url` or `MODEL_BASE_URL` to select a local or hosted service. Models with a configured official address can omit the URL. Qwen checkpoints, InternVL and unknown model names require an explicit service URL; there is no implicit local-server address.

In [runner.py](evaluation/inference/runner.py), `setup_api()` configures the model family, service address and API key, and calls `prepare_request()` in `adapters/__init__.py`. The protocol adapters are `openai.py` for Chat Completions and Responses, `anthropic.py` for Messages, and `gemini.py` for generateContent. `model_settings.py` holds GPT, Qwen, GLM, Seed, DeepSeek and Gemini parameter differences; `qwen_audio.py` only handles audio splitting.

The inference entry builds common `messages`. [clients.py](evaluation/clients.py) delegates payload encoding and response parsing to the selected adapter, and handles shared HTTP sending and validation. Complete endpoints are preserved; protocol selection does not require a separate transport for each model family. Official addresses are centralized in `OFFICIAL_API_URLS`. Additional configured Chat Completions services are:

| Service | Official base URL | Selection |
|---|---|---|
| Qwen hosted API | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Set `--base-url` explicitly; Qwen checkpoint names have no default service URL. |
| MiniMax | `https://api.minimax.io/v1` | Default for MiniMax model names; China endpoint: `https://api.minimaxi.com/v1`. |
| Kimi | `https://api.moonshot.ai/v1` | Default for Kimi/Moonshot model names; China endpoint: `https://api.moonshot.cn/v1`. |
| GLM | `https://open.bigmodel.cn/api/paas/v4` | Default for GLM model names. |
| Doubao-Seed | `https://ark.cn-beijing.volces.com/api/v3` | Default for Seed/Doubao names; native video is used by default and `--sample-frames` selects sampled images. |
| DeepSeek | `https://api.deepseek.com/v1` | Default for DeepSeek model names. |
| OpenRouter | `https://openrouter.ai/api/v1` | Set `--base-url` explicitly and use the full `provider/model` ID. |

An explicit `--base-url` takes precedence. MiniMax and Kimi address/protocol routing does not establish support for every model's media inputs or generation settings. MiniMax's official text API requires `--temperature 1` with the current runner, whose default `0` is outside that API's allowed range. Kimi parameters depend on the model. [Qwen endpoints](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/base-url), [MiniMax API](https://platform.minimax.io/docs/api-reference/text-openai-api), [MiniMax regional endpoints](https://platform.minimax.io/docs/token-plan/cursor), [Kimi international API](https://platform.kimi.ai/docs/overview), [Kimi China API](https://platform.kimi.com/docs/get-api-key).

Pass the key with `--api-key API_KEY`. For services other than OpenRouter, omitting it selects `MODEL_API_KEY` first, then a standard key variable for the model family, falling back to the API format when the family is unknown. Supported variables include `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, `MINIMAX_API_KEY`, `MOONSHOT_API_KEY`, `ZHIPUAI_API_KEY`, `ARK_API_KEY` and `DEEPSEEK_API_KEY`. Credentials are not saved in run configuration.

OpenRouter uses the same `evaluation.inference.run` entry and Chat Completions adapter. Provide its own key through `OPENROUTER_API_KEY`; it takes precedence over the generic `MODEL_API_KEY` fallback. Vendor keys and keys from other relay services are not reused. An explicit `--api-key` still takes precedence. [OpenRouter API setup](https://openrouter.ai/docs/quickstart).

The GLM, Kimi and MiniMax OpenRouter debug configurations use these image-capable models and explicitly select `--sample-frames`:

| Debug model | Full model ID | Catalog input modalities |
|---|---|---|
| [GLM 5.3 Flash](https://openrouter.ai/z-ai/glm-5.3-flash) | `z-ai/glm-5.3-flash` | Text, images, video |
| [Kimi K3](https://openrouter.ai/moonshotai/kimi-k3) | `moonshotai/kimi-k3` | Text, images, video |
| [MiniMax M3](https://openrouter.ai/minimax/minimax-m3) | `minimax/minimax-m3` | Text, images, video |

All three routes returned HTTP 200 through the standard urllib client. The debug configurations use the largest successful sample settings from this run:

| OpenRouter debug model | FPS | Maximum frames | Per-frame pixel cap | Total pixel cap | Tested dimensions |
|---|---:|---:|---:|---:|---|
| GLM | 2 | 64 | 100,352 | 6,422,528 | 420×224 |
| Kimi | 2 | 32 | 100,352 | 3,211,264 | 420×224 |
| MiniMax | 2 | 32 | 100,352 | 3,211,264 | 420×224 |

These overrides apply to the OpenRouter debug entries, not global vendor limits. Larger requests encountered transport failures; maximum capacity and repeatability remain unverified. GLM's 64-frame retry succeeded after an earlier TLS failure at the same settings. Kimi and MiniMax subsequently passed 32-frame requests; their 40-, 48- and 64-frame attempts failed during connection, upload or response handling, without a model rejection. In these tests, `direct` disables the explicit HTTP proxy only; it does not prove that VPN/TUN routing was bypassed. The debug entries contain the supplied OpenRouter key; CLI runs can use `OPENROUTER_API_KEY`. [Probe results](docs/model_input_limits.md#45-openrouter-实测).

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_API_KEY"
python -m evaluation.inference.run --data-path /dataset/clip \
  --model z-ai/glm-5.3-flash --base-url https://openrouter.ai/api/v1 \
  --sample-frames --max-frames 64 --frame-max-pixels 100352 --total-pixels 6422528
```

### Clip video

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

Replace `GPT_MODEL_NAME` and `CLAUDE_MODEL_NAME` with image-capable model names available to your account, and `API_KEY` with the service key. The Qwen examples require the named weights to be served at the specified URL; replace it with your deployed service address.

Video input has **no appended subtitles by default**. Add `--with-subtitle` to include them after the media and before the question. Clip subtitles are read from the question’s `subtitles` field. Episode subtitles are read from `subtitles/<video_id>.json`, in the directory beside `evaluation/`; each file contains the subtitle list. Paths are resolved relative to this code layout, independently of the terminal working directory. No subtitle-directory argument is required. Text mode requires non-empty subtitles; missing optional clip subtitles add no subtitle block.

For `clip`, the model and API select native video or sampled frames as above. Seed uses native video by default. For `episode`, video mode always sends sampled frames. Add `--sample-frames` to use frames for clip input as well. Timestamped JPEG images are sampled across the complete video using these defaults:

| Model family | FPS | Maximum frames | Pixels per frame | Total sampled pixels |
|---|---:|---:|---:|---:|
| GPT | 1 | 128 | 262,144 | 33,554,432 |
| Claude | 2 | 80 | 262,144 | 8,028,160 |
| DeepSeek | 2 | 600 | 602,112 | 361,267,200 |
| Other models | 2 | 768 | 602,112 | Qwen budget formula below |

Explicit CLI values override the defaults. The effective frame cap also respects the configured model policy: GPT 1500, Claude and DeepSeek 600, and Seed 1280. Defaults are not universal service limits. Frame count starts from `round(duration × fps)`, then respects the frame cap, available source frames and pixel budget; the selected frames span the complete video. Native video uses the serving endpoint's own decoding rules. See [model input limits](docs/model_input_limits.md) for sources and API probes.

For 10-, 120-, and 600-second videos, GPT defaults yield about 10, 120 and 128 frames; Claude yields about 20, 80 and 80; DeepSeek yields about 20, 240 and 600. Other models yield about 20, 240 and 768 before a service-specific cap. `media` records the requested, source and effective FPS, frame counts and actual timestamps.

DeepSeek's official API accepted 600 frames at 1008×560 in a 46.583 MiB request and returned HTTP 200. The requested model was `deepseek-v4-flash`; the response reported `deepseek-flash`. Claude `claude-sonnet-4-6` succeeded through the configured relay with 80 frames at 420×224: 1.915 MiB, HTTP 200, 90.81 seconds. Repeating that request disconnected, while larger probes failed during transport or returned HTTP 524. The 80-frame default reflects the largest successful sample in this run, not a stable capacity guarantee or a replacement for the published 600-image limit. These tests do not establish answer accuracy. [Test details](docs/model_input_limits.md).

Adaptive sampling follows [Qwen-Omni’s FPS-plus-frame-cap approach](https://github.com/QwenLM/Qwen2.5-Omni/blob/d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py#L149), while retaining the existing FFmpeg decoder. Generic HTTP input permits a single frame and does not impose Qwen’s model-specific minimum or even-frame alignment. Temporary frames leave the source MP4 unchanged. Native video is sent with its original audio track; sampled-frame input has no audio unless `--with-audio` is provided. Reducing the image count does not shorten the source audio.

Frame resolution uses pixel area rather than a fixed longest edge or square size. The tested widescreen video produces 672×364 frames with GPT defaults, 420×224 when Claude reaches 80 frames, and 1008×560 with DeepSeek defaults. Shorter Claude inputs with fewer frames can use a larger per-frame allocation, up to 262,144 pixels. The decoder preserves aspect ratio approximately while aligning dimensions to multiples of 28; that alignment is a local preprocessing choice. GPT, Claude and DeepSeek use a strict budget for the sum of sampled image pixels. More frames share that budget, reducing the per-frame allocation; when the allocation reaches the 100,352-pixel minimum, the sampler reduces the frame count. `--total-pixels` overrides this budget. Pixel limits do not guarantee a JPEG size, request size, or recognition accuracy.

Other models retain the Qwen-Omni utility defaults: at least 100,352 and at most 602,112 pixels per frame, with a video-budget parameter of 90,316,800. Unless `--total-pixels` is supplied, the Qwen formula multiplies that parameter by 2 before dividing by frame count; after reaching the minimum size, more frames can increase total pixels. Small inputs may be enlarged, and minimum-size or grid rounding can overshoot a tight per-frame target in this default mode. Lowering `--frame-max-pixels` alone does not increase the frame count. Metadata records requested/effective budgets, resized dimensions and actual total pixels.

Native video is always sent as the original file, including its embedded audio track. Sampled-frame input contains no audio by default; use `--with-audio` to attach a separate audio track.

Add `--with-audio` when sampled frames need a separate audio track. Native video is sent unchanged, including its embedded audio. This flag is independent of subtitles and text output settings.

Extracted audio uses 16 kHz mono PCM16 WAV and the same zero point and duration as the video frames. Delayed audio retains its delay. Media precedes subtitles and the question. Files are temporary; original videos stay unchanged. `media.audio` and `media.audio_details` record the transmitted representation and extraction settings. 

```bash
# Add to an existing native Gemini or supported Qwen inference command:
--sample-frames
# Sampled frames with a separate audio track:
--sample-frames --with-audio
```

Native-video mode always forwards the original video, including its embedded audio track. Sampled-frame mode sends images only unless `--with-audio` is provided, in which case the extracted audio is attached separately; the selected endpoint must support that combination. [Qwen serving examples](https://github.com/QwenLM/Qwen3-Omni#vllm-serve-usage), [DashScope modality combinations](https://help.aliyun.com/en/model-studio/qwen-omni), [Gemini multimodal content](https://ai.google.dev/api/generate-content#Content).

Native input sends the prepared MP4 as Base64. DashScope requires the encoded video string to be below 10 MB; hosted `qwen3-omni-flash` also limits each video to 150 seconds. Gemini inline input is intended for modest clips; this implementation does not upload videos through Gemini Files API. Use frame input when appropriate and keep native requests within the serving endpoint's limits. [Qwen limits](https://www.alibabacloud.com/help/en/model-studio/qwen-omni), [Gemini video input](https://ai.google.dev/gemini-api/docs/video-understanding).

### Episode video

```bash
python evaluation/inference/run.py \
  --data-path /dataset/episode/questions -g episode --modality video \
  --videos-dir /dataset/episode/videos \
  --model gemini-3-flash-preview \
  --output-dir output/episode/video/gemini3-flash
```

Episode video uses the same entry point and automatically selects timestamped frames; `--sample-frames` is not required. The question specifies the answer requirements, without a separate G2 category template. Add `--with-audio` when the selected frame-based route should receive a separate audio track. Optional subtitles use `--with-subtitle` and the episode subtitle files described above.

The model defaults above also apply to episodes. For 20- and 40-minute videos, GPT's 128 frames yield about 0.107 and 0.053 FPS; Claude's 80 frames yield about 0.067 and 0.033 FPS; DeepSeek's 600 frames yield about 0.5 and 0.25 FPS. Other models' 768-frame default yields about 0.64 and 0.32 FPS before a service-specific cap. Frames span the complete video. Sampling can miss brief expressions or reactions. The full image request, and audio when included, must still fit the service's input size, duration and context limits. Adjust the sampling and pixel budgets to the experiment; an accepted request does not establish recognition accuracy.

Before sending to the official Ark or DeepSeek endpoint, the client checks the actual encoded JSON size, including inline media: Ark is limited to 64,000,000 bytes (a conservative decimal interpretation of 64 MB), and DeepSeek to 48 × 1024 × 1024 bytes (48 MiB). An oversized request fails locally with guidance to reduce `--max-frames` or `--frame-max-pixels`; it is not sent and the video tail is not discarded. Context and other service limits still apply.

### Subtitle-only text

```bash
python evaluation/inference/run.py \
  --data-path /dataset/episode/questions \
  --granularity episode --modality text \
  --model TEXT_MODEL_NAME \
  --base-url http://localhost:8000/v1 \
  --output-dir output/episode/text/model
```

Use `--granularity clip` for clip questions. Text mode always uses subtitles and requires no `--with-subtitle`. It sends subtitle text in stored order, followed by the question; it sends no video, speaker annotations, subtitle IDs or timestamps. Missing or empty subtitles are input errors. The client does not truncate subtitles; input capacity is determined by the selected model service.

Text mode builds messages directly with `prompts.text_messages()` and uses the same four API transports as video and audio. Episode video baselines use sampled frames; external long-video methods can also submit their predictions directly for evaluation.

### InternVL and audio-only input

`OpenGVLab/InternVL3_5-241B-A28B` is served separately with vLLM or LMDeploy and called through this entry point; no separate `emollm` wrapper is required. Use the model name exposed by the deployed service:

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

Audio mode reads the corresponding video using the usual `--videos-dir` rule, extracts its audio and sends no images. Qwen2-Audio input is split into chunks of at most 30 seconds; all chunks and their time positions are sent in one request for one answer. The vLLM service's `--limit-mm-per-prompt '{"audio": N}'` must permit at least the largest chunk count. Context or service limits can still reject long recordings; the client does not discard later chunks. The original `Qwen-Audio-Chat` custom-code model is not covered by this adapter. Native Gemini and supported local Qwen-Omni services can also use audio mode. [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio), [InternVL deployment](https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B#deployment).

Select a vision-capable GLM or Seed model for video. DeepSeek's current `deepseek-flash` supports images and has passed the official API probe described above; `deepseek-v4-pro` is listed as text-only at the verification date. Other newly added routes have local request tests but no authenticated inference result unless explicitly recorded. No weights were downloaded or GPU inference performed. InternVL services must also allow the configured number of image items. [Detailed capabilities and validation scope](docs/model_input_limits.md).

### Prompt and runtime options

Common instructions and answer-format templates use English. Question content is passed through unchanged.

| Option | Behavior |
|---|---|
| `--modality video\|text\|audio` | Video input, subtitle-only input, or audio extracted from the video; default `video`. |
| `--api-key KEY` | API key for the selected model service. |
| `--prompt-mode system` | Send the common prompt as a separate system instruction. Default for video, text and audio. |
| `--prompt-mode user` | Place the same instructions in the user message. |
| `--qid ID` | Select a question; repeat for multiple IDs. |
| `--limit N` | Run the first N questions after granularity and question-ID filtering; omit to run all selected questions. |
| `--output-dir DIR` | Directory for inference results. |
| `--force` | Rerun all selected inference records, including previous successful predictions. |
| `--workers N` | Concurrent requests; default 1. |
| `--tries N` | Maximum attempts per request; default 3. |
| `--max-tokens N` | Output token limit; default 16384 (16K). |
| `--temperature N` | Sampling temperature (default: `0`; the official Seed 2.0 Pro/Lite 260215 endpoints use their fixed value `1`). |
| `--timeout N` | Request timeout in seconds; default 180. |
| `--sample-frames` | Force sampled frames for clip video; episode video always uses frames. |
| `--with-audio` | Attach a separate audio track when sampled frames are used. |
| `--fps N` | Target sampling rate, a finite positive number; model defaults are listed above. |
| `--max-frames N` | Requested positive frame-count cap, subject to model and pixel-budget limits; model defaults are listed above. |
| `--frame-max-pixels N` | Per-frame pixel-area cap; minimum 100,352, further bounded by the total pixel budget. |
| `--total-pixels N` | Strict total sampled-pixel cap. GPT, Claude and DeepSeek defaults are listed above; other models retain the Qwen budget formula when omitted. |

For a quick test, add `--limit 10 --output-dir output/debug/test` to an inference command above. Directory inputs follow filename order; JSON arrays and JSONL files preserve record order.

No extra configuration file is required for normal runs. Use `--config FILE` to supply optional service-specific parameters in a JSON file.

`--thinking default` leaves provider thinking settings unchanged. Qwen3-VL Plus/Flash and Qwen3-Omni Flash support `on`/`off`; Instruct and Thinking checkpoints use separate weights, so incompatible toggles are rejected. Qwen-Omni defaults to text output. The generic entry uses non-streaming requests with one complete JSON response and has no streaming switch. Services that require streaming are not supported by this entry. Gemini 3 maps `on` to high thinking and rejects `off`; Flash's `minimal` level can be specified in `--config` and does not guarantee zero thinking. Custom GPT/Claude reasoning settings can also be supplied through `--config` with `--thinking default`.

`predictions.jsonl` contains IDs, parsed predictions, raw model responses, input references and status. Video records include the selected representation and audio status; frame records also include sampling settings and timestamps. Inference reuses successful records with the same question ID; add `--force` to rerun all selected records. `--qid` and `--limit` control the selected questions.

## Evaluation

The three specialist models keep their official source trees in `evaluation/inference/emollm/`; their entry points and the general-purpose model adapters live in `evaluation/inference/adapters/`. Each specialist model uses its own dependency environment. See the [official-model setup guide](evaluation/inference/emollm/README.md) for weights, configuration and commands. The public `run.py` entry remains dedicated to general-purpose model services. Official source trees are kept unchanged; local wrappers adapt benchmark inputs and predictions. The upstream trees and weights are not bundled into the wheel; installed wrappers require `repo_dir` pointing to a prepared checkout.

```bash
python evaluation/eval.py \
  --data-path /dataset/clip/questions --granularity clip \
  --predictions output/clip/video/model/predictions.jsonl \
  --model JUDGE_MODEL_NAME --api-key API_KEY \
  --base-url http://localhost:8000/v1 \
  --output-dir output/clip/scores/model
```

Each evaluation invocation scores every question of the chosen granularity in `--data-path` again; evaluation has no question-ID filter.

The presence of `rubric` determines the scoring route:

- **`rubric: null`:** code computes label Precision, Recall, F1 and EM. Transition scores the before/after sets separately and averages each metric. Duplicate labels are removed; unknown labels remain incorrect predictions.
- **A provided rubric:** an LLM evaluates the question, reference answer, `answer_details` and prediction using that rubric. The allowed scores come directly from the question's rubric.

During annotation, store a rubric for every LLM-scored question, including G2 explanation and result questions. Only label questions use `null`.

`answer_details` explains required content or alternative valid answers. It does not itself assign points. Its description and items are included with the reference answer. The judge System Prompt includes all four examples and requires only `score` and `reason` in its JSON output.

Label-only evaluation does not initialize an API client and requires no `--model`, `--base-url` or API key. The judge client is configured only when LLM scoring is needed. The independent AffectGPT, Emotion-LLaMA and R1-Omni entries already save predictions in this evaluator's format. Other external methods and agents can submit at least the following fields for each answer:

```json
{"question_id": "Q000001", "prediction": "3"}
```

Predictions are matched only by `question_id` to questions selected by `--granularity`; prediction fields `granularity`, `video_id`, `status` and `error` are ignored. The evaluator uses `prediction` when non-null and non-blank, otherwise `pred_answer`. Every open answer is submitted as text. Label predictions may be label lists; transition predictions may use `before` and `after` lists. See [data_format.md](docs/data_format.md) for details.

### Scores and saved outputs

Each evaluation run creates a new timestamp-named subdirectory under `--output-dir`, containing the files below. The actual directory is printed at runtime, and previous evaluation runs are preserved. Evaluation always processes the complete question set for the chosen granularity and does not use `--force`.

| Output | Contents |
|---|---|
| `scores.jsonl` | Per-question scores, evaluation inputs, raw judge responses and errors. |
| `metrics.json` | Metrics by task `type`, raw/normalized/percentage scores, coverage and status counts. |
| `summary.md` | Readable results by task `type`. |

Label metrics are averaged per question. Trajectory reports the raw mean out of 4, normalized mean divided by 4, and percentage score multiplied by 25. Cause explanations report the raw mean out of 3 and its normalized/percentage equivalents. Result questions report ACC.

Within G2 emotional reasoning, the 0/1 results and graded explanations retain their separate scores and the agreed weighting: `0.6 × result ACC + 0.4 × normalized explanation mean`. Each LLM score is normalized by the maximum in its rubric. The unweighted mean includes every scored question equally. If only one part has scored answers, use that part's normalized mean; absent scores are `null`, not zero. The overall unweighted aggregate uses label F1 and normalized LLM scores.

Reports group questions directly by `type`, without additional answer-type groups or mapping files. Score ranges do not change a question's task type.

If neither answer field contains a response, the prediction is missing and unscored; null, empty and whitespace-only strings count as missing. Missing predictions and evaluation failures lower reported coverage. Invalid judge JSON or an out-of-range score is retried, then reported as an evaluation error rather than a wrong model answer. Exit status is nonzero when any selected question remains unscored because of missing input or execution failure.

## Verification

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic questions and a local HTTP service to verify request assembly, response parsing, label/LLM routing, score aggregation, failures and repeated runs. They do not call live model providers or evaluate the actual dataset; endpoint-specific model/media support must be checked when running that service.

### Direct Transformers inference

For a locally loaded Qwen checkpoint, use the separate Transformers entry point
instead of the HTTP deployment entry point:

```bash
python -m evaluation.inference.transformers \
  --data-path /path/to/questions \
  --model /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --videos-dir /path/to/videos \
  --output-dir output/transformers
```

The model name selects the native family automatically: Qwen2.5/Qwen3 Omni uses
`qwen_omni_utils` and `use_audio_in_video=True`, Qwen2/Qwen2.5/Qwen3 VL uses
`qwen_vl_utils` with native video, and Qwen2-Audio extracts a temporary WAV for the
official audio processor. InternVL checkpoints use the official frame-based
`model.chat` path. This entry point is separate from
`evaluation.inference.run` (the HTTP/API runner). Install the packages and versions
required by the selected checkpoint; audio/video processor utilities are loaded only
when that family is selected.
