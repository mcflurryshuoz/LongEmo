# 官方情绪模型的独立推理入口

本目录统一保存 AffectGPT、Emotion-LLaMA 和 R1-Omni 的官方源码；模型适配代码位于相邻的 `../adapters/`。三个独立入口分别运行在各自的 Python/CUDA 环境中，直接调用本目录内的官方模型代码。它们读取相同的 LongEmoBench clip 问题和 `<video_id>.mp4`，输出标准 `predictions.jsonl`。benchmark 的公共说明、原始问题、回答格式一起传给模型；推理 prompt 不包含参考答案或评分 rubric。

| 独立入口 | 官方调用 | 官方代码目录 |
| --- | --- | --- |
| [affectgpt.py](../adapters/affectgpt.py) | `BaseDataset` 预处理、`Chat.answer_sample` | [AffectGPT/AffectGPT](AffectGPT/AffectGPT) |
| [emotion_llama.py](../adapters/emotion_llama.py) | `EmotionLLaMARuntime.analyze` | [Emotion-LLaMA](Emotion-LLaMA) |
| [r1_omni.py](../adapters/r1_omni.py) | `model_init`、processor、`mm_infer` | [R1-Omni](R1-Omni) |

以下命令均在 `LongEmoBenchv1/` 根目录执行。示例中的 `/models`、`/configs`、`/dataset` 需要换成实际本地路径。`--data-path` 接受 JSON 文件、JSONL 文件或平铺的 JSON/JSONL 题目目录。省略 `--videos-dir` 时，默认使用 `--data-path` 目录下的 `videos/`；若传入题目文件，则使用该文件所在目录下的 `videos/`。显式指定 `--videos-dir` 可覆盖默认值。`-g` 是 `--granularity` 的简写，这三个推理入口使用 `clip`。JSON 中的相对 `*_path`、`*_dir` 和 `checkpoint` 相对于 JSON 文件所在目录解析。省略 `repo_dir` 时使用上述随附源码；省略 `config_path` 时使用各适配器的官方默认 YAML。

## AffectGPT

官方 [环境文件](AffectGPT/AffectGPT/environment.yml) 是 Linux/CUDA 导出环境，主要版本为 Python 3.10、Torch 2.4.0+cu121、Transformers 4.49.0。环境准备命令如下；本次代码接入没有执行安装。

```bash
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 conda env create \
  -n longemobench-affectgpt \
  -f evaluation/inference/emollm/AffectGPT/AffectGPT/environment.yml
conda activate longemobench-affectgpt
```

准备完整的 [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)、[CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)、[chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large) 目录，以及官方 **raw-frame** checkpoint。该 checkpoint 对应 `mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz`，下载位置见官方 [单视频推理说明](AffectGPT/AffectGPT/README.md#-inference-for-single-video)。`model_path` 填该版本实际的 `.pth` 文件。

适配器默认选用 [frame YAML](AffectGPT/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml)，并检查 attention fusion、CLIP、HuBERT 和 Qwen25 配置。常规 `emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz` 是依赖 OpenFace 的 face 版本，其权重不能用于这个入口。

保存 `/configs/affectgpt.json`：

```json
{
  "model_path": "/models/affectgpt-frame/checkpoint_30.pth",
  "llm_path": "/models/Qwen2.5-7B-Instruct",
  "visual_encoder_path": "/models/clip-vit-large-patch14",
  "audio_encoder_path": "/models/chinese-hubert-large",
  "device": "cuda:0",
  "generation": {
    "max_new_tokens": 1024,
    "do_sample": false
  }
}
```

```bash
python evaluation/inference/adapters/affectgpt.py \
  --runtime-config /configs/affectgpt.json \
  --data-path /dataset/clip/questions \
  --videos-dir /dataset/clip/videos \
  --output-dir output/clip/video/affectgpt
```

需要 PATH 中有 `ffmpeg`，或在 JSON 中设置 `ffmpeg_path`。也可设置 `audio_dir`，使用对应的 `<video_id>.wav`。实际模型输入是覆盖视频的均匀 **8 帧、224×224** 和分布在音轨中的 **8 段、每段 2 秒音频**；短视频重复末帧，短音频补零。适配器不生成 ASR 文本，不读取官方训练标注。输入及输出预算超过 LLM 上下文长度时返回错误。

## Emotion-LLaMA

按官方 [environment.yaml](Emotion-LLaMA/environment.yaml) 和 [demo 说明](Emotion-LLaMA/README.md#local-demo) 创建独立环境：

```bash
conda env create -n longemobench-emotion-llama \
  -f evaluation/inference/emollm/Emotion-LLaMA/environment.yaml
conda activate longemobench-emotion-llama
python -m pip install moviepy==1.0.3 soundfile==0.12.1 opencv-python==4.7.0.72
```

官方环境使用 Python 3.9、Torch 2.0.0、Transformers 4.30.0、PEFT 0.2.0、timm 0.6.13。源码直接入口支持这套 Python 3.9 用法；根项目的 `pip install -e .` 声明 Python ≥3.10，因此在该模型环境中直接运行下面的脚本。官方 `requirements.txt` 与 `environment.yaml` 的部分版本不同，应分别维护环境，不能混装三个模型的依赖。

必需资产：

- 完整的 [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) 目录。
- 官方 demo 的 [Emotion-LLaMA checkpoint](https://drive.google.com/file/d/1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY/view)；官方文件名为 `Emoation_LLaMA.pth`。
- 完整的 [chinese-hubert-large](https://huggingface.co/TencentGameMate/chinese-hubert-large) 目录。
- [EVA ViT-G 权重](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth)：放在该环境 `timm.models.hub.get_cache_dir()` 返回的缓存目录下，文件名为 `eva_vit_g.pth`。上游在缓存缺失时会下载此文件；离线运行需预先准备。此缓存通常为 Torch hub 的 `checkpoints` 目录，可在模型环境中用 `python -c 'from timm.models.hub import get_cache_dir; print(get_cache_dir())'` 查看。

保存 `/configs/emotion_llama.json`：

```json
{
  "model_path": "/models/Llama-2-7b-chat-hf",
  "checkpoint": "/models/Emoation_LLaMA.pth",
  "audio_model_path": "/models/chinese-hubert-large",
  "device": "cuda:0",
  "generation": {
    "temperature": 0.2,
    "max_new_tokens": 500,
    "max_length": 4096,
    "seed": 42
  }
}
```

```bash
python evaluation/inference/adapters/emotion_llama.py \
  --runtime-config /configs/emotion_llama.json \
  --data-path /dataset/clip/questions \
  --videos-dir /dataset/clip/videos \
  --output-dir output/clip/video/emotion_llama
```

默认使用 [demo.yaml](Emotion-LLaMA/eval_configs/demo.yaml) 的 8-bit CUDA 模式。`options` 可传官方 OmegaConf `key=value` 列表，例如 `model.low_resource=false`；上述三个模型路径通过配置覆盖。

模型实际看到**首帧**和**完整音轨经 HuBERT 最后一层均值池化的特征**，face 和 temporal feature 的两个槽位为零。视频必须有可解码音轨，MoviePy 需要可用的 ffmpeg。该入口不添加长视频分块或额外时序编码。官方生成固定使用采样，`temperature` 必须大于零。`max_length` 控制上下文预算，须容纳输入与输出预算之和且符合 Llama-2 的容量；适配器在官方裁剪前检查实际输入长度，超限时报告错误。

## R1-Omni

官方 [README](R1-Omni/README.md#️-environment-setup) 指向 R1-V 环境，列出 Torch 2.5.1+cu124、Transformers 4.49.0 和 FlashAttention 2.7.4；[setup.sh](R1-Omni/setup.sh) 安装随附 `src/r1-v[dev]`，该包要求 Python ≥3.10.9，且把 Transformers 指向一个 Git revision。两处说明并非完全一致，以下保留官方安装路径，并补充 HumanOmni 源码直接导入的媒体依赖；这组安装命令尚未在 GPU 环境执行验证。

```bash
conda create -n longemobench-r1-omni python=3.10.16
conda activate longemobench-r1-omni
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e "evaluation/inference/emollm/R1-Omni/src/r1-v[dev]"
python -m pip install decord==0.6.0 moviepy==1.0.3 opencv-python imageio ipdb h5py timm
```

准备完整的 [R1-Omni-0.5B](https://huggingface.co/StarJiaxing/R1-Omni-0.5B)、[SigLIP](https://huggingface.co/google/siglip-base-patch16-224)、[Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) 和 [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 目录，包括各自的配置、模型权重和 tokenizer/processor 文件。BERT 是官方模型架构的必要组件。

保存 `/configs/r1_omni.json`：

```json
{
  "model_path": "/models/R1-Omni-0.5B",
  "vision_model_path": "/models/siglip-base-patch16-224",
  "audio_model_path": "/models/whisper-large-v3",
  "bert_model_path": "/models/bert-base-uncased",
  "device": "cuda:0",
  "modal": "video_audio",
  "use_flash_attn": false,
  "generation": {
    "do_sample": false,
    "max_new_tokens": 2048
  },
  "seed": 42
}
```

```bash
python evaluation/inference/adapters/r1_omni.py \
  --runtime-config /configs/r1_omni.json \
  --data-path /dataset/clip/questions \
  --videos-dir /dataset/clip/videos \
  --output-dir output/clip/video/r1_omni
```

本地视觉和音频目录名必须分别包含 `siglip`、`whisper`，供上游选择 encoder。适配器在内存中覆盖官方硬编码资产路径。`use_flash_attn=false` 使用普通 attention；启用时需安装与 CUDA/Torch 匹配的官方 FlashAttention 依赖。

视频按 checkpoint 的 `num_frames` 在整段范围内均匀采样；Whisper 音频窗口通常只覆盖**开头 30 秒**，短音频按官方 processor 补齐，没有音频分块汇总。音频解码失败会记录错误，不采用上游的静音替代结果。`modal=video` 可关闭音频输入，但 checkpoint 中存在的 audio tower 仍会加载，仍需相应权重。完整 benchmark prompt 同时传给 LLM 和 BERT question 分支；BERT 分支保留上游 tokenizer 的截断行为。

## 运行、续跑和评分

三个入口均默认关闭字幕。需要显式加入字幕时，在对应命令上添加 `--with-subtitle`，字幕读取自题目 JSON 的 `subtitles` 字段。公共说明和回答格式统一使用英文，题目内容按原文传入。当前三个入口支持 `clip`，视频需提前准备为题目对应的片段。提示词沿用各模型官方入口的封装。

推理支持 `--limit N`：先按粒度和题号筛选，再取前 N 题；未指定时处理全部筛选结果。目录输入按文件名排序读取，JSON 数组和 JSONL 按记录原顺序读取。快速测试可添加 `--limit 10 --output-dir output/debug/test`。

重复运行会复用 `predictions.jsonl` 中按 `question_id` 匹配的已有成功记录，并重试失败项。首次读取大模型权重需要相应时间。`--force` 会重跑本次选中的全部题目；需要在输入内容或运行设置变化后重新推理时应显式使用该选项。输出保存到 `predictions.jsonl`；成功记录保存原始回答、解析后答案、实际请求和模态元数据。

完成推理后，在 benchmark 的评分环境中使用现有入口；三个模型使用相同预测格式和相同评分规则。下面的 judge 地址、模型名与 `API_KEY` 需替换成所选评审服务的信息：

```bash
python -m evaluation.eval \
  --data-path /dataset/clip/questions \
  --predictions output/clip/video/affectgpt/predictions.jsonl \
  --granularity clip \
  --model JUDGE_MODEL \
  --base-url http://localhost:8000/v1 --api-key API_KEY \
  --output-dir output/clip/scores/affectgpt
```

## 来源和验证边界

[sources.json](sources.json) 保存三个官方仓库 URL 与固定 commit：

| 模型 | Commit |
| --- | --- |
| AffectGPT | `fffca794c6792023234565370e3f69f0488aede9` |
| Emotion-LLaMA | `e69faadfd3824b94dd334f463cc5bbdb27229c6d` |
| R1-Omni | `17cafcae0b0a1c454896f4eb7a1c006bf13e1f4d` |

各仓库保留原始源码、Git revision 信息和各自许可证/使用条款；本目录没有为第三方代码或模型重新指定统一许可证。`predictions.jsonl` 中成功记录的 `media` 保存适配器运行时元数据，包括来源 revision 和模态处理信息。

本次验证覆盖无权重接口测试，尚未执行真实 GPU 模型推理，也未下载额外权重、数据集或安装模型环境。正式结果应结合实际运行输出确认资产版本、模态覆盖和生成设置。
