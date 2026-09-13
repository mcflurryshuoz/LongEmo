"""Shared CLI configuration, checkpointing and bounded request retries."""

from __future__ import annotations
import json
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from ..clients import Client
from ..io_utils import write_records, load_records
from .adapters import prepare_request


OFFICIAL_API_URLS = {
    "gpt": "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "minimax": "https://api.minimax.io/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "seed": "https://ark.cn-beijing.volces.com/api/v3",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
API_KEY_ENVS = {
    "gpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "minimax": "MINIMAX_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "glm": "ZHIPUAI_API_KEY",
    "seed": "ARK_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def model_args(parser):
    parser.add_argument("--model")
    parser.add_argument(
        "--base-url",
        default=os.getenv("MODEL_BASE_URL"),
        help="Service URL or full endpoint; model and URL determine the API automatically",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Credential; otherwise use the service's API key environment variable or MODEL_API_KEY; never written to results",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--thinking", choices=("default", "on", "off"), default="default"
    )
    parser.add_argument(
        "--config",
        help="Optional JSON file overriding model request settings",
    )


def setup_api(args, config):
    """Configure the model service, request settings and API key together."""
    args.model_family = detect_model_family(args.model)
    args.base_url = args.base_url or OFFICIAL_API_URLS.get(args.model_family)
    if not args.base_url:
        raise ValueError(
            "this model has no default service address; provide --base-url "
            "or MODEL_BASE_URL, for example http://localhost:8000/v1"
        )

    prepare_request(args, config)

    if args.api_key is None:
        if urlparse(args.base_url).hostname == "openrouter.ai":
            args.api_key = os.getenv(API_KEY_ENVS["openrouter"]) or os.getenv(
                "MODEL_API_KEY", ""
            )
            return
        provider = "qwen" if args.model_family.startswith("qwen") else args.model_family
        fallback = {"gemini": "gemini", "anthropic": "claude"}.get(
            args.api_format, "gpt"
        )
        key_env = API_KEY_ENVS.get(provider, API_KEY_ENVS[fallback])
        args.api_key = os.getenv("MODEL_API_KEY") or os.getenv(key_env, "")


def detect_model_family(model):
    """Identify the model series from its name, independently of the service URL."""
    model = (model or "").lower().rsplit("/", 1)[-1]
    if "qwen" in model and "audio" in model:
        return "qwen_audio"
    if "qwen" in model and "omni" in model:
        return "qwen_omni"
    if "qwen" in model and "vl" in model:
        return "qwen_vl"
    if "gemini" in model:
        return "gemini"
    if "claude" in model:
        return "claude"
    if model.startswith("gpt-") or re.match(r"^o[134](?:-|$)", model):
        return "gpt"
    if model.startswith("minimax"):
        return "minimax"
    if model.startswith(("kimi", "moonshot")):
        return "kimi"
    if model.startswith(("glm", "chatglm")):
        return "glm"
    if model.startswith(("doubao-", "seed")):
        return "seed"
    if model.startswith("internvl"):
        return "internvl"
    if model.startswith("deepseek"):
        return "deepseek"
    return "other"


def init_client(args):
    """Initialize the model API client from CLI arguments and optional JSON settings."""
    if args.tries < 1 or args.workers < 1:
        raise ValueError("tries and workers must be positive")
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("--config must contain a JSON object")
    setup_api(args, config)
    return Client(
        args.model or "MODEL_NAME",
        args.base_url,
        args.api_format,
        args.api_key,
        args.timeout,
        args.max_tokens,
        args.temperature,
        config,
    )


def retry(request_fn, tries):
    for attempt in range(tries):
        try:
            return request_fn()
        except Exception as exc:
            if attempt + 1 == tries:
                raise
            print(
                f"Request failed ({type(exc).__name__}); retrying {attempt + 2}/{tries}",
                flush=True,
            )
            time.sleep(min(attempt + 1, 3))


def execute_tasks(tasks, task_runner, output_path, *, workers=1, force=False):
    """Run tasks and reuse successful records already present in the output file."""
    old = (
        {}
        if force or not Path(output_path).exists()
        else {x["key"]: x for x in load_records(output_path)}
    )
    records = {}
    pending = []
    for task in tasks:
        key = task[0]
        previous = old.get(key)
        if previous and previous.get("status") == "ok" and not force:
            records[key] = previous
        else:
            pending.append(task)

    def finish_task(task):
        key, base = task[0], task[-1]
        print(f"[{key}] started", flush=True)
        try:
            result = task_runner(task)
        except Exception as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        return key, {**base, **result, "key": key}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(finish_task, task): task for task in pending}
        for future in as_completed(futures):
            key, record = future.result()
            records[key] = record
            write_records(
                output_path, [records[t[0]] for t in tasks if t[0] in records]
            )
            print(
                f"[{len(records)}/{len(tasks)}] {key}: {record['status']}", flush=True
            )
    ordered = [records[t[0]] for t in tasks]
    write_records(output_path, ordered)
    return ordered
