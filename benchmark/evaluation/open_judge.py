#!/usr/bin/env python3
"""LLM-as-judge for open answers with a single correct answer (binary 0/1 verdict).

Also serves as the client for OV-MER GPT-based grouping so one judge model backs
all quality judgments. Any OpenAI-compatible endpoint works; configuration is
CLI args > JUDGE_MODEL / JUDGE_BASE_URL / JUDGE_API_KEY env vars > defaults
(deepseek-v4-flash @ api.deepseek.com).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import io_utils  # noqa: E402

DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com"


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_env_or_args(cls, args: Any) -> "OpenAICompatibleJudge":
        return cls(
            model=getattr(args, "judge_model", None) or os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
            base_url=getattr(args, "judge_base_url", None) or os.environ.get("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
            api_key=getattr(args, "judge_api_key", None) or os.environ.get("JUDGE_API_KEY", "EMPTY"),
            timeout=float(getattr(args, "judge_timeout", 180)),
            temperature=float(getattr(args, "judge_temperature", 0.0)),
            max_tokens=int(getattr(args, "judge_max_tokens", 4096)),
        )

    def judge(self, q: dict[str, Any], pred_answer: Any, *, tries: int = 3) -> dict[str, Any]:
        # the verdict only needs the question for context plus the two answers
        payload = {
            "question": io_utils.question_text(q),
            "gold_answer": q.get("answer"),
            "model_answer": pred_answer,
        }
        prompt = (
            "You are a strict evaluator for video emotion QA. Judge whether the model_answer should be "
            "accepted as correct for the gold_answer. Do not require exact wording; judge semantic "
            "equivalence.\n\n"
            "Rules by answer type:\n"
            "- Character name: correct only if it refers to the same person (nicknames and full names "
            "count); the wrong person or several people is wrong.\n"
            "- Moment/event phrase: correct only if it clearly refers to the same moment or event as the "
            "gold answer, even in different words; a different scene, a vague description that fits many "
            "moments, or mere background plot is wrong.\n"
            "- Other short answers: correct only if semantically equivalent to the gold; reject answers "
            "about the wrong person, missing the required emotion or its direction, or omitting the "
            "required cue/reason.\n\n"
            "Return ONLY JSON with keys: correct (true/false), score (0 or 1), reason (short), matched (short).\n\n"
            f"Evaluation item:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        last = None
        for k in range(tries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content or ""
                parsed = io_utils.parse_jsonish(raw)
                if isinstance(parsed, dict):
                    parsed["raw_judge"] = raw
                    return parsed
                last = {"raw_judge": raw, "error": "judge returned non-json"}
            except Exception as e:
                last = {"error": f"{type(e).__name__}: {e}"}
                time.sleep(2 * (k + 1))
        return last or {"error": "judge failed"}


def judge_open_answer(q: dict[str, Any], pred_answer: Any, judge: OpenAICompatibleJudge) -> dict[str, Any]:
    """Binary accept/reject verdict for an open answer; returns correct + full judge verdict."""
    verdict = judge.judge(q, pred_answer)
    correct_raw = verdict.get("correct")
    score_raw = verdict.get("score")
    correct = bool(correct_raw) if isinstance(correct_raw, bool) else None
    if correct is None and isinstance(score_raw, (int, float)):
        correct = float(score_raw) >= 0.5
    return {"correct": correct, "score_method": "llm_judge", "judge": verdict}
