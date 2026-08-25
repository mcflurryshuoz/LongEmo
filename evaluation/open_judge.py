#!/usr/bin/env python3
"""LLM-as-judge for open answers with a single correct answer (binary 0/1 verdict).

Any OpenAI-compatible endpoint works. Model, endpoint, and API key are supplied
explicitly by the scoring command.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation import io_utils  # noqa: E402


JUDGE_FIELDS = {"correct", "score", "reason", "matched"}

def build_judge_prompt(q: dict[str, Any], pred_answer: Any) -> str:
    payload = {
        "question": io_utils.question_text(q),
        "gold_answer": q.get("answer"),
        "model_answer": pred_answer,
    }
    return (
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


class OpenAICompatibleJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180,
        thinking: str = "default",
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY", "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model
        self.thinking = thinking
        host = (urlparse(base_url or "").hostname or "").lower()
        if thinking != "default" and host != "api.deepseek.com":
            raise ValueError("--thinking on/off is supported only for the official DeepSeek endpoint")

    @classmethod
    def from_args(cls, args: Any) -> "OpenAICompatibleJudge":
        missing = [
            flag
            for flag, value in (
                ("--model", getattr(args, "model", None)),
                ("--base-url", getattr(args, "base_url", None)),
                ("--api-key", getattr(args, "api_key", None)),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"open-answer scoring requires: {', '.join(missing)}")
        return cls(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=float(getattr(args, "timeout", 180)),
            thinking=str(getattr(args, "thinking", "default")),
        )

    def judge(self, q: dict[str, Any], pred_answer: Any, *, tries: int = 3) -> dict[str, Any]:
        prompt = build_judge_prompt(q, pred_answer)
        last = None
        for k in range(tries):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.thinking != "default":
                    request["extra_body"] = {
                        "thinking": {
                            "type": "enabled" if self.thinking == "on" else "disabled"
                        }
                    }
                resp = self.client.chat.completions.create(**request)
                raw = resp.choices[0].message.content or ""
                parsed = json.loads(raw)
                if (
                    isinstance(parsed, dict)
                    and set(parsed) == JUDGE_FIELDS
                    and isinstance(parsed["correct"], bool)
                    and type(parsed["score"]) is int
                    and parsed["score"] in {0, 1}
                    and parsed["score"] == int(parsed["correct"])
                    and isinstance(parsed["reason"], str)
                    and isinstance(parsed["matched"], str)
                ):
                    parsed["raw_judge"] = raw
                    parsed["judge_model"] = self.model
                    parsed["thinking"] = self.thinking
                    return parsed
                last = {"raw_judge": raw, "error": "judge returned invalid JSON fields"}
            except Exception as e:
                last = {"error": f"{type(e).__name__}: {e}"}
            if k + 1 < tries:
                time.sleep(2 * (k + 1))
        return last or {"error": "judge failed"}


def judge_open_answer(q: dict[str, Any], pred_answer: Any, judge: OpenAICompatibleJudge) -> dict[str, Any]:
    """Binary accept/reject verdict for an open answer; returns correct + full judge verdict."""
    verdict = judge.judge(q, pred_answer)
    correct_raw = verdict.get("correct")
    correct = correct_raw if isinstance(correct_raw, bool) else None
    return {"correct": correct, "score_method": "llm_judge", "judge": verdict}
