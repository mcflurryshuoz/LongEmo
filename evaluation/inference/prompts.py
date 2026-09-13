"""Shared open-QA inference templates; no model calls or artifact generation."""

from __future__ import annotations
import copy
import re
import json
from ..io_utils import EMOTIC_26_LABELS, norm_label

labels = EMOTIC_26_LABELS
vocab = ", ".join(labels)
multimodal_prompt = f"""You are an expert in multimodal emotion understanding.

### Task
Answer the question using the available visual, audio, and textual evidence.

### Response Requirements
1. When emotion labels are requested, select one or more labels from the emotion vocabulary below. Use the label names exactly as written.
2. For all other questions, provide the information requested by the question. These answers are not restricted to the emotion vocabulary.
3. Follow the response format specified with the question and do not add unrelated content.

### Emotion Vocabulary
{vocab}"""

g1_layout = "{{subtitle_block}}\n\n### Question\n{{question}}\n\n### Answer Format\n{{answer_format}}"
formats = {
    "emotion_labels": "Your answer should consist of one or more labels from the provided emotion vocabulary. Separate multiple labels with commas.",
    "emotion_before_after": "Your answer should consist of two lines in the format below. Each line should contain one or more labels from the provided emotion vocabulary, separated by commas:\nBefore: <emotion labels>\nAfter: <emotion labels>",
    "free_text": "Your answer should be expressed in natural language.",
}
g1_tasks = {
    "contextual emotion": ("g1_contextual", "Contextual emotion", "emotion_labels"),
    "emotion transition": ("g1_transition", "Transition", "emotion_before_after"),
    "emotion trajectory": ("g1_trajectory", "Trajectory", "free_text"),
    "emotion cause": ("g1_cause", "Cause", "free_text"),
    "emotion influence": ("g1_influence", "Influence", "emotion_labels"),
}
g1_system_templates = {
    task: g1_layout.replace("{{answer_format}}", formats[format_key])
    for task, (_, _, format_key) in g1_tasks.items()
}


def fill_template(template, values):
    return re.sub(r"\{\{([a-z_]+)\}\}", lambda m: values[m.group(1)], template)


def build_messages(
    q,
    *,
    subtitle_text=None,
    multimodal=True,
    answer_requirements=None,
    prompt_mode="system",
    media_content=None,
):
    """Assemble the prompt text, optional subtitles, and media parts.

    The question record already follows the benchmark schema. This helper only
    arranges the provided values; it does not validate or reinterpret them.
    The caller provides subtitle text only when subtitles are enabled.
    """
    subtitle_text = (subtitle_text or "").strip()
    subtitle_block = "### Subtitles\n" + subtitle_text if subtitle_text else ""
    if q["granularity"] == "clip":
        if q["type"] not in g1_system_templates:
            raise ValueError("Unsupported G1 task.")
        user = fill_template(
            g1_system_templates[q["type"]],
            {"question": q["question"], "subtitle_block": subtitle_block},
        ).strip()
    else:
        # G2 requirements come with the question; never derive them from gold fields.
        parts = [subtitle_block] if subtitle_block else []
        parts.append(q["question"].strip())
        if answer_requirements and answer_requirements.strip():
            parts.append("Answer requirements:\n" + answer_requirements.strip())
        user = "\n\n".join(parts)
    system_text = multimodal_prompt if multimodal else subtitle_prompt
    if prompt_mode == "system":
        messages = [{"role": "system", "content": system_text}]
        prefix = ""
    else:
        messages = []
        prefix = system_text
    if media_content:
        content = (
            ([{"type": "text", "text": prefix}] if prefix else [])
            + copy.deepcopy(media_content)
            + [{"type": "text", "text": user}]
        )
    else:
        content = prefix + "\n\n" + user if prefix else user
    return messages + [{"role": "user", "content": content}]


subtitle_prompt = multimodal_prompt.replace(
    "You are an expert in multimodal emotion understanding.",
    "You are an expert in emotion understanding.",
).replace(
    "Answer the question using the available visual, audio, and textual evidence.",
    "Answer the question using only the provided subtitles. No video or audio is provided.",
)


def text_messages(question, transcript, *, prompt_mode="system"):
    """Build text-only messages without reading answers or reference annotations."""
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Text inference requires non-empty subtitles.")
    return build_messages(
        question,
        subtitle_text=transcript,
        multimodal=False,
        prompt_mode=prompt_mode,
    )


def final_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    match = re.fullmatch(r"<answer>(.*?)</answer>", text, flags=re.S)
    return (match.group(1) if match else text).strip()


def json_object(text):
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1)
    return json.loads(text)


def label_terms(value):
    if isinstance(value, str):
        value = [s for s in re.split(r"[,，;；\n]", value) if s.strip()]
    if not isinstance(value, list):
        return ["__invalid_label_format__"]
    result = []
    for term in value:
        if isinstance(term, str):
            term = norm_label(term.strip(" \t\r\n\"'`.,;:，；。"))
            if term:
                result.append(term)
        else:
            result.append("__invalid_label_value__")
    return list(dict.fromkeys(result))


def parse_prediction(q, value):
    if q.get("rubric") is not None:
        if not isinstance(value, str):
            raise ValueError("open-ended predictions must be text")
        return final_text(value), None
    if isinstance(value, str):
        value = final_text(value)
        try:
            value = json_object(value)
        except (ValueError, TypeError):
            pass
    if q["type"] != "emotion transition":
        parsed = label_terms(value)
        invalid = [x for x in parsed if x not in EMOTIC_26_LABELS]
        return parsed, "unrecognized labels: " + ", ".join(invalid) if invalid else None
    extra = False
    if isinstance(value, str):
        matches = list(re.finditer(r"(?im)^\s*(before|after)\s*:\s*([^\n]*)", value))
        parts = {m.group(1).lower(): m.group(2) for m in matches}
        extra = bool(
            re.sub(r"(?im)^\s*(before|after)\s*:\s*[^\n]*", "", value).strip()
        ) or len(matches) != len(parts)
    elif isinstance(value, dict):
        parts = value
        extra = bool(set(parts) - {"before", "after"})
    else:
        parts, extra = {}, True
    result = {slot: label_terms(parts.get(slot, [])) for slot in ("before", "after")}
    if extra:
        for values in result.values():
            values.append("__unassigned_transition_content__")
    invalid = (
        extra
        or set(parts) != {"before", "after"}
        or any(x not in EMOTIC_26_LABELS for values in result.values() for x in values)
    )
    return result, "invalid or incomplete before/after format" if invalid else None
