"""Prompt text for the model-controlled coarse-to-fine loop."""

SYSTEM_PROMPT = """You analyze a video question using timestamped image frames.
The complete video is first shown at low resolution. You may request additional
frames from selected time ranges before answering. Use only the supplied video,
optional audio, optional subtitles, and the question. Do not invent events.
Reply with exactly one JSON object: either {\"action\":\"inspect\",\"ranges\":[...]} or
{\"action\":\"answer\",\"answer\":\"...\"}. An inspect range has start, end, fps,
max_frames and max_pixels. Do not include audio or subtitle switches: those
inputs are fixed for this run. Request only evidence that is still necessary.
"""


def initial_prompt(question: dict, *, with_audio: bool, with_subtitle: bool) -> str:
    return (
        "Review the initial low-resolution overview and the question below. "
        "Request focused frame ranges if needed; otherwise answer directly.\n\n"
        f"Question: {question.get('question', '')}\n"
        f"Audio provided: {'yes' if with_audio else 'no'}; "
        f"subtitles provided: {'yes' if with_subtitle else 'no'}"
    )


def followup_prompt() -> str:
    return "Here is the requested additional visual evidence. Continue the analysis and return the next JSON action."
