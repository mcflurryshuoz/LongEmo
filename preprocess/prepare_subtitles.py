#!/usr/bin/env python3
"""Export dialogue-only subtitles from each series' s1_perception files.

The output mirrors the benchmark data layout:

  subtitles/<series>/<episode>.json

Speaker labels, subtitle credits, stage directions, sound-effect captions, and
pure non-dialogue lines are removed. Spoken dialogue and its original unit
boundaries are preserved.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Iterable


_BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]|（[^（）]*）|【[^【】]*】")
_HTML_TAG = re.compile(r"<[^>]+>")
_VISIBLE_TEXT = re.compile(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]")
_CAPTION_CREDIT = re.compile(
    r"(?i)^(?:caption(?:ed|ing)?|subtitles?|subbed|sync(?:ed|ing)?)\s+(?:by|and)|"
    r"^english\s*[-–—]\s*us\s*[-–—]\s*psdh$|"
    r"^(?:https?://|www\.)"
)
_NON_DIALOGUE_LINE = re.compile(
    r"(?ix)^\s*(?:"
    r"applause|audience\s+(?:laughs?|laughter|applause)|"
    r"laughs?|laughter|laughing|chuckles?|chuckling|giggles?|giggling|"
    r"sighs?|sighing|gasps?|gasping|groans?|groaning|grunts?|grunting|"
    r"sobs?|sobbing|sniffles?|sniffling|scoffs?|scoffing|"
    r"screams?|screaming|shrieks?|shrieking|whimpers?|whimpering|wailing|"
    r"cheers?|cheering|claps?|clapping|crowd\s+(?:cheers?|cheering)|"
    r"music|theme\s+music|singing|song|"
    r"doorbell(?:\s+rings?|\s+ringing)?|knocking|footsteps|"
    r"phone(?:\s+rings?|\s+ringing)|line\s+ringing|"
    r"beeps?|beeping|buzz(?:es|ing)?|whistles?|whistling|"
    r"thud|crash|bang|gunshot|rustling|clattering|"
    r"掌声|笑声|大笑|轻笑|叹气|叹息|哭声|哭泣|抽泣|"
    r"喘气|尖叫|喊叫|咳嗽|打喷嚏|欢呼|音乐|歌声|唱歌|演唱|"
    r"门铃|敲门声|脚步声|电话铃声|铃声|音效"
    r")\s*[.!?。！？…-]*\s*$"
)

_GENERIC_SPEAKERS = {
    "all",
    "announcer",
    "audience",
    "both",
    "children",
    "crowd",
    "christy",
    "dr. koothrappali",
    "engineer",
    "everyone",
    "girl",
    "girls",
    "guy",
    "guys",
    "host",
    "horneck",
    "man",
    "men",
    "narrator",
    "mrs. koothrappali",
    "mrs. wolowitz",
    "officer",
    "others",
    "rajesh",
    "radio",
    "reporter",
    "together",
    "tv",
    "voice",
    "woman",
    "women",
    "waitress",
    "gablehauser",
}

_NON_CHARACTER_SPEAKERS = {
    "barely audible",
    "british accent",
    "chants",
    "chuckling",
    "clears throat",
    "in unison",
    "laughing",
    "mimicking pirate",
    "mockingly",
    "mouths",
    "nasally",
    "on tape",
    "over machine",
    "sing-songy",
    "sobs",
    "system",
    "unknown",
    "whistles",
    "yelling",
}


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?。；，！？：])", r"\1", text)
    text = re.sub(r"([.!?。！？])\s*[:：]\s*", r"\1 ", text)
    text = re.sub(r"^\s*[:：]\s*", "", text)
    return text.strip()


def speaker_names(units: Iterable[dict[str, Any]]) -> set[str]:
    names = set(_GENERIC_SPEAKERS)
    for unit in units:
        raw = str(unit.get("speaker") or "").strip()
        if not raw or raw.lower() in {"unknown", "system"}:
            continue
        cleaned = normalize_space(_BRACKETED.sub(" ", raw))
        if cleaned:
            names.add(cleaned)
        for part in re.split(r"\s*(?:&|/|,|\band\b|\bwith\b)\s*", cleaned, flags=re.I):
            if part:
                names.add(part)
    return names


def clean_speaker_label(value: Any) -> str:
    speaker = html.unescape(str(value or ""))
    speaker = _HTML_TAG.sub(" ", speaker)
    speaker = normalize_space(_BRACKETED.sub(" ", speaker))
    if not speaker or speaker.lower().strip(".:") in _NON_CHARACTER_SPEAKERS:
        return "unknown"
    return speaker


def speaker_label_pattern(names: Iterable[str]) -> re.Pattern[str]:
    choices = sorted({name for name in names if name}, key=len, reverse=True)
    escaped = "|".join(re.escape(name) for name in choices)
    if not escaped:
        return re.compile(r"(?!x)x")
    return re.compile(rf"(?<![\w])(?:{escaped})\s*[:：]\s*", re.I)


def clean_subtitle_text(text: str, *, labels: re.Pattern[str]) -> str | None:
    out = html.unescape(str(text)).replace("\u200b", " ").replace("\ufeff", " ")
    out = _HTML_TAG.sub(" ", out)
    out = out.replace("♪", " ").replace("♫", " ").replace("♬", " ")

    for _ in range(8):
        cleaned = _BRACKETED.sub(" ", out)
        if cleaned == out:
            break
        out = cleaned

    out = labels.sub("", out)
    out = normalize_space(out)
    if not out or not _VISIBLE_TEXT.search(out):
        return None
    if _CAPTION_CREDIT.search(out) or _NON_DIALOGUE_LINE.fullmatch(out):
        return None
    return out


def export_episode(
    source: Path, *, known_speakers: Iterable[str] = ()
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    series = source.parents[2].name
    declared_series = str(payload.get("series") or "")
    ep = source.parent.name
    units = payload.get("units")
    if not isinstance(units, list):
        raise ValueError(f"{source}: 'units' must be a list")

    labels = speaker_label_pattern(speaker_names(units) | set(known_speakers))
    rows: list[dict[str, Any]] = []
    changed = 0
    removed = 0
    previous_start = float("-inf")

    for position, unit in enumerate(units, 1):
        if not isinstance(unit, dict):
            raise ValueError(f"{source}: unit {position} is not an object")
        start = float(unit["start"])
        end = float(unit["end"])
        if end < start:
            raise ValueError(f"{source}: unit {position} ends before it starts")
        if start < previous_start:
            raise ValueError(f"{source}: units are not ordered by start time")
        previous_start = start

        raw_text = str(unit.get("text") or "")
        text = clean_subtitle_text(raw_text, labels=labels)
        if text is None:
            removed += 1
            continue
        if text != raw_text:
            changed += 1
        rows.append(
            {
                "series": series,
                "ep": ep,
                "unit_id": str(unit.get("unit_id") or f"u{position:04d}"),
                "t": [start, end],
                "speaker": clean_speaker_label(unit.get("speaker")),
                "text": text,
            }
        )

    report: dict[str, int | str] = {
        "series": series,
        "ep": ep,
        "source_units": len(units),
        "kept_units": len(rows),
        "changed_units": changed,
        "removed_units": removed,
    }
    if declared_series and declared_series != series:
        report["declared_series"] = declared_series
    return rows, report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_sources(inputs_root: Path, wanted_series: set[str], wanted_eps: set[str]) -> list[Path]:
    sources = []
    for series_dir in sorted(path for path in inputs_root.iterdir() if path.is_dir()):
        if wanted_series and series_dir.name not in wanted_series:
            continue
        s1_dir = series_dir / "s1_perception"
        if not s1_dir.is_dir():
            continue
        for source in sorted(s1_dir.glob("*/perceptions_reviewed.json")):
            if not wanted_eps or source.parent.name in wanted_eps:
                sources.append(source)
    if not sources:
        raise SystemExit(f"no perceptions_reviewed.json files found under {inputs_root}")
    return sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs-root", required=True, help="Root containing <series>/s1_perception/<episode>/perceptions_reviewed.json.")
    ap.add_argument("--out", required=True, help="Output subtitle root.")
    ap.add_argument("--series", action="append", default=None, help="Only export selected series; repeat for multiple series.")
    ap.add_argument("--episode", action="append", default=None, help="Only export selected episode id(s); repeat for multiple episodes.")
    args = ap.parse_args()

    sources = discover_sources(
        Path(args.inputs_root), set(args.series or []), set(args.episode or [])
    )
    out_root = Path(args.out)
    exported_series: set[str] = set()
    reports: list[dict[str, int | str]] = []
    series_speakers: dict[str, set[str]] = {}
    for source in sources:
        payload = json.loads(source.read_text(encoding="utf-8"))
        units = payload.get("units")
        if isinstance(units, list):
            series_speakers.setdefault(source.parents[2].name, set()).update(speaker_names(units))

    for position, source in enumerate(sources, 1):
        canonical_series = source.parents[2].name
        rows, report = export_episode(
            source, known_speakers=series_speakers.get(canonical_series, set())
        )
        series = str(report["series"])
        ep = str(report["ep"])
        filename = f"{ep.lower()}.json"
        print(f"[{position}/{len(sources)}] {series}/{ep}", flush=True)
        write_json(out_root / series / filename, rows)
        exported_series.add(series)
        reports.append(report)

    summary = {
        "series": len(exported_series),
        "episodes": len(reports),
        "source_units": sum(int(row["source_units"]) for row in reports),
        "kept_units": sum(int(row["kept_units"]) for row in reports),
        "changed_units": sum(int(row["changed_units"]) for row in reports),
        "removed_units": sum(int(row["removed_units"]) for row in reports),
        "episodes_detail": reports,
    }
    write_json(out_root / "cleaning_report.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
