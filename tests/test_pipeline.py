import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evaluation import eval as eval_module, io_utils
from evaluation.clients import Client
from evaluation.inference import run
from evaluation.inference.adapters import emollm_runner
from helpers import question, prediction, service


def invoke(module, args):
    with contextlib.redirect_stdout(io.StringIO()):
        return module.run(module.parser().parse_args(args))


def scoring_runs(output):
    return sorted(path.parent for path in output.glob("*/scores.jsonl"))


def episode_frames(*_args, **_kwargs):
    timestamps = [0.0, 900.0, 1799.5]
    parts = []
    for timestamp in timestamps:
        parts.extend(
            [
                {"type": "text", "text": f"Frame at {timestamp:.3f} seconds:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,c3ludGhldGlj"},
                },
            ]
        )
    return parts, {
        "timestamps_seconds": timestamps,
        "num_frames": len(timestamps),
        "duration_seconds": 1800.0,
    }


class PipelineTest(unittest.TestCase):
    def test_video_directory_defaults_and_explicit_override(self):
        for layout in ("json", "jsonl", "directory", "override"):
            with (
                self.subTest(layout=layout),
                tempfile.TemporaryDirectory() as td,
                patch.object(
                    run,
                    "encode_video",
                    return_value={
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64,dGVzdA=="},
                    },
                ) as video,
                patch.object(
                    Client,
                    "generate",
                    return_value={
                        "content": "happiness",
                        "raw_response": {},
                        "usage": None,
                    },
                ) as generate,
            ):
                root = Path(td)
                data_dir = root / "data"
                data_dir.mkdir()
                source = data_dir / ("q.jsonl" if layout == "jsonl" else "q.json")
                source.write_text(json.dumps(question()) + "\n")
                videos = (
                    root / "custom_videos"
                    if layout == "override"
                    else data_dir / "videos"
                )
                videos.mkdir()
                video_path = videos / "V1.mp4"
                video_path.write_bytes(b"synthetic video")
                args = [
                    "--data-path",
                    str(data_dir if layout == "directory" else source),
                    "--model",
                    "test",
                    "--base-url",
                    "http://localhost/v1",
                    "--output-dir",
                    str(root / "out"),
                    "--tries",
                    "1",
                ]
                if layout == "override":
                    args += ["--videos-dir", str(videos)]
                self.assertEqual(invoke(run, args), 0)
                self.assertEqual(
                    video.call_args.args[0]["path"], str(video_path.resolve())
                )
                generate.assert_called_once()

    def test_limit_selects_first_questions_after_id_filtering(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(
                Client,
                "generate",
                return_value={
                    "content": "happiness",
                    "raw_response": {},
                    "usage": None,
                },
            ) as generate,
        ):
            root = Path(td)
            source = root / "q.json"
            source.write_text(
                json.dumps(
                    [
                        question("Q3"),
                        question("E1", granularity="episode"),
                        question("Q1"),
                        question("Q2"),
                    ]
                )
            )
            base = [
                "--data-path",
                str(source),
                "--modality",
                "text",
                "--model",
                "test",
                "--base-url",
                "http://localhost:8000/v1",
            ]
            cases = [
                ([], ["Q3", "Q1", "Q2"]),
                (["--force", "--limit", "2"], ["Q3", "Q1"]),
                (["--force", "--qid", "Q2", "--qid", "Q1", "--limit", "1"], ["Q1"]),
            ]
            call_count = 0
            out = root / "out"
            for selection, expected in cases:
                with self.subTest(selection=selection):
                    self.assertEqual(
                        invoke(run, base + selection + ["--output-dir", str(out)]), 0
                    )
                    records = io_utils.load_records(out / "predictions.jsonl")
                    self.assertEqual([r["question_id"] for r in records], expected)
                    self.assertTrue(all(r["status"] == "ok" for r in records))
                    call_count += len(expected)
                    self.assertEqual(generate.call_count, call_count)

    def test_subtitles_directory_cli_option_is_removed(self):
        parsers = [(run.parser(), ["--data-path", "questions.json"])] + [
            (
                emollm_runner.parser(model),
                [
                    "--data-path",
                    "questions.json",
                    "--videos-dir",
                    "videos",
                    "--runtime-config",
                    "runtime.json",
                ],
            )
            for model in emollm_runner.REPOSITORIES
        ]
        for parser, required in parsers:
            with self.subTest(description=parser.description):
                self.assertFalse(parser.parse_args(required).with_subtitle)
                self.assertNotIn("--subtitles-dir", parser.format_help())
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parser.parse_args(required + ["--subtitles-dir", "subtitles"])

    def test_text_modes_protocols_and_input_isolation(self):
        for api in ("chat", "responses", "gemini", "anthropic"):
            for granularity in ("clip", "episode"):
                with (
                    self.subTest(api=api, granularity=granularity),
                    tempfile.TemporaryDirectory() as td,
                    service(lambda *_: "happiness") as (url, calls),
                    patch.object(
                        io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "subtitles"
                    ),
                ):
                    root = Path(td)
                    source = root / "questions.json"
                    q = question(
                        granularity=granularity,
                        answer=["sadness"]
                        if granularity == "clip"
                        else "REFERENCE_SECRET",
                    )
                    if granularity == "episode":
                        io_utils.EPISODE_SUBTITLES_DIR.mkdir()
                        (io_utils.EPISODE_SUBTITLES_DIR / "V1.json").write_text(
                            json.dumps(q["subtitles"])
                        )
                        q["subtitles"] = None
                    source.write_text(json.dumps(q))
                    original = source.read_bytes()
                    args = [
                        "--data-path",
                        str(source),
                        "-g",
                        granularity,
                        "--modality",
                        "text",
                        "--model",
                        "test/model",
                        "--base-url",
                        url
                        + {
                            "chat": "/v1",
                            "responses": "/v1/responses",
                            "gemini": "/v1beta",
                            "anthropic": "/v1/messages",
                        }[api],
                        "--output-dir",
                        str(root / "out"),
                        "--tries",
                        "1",
                    ]
                    self.assertEqual(invoke(run, args), 0)
                    self.assertEqual(len(calls), 1)
                    payload = json.dumps(calls[0][1])
                    for secret in (
                        "SYNTHETIC_SOURCE",
                        "PRIVATE_SPEAKER",
                        "REFERENCE_SECRET",
                        "U1",
                        "video_url",
                        "inline_data",
                    ):
                        self.assertNotIn(secret, payload)
                    self.assertLess(
                        payload.index("First subtitle."), payload.index(q["question"])
                    )
                    self.assertNotIn("/v1beta/v1beta", calls[0][0])
                    self.assertEqual(source.read_bytes(), original)
                    self.assertEqual(invoke(run, args), 0)
                    self.assertEqual(len(calls), 1)

    def test_video_content_order_subtitle_switch_and_complete_response(self):
        for api, model_family in [("chat", "qwen_omni"), ("gemini", "gemini")]:
            for subtitles, prompt_mode in (
                (False, None),
                (True, None),
                (False, "merge"),
                (True, "merge"),
            ):
                with (
                    self.subTest(api=api, subtitles=subtitles, prompt_mode=prompt_mode),
                    tempfile.TemporaryDirectory() as td,
                    service(lambda *_: "happiness") as (url, calls),
                ):
                    root = Path(td)
                    q = question()
                    source = root / "q.json"
                    source.write_text(json.dumps(q))
                    videos = root / "videos"
                    videos.mkdir()
                    (videos / "V1.mp4").write_bytes(b"synthetic-media-content")
                    args = [
                        "--data-path",
                        str(source),
                        "--videos-dir",
                        str(videos),
                        "--model",
                        "qwen3-omni-flash"
                        if model_family == "qwen_omni"
                        else "gemini-3-flash-preview",
                        "--base-url",
                        url + ("/v1beta" if api == "gemini" else "/v1"),
                        "--output-dir",
                        str(root / "out"),
                        "--tries",
                        "1",
                    ]
                    if prompt_mode:
                        args += ["--prompt-mode", prompt_mode]
                    if subtitles:
                        args += ["--with-subtitle"]
                    if model_family == "qwen_omni":
                        args += ["--thinking", "on"]
                    self.assertEqual(invoke(run, args), 0)
                    payload = calls[0][1]
                    self.assertNotIn("stream", payload)
                    text = json.dumps(payload)
                    self.assertEqual("First subtitle." in text, subtitles)
                    self.assertNotIn("SYNTHETIC_SOURCE", text)
                    self.assertEqual(
                        text.count(
                            "You are an expert in multimodal emotion understanding."
                        ),
                        1,
                    )
                    media_index = 1 if prompt_mode == "merge" else 0
                    if api == "chat":
                        self.assertTrue(payload["enable_thinking"])
                        self.assertEqual(
                            [message["role"] for message in payload["messages"]],
                            ["user"] if prompt_mode == "merge" else ["system", "user"],
                        )
                        parts = payload["messages"][-1]["content"]
                        self.assertEqual(parts[media_index]["type"], "video_url")
                    else:
                        self.assertEqual(
                            "systemInstruction" in payload, prompt_mode is None
                        )
                        self.assertIn(
                            "inline_data", payload["contents"][0]["parts"][media_index]
                        )
                    pred = io_utils.load_records(root / "out/predictions.jsonl")[0]
                    self.assertEqual(pred["prediction"], ["happiness"])
                    self.assertNotIn("base64,", json.dumps(pred))

    def test_episode_video_frames_protocols_prompts_and_subtitles(self):
        cases = [
            (
                "gemini-3-flash-preview",
                "/v1beta",
                "system",
                True,
                False,
                "emotion trajectory",
            ),
            (
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "/v1",
                "merge",
                False,
                True,
                "emotional reasoning",
            ),
            (
                "claude-sonnet-4-6",
                "/v1/messages",
                "system",
                False,
                False,
                "emotional intensity comparison",
            ),
            ("gpt-4.1", "/v1/responses", "merge", True, False, "emotional reasoning"),
        ]
        for model, endpoint, prompt_mode, subtitles, no_audio, task in cases:
            with (
                self.subTest(model=model),
                tempfile.TemporaryDirectory() as td,
                service(lambda *_: "A complete natural-language answer.") as (
                    url,
                    calls,
                ),
                patch.object(io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "subtitles"),
                patch.object(
                    run, "sample_video_frames", side_effect=episode_frames
                ) as frames,
                patch.object(
                    run,
                    "encode_video",
                    side_effect=AssertionError("native video used"),
                ),
                patch.object(
                    run,
                    "extract_audio",
                    return_value=(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "c3ludGhldGlj", "format": "wav"},
                        },
                        {"audio": "separate_audio", "duration_seconds": 1800.0},
                    ),
                ) as audio,
            ):
                root = Path(td)
                data = root / "data"
                data.mkdir()
                q = question(
                    granularity="episode",
                    task=task,
                    question="How does Alex's frustration develop across the competition?",
                    answer="REFERENCE_SECRET",
                    answer_details={
                        "description": "DETAILS_SECRET",
                        "items": [{"id": "R1", "description": "ITEM_SECRET"}],
                    },
                    subtitles=None,
                )
                if task == "emotion trajectory":
                    q["answer_details"]["items"][0].update(
                        anchor=None, emotion="EMOTION_SECRET", intensity=None
                    )
                q["rubric"]["criterion"] = "RUBRIC_SECRET"
                source = data / "q.json"
                source.write_text(json.dumps(q))
                original = source.read_bytes()
                videos = data / "videos"
                videos.mkdir()
                video = videos / "V1.mp4"
                video.write_bytes(b"synthetic full episode")
                io_utils.EPISODE_SUBTITLES_DIR.mkdir()
                (io_utils.EPISODE_SUBTITLES_DIR / "V1.json").write_text(
                    json.dumps(question()["subtitles"])
                )
                out = root / "out"
                args = [
                    "--data-path",
                    str(source),
                    "-g",
                    "episode",
                    "--model",
                    model,
                    "--base-url",
                    url + endpoint,
                    "--prompt-mode",
                    prompt_mode,
                    "--fps",
                    "1",
                    "--max-frames",
                    "3",
                    "--output-dir",
                    str(out),
                    "--tries",
                    "1",
                ]
                if subtitles:
                    args += ["--with-subtitle"]
                if no_audio:
                    args += ["--no-audio"]
                self.assertEqual(invoke(run, args), 0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(frames.call_args.args[0]["path"], str(video.resolve()))
                self.assertEqual(frames.call_args.kwargs["fps"], 1.0)
                self.assertEqual(frames.call_args.kwargs["max_frames"], 3)
                payload = json.dumps(calls[0][1])
                for secret in (
                    "REFERENCE_SECRET",
                    "DETAILS_SECRET",
                    "ITEM_SECRET",
                    "EMOTION_SECRET",
                    "RUBRIC_SECRET",
                    "SYNTHETIC_SOURCE",
                    "PRIVATE_SPEAKER",
                ):
                    self.assertNotIn(secret, payload)
                self.assertNotIn("### Answer Format", payload)
                self.assertNotIn("Before: <emotion labels>", payload)
                self.assertNotIn("video_url", payload)
                self.assertEqual("First subtitle." in payload, subtitles)
                self.assertLess(
                    payload.index("Frame at 1799.500 seconds:"),
                    payload.index(q["question"]),
                )
                if subtitles:
                    self.assertLess(
                        payload.index("Frame at 1799.500 seconds:"),
                        payload.index("First subtitle."),
                    )
                    self.assertLess(
                        payload.index("Last subtitle."), payload.index(q["question"])
                    )
                self.assertEqual(
                    payload.count(
                        "You are an expert in multimodal emotion understanding."
                    ),
                    1,
                )
                result = io_utils.load_records(out / "predictions.jsonl")[0]
                self.assertEqual(result["granularity"], "episode")
                self.assertEqual(
                    result["prediction"], "A complete natural-language answer."
                )
                self.assertEqual(
                    [message["role"] for message in result["messages"]],
                    ["system", "user"] if prompt_mode == "system" else ["user"],
                )
                references = [
                    part
                    for part in result["messages"][-1]["content"]
                    if part["type"] == "frame_reference"
                ]
                self.assertEqual(
                    [part["timestamp_seconds"] for part in references],
                    [0.0, 900.0, 1799.5],
                )
                self.assertTrue(
                    all(part["path"] == str(video.resolve()) for part in references)
                )
                self.assertEqual(result["media"]["representation"], "frames")
                self.assertEqual(result["media"]["duration_seconds"], 1800.0)
                self.assertNotIn("base64,", json.dumps(result))
                if model.startswith("gemini"):
                    audio.assert_called_once()
                    self.assertEqual(result["media"]["audio"], "separate_audio")
                    self.assertIn("audio/wav", payload)
                else:
                    audio.assert_not_called()
                    self.assertEqual(result["media"]["audio"], "not_sent")
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual(invoke(run, args), 0)
                self.assertEqual(len(calls), 1)

    def test_episode_frame_cap_can_exceed_default(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(run, "sample_video_frames", side_effect=episode_frames) as frames,
            patch.object(
                Client,
                "generate",
                return_value={"content": "Yes", "raw_response": {}, "usage": None},
            ) as generate,
        ):
            root = Path(td)
            source = root / "q.json"
            source.write_text(
                json.dumps(question(granularity="episode", subtitles=None))
            )
            (root / "videos").mkdir()
            (root / "videos/V1.mp4").write_bytes(b"synthetic full episode")
            out = root / "out"
            self.assertEqual(
                invoke(
                    run,
                    [
                        "--data-path",
                        str(source),
                        "-g",
                        "episode",
                        "--model",
                        "test",
                        "--base-url",
                        "http://localhost:8000/v1",
                        "--fps",
                        "1",
                        "--max-frames",
                        "1024",
                        "--frame-max-pixels",
                        "200704",
                        "--output-dir",
                        str(out),
                        "--tries",
                        "1",
                    ],
                ),
                0,
            )
            self.assertEqual(frames.call_args.kwargs["max_frames"], 1024)
            self.assertEqual(frames.call_args.kwargs["max_pixels"], 200704)
            result = io_utils.load_records(out / "predictions.jsonl")[0]
            self.assertEqual(result["media"]["num_frames"], 3)
            generate.assert_called_once()

    def test_external_subtitle_change_invalidates_cached_prediction(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: "Yes") as (url, calls),
            patch.object(io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "subtitles"),
        ):
            root = Path(td)
            source = root / "q.json"
            source.write_text(
                json.dumps(question(granularity="episode", subtitles=None))
            )
            subs = io_utils.EPISODE_SUBTITLES_DIR
            subs.mkdir()
            subfile = subs / "V1.json"
            rows = question()["subtitles"]
            subfile.write_text(json.dumps(rows))
            args = [
                "--data-path",
                str(source),
                "--granularity",
                "episode",
                "--modality",
                "text",
                "--model",
                "test",
                "--base-url",
                url + "/v1",
                "--output-dir",
                str(root / "out"),
                "--tries",
                "1",
            ]
            self.assertEqual(invoke(run, args), 0)
            self.assertEqual(invoke(run, args), 0)
            self.assertEqual(len(calls), 1)
            rows[0]["text"] = "Changed dialogue."
            subfile.write_text(json.dumps(rows))
            self.assertEqual(invoke(run, args), 0)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("Changed dialogue.", json.dumps(calls[0][1]))

    def test_absent_optional_subtitles_do_not_block_video(self):
        for embedded in (None, []):
            for external in ("missing", "present"):
                with (
                    self.subTest(embedded=embedded, external=external),
                    tempfile.TemporaryDirectory() as td,
                    service(lambda *_: "happiness") as (url, calls),
                    patch.object(
                        io_utils, "EPISODE_SUBTITLES_DIR", Path(td) / "subtitles"
                    ),
                ):
                    root = Path(td)
                    q = question(subtitles=embedded)
                    (root / "q.json").write_text(json.dumps(q))
                    (root / "videos").mkdir()
                    (root / "videos/V1.mp4").write_bytes(b"synthetic-media-content")
                    (root / "subtitles").mkdir()
                    if external == "present":
                        (root / "subtitles/V1.json").write_text(
                            json.dumps(question()["subtitles"])
                        )
                    args = [
                        "--data-path",
                        str(root / "q.json"),
                        "--videos-dir",
                        str(root / "videos"),
                        "--with-subtitle",
                        "--model",
                        "test",
                        "--base-url",
                        url + "/v1",
                        "--output-dir",
                        str(root / "video"),
                        "--tries",
                        "1",
                    ]
                    self.assertEqual(invoke(run, args), 0)
                    self.assertEqual(len(calls), 1)
                    payload = json.dumps(calls[0][1])
                    self.assertIn("video_url", payload)
                    self.assertNotIn("### Subtitles", payload)
                    args[args.index(str(root / "video"))] = str(root / "text")
                    self.assertEqual(invoke(run, args + ["--modality", "text"]), 1)
                    self.assertEqual(len(calls), 1)
                    record = io_utils.load_records(root / "text/predictions.jsonl")[0]
                    self.assertEqual(record["status"], "input_error")

    def test_input_failures_for_clip_and_episode(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(
                Client, "generate", side_effect=AssertionError("network used")
            ),
        ):
            root = Path(td)
            source = root / "q.json"
            source.write_text(json.dumps(question()))
            args = [
                "--data-path",
                str(source),
                "--modality",
                "text",
                "--model",
                "test",
                "--base-url",
                "http://localhost:8000/v1",
                "--output-dir",
                str(root / "out"),
            ]
            with self.assertRaises(FileExistsError):
                invoke(run, args[:-2] + ["--output-dir", str(source)])
            q = question(granularity="episode", subtitles=None)
            source.write_text(json.dumps(q))
            self.assertEqual(
                invoke(
                    run,
                    [
                        "--data-path",
                        str(source),
                        "--granularity",
                        "episode",
                        "--model",
                        "test",
                        "--base-url",
                        "http://localhost:8000/v1",
                        "--output-dir",
                        str(root / "missing_video"),
                    ],
                ),
                1,
            )
            record = io_utils.load_records(root / "missing_video/predictions.jsonl")[0]
            self.assertEqual(record["status"], "input_error")
            self.assertIn("missing or empty prepared video", record["error"])
            self.assertEqual(
                invoke(
                    run,
                    [
                        "--data-path",
                        str(source),
                        "--granularity",
                        "episode",
                        "--modality",
                        "text",
                        "--model",
                        "test",
                        "--base-url",
                        "http://localhost:8000/v1",
                        "--output-dir",
                        str(root / "missing"),
                    ],
                ),
                1,
            )
            self.assertEqual(
                io_utils.load_records(root / "missing/predictions.jsonl")[0]["status"],
                "input_error",
            )

    def test_all_clip_scoring_routes_and_statistics(self):
        def judge(_path, body):
            self.assertEqual(len(body["messages"]), 2)
            user = body["messages"][1]["content"]
            self.assertNotIn("SYNTHETIC_SOURCE", user)
            self.assertNotIn("First subtitle.", user)
            return json.dumps({"score": 2, "reason": "Synthetic partial answer."})

        with tempfile.TemporaryDirectory() as td, service(judge) as (url, calls):
            root = Path(td)
            qs = [
                question("Q1", answer=["happiness", "sadness"]),
                question("Q2", task="emotion transition"),
                question("Q3", task="emotion trajectory"),
                question("Q4", task="emotion cause"),
                question("Q5", task="emotion influence"),
            ]
            answers = [
                "happiness, nonexistent",
                "Before: fear\nAfter: peace",
                "Partial trajectory",
                "Partial cause",
                "happiness",
            ]
            (root / "q.json").write_text(json.dumps(qs))
            (root / "p.json").write_text(
                json.dumps([prediction(q, a) for q, a in zip(qs, answers)])
            )
            (root / "scores").mkdir()
            legacy_scores = root / "scores/scores.jsonl"
            legacy_scores.write_text('{"question_id":"previous","score":0}\n')
            legacy_bytes = legacy_scores.read_bytes()
            args = [
                "--data-path",
                str(root / "q.json"),
                "--predictions",
                str(root / "p.json"),
                "--model",
                "judge",
                "--base-url",
                url + "/v1",
                "--output-dir",
                str(root / "scores"),
                "--tries",
                "1",
            ]
            self.assertEqual(invoke(eval_module, args), 0)
            self.assertEqual(len(calls), 2)
            first_run = scoring_runs(root / "scores")[0]
            first_files = {p.name: p.read_bytes() for p in first_run.iterdir()}
            m = json.loads((first_run / "metrics.json").read_text())
            self.assertAlmostEqual(m["overall_unweighted"]["normalized_mean"], 0.7)
            self.assertAlmostEqual(
                m["tasks"]["emotion transition"]["labels"]["em"], 0.5
            )
            self.assertEqual(m["tasks"]["emotion trajectory"]["percent_score"], 50)
            self.assertEqual(invoke(eval_module, args), 0)
            self.assertEqual(len(calls), 4)
            runs = scoring_runs(root / "scores")
            self.assertEqual(len(runs), 2)
            second_run = runs[-1]
            self.assertNotEqual(first_run, second_run)
            second_files = {p.name: p.read_bytes() for p in second_run.iterdir()}
            self.assertEqual(invoke(eval_module, args), 0)
            self.assertEqual(len(calls), 6)
            runs = scoring_runs(root / "scores")
            self.assertEqual(len(runs), 3)
            scored = io_utils.load_records(runs[-1] / "scores.jsonl")
            self.assertEqual(
                [r["question_id"] for r in scored], [q["question_id"] for q in qs]
            )
            current = json.loads((runs[-1] / "metrics.json").read_text())
            self.assertEqual(set(current["tasks"]), {q["type"] for q in qs})
            self.assertEqual(
                {p.name: p.read_bytes() for p in first_run.iterdir()}, first_files
            )
            self.assertEqual(
                {p.name: p.read_bytes() for p in second_run.iterdir()}, second_files
            )
            self.assertEqual(legacy_scores.read_bytes(), legacy_bytes)

    def test_yes_no_uses_judge_and_missing_predictions_are_not_zero(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(
                lambda *_: json.dumps({"score": 0, "reason": "Judge verdict."})
            ) as (url, calls),
        ):
            root = Path(td)
            qs = [question(f"Q{i}", granularity="episode") for i in range(1, 5)]
            preds = [
                prediction(qs[0], "Yes"),
                prediction(qs[1], ""),
                prediction(qs[2], None, status="error", error="timeout"),
            ]
            (root / "q.json").write_text(json.dumps(qs))
            (root / "p.json").write_text(json.dumps(preds))
            args = [
                "--data-path",
                str(root / "q.json"),
                "--predictions",
                str(root / "p.json"),
                "--granularity",
                "episode",
                "--model",
                "judge",
                "--base-url",
                url + "/v1",
                "--output-dir",
                str(root / "scores"),
                "--tries",
                "1",
            ]
            self.assertEqual(invoke(eval_module, args), 1)
            self.assertEqual(len(calls), 1)
            metrics = json.loads(
                (scoring_runs(root / "scores")[0] / "metrics.json").read_text()
            )
            self.assertEqual(metrics["emotional_reasoning"]["result"]["n_scored"], 1)
            self.assertEqual(metrics["emotional_reasoning"]["result"]["acc"], 0)
            self.assertEqual(
                metrics["overall_unweighted"]["statuses"],
                {"ok": 1, "missing_prediction": 3},
            )

    def test_external_prediction_aliases_and_empty_answers(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: '{"score":1,"reason":"Judge verdict."}') as (
                url,
                calls,
            ),
        ):
            root = Path(td)
            qs = [question(f"Q{i}", granularity="episode") for i in range(1, 9)]
            preds = [
                {"question_id": "Q1", "pred_answer": "Yes"},
                {
                    "question_id": "Q2", "prediction": "Yes",
                    "status": "error", "error": "ignored external metadata",
                },
                {"question_id": "Q3", "prediction": "", "pred_answer": "Yes"},
                {"question_id": "Q4", "prediction": None, "pred_answer": "Yes"},
                {
                    "question_id": "Q5", "prediction": "No", "pred_answer": "Yes",
                    "status": "error",
                },
                {"question_id": "Q6"},
                {"question_id": "Q7", "prediction": " \n", "pred_answer": "\t"},
                {"question_id": "Q8", "prediction": " \n", "pred_answer": "Yes"},
            ]
            (root / "q.json").write_text(json.dumps(qs))
            prediction_path = root / "p.json"
            prediction_path.write_text(json.dumps(preds))
            original_predictions = prediction_path.read_bytes()
            self.assertEqual(
                invoke(
                    eval_module,
                    [
                        "--data-path", str(root / "q.json"),
                        "--predictions", str(prediction_path),
                        "--granularity", "episode",
                        "--model", "judge",
                        "--base-url", url + "/v1",
                        "--output-dir", str(root / "scores"),
                        "--tries", "1",
                    ],
                ),
                1,
            )
            self.assertEqual(len(calls), 6)
            result_dir = scoring_runs(root / "scores")[0]
            scored = io_utils.load_records(result_dir / "scores.jsonl")
            submitted = {
                row["question_id"]: row["prediction"]
                for row in scored if row["status"] == "ok"
            }
            self.assertEqual(
                submitted,
                {"Q1": "Yes", "Q2": "Yes", "Q3": "Yes", "Q4": "Yes", "Q5": "No", "Q8": "Yes"},
            )
            self.assertEqual(
                [row["question_id"] for row in scored if row["status"] == "missing_prediction"],
                ["Q6", "Q7"],
            )
            metrics = json.loads((result_dir / "metrics.json").read_text())
            self.assertEqual(metrics["overall_unweighted"]["n_scored"], 6)
            self.assertEqual(metrics["overall_unweighted"]["normalized_mean"], 1)
            self.assertEqual(prediction_path.read_bytes(), original_predictions)

    def test_invalid_judge_score_retried_and_not_counted_as_wrong_answer(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: '{"score":9,"reason":"Bad range"}') as (url, calls),
        ):
            root = Path(td)
            q = question(granularity="episode")
            (root / "q.json").write_text(json.dumps(q))
            (root / "p.json").write_text(json.dumps(prediction(q, "Yes")))
            args = [
                "--data-path",
                str(root / "q.json"),
                "--predictions",
                str(root / "p.json"),
                "--granularity",
                "episode",
                "--model",
                "judge",
                "--base-url",
                url + "/v1",
                "--output-dir",
                str(root / "scores"),
                "--tries",
                "2",
            ]
            self.assertEqual(invoke(eval_module, args), 1)
            self.assertEqual(len(calls), 2)
            result_dir = scoring_runs(root / "scores")[0]
            result = json.loads((result_dir / "metrics.json").read_text())
            self.assertIsNone(result["overall_unweighted"]["normalized_mean"])
            scored = io_utils.load_records(result_dir / "scores.jsonl")[0]
            self.assertEqual(len(scored["judge_responses"]), 2)
            for attempt in scored["judge_responses"]:
                self.assertEqual(
                    json.loads(attempt["choices"][0]["message"]["content"])["score"], 9
                )

    def test_label_task_with_rubric_uses_judge_and_its_own_scale(self):
        rubric = {
            "criterion": "Grade the requested emotional description.",
            "scores": {str(i): f"Calibration level {i}." for i in range(6)},
        }
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda *_: '{"score":3,"reason":"Partial description."}') as (
                url,
                calls,
            ),
        ):
            root = Path(td)
            q = question(rubric=rubric)
            (root / "q.json").write_text(json.dumps(q))
            (root / "p.json").write_text(json.dumps(prediction(q, "happiness")))
            self.assertEqual(
                invoke(
                    eval_module,
                    [
                        "--data-path", str(root / "q.json"),
                        "--predictions", str(root / "p.json"),
                        "--model", "judge",
                        "--base-url", url + "/v1",
                        "--output-dir", str(root / "out"),
                        "--tries", "1",
                    ],
                ),
                0,
            )
            self.assertEqual(len(calls), 1)
            result_dir = scoring_runs(root / "out")[0]
            scored = io_utils.load_records(result_dir / "scores.jsonl")[0]
            self.assertEqual(scored["type"], "contextual emotion")
            self.assertEqual(scored["prediction"], "happiness")
            self.assertEqual(scored["evaluation_input"]["rubric"], rubric)
            self.assertEqual(scored["score"], 3)
            self.assertEqual(scored["max_score"], 5)
            self.assertAlmostEqual(scored["normalized_score"], 0.6)
            metrics = json.loads((result_dir / "metrics.json").read_text())
            self.assertEqual(set(metrics["tasks"]), {"contextual emotion"})
            self.assertNotIn("labels", metrics["tasks"]["contextual emotion"])

    def test_episode_mixed_rubrics_separate_and_weighted_scores(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trajectory = question("Q1", "episode", "emotion trajectory")
            intensity = question("Q2", "episode", "emotional intensity comparison")
            result = question("Q3", "episode", answer="Alice")
            rubric = {
                "criterion": "Grade the completeness of the requested observation.",
                "scores": {str(i): f"Calibration level {i}." for i in range(6)},
            }
            explanation = question(
                "Q4", "episode", answer="The person is more confident.", rubric=rubric
            )
            missing_result = question("Q5", "episode")
            failed_result = question("Q6", "episode")
            missing_explanation = question("Q7", "episode", rubric=rubric)
            failed_explanation = question("Q8", "episode", rubric=rubric)
            qs = [
                trajectory, intensity, result, explanation,
                missing_result, failed_result, missing_explanation, failed_explanation,
            ]
            (root / "q.json").write_text(json.dumps(qs))
            (root / "p.json").write_text(
                json.dumps(
                    [prediction(q, "submitted answer") for q in qs[:4]]
                    + [
                        prediction(q, None, status="error", error="timeout")
                        for q in (failed_result, failed_explanation)
                    ]
                )
            )
            marks = iter([3, 1, 1, 2])
            with service(
                lambda *_: json.dumps(
                    {"score": next(marks), "reason": "Synthetic verdict."}
                )
            ) as (url, calls):
                args = [
                    "--data-path",
                    str(root / "q.json"),
                    "--predictions",
                    str(root / "p.json"),
                    "--granularity",
                    "episode",
                    "--model",
                    "judge",
                    "--base-url",
                    url + "/v1",
                    "--output-dir",
                    str(root / "out"),
                    "--tries",
                    "1",
                ]
                self.assertEqual(invoke(eval_module, args), 1)
                self.assertEqual(len(calls), 4)
                result_dir = scoring_runs(root / "out")[0]
                metrics = json.loads((result_dir / "metrics.json").read_text())
                reasoning = metrics["emotional_reasoning"]
                self.assertEqual(reasoning["result"]["acc"], 1)
                self.assertEqual(reasoning["explanation"]["raw_mean"], 2)
                self.assertEqual(reasoning["explanation"]["max_score"], 5)
                self.assertAlmostEqual(reasoning["weighted"], 0.6 + 0.4 * 2 / 5)
                self.assertAlmostEqual(reasoning["unweighted"], 0.7)
                for group in ("result", "explanation"):
                    self.assertEqual(reasoning[group]["n_total"], 3)
                    self.assertEqual(reasoning[group]["n_scored"], 1)
                    self.assertEqual(
                        reasoning[group]["statuses"],
                        {"ok": 1, "missing_prediction": 2},
                    )
                self.assertEqual(
                    metrics["tasks"]["emotion trajectory"]["percent_score"], 75
                )
                self.assertEqual(set(metrics["tasks"]), {q["type"] for q in qs})
                self.assertNotIn("answer_types", metrics)
                self.assertNotIn("answer_types", reasoning)
                scored = io_utils.load_records(result_dir / "scores.jsonl")
                for q, row in zip(qs, scored):
                    self.assertEqual(row["type"], q["type"])
                    self.assertEqual(
                        row["max_score"], max(map(int, q["rubric"]["scores"]))
                    )
                    self.assertNotIn("score_kind", row)
                    self.assertNotIn("answer_type", row)

    def test_label_only_scoring_needs_no_model_and_preserves_external_predictions(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(
                Client, "generate", side_effect=AssertionError("network used")
            ),
        ):
            root = Path(td)
            q = question()
            p = {
                "question_id": q["question_id"], "pred_answer": "happiness",
                "status": "error", "error": "ignored external metadata",
            }
            (root / "q.json").write_text(json.dumps(q))
            (root / "p.json").write_text(json.dumps(p))
            args = [
                "--data-path",
                str(root / "q.json"),
                "--predictions",
                str(root / "p.json"),
                "--output-dir",
                str(root / "scores"),
            ]
            for model_options in ([], ["--model", "unavailable"]):
                with self.subTest(model_options=model_options):
                    self.assertEqual(invoke(eval_module, args + model_options), 0)
            self.assertEqual(json.loads((root / "p.json").read_text()), p)

    def test_existing_output_is_reused_without_persisting_credentials(self):
        with (
            tempfile.TemporaryDirectory() as td,
            service(lambda _path, body: body["model"]) as (url, calls),
        ):
            root = Path(td)
            source = root / "q.json"
            source.write_text(json.dumps(question()))
            args = [
                "--data-path",
                str(source),
                "--modality",
                "text",
                "--model",
                "model1",
                "--base-url",
                url + "/v1",
                "--api-key",
                "NEVER_PERSIST_THIS",
                "--output-dir",
                str(root / "out"),
                "--tries",
                "1",
            ]
            self.assertEqual(invoke(run, args), 0)
            for f in (root / "out").iterdir():
                self.assertNotIn("NEVER_PERSIST_THIS", f.read_text())
            changed = list(args)
            changed[changed.index("model1")] = "model2"
            self.assertEqual(invoke(run, changed), 0)
            self.assertEqual(len(calls), 1)
            records = io_utils.load_records(root / "out/predictions.jsonl")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prediction"], ["model1"])
            for f in (root / "out").iterdir():
                self.assertNotIn("NEVER_PERSIST_THIS", f.read_text())
