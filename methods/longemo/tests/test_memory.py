import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from methods.longemo.common import manifest, LoggedClient, usage_summary
from methods.longemo.memory import empty_memory, apply_window, state_index
from methods.longemo.prepare import parse_srt
from methods.longemo.retrieval import retrieve, validate_plan
from methods.longemo.runner import _answer_one
from evaluation import eval as scorer
from evaluation.io_utils import write_json, write_records
from methods.agentic import runner as agentic


def observation_payload():
    return {"entities": [{"id": "new_1", "name": None, "description": "woman in blue"}],
            "observations": [{"id": "o1", "subject": "new_1", "span": [1, 3], "cue": "smiles after hearing good news", "modality": "visual"}],
            "events": [{"id": "e1", "span": [1, 3], "summary": "Receives news", "observation_ids": ["o1"], "continues_event": None,
                        "states": [{"subject": "new_1", "target": "news", "emotion": "relief", "intensity": "brief smile", "evidence_ids": ["o1"], "uncertainty": "", "appraisal": None}]}],
            "relations": [], "corrections": []}


def commit(memory, payload=None, **kwargs):
    return apply_window(memory, payload or observation_payload(), window_id=kwargs.pop("window_id", "W1"),
                        core=kwargs.pop("core", [0, 20]), media=kwargs.pop("media", [0, 22]),
                        metadata={"audio": False, "subtitle_ids": []}, **kwargs)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = empty_memory("video", 60, "sourcehash")

    def test_single_source_of_truth_and_atomicity(self):
        updated = commit(self.memory)
        self.assertEqual(self.memory["events"], [])
        self.assertEqual(updated["events"][0]["states"][0]["evidence_refs"], ["O1"])
        self.assertIs(state_index(updated)["S1"][1], updated["events"][0]["states"][0])
        invalid = observation_payload()
        invalid["events"][0]["states"][0]["evidence_ids"] = ["invented"]
        with self.assertRaises(ValueError): commit(self.memory, invalid)
        self.assertEqual(self.memory["observations"], [])

    def test_temporal_and_modality_validation(self):
        payload = observation_payload()
        payload["observations"][0]["span"] = [1, float("nan")]
        with self.assertRaises(ValueError): commit(self.memory, payload)
        payload = observation_payload()
        payload["observations"][0]["modality"] = "audio"
        with self.assertRaises(ValueError): commit(self.memory, payload)
        with self.assertRaises(ValueError): commit(commit(self.memory))

    def test_continuation_preserves_one_occurrence(self):
        memory = commit(self.memory)
        payload = observation_payload()
        payload["entities"][0]["id"] = "P1"
        payload["observations"][0].update(subject="P1", span=[21, 23])
        payload["events"][0].update(span=[21, 23], continues_event="E1")
        payload["events"][0]["states"][0]["subject"] = "P1"
        updated = commit(memory, payload, window_id="W2", core=[20, 40], media=[18, 42])
        self.assertEqual(len(updated["events"]), 1)
        self.assertEqual(len(updated["events"][0]["states"]), 2)
        self.assertEqual(len(updated["entities"]), 1)

    def test_correction_is_versioned_not_a_new_transition(self):
        memory = commit(self.memory)
        payload = observation_payload()
        payload["entities"][0]["id"] = "P1"
        payload["observations"][0].update(subject="P1", span=[21, 23])
        payload["events"] = []
        payload["corrections"] = [{"state_id": "S1", "expected_version": 1, "emotion": "polite smile despite concern",
            "uncertainty": "later self-report", "evidence_ids": ["o1"], "reason": "speaker clarifies the earlier reaction"}]
        with self.assertRaises(ValueError): commit(memory, payload, window_id="W2", core=[20, 40], media=[18, 42])
        updated = commit(memory, payload, window_id="W2", core=[20, 40], media=[18, 42], allow_revisions=True)
        self.assertEqual(len(updated["events"]), 1)
        self.assertEqual(updated["events"][0]["states"][0]["span"], [1, 3])
        self.assertEqual(updated["events"][0]["states"][0]["version"], 2)
        self.assertEqual(updated["state_history"][0]["state"]["emotion"], "relief")
        self.assertTrue(updated["events"][0]["summary_stale"])
        with self.assertRaises(ValueError): commit(updated, payload, window_id="W3", core=[20, 40], media=[18, 42], allow_revisions=True)

    def test_subtitle_timing_and_unknown_speaker(self):
        rows = parse_srt("1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i>\n\n2\n00:00:03.000 --> 00:00:04.000\nWorld")
        self.assertEqual(rows[0]["t"], [1, 2.5])
        self.assertEqual(rows[0]["text"], "Hello")
        self.assertIsNone(rows[0]["speaker"])

    def test_manifest_rejects_stale_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.json"
            first = manifest(path, {"model": "a"})
            self.assertEqual(first, manifest(path, {"model": "a"}))
            with self.assertRaises(ValueError): manifest(path, {"model": "b"})

    def test_retrieval_does_not_mutate_and_obeys_budget(self):
        memory = commit(self.memory)
        before = copy.deepcopy(memory)
        plan = {"mode": "trajectory", "entity_terms": ["woman"], "target_terms": ["news"], "query_terms": [], "time_range": None}
        for mode in ("flat", "graph"):
            result = retrieve(memory, "How does she feel?", plan, mode=mode, budget_chars=2000, dense_scores={"E1": .9})
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 2000)
            self.assertEqual(result["coverage"]["candidate_events"], 1)
        self.assertEqual(memory, before)
        plan["time_range"] = [0, float("inf")]
        with self.assertRaises(ValueError): validate_plan(plan)

    def test_event_comparison_fields_are_validated_and_preserved(self):
        payload = observation_payload()
        payload["events"][0].update(event_type="evaluation", action="rates the dish",
            objects=["dish"], participants=["new_1"],
            signals=[{"kind": "explicit_score", "value": "8.5", "text": "rates it 8.5"}],
            confidence=0.9)
        updated = commit(self.memory, payload)
        event = updated["events"][0]
        self.assertEqual(updated["schema_version"], 2)
        self.assertEqual(event["event_type"], "evaluation")
        self.assertEqual(event["actions"], ["rates the dish"])
        self.assertEqual(event["objects"], ["dish"])
        self.assertEqual(event["signals"][0]["value"], "8.5")
        self.assertEqual(event["confidence"], 0.9)
        bad = observation_payload()
        bad["events"][0]["signals"] = [{"kind": "made_up", "value": "1", "text": "x"}]
        with self.assertRaises(ValueError): commit(self.memory, bad)


class IntegrationTests(unittest.TestCase):
    def test_question_answering_excludes_gold_and_preserves_memory(self):
        memory = commit(empty_memory("video", 60, "hash"))
        memory["build_fingerprint"] = "build"
        previous = copy.deepcopy(memory)
        q = {"question_id": "q", "video_id": "video", "granularity": "episode", "type": "emotion trajectory", "question": "How does the woman feel?",
             "answer": "GOLD_SECRET", "answer_details": "DETAIL_SECRET", "rubric": "RUBRIC_SECRET"}
        responses = [{"mode": "trajectory", "entity_terms": ["woman"], "target_terms": [], "query_terms": [], "time_range": None},
                     {"answer": "She feels relief after the news.", "evidence_ids": ["E1"], "uncertainty": "", "inspect": []}]

        class Client:
            model = "mock"
            def generate(self, messages):
                body = json.dumps(messages)
                for marker in ("GOLD_SECRET", "DETAIL_SECRET", "RUBRIC_SECRET"):
                    assert marker not in body
                return {"content": json.dumps(responses.pop(0)), "usage": {"totalTokenCount": 10}}

        args = type("Args", (), {"tries": 1, "retrieval": "graph", "evidence_chars": 48000, "top_k": 12,
                               "max_inspections": 0, "inspection_seconds": 120})()
        with tempfile.TemporaryDirectory() as directory:
            index = type("Index", (), {"rank": lambda self,q: {"E1": .9}})()
            result = _answer_one(q, memory, args, Client(), Path(directory), index)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["usage"]["total_tokens"], 20)
        self.assertEqual(memory, previous)

    def test_duplicate_submission_rejected_before_any_api(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            write_json(p/"q.json", [{"question_id": "q", "video_id": "v", "granularity": "clip", "type": "contextual emotion", "answer": ["happiness"], "rubric": None}])
            write_records(p/"p.jsonl", [{"question_id": "q", "prediction": "happiness"}]*2)
            args = scorer.parser().parse_args(["--data-path", str(p/"q.json"), "--predictions", str(p/"p.jsonl")])
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                scorer.run(args)

    def test_agentic_import_cli_and_total_usage(self):
        args = agentic.parser().parse_args(["--data-path", "unused", "--output-dir", "unused", "--model", "mock"])
        self.assertEqual((args.tries, args.workers), (3, 1))

        class Client:
            def __init__(self): self.count = 0
            def generate(self, messages):
                self.count += 1
                value = {"action": "answer", "answer": "ok"} if self.count == 2 else {"action": "inspect", "ranges": [
                    {"start": 1, "end": 2, "fps": 1, "max_frames": 1, "max_pixels": 200704}]}
                return {"content": json.dumps(value), "usage": {"total_tokens": self.count*10}}

        with patch.object(agentic, "sample_video_frames", return_value=([], {})), patch.object(agentic, "sample_interval", return_value=([], {})):
            result = agentic._run_one(Client(), {"question": "q"}, {}, with_audio=False, with_subtitle=False,
                initial_fps=.5, initial_max_frames=32, initial_max_pixels=200704, max_rounds=3, tries=1)
        self.assertEqual(result["total_tokens_all_calls"], 30)
        self.assertEqual(len(result["calls"]), 2)


if __name__ == "__main__":
    unittest.main()
