import copy
import json
import unittest

from methods.longemo.noevent_memory import apply_window, empty_memory, perception_context
from methods.longemo.noevent_retrieval import retrieve


def window_payload(cue="speaks softly", subject="new_1", span=(1, 5)):
    return {"entities": [{"id": subject, "name": "Taylor", "description": "person in a blue shirt"}],
            "observations": [{"id": "o1", "subject": subject, "span": list(span),
                              "cue": cue, "modality": "visual"}],
            "summary": "Taylor speaks.", "actions": ["speaks"], "objects": [],
            "signals": [], "participants": [subject],
            "emotion_cues": [{"subject": subject, "target": "the conversation",
                              "emotion": "uneasy", "intensity": "quiet voice", "evidence_ids": ["o1"]}]}


def add_window(memory, payload, index=1):
    core = [(index - 1) * 20, index * 20]
    return apply_window(memory, payload, window_id=f"W{index:05d}", core=core, media=core,
                        metadata={"audio": False, "subtitle_ids": []})


PLAN = {"mode": "trajectory", "entity_terms": ["Taylor"], "target_terms": [],
        "query_terms": ["speaks"], "time_range": None}


class NoEventMemoryTests(unittest.TestCase):
    def test_window_schema_has_no_graph_objects(self):
        memory = empty_memory("V1", 20, "sha")
        payload = {"entities": [{"id": "new_1", "name": None, "description": "person"}],
                   "observations": [{"id": "o1", "subject": "new_1", "span": [1, 5],
                                     "cue": "speaks softly", "modality": "visual"}],
                   "summary": "A person speaks softly.", "actions": ["speaks"], "objects": [],
                   "signals": [], "participants": ["new_1"],
                   "emotion_cues": [{"subject": "new_1", "target": "the conversation",
                                     "emotion": "uneasy", "intensity": "quiet voice", "evidence_ids": ["o1"]}]}
        result = apply_window(memory, payload, window_id="W00001", core=[0, 20], media=[0, 20],
                              metadata={"audio": False, "subtitle_ids": []})
        self.assertEqual(result["representation"], "window_records")
        self.assertEqual(len(result["windows"]), 1)
        for key in ("events", "states", "relations", "continues_event"):
            self.assertNotIn(key, result)
        self.assertEqual(result["windows"][0]["observation_ids"], ["O1"])

    def test_bad_reference_is_atomic(self):
        memory = empty_memory("V1", 20, "sha")
        payload = {"entities": [{"id": "new_1", "name": None, "description": "person"}],
                   "observations": [{"id": "o1", "subject": "new_1", "span": [1, 5],
                                     "cue": "speaks", "modality": "visual"}],
                   "summary": "x", "emotion_cues": [{"subject": "new_1", "target": "x",
                                     "emotion": "sad", "intensity": "low", "evidence_ids": ["missing"]}]}
        with self.assertRaises(ValueError):
            apply_window(memory, payload, window_id="W00001", core=[0, 20], media=[0, 20],
                         metadata={"audio": False, "subtitle_ids": []})
        self.assertEqual(memory["completed_windows"], [])
        self.assertEqual(memory["entities"], [])

    def test_graph_fields_are_rejected_before_publishing(self):
        for location in ("top_level", "nested"):
            memory = empty_memory("V1", 20, "sha")
            payload = window_payload()
            if location == "top_level":
                payload["events"] = []
            else:
                payload["observations"][0]["event_id"] = "E1"
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "graph fields"):
                add_window(memory, payload)
            self.assertEqual(memory["completed_windows"], [])

    def test_unknown_participant_is_rejected_atomically(self):
        memory = empty_memory("V1", 20, "sha")
        payload = window_payload()
        payload["participants"].append("undeclared")
        with self.assertRaisesRegex(ValueError, "participant"):
            add_window(memory, payload)
        self.assertEqual(memory["entities"], [])

    def test_perception_receives_identity_only_without_prior_windows(self):
        memory = add_window(empty_memory("V1", 40, "sha"), window_payload("earlier private observation"))
        context = perception_context(memory, window_id="W00002", core=[20, 40], media=[18, 40])
        self.assertEqual(set(context), {"video_id", "window_id", "core_interval", "media_interval", "cast"})
        self.assertEqual(context["cast"], [{"id": "P1", "name": "Taylor", "description": "person in a blue shirt"}])
        self.assertNotIn("earlier private observation", json.dumps(context))
        self.assertNotIn("source_refs", json.dumps(context))


class NoEventRetrievalTests(unittest.TestCase):
    def test_answer_evidence_resolves_observations_and_people(self):
        cue = "Taylor whispers that the conversation makes them uneasy."
        memory = add_window(empty_memory("V1", 20, "sha"), window_payload(cue))
        before = copy.deepcopy(memory)
        packet = retrieve(memory, "How does Taylor feel?", PLAN, dense_scores={"W00001": 0.8})
        record = packet["windows"][0]
        self.assertEqual(record["observations"], memory["observations"])
        self.assertEqual(record["observations"][0]["cue"], cue)
        self.assertEqual(record["people"][0]["name"], "Taylor")
        self.assertEqual(record["observation_ids"], ["O1"])
        self.assertEqual(packet["evidence_ids"], ["W00001", "O1"])
        self.assertEqual(packet["coverage"]["used_characters"], len(json.dumps(packet, ensure_ascii=False)))
        self.assertEqual(memory, before)
        for key in ("events", "states", "relations", "continues_event"):
            self.assertNotIn(key, record)

    def test_budget_counts_resolved_observations_and_skips_large_window(self):
        memory = add_window(empty_memory("V1", 40, "sha"), window_payload("long observed cue " * 1000))
        memory = add_window(memory, window_payload("最后一句轻声道歉。", subject="P1", span=(21, 24)), index=2)
        packet = retrieve(memory, "How does Taylor feel?", PLAN,
                          dense_scores={"W00001": 0.9, "W00002": 0.8}, budget_chars=2000)
        self.assertEqual([record["id"] for record in packet["windows"]], ["W00002"])
        self.assertEqual(packet["windows"][0]["observations"][0]["cue"], "最后一句轻声道歉。")
        self.assertEqual(packet["evidence_ids"], ["W00002", "O2"])
        self.assertLessEqual(len(json.dumps(packet, ensure_ascii=False)), 2000)
        self.assertEqual(packet["coverage"]["used_characters"], len(json.dumps(packet, ensure_ascii=False)))
        self.assertEqual(packet["coverage"]["selected_windows"], 1)

    def test_oversized_evidence_is_not_sent_in_full_or_truncated(self):
        memory = add_window(empty_memory("V1", 20, "sha"), window_payload("observed " * 1000))
        with self.assertRaisesRegex(ValueError, "no complete window evidence fits"):
            retrieve(memory, "How does Taylor feel?", PLAN,
                     dense_scores={"W00001": 0.8}, budget_chars=2000)

    def test_missing_observation_reference_fails_explicitly(self):
        memory = add_window(empty_memory("V1", 20, "sha"), window_payload())
        memory["windows"][0]["observation_ids"].append("missing")
        with self.assertRaisesRegex(ValueError, "unknown observation"):
            retrieve(memory, "How does Taylor feel?", PLAN, dense_scores={"W00001": 0.8})

    def test_dense_route_cannot_silently_drop_missing_scores(self):
        memory = add_window(empty_memory("V1", 20, "sha"), window_payload())
        with self.assertRaisesRegex(ValueError, "missing/nonfinite"):
            retrieve(memory, "How does Taylor feel?", PLAN, dense_scores={})


if __name__ == "__main__":
    unittest.main()
