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

    def test_known_cast_id_can_be_referenced_without_redeclaration(self):
        memory = add_window(empty_memory("V1", 40, "sha"), window_payload())
        original = copy.deepcopy(memory)
        payload = window_payload(subject="P1", span=(21, 24))
        payload["entities"] = []
        payload["observations"].append({"id": "o2", "subject": "P1", "span": [25, 26],
                                        "cue": "looks down", "modality": "visual"})
        result = add_window(memory, payload, index=2)
        self.assertEqual(memory, original)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["source_refs"], ["W00001", "W00002"])
        self.assertEqual(result["windows"][1]["participants"], ["P1"])
        self.assertEqual(result["windows"][1]["emotion_cues"][0]["subject"], "P1")
        self.assertEqual([row["subject"] for row in result["observations"]], ["P1", "P1", "P1"])

    def test_known_cast_id_may_also_be_declared_once(self):
        memory = add_window(empty_memory("V1", 40, "sha"), window_payload())
        result = add_window(memory, window_payload(subject="P1", span=(21, 24)), index=2)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(result["entities"][0]["source_refs"], ["W00001", "W00002"])

    def test_same_window_local_ids_map_consistently(self):
        memory = empty_memory("V1", 20, "sha")
        payload = window_payload()
        payload["entities"].append({"id": "new_2", "name": None, "description": "person in red"})
        payload["observations"].append({"id": "o2", "subject": "new_2", "span": [6, 8],
                                        "cue": "smiles while listening", "modality": "visual"})
        payload["participants"].append("new_2")
        payload["emotion_cues"].append({"subject": "new_2", "target": "the conversation", "emotion": "pleased",
                                        "intensity": "small smile", "evidence_ids": ["o2"]})
        result = add_window(memory, payload)
        self.assertEqual([row["subject"] for row in result["observations"]], ["P1", "P2"])
        self.assertEqual(result["windows"][0]["participants"], ["P1", "P2"])
        self.assertEqual(result["windows"][0]["emotion_cues"][1]["subject"], "P2")
        self.assertEqual(result["windows"][0]["emotion_cues"][1]["evidence_refs"], ["O2"])
        self.assertEqual([person["source_refs"] for person in result["entities"]], [["W00001"], ["W00001"]])

    def test_unknown_subject_and_guessed_canonical_id_are_still_rejected(self):
        for subject in ("P1", "P999", "Taylor", "undeclared", None, {"id": "new_1"}):
            with self.subTest(subject=subject):
                memory = empty_memory("V1", 20, "sha")
                payload = window_payload()
                payload["observations"][0]["subject"] = subject
                with self.assertRaisesRegex(ValueError, "observation subject"):
                    add_window(memory, payload)
                self.assertEqual(memory["entities"], [])
                self.assertEqual(memory["completed_windows"], [])

    def test_unknown_emotion_subject_is_still_rejected(self):
        memory = add_window(empty_memory("V1", 40, "sha"), window_payload())
        original = copy.deepcopy(memory)
        payload = window_payload(subject="P1", span=(21, 24))
        payload["entities"] = []
        payload["emotion_cues"][0]["subject"] = "P999"
        with self.assertRaisesRegex(ValueError, "emotion cue subject"):
            add_window(memory, payload, index=2)
        self.assertEqual(memory, original)

    def test_duplicate_new_and_existing_declarations_are_rejected(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                memory = empty_memory("V1", 40, "sha")
                if existing:
                    memory = add_window(memory, window_payload())
                original = copy.deepcopy(memory)
                payload = window_payload(subject="P1" if existing else "new_1", span=(21, 24) if existing else (1, 5))
                payload["entities"].append(copy.deepcopy(payload["entities"][0]))
                with self.assertRaisesRegex(ValueError, "duplicate local entity ID"):
                    add_window(memory, payload, index=2 if existing else 1)
                self.assertEqual(memory, original)

    def test_environment_only_window_needs_no_fabricated_person(self):
        memory = empty_memory("V1", 20, "sha")
        payload = {"entities": [], "observations": [], "participants": [], "emotion_cues": [],
                   "summary": "An empty room is visible while instrumental music plays.",
                   "actions": ["Instrumental music plays."], "objects": ["empty room"], "signals": []}
        result = add_window(memory, payload)
        self.assertEqual(result["completed_windows"], ["W00001"])
        self.assertEqual(result["entities"], [])
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["windows"][0]["observation_ids"], [])
        self.assertEqual(result["windows"][0]["summary"], payload["summary"])
        self.assertEqual(result["windows"][0]["actions"], payload["actions"])
        self.assertNotIn("events", result)


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
