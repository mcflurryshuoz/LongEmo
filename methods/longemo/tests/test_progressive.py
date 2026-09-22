import copy
import unittest

from methods.longemo.memory import empty_memory
from methods.longemo.progressive_retrieval import expand_disclosure, initial_disclosure
from methods.longemo.stream_index import attach_events, build_stream_index
from methods.longemo.tests.test_memory import commit


class ProgressiveRetrievalTests(unittest.TestCase):
    def setUp(self):
        memory = commit(empty_memory("video", 120, "hash"))
        for index, start, target in ((1, 30, "news"), (2, 60, "news"), (3, 90, "news")):
            event = copy.deepcopy(memory["events"][0])
            event["id"] = f"E{index + 1}"
            event["spans"] = [[start, start + 4]]
            event["summary"] = f"The woman remains concerned about the {target}"
            event["states"][0]["target"] = target
            memory["events"].append(event)
        memory["completed_windows"] = ["W1", "W2", "W3", "W4"]
        self.memory = memory
        self.index = attach_events(build_stream_index(memory), memory)
        self.plan = {"mode": "trajectory", "entity_terms": ["woman"],
                     "target_terms": ["news"], "query_terms": [], "time_range": None}
        self.dense = {"E1": 0.99, "E2": 0.2, "E3": 0.1, "E4": 0.05}

    def test_initial_disclosure_is_small_and_tracks_anchors(self):
        trace = {}
        evidence, state = initial_disclosure(self.memory, "How does the woman feel over time?",
            self.plan, dense_scores=self.dense, stream_index=self.index, anchor_k=1, trace=trace)
        self.assertLessEqual(len(evidence["events"]), 3)
        self.assertEqual(len(state["anchor_ids"]), 1)
        self.assertEqual(trace["algorithm"], "progressive_anchor_stream_v1")
        self.assertEqual(state["revealed_ids"], [event["id"] for event in evidence["events"]])
        self.assertTrue(evidence["coverage_map"])
        self.assertTrue(all("event_hints" in bucket for bucket in evidence["coverage_map"]))

    def test_expansion_returns_new_page_only(self):
        evidence, state = initial_disclosure(self.memory, "How does the woman feel over time?",
            self.plan, dense_scores=self.dense, stream_index=self.index, anchor_k=1)
        anchor = state["anchor_ids"][0]
        more, state = expand_disclosure(self.memory, state, {
            "anchor_ids": [anchor], "scope": "same_target", "direction": "after", "page_size": 1,
            "reason": "check the later state"}, stream_index=self.index)
        self.assertTrue(more["events"])
        self.assertTrue(set(more["coverage"]) >= {"page_events", "revealed_events"})
        self.assertNotIn(more["events"][0]["id"], evidence["events"])
        self.assertEqual(len(state["revealed_ids"]), len(evidence["events"]) + 1)

    def test_oversized_anchor_is_not_silently_dropped(self):
        self.memory["events"][0]["summary"] = "x" * 100000
        evidence, state = initial_disclosure(self.memory, "How does the woman feel?",
            self.plan, dense_scores=self.dense, stream_index=self.index, anchor_k=1,
            budget_chars=1000)
        self.assertEqual(len(evidence["events"]), 1)
        self.assertEqual(len(state["revealed_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
