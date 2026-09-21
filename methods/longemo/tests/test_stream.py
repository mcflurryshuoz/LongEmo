import copy
import unittest

from methods.longemo.memory import empty_memory
from methods.longemo.stream_index import attach_events, build_stream_index, stream_event_ids
from methods.longemo.stream_retrieval import retrieve_stream
from methods.longemo.tests.test_memory import commit


class StreamRetrievalTests(unittest.TestCase):
    def setUp(self):
        memory = commit(empty_memory("video", 80, "hash"))
        event = copy.deepcopy(memory["events"][0])
        event["id"] = "E2"
        event["spans"] = [[30, 34]]
        event["summary"] = "The woman remains relieved about the news"
        event["states"][0]["target"] = "news"
        memory["events"].append(event)
        memory["completed_windows"] = ["W1", "W2"]
        self.memory = memory
        self.index = attach_events(build_stream_index(memory), memory)

    def test_stream_index_keeps_chronological_candidates(self):
        streams = list(self.index["streams"])
        self.assertTrue(streams)
        ids, total = stream_event_ids(self.index, streams, bounds=[0, 80], page=0, page_size=1)
        self.assertEqual(total, 2)
        self.assertEqual(ids, ["E1"])
        ids, _ = stream_event_ids(self.index, streams, bounds=[0, 80], page=1, page_size=1)
        self.assertEqual(ids, ["E2"])

    def test_retrieval_expands_seed_to_full_event_stream(self):
        plan = {"mode": "trajectory", "entity_terms": ["woman"], "target_terms": ["news"],
                "query_terms": [], "time_range": None}
        trace = {}
        result = retrieve_stream(self.memory, "How does the woman feel over time?", plan, top_k=1,
                                 dense_scores={"E1": 0.9, "E2": 0.1}, stream_index=self.index,
                                 page_size=1, trace=trace)
        self.assertEqual({event["id"] for event in result["events"]}, {"E1", "E2"})
        self.assertEqual(result["coverage"]["candidate_events"], 2)
        self.assertEqual(result["coverage"]["stream_events"], 2)
        self.assertEqual(trace["algorithm"], "semantic_dense_rrf_graph_stream_v1")
        self.assertEqual(trace["stream_pages"][0]["returned"], 1)

    def test_explicit_range_limits_stream(self):
        plan = {"mode": "local", "entity_terms": ["woman"], "target_terms": ["news"],
                "query_terms": [], "time_range": [25, 40]}
        result = retrieve_stream(self.memory, "What happens later?", plan, top_k=1,
                                 dense_scores={"E1": 0.9, "E2": 0.1}, stream_index=self.index)
        self.assertEqual([event["id"] for event in result["events"]], ["E2"])
        self.assertEqual(result["coverage"]["candidate_events"], 1)


if __name__ == "__main__":
    unittest.main()
