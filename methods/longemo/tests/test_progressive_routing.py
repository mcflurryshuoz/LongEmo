import copy
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from evaluation.io_utils import write_json
from methods.longemo.memory import empty_memory
from methods.longemo.progressive_retrieval import initial_disclosure
from methods.longemo.retrieval import retrieve
from methods.longemo.runner import _answer_one, answer, parser
from methods.longemo.stream_index import attach_events, build_stream_index
from methods.longemo.tests.test_memory import commit


class StubClient:
    model = "mock"

    def __init__(self, mode="trajectory"):
        self.responses = [
            {"mode": mode, "entity_terms": ["woman"], "target_terms": ["news"],
             "query_terms": [], "time_range": None},
            {"answer": "She feels relief.", "evidence_ids": ["E1"], "uncertainty": "",
             "inspect": [], "retrieve_more": []},
        ]

    def configuration(self):
        return {"model": self.model}

    def generate(self, messages):
        return {"content": json.dumps(self.responses.pop(0)), "usage": {"total_tokens": 10}}


class StubIndex:
    metadata = {"signature": "fixed"}

    def __init__(self, *args):
        pass

    def rank(self, question):
        return {"E1": 0.9}


class ProgressiveRoutingTests(unittest.TestCase):
    def args(self, directory, policy=None):
        argv = ["answer", "--data-path", str(directory / "questions.json"),
                "--memory-dir", str(directory / "memory"), "--output-dir", str(directory / "answers"),
                "--model", "mock", "--retrieval", "progressive", "--tries", "1"]
        if policy is not None:
            argv += ["--progressive-routing", policy]
        return parser().parse_args(argv)

    def memory(self):
        memory = commit(empty_memory("video", 60, "hash"))
        memory.update(build_fingerprint="fixed-build", complete=True)
        return memory

    def test_default_preserves_task_routing_and_none_is_pure_for_all_tasks(self):
        tasks = {"emotion trajectory": "trajectory", "emotional intensity comparison": "comparison",
                 "emotional reasoning": "causal"}
        for policy in (None, "none"):
            for task, mode in tasks.items():
                with self.subTest(policy=policy, task=task), tempfile.TemporaryDirectory() as tmp:
                    directory = Path(tmp)
                    args = self.args(directory, policy)
                    memory = self.memory()
                    before = copy.deepcopy(memory)
                    question = {"question_id": "q", "video_id": "video", "granularity": "episode",
                                "type": task, "question": "How does the woman feel about the news?"}
                    index = attach_events(build_stream_index(memory), memory)
                    expected = "progressive" if policy == "none" or task == "emotion trajectory" else "graph"
                    with patch("methods.longemo.runner.initial_disclosure", wraps=initial_disclosure) as progressive, \
                         patch("methods.longemo.runner.retrieve", wraps=retrieve) as graph:
                        result = _answer_one(question, memory, args, StubClient(mode), directory,
                                             StubIndex(), index)
                    self.assertEqual(progressive.call_count, int(expected == "progressive"))
                    self.assertEqual(graph.call_count, int(expected == "graph"))
                    self.assertEqual(result["routing"]["effective_retrieval"], expected)
                    self.assertEqual(result["routing"]["progressive_routing"], policy or "task")
                    if policy == "none":
                        self.assertEqual(result["routing"]["hybrid_graph_tasks"], [])
                    trace = json.loads((directory / "traces/q.json").read_text())
                    self.assertEqual(trace["routing"], result["routing"])
                    self.assertEqual(memory, before)

    def test_route_is_fingerprinted_and_cannot_reuse_another_policy_output(self):
        # Exercise the real answer manifest with local deterministic encoders;
        # no NumPy dependency or external model/embedding request is needed.
        embeddings = ModuleType("methods.longemo.embeddings")

        class Encoder:
            config = {"model": "fixed-encoder"}

            def __init__(self, *args, **kwargs):
                pass

        embeddings.Encoder = embeddings.APIEncoder = Encoder
        embeddings.EventIndex = StubIndex
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_json(directory / "questions.json", [{"question_id": "q", "video_id": "video",
                "granularity": "episode", "type": "emotion trajectory", "question": "How does she feel?"}])
            write_json(directory / "memory/video/memory.json", self.memory())
            args = self.args(directory, "none")
            with patch.dict(sys.modules, {"methods.longemo.embeddings": embeddings}), \
                 patch("methods.longemo.runner.client_for", return_value=StubClient()):
                self.assertEqual(answer(args), 0)
                path = directory / "answers/manifest.json"
                first_manifest = json.loads(path.read_text())
                self.assertEqual(first_manifest["configuration"]["progressive_routing"], "none")
                self.assertEqual(first_manifest["configuration"]["hybrid_graph_tasks"], [])
                args.progressive_routing = "task"
                with self.assertRaisesRegex(ValueError, "configuration changed"):
                    answer(args)
                self.assertEqual(json.loads(path.read_text()), first_manifest)


if __name__ == "__main__":
    unittest.main()
