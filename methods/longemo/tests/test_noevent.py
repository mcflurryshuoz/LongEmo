import unittest

from methods.longemo.noevent_memory import apply_window, empty_memory


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


if __name__ == "__main__":
    unittest.main()
