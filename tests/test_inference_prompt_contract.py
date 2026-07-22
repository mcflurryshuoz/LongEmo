#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.inference import inference_utils  # noqa: E402


class ParsePredictionTest(unittest.TestCase):
    def test_code_fenced_json_is_parsed(self):
        raw = '```json\n{"reason": "the delivery softens", "answer": "B"}\n```'
        self.assertEqual(inference_utils.parse_prediction(raw), ("B", "the delivery softens"))

    def test_inline_think_block_is_stripped(self):
        raw = (
            '<think>The clip shows {"maybe": "irrelevant"} braces inside thinking.</think>\n'
            '{"reason": "voice cracks on the last word", "answer": {"emotion": ["hurt"]}}'
        )
        answer, reason = inference_utils.parse_prediction(raw)
        self.assertEqual(answer, {"emotion": ["hurt"]})
        self.assertEqual(reason, "voice cracks on the last word")


class InferencePromptContractTest(unittest.TestCase):
    def test_wrapper_contract_is_present_and_reason_first(self):
        prompt = inference_utils.build_prompt(
            {
                "qid": "q1",
                "question_type": "open_ended",
                "emotion_answer": True,
                "answer": {"from_emotion": ["anxious"], "to_emotion": ["hurt"]},
                "question": 'How does the person change emotionally? Answer with a JSON object like {"from_emotion":["..."],"to_emotion":["..."]}.',
            }
        )
        self.assertIn("Return ONLY a JSON object with exactly these keys in this order: reason, answer.", prompt)
        self.assertIn("Write reason first", prompt)
        self.assertIn("Put no explanation inside answer.", prompt)

    def test_question_tail_is_the_single_format_authority(self):
        prompt = inference_utils.build_prompt(
            {
                "qid": "q2",
                "question_type": "open_ended",
                "emotion_answer": True,
                "answer": {"emotion": ["embarrassed", "hopeful"]},
                "question": 'Which emotions are mixed in the person? Answer with a JSON object like {"emotion":["..."]}.',
            }
        )
        # The contract defers to the question's own final instruction instead of
        # restating per-type formats (which would duplicate the guide tails).
        self.assertIn("The question's final instruction specifies the answer format", prompt)
        self.assertIn('Answer with a JSON object like {"emotion":["..."]}.', prompt)
        self.assertNotIn("For mixed/open emotion sets", prompt)
        self.assertNotIn("For emotion transitions", prompt)

    def test_internal_metadata_is_not_leaked_to_the_model(self):
        prompt = inference_utils.build_prompt(
            {
                "qid": "q3",
                "question_type": "single_choice",
                "emotion_answer": True,
                "question": "Which emotion dominates? Please select only one option. Respond with the option letter only.",
                "options": ["A. joy", "B. anger"],
            }
        )
        self.assertNotIn("Question type:", prompt)
        self.assertNotIn("Task type:", prompt)
        self.assertNotIn("Scope:", prompt)
        self.assertNotIn("Direct emotion answer:", prompt)
        self.assertNotIn("T4-A", prompt)
        self.assertNotIn("season", prompt)

    def test_options_are_always_included_for_choice_questions(self):
        record = {
            "question_type": "single_choice",
            "question": "Are the signals consistent or conflicting? Please select only one option.",
            "options": ["A. consistent", "B. conflicting", "C. unclear"],
        }
        prompt = inference_utils.build_prompt(record)
        self.assertIn("A. consistent", prompt)
        self.assertIn("B. conflicting", prompt)
        self.assertIn("C. unclear", prompt)

    def test_with_transcript_flag_controls_subtitle_feeding(self):
        record = {
            "question_type": "open_ended",
            "question": "Which delivery detail matters most? Answer with one short phrase.",
            "input_transcript": [
                {"speaker": "Person A", "text": "I'm fine."},
                {"speaker": "Person B", "text": "Are you sure?"},
            ],
        }
        with_t = inference_utils.build_prompt(record, with_transcript=True)
        self.assertIn("Subtitles for the same video span", with_t)
        self.assertIn("no speaker labels", with_t)
        self.assertIn("I'm fine.", with_t)
        self.assertIn("Are you sure?", with_t)
        # speaker attribution must come from the video, not the subtitles
        self.assertNotIn("Person A", with_t)
        self.assertNotIn("Person B", with_t)
        without_t = inference_utils.build_prompt(record, with_transcript=False)
        self.assertNotIn("Subtitles for the same video span", without_t)
        self.assertNotIn("I'm fine.", without_t)


if __name__ == "__main__":
    unittest.main()
