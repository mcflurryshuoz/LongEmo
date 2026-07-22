#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.common import contract  # noqa: E402
from benchmark.evaluation import emotion_grouping  # noqa: E402


def single_choice(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q001",
        "question_type": "single_choice",
        "question": "What emotion is most visible? Please select only one option. Respond with the option letter only.",
        "options": ["A. happy", "B. sad", "C. angry", "D. fearful"],
        "answer": "A",
    }
    q.update(overrides)
    return q


def multi_select(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q002",
        "question_type": "multi_select",
        "question": 'Choose the emotions shown. Please select all correct options. Respond with option letters only, such as "B D".',
        "options": ["A. angry", "B. happy", "C. calm", "D. surprised"],
        "answer": ["A", "D"],
    }
    q.update(overrides)
    return q


def ranking(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q003",
        "question_type": "ranking",
        "question": (
            "Choose the emotional stages she goes through, and order only the selected options chronologically. "
            'An option may be used more than once if the same emotion returns. Some options may be distractors. '
            'Respond with option letters only, such as "A B C".'
        ),
        "options": ["A. happy", "B. angry", "C. surprised", "D. sad"],
        "answer": ["A", "B"],
    }
    q.update(overrides)
    return q


def emotion(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q004",
        "question_type": "open_ended",
        "question": 'What is the emotion of the man? Answer with a JSON object like {"emotion":["..."]}.',
        "options": [],
        "answer": {"emotion": ["nervous"]},
        "emotion_answer": True,
    }
    q.update(overrides)
    return q


def transition(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q005",
        "question_type": "open_ended",
        "question": (
            "How does the emotion of the woman change? "
            'Answer with a JSON object like {"from_emotion":["..."],"to_emotion":["..."]}.'
        ),
        "options": [],
        "answer": {"from_emotion": ["calm"], "to_emotion": ["angry"]},
        "emotion_answer": True,
    }
    q.update(overrides)
    return q


def yes_no(**overrides):
    q = {
        "series": "friends",
        "qid": "g3_S01E01_q001",
        "question_type": "open_ended",
        "question": "By the end of the episode, has Ross gotten over the divorce? Answer Yes or No.",
        "options": [],
        "answer": "No",
    }
    q.update(overrides)
    return q


def open_qa(**overrides):
    q = {
        "series": "friends",
        "qid": "g1_2_S01E01_q006",
        "question_type": "open_ended",
        "question": "What visible action makes the boys react? Answer with one short phrase.",
        "options": [],
        "answer": "The wood chipper destroys the branch.",
    }
    q.update(overrides)
    return q


ALL_VALID = (single_choice, multi_select, ranking, emotion, transition, yes_no, open_qa)


class GoldFormatTest(unittest.TestCase):
    def test_valid_questions_pass(self):
        for make in ALL_VALID:
            self.assertEqual(contract.validate_gold_format(make()), [], make.__name__)

    def test_unknown_question_type_is_rejected(self):
        self.assertTrue(contract.validate_gold_format(single_choice(question_type="short_answer")))
        self.assertTrue(contract.validate_gold_format(single_choice(question_type=None)))

    def test_single_choice_gold_must_be_letter_in_options(self):
        self.assertTrue(contract.validate_gold_format(single_choice(answer="happy")))
        self.assertTrue(contract.validate_gold_format(single_choice(answer="E")))

    def test_choice_tails_are_required_verbatim(self):
        self.assertTrue(contract.validate_gold_format(single_choice(question="What emotion? Answer with one letter.")))
        self.assertTrue(contract.validate_gold_format(multi_select(question="Choose all. Respond with letters.")))

    def test_multi_select_gold_rules(self):
        self.assertTrue(contract.validate_gold_format(multi_select(answer=[])))
        self.assertTrue(contract.validate_gold_format(multi_select(answer=["A", "A"])))
        self.assertTrue(contract.validate_gold_format(multi_select(answer=["A", "E"])))
        self.assertTrue(contract.validate_gold_format(multi_select(answer="A D")))

    def test_ranking_allows_repeated_letters(self):
        self.assertEqual(contract.validate_gold_format(ranking(answer=["A", "B", "A"])), [])
        self.assertTrue(contract.validate_gold_format(ranking(answer=["A"])))

    def test_emotion_answer_flag_placement(self):
        self.assertTrue(contract.validate_gold_format(single_choice(emotion_answer=True)))
        self.assertTrue(contract.validate_gold_format(emotion(emotion_answer=False)))

    def test_emotion_gold_shapes(self):
        self.assertEqual(contract.validate_gold_format(emotion(answer={"emotion": ["sad", "angry"]})), [])
        self.assertTrue(contract.validate_gold_format(emotion(answer={"emotions": ["sad"]})))
        self.assertTrue(contract.validate_gold_format(emotion(answer={"emotion": []})))
        self.assertTrue(contract.validate_gold_format(emotion(answer="sad")))
        self.assertTrue(contract.validate_gold_format(transition(answer={"from_emotion": ["calm"]})))

    def test_open_question_must_not_carry_options(self):
        self.assertTrue(contract.validate_gold_format(open_qa(options=["A. x", "B. y", "C. z"])))

    def test_open_gold_must_be_string(self):
        self.assertTrue(contract.validate_gold_format(open_qa(answer=["phrase"])))

    def test_yes_no_gold_must_be_yes_or_no(self):
        self.assertEqual(contract.validate_gold_format(yes_no(answer="Yes")), [])
        self.assertTrue(contract.validate_gold_format(yes_no(answer="Not really")))

    def test_yes_no_parsing(self):
        self.assertTrue(contract.is_yes_no_question(yes_no()))
        self.assertFalse(contract.is_yes_no_question(open_qa()))
        self.assertEqual(contract.match_yes_no("No"), "no")
        self.assertEqual(contract.match_yes_no("No, he still misses her."), "no")
        self.assertIsNone(contract.match_yes_no("Yes and no"))
        self.assertIsNone(contract.match_yes_no("definitely"))


class LabelParsingTest(unittest.TestCase):
    def test_robust_labels(self):
        self.assertEqual(contract.robust_labels_from("B"), ["B"])
        self.assertEqual(contract.robust_labels_from("B D"), ["B", "D"])
        self.assertEqual(contract.robust_labels_from("b, d"), ["B", "D"])
        self.assertEqual(contract.robust_labels_from("Answer: B"), ["B"])
        self.assertEqual(contract.robust_labels_from({"answer": "C A D"}), ["C", "A", "D"])
        self.assertEqual(contract.robust_labels_from(["A", "C"]), ["A", "C"])
        self.assertEqual(contract.robust_labels_from(""), [])

    def test_emotion_terms(self):
        self.assertEqual(contract.emotion_terms({"emotion": ["nervous", "hopeful"]}), ["nervous", "hopeful"])
        self.assertEqual(contract.emotion_terms("embarrassed; hopeful"), ["embarrassed", "hopeful"])
        self.assertEqual(contract.emotion_terms("sad and angry"), ["sad", "angry"])
        self.assertEqual(contract.emotion_terms('{"emotion": ["hurt"]}'), ["hurt"])

    def test_parse_transition(self):
        self.assertEqual(contract.parse_transition("from anxious to hurt"), ("anxious", "hurt"))
        self.assertEqual(contract.parse_transition("calm -> angry"), ("calm", "angry"))
        self.assertIsNone(contract.parse_transition("just hurt"))


class EmotionSlotTest(unittest.TestCase):
    def test_direct_emotion_single_slot(self):
        slots = emotion_grouping.emotion_slots(emotion(), '{"emotion": ["anxious"]}')
        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["slot"], "emotion")
        self.assertEqual(slots[0]["gold_labels"], ["nervous"])
        self.assertEqual(slots[0]["pred_labels"], ["anxious"])

    def test_transition_two_slots(self):
        slots = emotion_grouping.emotion_slots(
            transition(), '{"from_emotion": ["calm"], "to_emotion": ["furious"]}'
        )
        self.assertEqual([s["slot"] for s in slots], ["from_emotion", "to_emotion"])
        self.assertEqual(slots[0]["pred_labels"], ["calm"])
        self.assertEqual(slots[1]["pred_labels"], ["furious"])

    def test_transition_free_text_fallback(self):
        slots = emotion_grouping.emotion_slots(transition(), "from calm to furious")
        self.assertEqual(slots[0]["pred_labels"], ["calm"])
        self.assertEqual(slots[1]["pred_labels"], ["furious"])

    def test_non_emotion_question_yields_no_slots(self):
        self.assertEqual(emotion_grouping.emotion_slots(open_qa(), "whatever"), [])


class PredictionFormatTest(unittest.TestCase):
    def test_clean_predictions_have_no_issues(self):
        self.assertEqual(contract.validate_prediction_format(single_choice(), "A"), [])
        self.assertEqual(contract.validate_prediction_format(multi_select(), "A D"), [])
        self.assertEqual(contract.validate_prediction_format(ranking(), "A B A"), [])
        self.assertEqual(contract.validate_prediction_format(emotion(), '{"emotion": ["sad"]}'), [])
        self.assertEqual(
            contract.validate_prediction_format(transition(), '{"from_emotion": ["calm"], "to_emotion": ["angry"]}'),
            [],
        )

    def test_yes_no_prediction_format(self):
        self.assertEqual(contract.validate_prediction_format(yes_no(), "No"), [])
        self.assertTrue(contract.validate_prediction_format(yes_no(), "No, he still misses her."))

    def test_format_issues_are_flagged(self):
        self.assertTrue(contract.validate_prediction_format(single_choice(), "the answer is A"))
        self.assertTrue(contract.validate_prediction_format(single_choice(), "E"))
        self.assertTrue(contract.validate_prediction_format(multi_select(), "A A"))
        self.assertTrue(contract.validate_prediction_format(emotion(), "sad"))
        self.assertTrue(contract.validate_prediction_format(transition(), "from calm to angry"))


if __name__ == "__main__":
    unittest.main()
