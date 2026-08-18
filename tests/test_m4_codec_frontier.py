import unittest

from vlm_diagnosis.exps.m4_codec_frontier import (
    question_metric_inputs,
    question_prompt,
    questions_from_row,
)


class M4CodecFrontierTest(unittest.TestCase):
    def test_coordinate_answers_route_to_grounding_metric(self):
        task, answers, bbox = question_metric_inputs({
            "primary_task_type": "grounding",
            "answer_type": "coordinate",
            "acceptable_answers": [],
            "target_bbox": [1, 2, 3, 4],
        })
        self.assertEqual(task, "grounding")
        self.assertEqual(answers, [])
        self.assertEqual(bbox, [1, 2, 3, 4])

    def test_text_answers_route_to_exact_match_family(self):
        task, answers, bbox = question_metric_inputs({
            "primary_task_type": "OCR",
            "answer_type": "text",
            "acceptable_answers": ["A7"],
        })
        self.assertEqual(task, "ocr")
        self.assertEqual(answers, ["A7"])
        self.assertIsNone(bbox)

    def test_coordinate_prompt_is_not_modified(self):
        question = {"question": "Return (x, y).", "answer_type": "coordinate"}
        self.assertEqual(question_prompt(question), question["question"])

    def test_t4_pilot_schema_preserves_ambiguous_ocr_label(self):
        questions = questions_from_row({
            "sample_id": "screen",
            "content_questions": [{
                "question_id": "q1", "question": "What text?", "answers": ["A"],
                "type_draft": "OCR/semantic",
            }],
            "location_questions": [{
                "question_id": "q2", "question": "Where?", "answers": ["top half"],
                "template": "half",
                "source_bbox": [1, 2, 3, 4],
            }],
        })
        self.assertEqual(questions[0]["primary_task_type"], "ocr_semantic_ambiguous")
        self.assertEqual(questions[1]["primary_task_type"], "layout")
        self.assertEqual(questions[1]["evidence_bboxes"], [[1, 2, 3, 4]])
        self.assertEqual(questions[1]["acceptable_answers"], ["top half", "top"])


if __name__ == "__main__":
    unittest.main()
