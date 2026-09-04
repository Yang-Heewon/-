import json
import tempfile
import unittest
from pathlib import Path

from vlm_diagnosis.scripts.summarize_memory_frontier import (
    load_question_results,
    parse_named_path,
    summarize_arm,
)


class MemoryFrontierSummaryTest(unittest.TestCase):
    def row(self, sample, question, prediction, em, package_bytes):
        return {
            "sample_id": sample,
            "question_id": question,
            "prediction": prediction,
            "em": em,
            "anls": em,
            "package_bytes": package_bytes,
        }

    def test_alignment_retention_and_unique_memory_bytes(self):
        source = {
            ("s1", "q1"): self.row("s1", "q1", "Blue", 1, 100),
            ("s1", "q2"): self.row("s1", "q2", "wrong", 0, 100),
            ("s2", "q1"): self.row("s2", "q1", "$2", 1, 200),
        }
        arm = {
            ("s1", "q1"): self.row("s1", "q1", "blue", 1, 25),
            ("s1", "q2"): self.row("s1", "q2", "right", 1, 25),
            ("s2", "q1"): self.row("s2", "q1", "$3", 0, 50),
        }
        result = summarize_arm(source, arm)
        self.assertAlmostEqual(result["em"], 2 / 3)
        self.assertEqual(result["conditional_retention"], 0.5)
        self.assertEqual(result["prediction_agreement_exact"], 0.0)
        self.assertAlmostEqual(result["prediction_agreement_normalized"], 1 / 3)
        self.assertEqual(result["package"]["mean_bytes"], 37.5)
        self.assertEqual(result["package"]["mean_ratio_to_source_container"], 0.25)

    def test_duplicate_question_or_inconsistent_size_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.jsonl"
            row = self.row("s", "q", "a", 1, 10)
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_question_results(path)

        source = {("s", "q1"): self.row("s", "q1", "a", 1, 10)}
        inconsistent = {
            ("s", "q1"): self.row("s", "q1", "a", 1, 10),
            ("s", "q2"): self.row("s", "q2", "b", 1, 11),
        }
        with self.assertRaisesRegex(ValueError, "changes within sample"):
            summarize_arm(source, inconsistent)

    def test_named_path_requires_name_and_path(self):
        with self.assertRaisesRegex(ValueError, "NAME=PATH"):
            parse_named_path("missing")


if __name__ == "__main__":
    unittest.main()
