import json
import tempfile
import unittest
from pathlib import Path

from vlm_diagnosis.exps.source_denial_kv import (
    assert_source_free,
    package_path,
    prepare_manifests,
    split_manifest_row,
)


class SourceDenialContractTest(unittest.TestCase):
    def test_split_hides_questions_from_writer_and_image_from_reader(self):
        row = {
            "dataset": "toy",
            "split": "test",
            "sample_id": "screen/1",
            "image": "data/private/screen.png",
            "image_sha256": "abc",
            "questions": [{
                "question_id": "q1",
                "question": "What changed?",
                "answers": ["on"],
            }],
        }
        write, read = split_manifest_row(row)
        self.assertNotIn("questions", write)
        self.assertNotIn("image", read)
        self.assertEqual(read["questions"][0]["question_id"], "q1")

    def test_source_bearing_read_key_is_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, "source-bearing key"):
            assert_source_free({"questions": [{"image_path": "secret.png"}]})

    def test_package_name_is_deterministic_and_path_safe(self):
        first = package_path(__import__("pathlib").Path("packages"), "a/b")
        second = package_path(__import__("pathlib").Path("packages"), "a/b")
        self.assertEqual(first, second)
        self.assertEqual(first.parent.name, "packages")
        self.assertNotIn("/", first.name)

    def test_prepare_materializes_declared_question_prefix(self):
        row = {
            "dataset": "toy",
            "split": "test",
            "sample_id": "screen-1",
            "image": "data/private/screen.png",
            "questions": [
                {"question_id": f"q{index}", "question": "q", "answers": ["a"]}
                for index in range(3)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            write = root / "write.jsonl"
            read = root / "read.jsonl"
            source.write_text(json.dumps(row) + "\n", encoding="utf-8")

            prepare_manifests(
                source,
                write,
                read,
                questions_per_image=2,
            )

            read_row = json.loads(read.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["question_id"] for item in read_row["questions"]],
                ["q0", "q1"],
            )
            self.assertEqual(
                read_row["question_selection"],
                {
                    "strategy": "manifest_order_prefix",
                    "requested_per_image": 2,
                    "source_question_count": 3,
                    "selected_question_count": 2,
                },
            )
            self.assertNotIn("image", read_row)


if __name__ == "__main__":
    unittest.main()
