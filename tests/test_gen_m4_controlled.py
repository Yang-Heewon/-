import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vlm_diagnosis.scripts.gen_m4_controlled import (
    DEFAULT_N_IMAGES,
    DOMAIN,
    IMAGE_SIZE,
    TASK_TYPES,
    generate_dataset,
    validate_manifest_row,
)


class M4ControlledGeneratorTest(unittest.TestCase):
    def _generate(self, root: Path, *, n_images: int = 2, seed: int = 19):
        return generate_dataset(
            root / "images",
            root / "manifest.jsonl",
            image_prefix="images",
            n_images=n_images,
            seed=seed,
        )

    def test_defaults_and_six_type_schema(self):
        self.assertEqual(DEFAULT_N_IMAGES, 24)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._generate(root, n_images=1)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            validate_manifest_row(row)
            self.assertTrue(row["synthetic"])
            self.assertEqual(row["domain"], DOMAIN)
            self.assertEqual([q["task_type"] for q in row["questions"]], list(TASK_TYPES))

            by_type = {q["task_type"]: q for q in row["questions"]}
            self.assertEqual(by_type["grounding"]["answer_type"], "coordinate")
            self.assertIsNotNone(by_type["grounding"]["target_bbox"])
            self.assertTrue(by_type["icon"]["requires_icon"])
            self.assertFalse(by_type["icon"]["requires_text"])
            self.assertTrue(by_type["semantic"]["requires_state"])
            self.assertFalse(by_type["semantic"]["requires_text"])
            self.assertTrue(by_type["count"]["requires_count"])
            self.assertEqual(
                len(by_type["count"]["evidence_bboxes"]),
                int(by_type["count"]["acceptable_answers"][0]),
            )
            with Image.open(root / row["image"]) as image:
                self.assertEqual(image.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_same_seed_is_byte_and_manifest_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            first_rows = self._generate(first)
            second_rows = self._generate(second)
            self.assertEqual(first_rows, second_rows)
            self.assertEqual(
                (first / "manifest.jsonl").read_bytes(),
                (second / "manifest.jsonl").read_bytes(),
            )
            for row in first_rows:
                first_payload = (first / row["image"]).read_bytes()
                second_payload = (second / row["image"]).read_bytes()
                self.assertEqual(hashlib.sha256(first_payload).digest(),
                                 hashlib.sha256(second_payload).digest())
                self.assertEqual(row["image_sha256"], hashlib.sha256(first_payload).hexdigest())

    def test_refuses_to_overwrite_unowned_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"dataset": "someone_else"}) + "\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_dataset(
                    root / "images",
                    manifest,
                    image_prefix="images",
                    n_images=1,
                    seed=1,
                    overwrite=True,
                )
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["dataset"],
                             "someone_else")


if __name__ == "__main__":
    unittest.main()
