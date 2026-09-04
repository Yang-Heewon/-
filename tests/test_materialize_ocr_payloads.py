import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from vlm_diagnosis.exps import source_denial_text as text_eval
from vlm_diagnosis.scripts.frontier_paddleocr import _package_line
from vlm_diagnosis.scripts.materialize_ocr_payloads import materialize_packages


class MaterializeOCRPayloadsTest(unittest.TestCase):
    def _ocr_manifest(self, root: Path) -> Path:
        path = root / "frontier_ocr.jsonl"
        path.write_bytes(
            _package_line(
                {
                    "record_type": "ocr_memory_package",
                    "sample_id": "화면/one",
                    "recognized_text": "가나다abc",
                    "layout_text": "[0,0,10,10] 가나다abc",
                    "source_sha256": "a" * 64,
                    "source_path_stored": False,
                }
            )
        )
        return path

    def test_materialized_files_are_minimal_exact_and_budgeted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._ocr_manifest(root)
            manifest = root / "packages.jsonl"
            payload_dir = root / "payloads"

            materialize_packages(
                input_manifest=source,
                out_manifest=manifest,
                payload_dir=payload_dir,
                byte_cap=5,
            )

            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(records[0]["record_type"], "run_metadata")
            package = records[1]
            self.assertNotIn("recognized_text", package)
            self.assertNotIn("layout_text", package)
            self.assertFalse(package["source_path_stored"])
            for representation, descriptor in package["representations"].items():
                payload_path = root / descriptor["payload_relpath"]
                payload = payload_path.read_bytes()
                self.assertLessEqual(len(payload), 5)
                self.assertEqual(descriptor["payload_bytes"], len(payload))
                self.assertEqual(
                    descriptor["payload_sha256"], hashlib.sha256(payload).hexdigest()
                )
                payload.decode("utf-8")
                self.assertEqual(descriptor["file_count"], 1)
                self.assertTrue(descriptor["truncated"])
                self.assertEqual(descriptor["byte_cap"], 5)

            packages = text_eval.load_text_packages(manifest)
            selected = text_eval.select_memory(packages["화면/one"], "plain", 5)
            self.assertTrue(selected.available)
            self.assertEqual(selected.used_text, "가")
            self.assertEqual(selected.package_bytes, 3)
            self.assertEqual(selected.used_bytes, 3)
            self.assertTrue(selected.truncated)

    def test_reader_cap_cannot_hide_part_of_a_larger_materialized_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._ocr_manifest(root)
            manifest = root / "packages.jsonl"
            materialize_packages(
                input_manifest=source,
                out_manifest=manifest,
                payload_dir=root / "payloads",
                representations=("plain",),
                byte_cap=None,
            )
            package = text_eval.load_text_packages(manifest)["화면/one"]
            selected = text_eval.select_memory(package, "plain", 5)
            self.assertFalse(selected.available)
            self.assertEqual(selected.package_bytes, len("가나다abc".encode()))
            self.assertIsNone(selected.used_text)
            self.assertIsNone(selected.used_bytes)
            self.assertIn("over byte_cap", selected.reason)

    def test_tampered_payload_fails_size_or_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._ocr_manifest(root)
            manifest = root / "packages.jsonl"
            materialize_packages(
                input_manifest=source,
                out_manifest=manifest,
                payload_dir=root / "payloads",
                representations=("plain",),
            )
            package = json.loads(manifest.read_text().splitlines()[1])
            payload = root / package["representations"]["plain"]["payload_relpath"]
            original = payload.read_bytes()
            payload.write_bytes(b"X" * len(original))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                text_eval.load_text_packages(manifest)

    def test_materializer_rejects_payload_directory_outside_manifest_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._ocr_manifest(root)
            with self.assertRaisesRegex(ValueError, "contained"):
                materialize_packages(
                    input_manifest=source,
                    out_manifest=root / "manifest" / "packages.jsonl",
                    payload_dir=root / "sibling-payloads",
                )


if __name__ == "__main__":
    unittest.main()
