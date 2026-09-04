import json
import tempfile
import unittest
from pathlib import Path

from vlm_diagnosis.exps.source_denial_raster import (
    condition_id,
    load_package_index,
    raster_package_path,
    resolve_memory_package,
)


class RasterSourceDenialContractTest(unittest.TestCase):
    def test_paths_are_stable_safe_and_budget_specific(self):
        root = Path("packages")
        first = raster_package_path(root, "../../screen 1", "jpeg", 32768)
        second = raster_package_path(root, "../../screen 1", "jpeg", 65536)
        self.assertEqual(first.parent, root)
        self.assertNotIn("..", first.name)
        self.assertNotEqual(first, second)
        self.assertEqual(first.suffix, ".jpeg")

    def test_condition_ids_do_not_hide_physical_budget(self):
        self.assertEqual(condition_id("copy", None), "SOURCE_CONTAINER_COPY")
        self.assertEqual(condition_id("avif", 32768), "AVIF@32768B")

    def test_package_index_filters_condition_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            rows = [
                {"record_type": "run_metadata"},
                {
                    "record_type": "package", "sample_id": "a",
                    "condition_id": "JPEG@32768B", "package": "p/a.jpeg",
                    "feasible": True,
                },
                {
                    "record_type": "package", "sample_id": "a",
                    "condition_id": "WEBP@32768B", "package": "p/a.webp",
                    "feasible": True,
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            index = load_package_index(path, "JPEG@32768B")
            self.assertEqual(list(index), ["a"])
            self.assertEqual(index["a"]["package"], "p/a.jpeg")

            with path.open("a") as handle:
                handle.write(json.dumps(rows[1]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate raster package"):
                load_package_index(path, "JPEG@32768B")

    def test_package_manifest_cannot_smuggle_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packages.jsonl"
            path.write_text(json.dumps({
                "record_type": "package", "sample_id": "a",
                "condition_id": "JPEG@32768B", "package": "p/a.jpeg",
                "image_path": "data/private.jpg", "feasible": True,
            }) + "\n")
            with self.assertRaisesRegex(ValueError, "source-bearing key"):
                load_package_index(path, "JPEG@32768B")

    def test_reader_refuses_package_alias_into_source_data(self):
        with self.assertRaisesRegex(ValueError, "source-data path"):
            resolve_memory_package("data/screenqa_pilot/private.jpg")


if __name__ == "__main__":
    unittest.main()
