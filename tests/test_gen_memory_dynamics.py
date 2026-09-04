import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops

from vlm_diagnosis.scripts.gen_memory_dynamics import (
    DEFAULT_N_EPISODES,
    FACTOR_CELLS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INTERFERENCE_DISTRACTOR_COUNTS,
    RETRIEVAL_CANDIDATE_COUNTS,
    TEMPORAL_MEMORY_CONDITIONS,
    build_episode,
    generate_dataset,
    validate_manifest_row,
)


class MemoryDynamicsGeneratorTest(unittest.TestCase):
    def _generate(self, root: Path, *, seed: int = 17):
        return generate_dataset(
            root / "images",
            root / "manifest.jsonl",
            image_prefix="images",
            n_episodes=4,
            seed=seed,
        )

    def test_default_and_balanced_2x2_factorial(self):
        self.assertEqual(DEFAULT_N_EPISODES, 32)
        rows = [build_episode(index, 7, "images")[1] for index in range(4)]
        cells = Counter(row["factorial"]["cell_id"] for row in rows)
        self.assertEqual(
            cells,
            Counter(f"{time_gap}__{state_change}" for time_gap, state_change in FACTOR_CELLS),
        )
        for row in rows:
            validate_manifest_row(row)
            self.assertTrue(row["factorial"]["axes_are_orthogonal_by_construction"])
            self.assertEqual(len(row["memories"]), 5)
            by_role = {memory["role"]: memory for memory in row["memories"]}
            self.assertEqual(
                by_role["target_old"]["entity_id"],
                by_role["target_current"]["entity_id"],
            )
            self.assertNotEqual(
                by_role["target_old"]["revision_id"],
                by_role["target_current"]["revision_id"],
            )

    def test_queried_field_is_rotated_across_every_factor_cell(self):
        rows = [build_episode(index, 11, "images")[1] for index in range(16)]
        fields_by_cell = {}
        for row in rows:
            fields_by_cell.setdefault(row["factorial"]["cell_id"], set()).add(
                row["question"]["queried_field"]
            )
        self.assertEqual(set(fields_by_cell), {
            "near__unchanged", "near__changed", "far__unchanged", "far__changed"
        })
        self.assertTrue(all(fields == {"status", "priority", "owner", "region"}
                            for fields in fields_by_cell.values()))

    def test_changed_state_has_conflicting_fact_and_stale_label(self):
        _, changed = build_episode(1, 23, "images")
        _, unchanged = build_episode(0, 23, "images")

        changed_by_role = {memory["role"]: memory for memory in changed["memories"]}
        field = changed["question"]["queried_field"]
        old_value = changed_by_role["target_old"]["facts"][field]
        current_value = changed_by_role["target_current"]["facts"][field]
        self.assertNotEqual(old_value, current_value)
        self.assertEqual(changed["question"]["current_answer"], current_value)
        self.assertEqual(changed["question"]["stale_answers"], [old_value])
        self.assertEqual(
            changed["question"]["conflicting_evidence"][0]["evidence_label"],
            "stale_conflict",
        )

        unchanged_by_role = {memory["role"]: memory for memory in unchanged["memories"]}
        stable_field = unchanged["question"]["queried_field"]
        self.assertEqual(
            unchanged_by_role["target_old"]["facts"][stable_field],
            unchanged_by_role["target_current"]["facts"][stable_field],
        )
        self.assertEqual(unchanged["question"]["stale_answers"], [])
        self.assertEqual(unchanged["question"]["conflicting_evidence"], [])

    def test_d3_retrieval_and_interference_are_separate(self):
        _, row = build_episode(3, 29, "images")
        current_id = next(
            memory["memory_id"] for memory in row["memories"]
            if memory["role"] == "target_current"
        )
        old_id = next(
            memory["memory_id"] for memory in row["memories"]
            if memory["role"] == "target_old"
        )

        retrieval = row["d3_trials"]["retrieval"]
        self.assertEqual(
            [trial["candidate_count"] for trial in retrieval],
            list(RETRIEVAL_CANDIDATE_COUNTS),
        )
        for trial in retrieval:
            self.assertFalse(trial["retrieval_bypassed"])
            self.assertEqual(trial["oracle_task_memory_ids"], [current_id])
            self.assertEqual(trial["inference_distractors_in_oracle_score"], 0)
            self.assertEqual(trial["failure_taxonomy"], "stored_not_retrieved")

        interference = row["d3_trials"]["interference"]
        self.assertEqual(
            [trial["distractor_count"] for trial in interference],
            list(INTERFERENCE_DISTRACTOR_COUNTS),
        )
        for trial in interference:
            self.assertTrue(trial["retrieval_bypassed"])
            self.assertTrue(trial["relevant_always_included"])
            self.assertIn(current_id, trial["preselected_memory_ids"])
            self.assertNotIn(old_id, trial["preselected_memory_ids"])
            self.assertEqual(trial["failure_taxonomy"], "retrieved_not_used")

        composition = row["d3_trials"]["composition"]
        self.assertEqual(composition[0]["memory_ids"], composition[1]["memory_ids"])
        self.assertNotEqual(composition[0]["composition_mode"],
                            composition[1]["composition_mode"])

    def test_d4_crosses_memory_condition_with_episode_factors(self):
        _, row = build_episode(3, 31, "images")  # far + changed
        self.assertEqual(row["factorial"]["cell_id"], "far__changed")
        trials = row["d4_trials"]
        self.assertEqual(
            [trial["memory_condition"] for trial in trials],
            list(TEMPORAL_MEMORY_CONDITIONS),
        )
        by_condition = {trial["memory_condition"]: trial for trial in trials}
        self.assertTrue(by_condition["old+current"]["cross_revision_conflict_present"])
        self.assertTrue(by_condition["old_only"]["stale_capture_eligible"])
        self.assertFalse(by_condition["current_only"]["stale_capture_eligible"])
        self.assertEqual(by_condition["no_memory"]["memory_ids"], [])
        self.assertTrue(all(trial["time_gap"] == "far" for trial in trials))
        self.assertTrue(all(trial["state_change"] == "changed" for trial in trials))

    def test_generation_is_byte_deterministic_and_images_are_revision_pairs(self):
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
            self.assertEqual(
                json.loads((first / "manifest.meta.json").read_text(encoding="utf-8"))[
                    "factorial_cells"
                ],
                {
                    "far__changed": 1,
                    "far__unchanged": 1,
                    "near__changed": 1,
                    "near__unchanged": 1,
                },
            )

            for row in first_rows:
                by_role = {memory["role"]: memory for memory in row["memories"]}
                for memory in row["memories"]:
                    first_payload = (first / memory["image"]).read_bytes()
                    second_payload = (second / memory["image"]).read_bytes()
                    self.assertEqual(first_payload, second_payload)
                    self.assertEqual(memory["image_sha256"],
                                     hashlib.sha256(first_payload).hexdigest())
                    with Image.open(first / memory["image"]) as image:
                        self.assertEqual(image.size, (IMAGE_WIDTH, IMAGE_HEIGHT))

                with Image.open(first / by_role["target_old"]["image"]) as old_image, \
                     Image.open(first / by_role["target_current"]["image"]) as current_image:
                    self.assertIsNotNone(ImageChops.difference(old_image, current_image).getbbox())

    def test_rejects_unbalanced_size_and_unowned_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "multiple of four"):
                generate_dataset(
                    root / "images", root / "manifest.jsonl",
                    image_prefix="images", n_episodes=2,
                )

            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"dataset": "someone_else"}) + "\n",
                                encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_dataset(
                    root / "images", manifest, image_prefix="images",
                    n_episodes=4, overwrite=True,
                )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["dataset"],
                "someone_else",
            )


if __name__ == "__main__":
    unittest.main()
