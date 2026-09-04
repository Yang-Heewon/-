import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image

from vlm_diagnosis.scripts.gen_action_proxy import (
    ACTION_TYPES,
    DEFAULT_N_EPISODES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    build_episode,
    generate_dataset,
    validate_manifest_row,
)


class ActionProxyGeneratorTest(unittest.TestCase):
    def _generate(self, root: Path, *, seed: int = 17):
        return generate_dataset(
            root / "images",
            root / "manifest.jsonl",
            image_prefix="images",
            n_episodes=6,
            seed=seed,
        )

    def test_defaults_balance_and_explicit_claim_scope(self):
        self.assertEqual(DEFAULT_N_EPISODES, 24)
        rows = [build_episode(index, 7, "images")[1] for index in range(6)]
        self.assertEqual(
            Counter(row["next_action"]["action_type"] for row in rows),
            Counter({"click": 2, "type": 2, "scroll": 2}),
        )
        for row in rows:
            validate_manifest_row(row)
            self.assertTrue(row["synthetic"])
            self.assertFalse(row["real_trajectory_claim_allowed"])
            self.assertIn("no real agent-trajectory", row["claim_scope"])
            self.assertEqual(
                [event["event_type"] for event in row["event_history"]],
                ["observation", "action", "observation"],
            )
            self.assertEqual(
                row["episode_order"]["next_action_sequence_index"],
                row["next_action"]["sequence_index"],
            )

    def test_targets_revisions_and_invalidated_action_are_consistent(self):
        for index in range(3):
            _, row = build_episode(index, 23, "images")
            old_observation, current_observation = row["observations"]
            old_elements = {
                element["element_id"]: element for element in old_observation["elements"]
            }
            current_elements = {
                element["element_id"]: element for element in current_observation["elements"]
            }

            previous = row["action_history"][0]
            invalidated = row["invalidated_actions"][0]
            next_action = row["next_action"]
            self.assertIn(previous["target_element_id"], old_elements)
            self.assertNotIn(previous["target_element_id"], current_elements)
            self.assertEqual(previous["action_id"], invalidated["action_id"])
            self.assertTrue(invalidated["must_not_be_replayed"])
            self.assertEqual(
                invalidated["valid_in_revision_id"], old_observation["revision_id"]
            )
            self.assertEqual(
                invalidated["invalid_in_revision_id"], current_observation["revision_id"]
            )

            self.assertIn(next_action["target_element_id"], current_elements)
            self.assertEqual(
                next_action["target_bbox"],
                current_elements[next_action["target_element_id"]]["bbox"],
            )
            self.assertEqual(
                next_action["supporting_old_observation_id"],
                old_observation["observation_id"],
            )
            self.assertEqual(
                [trial["condition"] for trial in row["offline_action_trials"]],
                ["ordered_history", "current_only_control"],
            )

    def test_each_action_label_uses_the_old_visual_cue(self):
        rows = [build_episode(index, 31, "images")[1] for index in range(12)]
        scroll_directions = []
        for row in rows:
            old_cue = row["observations"][0]["state_facts"]["saved_cue_value"]
            next_action = row["next_action"]
            action_type = next_action["action_type"]
            self.assertIn(action_type, ACTION_TYPES)
            if action_type == "click":
                self.assertEqual(next_action["action_label"], f"click:{old_cue}")
                self.assertEqual(next_action["memory_dependency"], "old_visual_action_label")
            elif action_type == "type":
                self.assertEqual(next_action["arguments"]["text"], old_cue)
                self.assertEqual(next_action["memory_dependency"], "old_visual_text_payload")
            else:
                direction = next_action["arguments"]["direction"]
                scroll_directions.append(direction)
                expected = "up" if old_cue in {"OVERVIEW", "PLANNING"} else "down"
                self.assertEqual(direction, expected)
                self.assertEqual(
                    next_action["memory_dependency"],
                    "old_visual_waypoint_plus_current_scroll_state",
                )
        self.assertEqual(set(scroll_directions), {"up", "down"})

    def test_generation_is_byte_and_manifest_deterministic(self):
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
            meta = json.loads((first / "manifest.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["action_type_counts"], {"click": 2, "scroll": 2, "type": 2})
            self.assertFalse(meta["real_trajectory_claim_allowed"])
            self.assertFalse(meta["interactive_environment"])

            for row in first_rows:
                for observation in row["observations"]:
                    first_payload = (first / observation["image"]).read_bytes()
                    second_payload = (second / observation["image"]).read_bytes()
                    self.assertEqual(first_payload, second_payload)
                    self.assertEqual(
                        observation["image_sha256"], hashlib.sha256(first_payload).hexdigest()
                    )
                    self.assertEqual(observation["source_image_bytes"], len(first_payload))
                    with Image.open(first / observation["image"]) as image:
                        self.assertEqual(image.size, (IMAGE_WIDTH, IMAGE_HEIGHT))

    def test_rejects_unbalanced_size_and_unowned_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "multiple of three"):
                generate_dataset(
                    root / "images",
                    root / "manifest.jsonl",
                    image_prefix="images",
                    n_episodes=4,
                )

            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"dataset": "someone_else"}) + "\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                generate_dataset(
                    root / "images",
                    manifest,
                    image_prefix="images",
                    n_episodes=3,
                    overwrite=True,
                )
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["dataset"],
                "someone_else",
            )


if __name__ == "__main__":
    unittest.main()
